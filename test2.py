#!/usr/bin/env python3
import os, re, sys, hashlib
from typing import List, Tuple, Dict, Optional
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import tensorflow as tf

HEADER = "TensorflowToC/model_inference.h"
TOL = 1e-6
PREFIXES = ("sequential_","model_","module_","block_","net_","cnn_")

# ---------- discovery ----------
def discover_models(root="experiments") -> List[Tuple[float,str,str]]:
    found: List[Tuple[float,str,str]] = []
    if not os.path.isdir(root):
        return found
    for base, _, files in os.walk(root):
        if "saved_model.pb" in files:
            p = os.path.join(base)
            try: found.append((os.path.getmtime(os.path.join(base,"saved_model.pb")), p, "savedmodel"))
            except OSError: pass
        for fn in files:
            if fn in ("model.h5","model.keras"):
                p = os.path.join(base, fn)
                try: found.append((os.path.getmtime(p), p, "keras"))
                except OSError: pass
    return sorted(found, key=lambda x: x[0], reverse=True)

# ---------- parsing / matching ----------
def parse_c_arrays(header_path: str) -> Dict[str, np.ndarray]:
    txt = open(header_path,"r").read()
    pat = re.compile(r"static\s+const\s+float\s+(\w+)\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\};", re.S)
    out = {}
    for name, body in pat.findall(txt):
        cleaned = body.replace("{"," ").replace("}"," ").replace("\n"," ").replace("\r"," ")
        toks = [t for t in cleaned.split(",") if t.strip()]
        try:
            out[name] = np.array([float(t) for t in toks], dtype=np.float32)
        except ValueError as e:
            print(f"Failed to parse array {name}: {e}", file=sys.stderr)
    return out

def collect_tf(model_path: str) -> Dict[str, np.ndarray]:
    m = tf.keras.models.load_model(model_path, compile=False)
    out = {}
    for lyr in m.layers:
        ws = lyr.get_weights()
        if not ws: continue
        n = lyr.name
        if isinstance(lyr, tf.keras.layers.Conv2D):
            out[f"{n}_kernel"] = ws[0].astype(np.float32).reshape(-1)
            if lyr.use_bias and len(ws)>1: out[f"{n}_bias"] = ws[1].astype(np.float32).reshape(-1)
        elif isinstance(lyr, tf.keras.layers.Dense):
            out[f"{n}_W"] = ws[0].astype(np.float32).reshape(-1)
            if len(ws)>1: out[f"{n}_b"] = ws[1].astype(np.float32).reshape(-1)
    return out

def normalize_key(k: str) -> str:
    s = k.lower()
    for p in PREFIXES:
        if s.startswith(p): s = s[len(p):]
    return s

def candidates(tf_key: str) -> List[str]:
    alts = {tf_key, f"sequential_{tf_key}"}
    if tf_key.endswith("_kernel"):
        w = tf_key[:-7] + "W"; alts |= {w, f"sequential_{w}"}
    if tf_key.endswith("_W"):
        k = tf_key[:-2] + "kernel"; alts |= {k, f"sequential_{k}"}
    if tf_key.endswith("_bias"):
        b = tf_key[:-5] + "b"; alts |= {b, f"sequential_{b}"}
    if tf_key.endswith("_b"):
        bb = tf_key[:-1] + "ias"; alts |= {bb, f"sequential_{bb}"}
    return list(alts)

def pick_c_name(tf_key: str, c_dict: Dict[str,np.ndarray]) -> Optional[str]:
    for cand in candidates(tf_key):
        if cand in c_dict: return cand
    n = normalize_key(tf_key)
    for k in c_dict:
        if normalize_key(k) == n: return k
    for k in c_dict:
        if k.endswith(tf_key) or normalize_key(k).endswith(n): return k
    for k in c_dict:
        if tf_key in k or n in normalize_key(k): return k
    return None

# ---------- compare ----------
def cmp_vec(a,b):
    if a.shape[0] != b.shape[0]:
        return False, f"size mismatch TF {a.shape[0]} vs C {b.shape[0]}"
    if np.allclose(a,b,atol=TOL): return True, ""
    diff = np.abs(a-b); i = int(np.argmax(diff))
    return False, f"maxΔ={float(diff[i]):.3e} meanΔ={float(diff.mean()):.3e} idx={i}"

# ---------- main ----------
def main():
    print("Working dir:", os.path.abspath(os.getcwd()))
    print("Header    :", os.path.abspath(HEADER), "[exists]" if os.path.exists(HEADER) else "[missing]")

    models = discover_models("experiments")
    if not models:
        print("No model exports found anywhere under ./experiments.", file=sys.stderr)
        print("Run this to list what exists:\n  find experiments -type f -name 'model.*' -o -name 'saved_model.pb'")
        sys.exit(2)

    print("\nDiscovered model exports (newest first):")
    for mtime, path, kind in models[:10]:
        print(" ", kind.ljust(10), os.path.abspath(path))

    # choose newest
    _, model_path, kind = models[0]
    print("\nUsing model:", os.path.abspath(model_path))

    if not os.path.exists(HEADER):
        print("Header not found. Generate it, then rerun.", file=sys.stderr)
        sys.exit(2)

    # Load
    c_w = parse_c_arrays(HEADER)
    tf_w = collect_tf(model_path)

    print(f"\nCounts -> TF tensors: {len(tf_w)}  |  C arrays: {len(c_w)}")

    ok = bad = 0
    for tf_name, tf_vec in tf_w.items():
        c_name = pick_c_name(tf_name, c_w)
        if not c_name:
            print(f"MISS  {tf_name}  -> not found in header")
            bad += 1
            continue
        same, msg = cmp_vec(tf_vec, c_w[c_name])
        if same:
            print(f"OK    {tf_name}  ==  {c_name}  ({tf_vec.size})")
            ok += 1
        else:
            print(f"DIFF  {tf_name}  vs  {c_name}: {msg}")
            bad += 1

    print("\nSummary: OK =", ok, " | DIFF =", bad)
    sys.exit(0 if bad == 0 else 1)

if __name__ == "__main__":
    main()
