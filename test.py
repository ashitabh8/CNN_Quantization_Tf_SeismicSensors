#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Dict, Tuple, List, Optional
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa

# ------------- matching helpers -------------
PREFIXES = ("sequential_", "model_", "module_", "block_", "net_", "cnn_")

def normalize_key(k: str) -> str:
    s = k.lower()
    for p in PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
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

def match_c_name(tf_key: str, c_dict: Dict[str, np.ndarray]) -> Optional[str]:
    for cand in candidates(tf_key):
        if cand in c_dict:
            return cand
    n = normalize_key(tf_key)
    for k in c_dict:
        if normalize_key(k) == n:
            return k
    for k in c_dict:
        if k.endswith(tf_key) or normalize_key(k).endswith(n):
            return k
    for k in c_dict:
        if tf_key in k or n in normalize_key(k):
            return k
    return None

# ------------- loaders -------------
def load_h5_weights(model_path: str) -> Dict[str, np.ndarray]:
    model = tf.keras.models.load_model(model_path, compile=False)
    out: Dict[str, np.ndarray] = {}
    for lyr in model.layers:
        ws = lyr.get_weights()
        if not ws:
            continue
        n = lyr.name
        if isinstance(lyr, tf.keras.layers.Conv2D):
            out[f"{n}_kernel"] = ws[0].astype(np.float32, copy=False).reshape(-1)
            if lyr.use_bias and len(ws) > 1:
                out[f"{n}_bias"] = ws[1].astype(np.float32, copy=False).reshape(-1)
        elif isinstance(lyr, tf.keras.layers.Dense):
            out[f"{n}_W"] = ws[0].astype(np.float32, copy=False).reshape(-1)
            if len(ws) > 1:
                out[f"{n}_b"] = ws[1].astype(np.float32, copy=False).reshape(-1)
    return out

def parse_c_arrays(header_path: str) -> Dict[str, np.ndarray]:
    txt = open(header_path, "r").read()
    pat = re.compile(r"static\s+const\s+float\s+(\w+)\s*\[\s*\d+\s*\]\s*=\s*\{(.*?)\};", re.S)
    arrays: Dict[str, np.ndarray] = {}
    for name, body in pat.findall(txt):
        cleaned = body.replace("{", " ").replace("}", " ").replace("\n", " ").replace("\r", " ")
        toks = [t for t in cleaned.split(",") if t.strip()]
        try:
            vals = np.array([float(t) for t in toks], dtype=np.float32)
        except ValueError as e:
            print(f"Failed to parse array {name}: {e}", file=sys.stderr)
            continue
        arrays[name] = vals
    return arrays

# ------------- metrics -------------
def metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float, float]:
    if a.size != b.size:
        return np.inf, np.inf, np.inf, np.inf, -1.0
    d = a - b
    max_abs = float(np.max(np.abs(d)))
    mean_abs = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    denom = float(np.sqrt(np.mean(a * a)) + 1e-12)
    rrmse = rmse / denom
    # cosine similarity
    da = float(np.dot(a, a)) ** 0.5 + 1e-20
    db = float(np.dot(b, b)) ** 0.5 + 1e-20
    cos = float(np.dot(a, b) / (da * db))
    return max_abs, mean_abs, rmse, rrmse, cos

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="Compare .h5 Keras weights against C header arrays with detailed error metrics.")
    ap.add_argument("--model", required=True, help="Path to .h5 Keras model")
    ap.add_argument("--header", required=True, help="Path to generated C header")
    ap.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for allclose and pass/fail")
    ap.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance (not usually needed for weights)")
    ap.add_argument("--cos_min", type=float, default=0.999999, help="Minimum cosine similarity to pass")
    ap.add_argument("--quiet", action="store_true", help="Only print summary")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model .h5 not found: {args.model}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.header):
        print(f"Header not found: {args.header}", file=sys.stderr)
        sys.exit(2)

    tf_w = load_h5_weights(args.model)
    c_w = parse_c_arrays(args.header)

    if not tf_w:
        print("No weights found in .h5", file=sys.stderr)
        sys.exit(2)
    if not c_w:
        print("No arrays found in header", file=sys.stderr)
        sys.exit(2)

    total_ok = 0
    total_bad = 0
    worst = {"name": "", "max_abs": 0.0, "rmse": 0.0, "rrmse": 0.0, "cos": 1.0}

    if not args.quiet:
        print(f"Comparing {os.path.abspath(args.model)} vs {os.path.abspath(args.header)}")
        print("name".ljust(26), "max_abs", "mean_abs", "rmse", "rrmse", "cosine")

    for tf_name, a in tf_w.items():
        c_name = match_c_name(tf_name, c_w)
        if not c_name:
            if not args.quiet:
                print(f"{tf_name.ljust(26)} MISSING in header")
            total_bad += 1
            continue
        b = c_w[c_name]
        max_abs, mean_abs, rmse, rrmse, cos = metrics(a, b)
        passed = (np.allclose(a, b, atol=args.atol, rtol=args.rtol) and cos >= args.cos_min)
        if not args.quiet:
            print(f"{tf_name.ljust(26)} {max_abs:.3e} {mean_abs:.3e} {rmse:.3e} {rrmse:.3e} {cos:.8f} {'OK' if passed else 'DIFF'}")
        if passed:
            total_ok += 1
        else:
            total_bad += 1
        # track worst by rmse
        if rmse > worst["rmse"]:
            worst = {"name": tf_name, "max_abs": max_abs, "rmse": rmse, "rrmse": rrmse, "cos": cos}

    print("\nSummary")
    print(f"OK = {total_ok}  DIFF = {total_bad}")
    if worst["name"]:
        print(f"Worst layer: {worst['name']} | max_abs={worst['max_abs']:.3e} rmse={worst['rmse']:.3e} rrmse={worst['rrmse']:.3e} cos={worst['cos']:.8f}")

    # CI-friendly exit
    sys.exit(0 if total_bad == 0 else 1)

if __name__ == "__main__":
    main()
