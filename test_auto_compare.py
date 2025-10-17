# test_auto_compare.py
import argparse, os, subprocess, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

def load_model_any(path):
    p = Path(path)
    if p.suffix in [".h5", ".keras"]:
        return tf.keras.models.load_model(str(p))
    raise ValueError("Use a .h5 or .keras file for Keras 3. Given: %s" % path)

def shape_wo_batch(shape):
    dims = list(shape)
    return dims[1:] if dims and dims[0] is None else dims[1:] if dims else []

def softmax(v):
    v = np.asarray(v, dtype=np.float32)
    m = np.max(v)
    ex = np.exp(v - m)
    return ex / np.sum(ex)

def is_prob_vector(y, tol=1e-3):
    y = np.asarray(y, dtype=np.float32)
    s = float(np.sum(y))
    return np.all(y >= -tol) and np.all(y <= 1+tol) and abs(s - 1.0) < 5e-3

def run_c(exe, in_path, out_path, in_size, out_size):
    cmd = [str(exe), str(in_path), str(out_path), str(in_size), str(out_size)]
    subprocess.run(cmd, check=True)
    return np.fromfile(out_path, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".h5 or .keras")
    ap.add_argument("--zeros", action="store_true", help="use all-zero input instead of random")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--exe", default="TensorflowToC/test_model/run_c_model")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--save-dir", default="TensorflowToC/test_model")
    args = ap.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        print("C runner not found at", exe, file=sys.stderr)
        sys.exit(2)

    model = load_model_any(args.model)
    in_dims = shape_wo_batch(tf.keras.backend.int_shape(model.inputs[0]))
    out_dims = shape_wo_batch(tf.keras.backend.int_shape(model.outputs[0]))
    assert len(in_dims) == 3, f"Expect 3D input [*,*,*]; got {in_dims}"
    in_size = int(np.prod(in_dims))
    out_size = int(np.prod(out_dims)) if out_dims else 1

    # Build TF input
    rng = np.random.RandomState(args.seed)
    if args.zeros:
        x = np.zeros((1, *in_dims), dtype=np.float32)
    else:
        x = rng.uniform(-1.0, 1.0, size=(1, *in_dims)).astype(np.float32)

    # TF forward
    tf_out = np.asarray(model.predict(x, verbose=0), dtype=np.float32).reshape(-1)
    tf_is_prob = is_prob_vector(tf_out)

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best = None
    perms = [
        (0,1,2),
        (0,2,1),
        (1,0,2),
        (1,2,0),
        (2,0,1),
        (2,1,0),
    ]

    trials = []
    for perm in perms:
        # Permute input before writing to C, but keep TF input as is
        # x has shape (1, d0, d1, d2). Permute the inner 3 dims.
        xp = x.transpose([0, perm[0]+1, perm[1]+1, perm[2]+1])
        in_path = save_dir / f"input_{perm}.bin"
        out_path = save_dir / f"c_output_{perm}.bin"
        xp.ravel().astype(np.float32).tofile(in_path)

        # Run C once
        c_raw = run_c(exe, in_path, out_path, in_size, out_size)

        # Compare both raw and softmaxed C
        for apply_sm in ([False, True] if tf_is_prob else [False]):
            c_cmp = softmax(c_raw) if apply_sm else c_raw
            if c_cmp.size != tf_out.size:
                continue
            abs_err = np.abs(c_cmp - tf_out)
            max_err = float(abs_err.max(initial=0.0))
            mean_err = float(abs_err.mean())
            rel_err = np.abs(abs_err / (np.abs(tf_out) + 1e-12))
            max_rel = float(rel_err.max(initial=0.0))
            ok = np.allclose(c_cmp, tf_out, atol=args.atol, rtol=args.rtol)
            trials.append({
                "perm": perm,
                "softmax": apply_sm,
                "max_abs": max_err,
                "mean_abs": mean_err,
                "max_rel": max_rel,
                "ok": ok,
                "first_tf": tf_out[:4].copy(),
                "first_c": c_cmp[:4].copy(),
            })

    trials.sort(key=lambda t: t["max_abs"])
    top = trials[0]
    print("\n=== Auto comparison report ===")
    print(f"TF output looks like probabilities: {tf_is_prob}")
    print(f"Best perm: {top['perm']}  C softmax applied: {top['softmax']}")
    print(f"max abs err: {top['max_abs']:.6g}  mean abs err: {top['mean_abs']:.6g}  max rel err: {top['max_rel']:.6g}")
    print(f"first 4 TF: {top['first_tf']}")
    print(f"first 4  C: {top['first_c']}")
    print("RESULT:", "PASS" if top["ok"] else "FAIL")

    # Also print a small leaderboard
    print("\nTop 6 trials by max abs error:")
    for t in trials[:6]:
        print(f"perm={t['perm']} softmax={t['softmax']}  max_abs={t['max_abs']:.6g}  mean_abs={t['mean_abs']:.6g}  ok={t['ok']}")

if __name__ == "__main__":
    main()
