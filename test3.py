# test3.py
# Usage:
#   python3 test3.py
# Optional flags:
#   --model <path to SavedModel dir or .h5/.keras>
#   --seed 123
#   --atol 1e-5
#   --rtol 1e-4
#   --save-input TensorflowToC/test_model/input.bin
#   --save-tf-out TensorflowToC/test_model/tf_output.bin
#   --save-c-out TensorflowToC/test_model/c_output.bin
#   --skip-build to only run compare if you already compiled
#
# Notes:
# - Assumes generated C sources are in TensorflowToC/test_model/*.c and headers in TensorflowToC/.
# - If your C API function is not model_inference, change it in test_model.c and rebuild.

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Silence TF logs a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


def find_default_model():
    # Prefer the SavedModel under the concrete experiment path seen in your tree
    exp_saved = Path("experiments/20250923_211548_ultralightweightcnn_4classes_fff791a8/model")
    if exp_saved.exists():
        return str(exp_saved)
    # Fall back to common names
    for cand in ["model.keras", "model.h5", "experiments/latest/model", "experiments/model"]:
        p = Path(cand)
        if p.exists():
            return str(p)
    # As a last resort, pick any SavedModel dir with saved_model.pb
    for p in Path(".").rglob("saved_model.pb"):
        return str(p.parent)
    raise FileNotFoundError("Could not find a model. Provide --model <path>.")


def load_model_any(path):
    path = Path(path)
    if path.is_dir():
        # SavedModel
        return tf.keras.models.load_model(str(path))
    if path.suffix in [".h5", ".keras"]:
        return tf.keras.models.load_model(str(path))
    raise ValueError(f"Unsupported model path: {path}")


def shape_to_size(shape):
    # shape includes batch dim at index 0, usually None
    dims = list(shape)
    if dims[0] is None:
        dims = dims[1:]
    else:
        # enforce single batch
        dims = dims[1:] if len(dims) > 0 else []
    size = int(np.prod(dims)) if dims else 1
    return size, dims


def build_c_runner(skip_build=False):
    exe = Path("TensorflowToC/test_model/run_c_model")
    if skip_build and exe.exists():
        return str(exe)

    sources = []
    sources += glob.glob("TensorflowToC/test_model/*.c")
    # Include top-level generated sources if any
    sources += glob.glob("TensorflowToC/*.c")

    if not any(Path(s).name != "test_model.c" for s in sources):
        print("WARN: only found test_model.c. Make sure your generated C sources are in TensorflowToC/test_model/*.c")

    cmd = [
        "gcc",
        "-O3",
        "-ffast-math",
        "-std=c11",
        "-I",
        "TensorflowToC",
        "-I",
        "TensorflowToC/test_model",
        "TensorflowToC/test_model.c",
    ] + sources + [
        "-o",
        str(exe),
        "-lm",
    ]

    # Remove duplicate paths if we added test_model.c twice
    seen = set()
    dedup = []
    for x in cmd:
        key = (x, cmd.count(x)) if x.endswith(".c") else (x, None)
        if x.endswith(".c"):
            if x in seen:
                continue
            seen.add(x)
        dedup.append(x)
    cmd = dedup

    print("Compiling C runner...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Build failed. Fix compile errors in your generated C or headers.")
        sys.exit(e.returncode)

    if not exe.exists():
        print("Build did not produce the expected executable:", exe)
        sys.exit(1)
    return str(exe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--save-input", type=str, default="TensorflowToC/test_model/input.bin")
    parser.add_argument("--save-tf-out", type=str, default="TensorflowToC/test_model/tf_output.bin")
    parser.add_argument("--save-c-out", type=str, default="TensorflowToC/test_model/c_output.bin")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    model_path = args.model or find_default_model()
    print(f"Loading TF model from: {model_path}")
    model = load_model_any(model_path)

    # Resolve input and output sizes
    in_size, in_dims = shape_to_size(tf.keras.backend.int_shape(model.inputs[0]))
    out_size, out_dims = shape_to_size(tf.keras.backend.int_shape(model.outputs[0]))
    if in_size <= 0 or out_size <= 0:
        raise RuntimeError(f"Bad model shapes. input {in_dims} out {out_dims}")

    print(f"Model input dims (without batch): {in_dims}  size {in_size}")
    print(f"Model output dims (without batch): {out_dims}  size {out_size}")

    # Deterministic input
    rng = np.random.RandomState(args.seed)
    x = rng.uniform(low=-1.0, high=1.0, size=(1, *in_dims)).astype(np.float32)

    # Save flat input for C
    in_path = Path(args.save_input)
    in_path.parent.mkdir(parents=True, exist_ok=True)
    x.ravel().astype(np.float32).tofile(in_path)
    print(f"Wrote input to {in_path} ({in_size} float32)")

    # TF forward
    tf_out = model.predict(x, verbose=0)
    tf_out = np.asarray(tf_out, dtype=np.float32).reshape(-1)
    Path(args.save_tf_out).parent.mkdir(parents=True, exist_ok=True)
    tf_out.astype(np.float32).tofile(args.save_tf_out)
    print(f"Wrote TF output to {args.save_tf_out} ({tf_out.size} float32)")

    # Build and run C model
    exe = build_c_runner(skip_build=args.skip_build)
    c_out_path = Path(args.save_c_out)
    run_cmd = [
        exe,
        str(in_path),
        str(c_out_path),
        str(in_size),
        str(out_size),
    ]
    print("Running C model:", " ".join(run_cmd))
    subprocess.run(run_cmd, check=True)

    # Load C output
    c_out = np.fromfile(c_out_path, dtype=np.float32)
    if c_out.size != tf_out.size:
        print(f"Size mismatch. C {c_out.size} vs TF {tf_out.size}")
        sys.exit(2)

    def _is_prob_vector(y, tol=1e-3):
        y = np.asarray(y, dtype=np.float32)
        s = float(np.sum(y))
        return np.all(y >= -tol) and np.all(y <= 1 + tol) and abs(s - 1.0) < 5e-3

    def _softmax(v):
        v = np.asarray(v, dtype=np.float32)
        m = np.max(v)
        ex = np.exp(v - m)
        return ex / np.sum(ex)

    # If TF looks like probabilities, apply softmax to C logits
    applied_softmax = False
    if _is_prob_vector(tf_out):
        c_out = _softmax(c_out)
        applied_softmax = True

    # Compare
    abs_err = np.abs(c_out - tf_out)
    max_err = float(abs_err.max(initial=0.0))
    mean_err = float(abs_err.mean())
    rel_err = np.abs(abs_err / (np.abs(tf_out) + 1e-12))
    max_rel = float(rel_err.max(initial=0.0))

    rtol = args.rtol
    atol = args.atol
    ok = np.allclose(c_out, tf_out, rtol=rtol, atol=atol)

    print("\n=== Comparison ===")
    print(f"atol {atol}  rtol {rtol}  applied_softmax_on_C={applied_softmax}")
    print(f"max abs err: {max_err:.6g}")
    print(f"mean abs err: {mean_err:.6g}")
    print(f"max rel err: {max_rel:.6g}")
    print(f"first 8 TF: {tf_out[:8]}")
    print(f"first 8  C: {c_out[:8]}")
    print("RESULT:", "PASS" if ok else "FAIL")

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
