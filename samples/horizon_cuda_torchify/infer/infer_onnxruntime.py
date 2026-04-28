#!/usr/bin/env python3
import pathlib

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "input.npy"
CUDA_ONNX = ROOT / "export" / "model_cuda.onnx"
TORCH_ONNX = ROOT / "export" / "model_torch.onnx"
OUT_DIR = ROOT / "infer"


def run_onnx(model_path: pathlib.Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out = sess.run(["output"], {"x": x, "y": y})[0]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {DATA_PATH}. Run `python3 data/generate_input.py` first."
        )

    data = np.load(DATA_PATH, allow_pickle=True).item()
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)

    out_cuda = run_onnx(CUDA_ONNX, x, y)
    out_torch = run_onnx(TORCH_ONNX, x, y)

    np.save(OUT_DIR / "output_cuda.npy", out_cuda)
    np.save(OUT_DIR / "output_torch.npy", out_torch)

    print("CUDA ONNX output saved to infer/output_cuda.npy")
    print("Torch ONNX output saved to infer/output_torch.npy")


if __name__ == "__main__":
    main()
