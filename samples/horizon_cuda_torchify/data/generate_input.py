#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic input.npy for sample")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--output", type=str, default="input.npy", help="Output npy file name")
    args = parser.parse_args()

    np.random.seed(args.seed)
    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    y = np.random.randn(1, 3, 8, 8).astype(np.float32)

    out_path = pathlib.Path(__file__).resolve().parent / args.output
    np.save(out_path, {"x": x, "y": y})
    print(f"Saved input data to: {out_path}")


if __name__ == "__main__":
    main()
