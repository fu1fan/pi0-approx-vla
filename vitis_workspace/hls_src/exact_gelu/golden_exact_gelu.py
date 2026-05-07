#!/usr/bin/env python3
"""Python golden/error check for exact tanh GELU baseline."""

from __future__ import annotations

import json

import numpy as np


GELU_LEN = 4096


def exact_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def main() -> int:
    idx = np.arange(GELU_LEN, dtype=np.float64)
    x = 0.0025 * ((idx * 37) % 4001) - 5.0
    x_f32 = x.astype(np.float32)
    ref = exact_gelu(x_f32.astype(np.float64))
    approx = exact_gelu(x_f32).astype(np.float32)
    diff = approx.astype(np.float64) - ref
    eps = 1e-12
    metrics = {
        "kernel": "exact_gelu",
        "variant": "tanh",
        "dtype": "float32",
        "shape": f"len{GELU_LEN}",
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "cosine": float(np.dot(ref, approx) / (np.linalg.norm(ref) * np.linalg.norm(approx) + eps)),
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(ref) + eps)),
        "non_finite": int((~np.isfinite(approx)).sum()),
    }
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["non_finite"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
