#!/usr/bin/env python3
"""Python golden/error check for PWL GELU."""

from __future__ import annotations

import json

import numpy as np


GELU_LEN = 4096
X_BREAKS = np.linspace(-4.0, 4.0, 17)
Y_BREAKS = 0.5 * X_BREAKS * (
    1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (X_BREAKS + 0.044715 * X_BREAKS**3))
)


def exact_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def pwl_gelu(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -4.0, 4.0)
    approx = np.interp(clipped, X_BREAKS, Y_BREAKS)
    approx = np.where(x <= -4.0, 0.0, approx)
    approx = np.where(x >= 4.0, x, approx)
    return approx


def main() -> int:
    idx = np.arange(GELU_LEN, dtype=np.float64)
    x = 0.0025 * ((idx * 37) % 4001) - 5.0
    ref = exact_gelu(x)
    approx = pwl_gelu(x)
    diff = approx - ref
    eps = 1e-12
    metrics = {
        "kernel": "gelu_pwl",
        "dtype": "fixed16x6",
        "shape": f"len{GELU_LEN}",
        "segments": 16,
        "range": [-4.0, 4.0],
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
