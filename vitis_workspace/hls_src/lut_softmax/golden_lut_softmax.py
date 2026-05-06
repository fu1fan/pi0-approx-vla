#!/usr/bin/env python3
"""Python golden/error check for LUT softmax."""

from __future__ import annotations

import json

import numpy as np


ROWS = 4
LENGTH = 128
LUT_SIZE = 64
EXP_LUT = np.exp(np.linspace(-8.0, 0.0, LUT_SIZE))


def lut_exp(x: np.ndarray) -> np.ndarray:
    clamped = np.clip(x, -8.0, 0.0)
    idx = np.floor((clamped + 8.0) * ((LUT_SIZE - 1) / 8.0)).astype(np.int64)
    return EXP_LUT[np.clip(idx, 0, LUT_SIZE - 1)]


def main() -> int:
    r = np.arange(ROWS)[:, None]
    c = np.arange(LENGTH)[None, :]
    scores = 0.03125 * ((r * 17 + c * 13) % 257).astype(np.float64) - 4.0
    shifted = scores - np.max(scores, axis=1, keepdims=True)

    exact = np.exp(shifted)
    exact /= np.sum(exact, axis=1, keepdims=True)

    approx = lut_exp(shifted)
    approx /= np.sum(approx, axis=1, keepdims=True)

    diff = approx - exact
    eps = 1e-12
    metrics = {
        "kernel": "lut_softmax",
        "dtype": "fixed16x6_prob18x2",
        "shape": f"rows{ROWS}_len{LENGTH}",
        "lut_size": LUT_SIZE,
        "clamp": [-8.0, 0.0],
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "kl": float(np.mean(np.sum(exact * np.log((exact + eps) / (approx + eps)), axis=1))),
        "cosine": float(np.dot(exact.ravel(), approx.ravel()) / (np.linalg.norm(exact) * np.linalg.norm(approx) + eps)),
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(exact) + eps)),
        "row_sum_max_abs_err": float(np.max(np.abs(np.sum(approx, axis=1) - 1.0))),
        "non_finite": int((~np.isfinite(approx)).sum()),
    }
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["non_finite"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
