#!/usr/bin/env python3
"""Python golden/error check for exact float softmax baseline."""

from __future__ import annotations

import json

import numpy as np


ROWS = 4
LENGTH = 128


def main() -> int:
    r = np.arange(ROWS)[:, None]
    c = np.arange(LENGTH)[None, :]
    scores = 0.03125 * ((r * 17 + c * 13) % 257).astype(np.float64) - 4.0
    scores_f32 = scores.astype(np.float32)

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exact = np.exp(shifted)
    exact /= np.sum(exact, axis=1, keepdims=True)

    shifted_f32 = (scores_f32 - np.max(scores_f32, axis=1, keepdims=True)).astype(np.float32)
    exp_f32 = np.exp(shifted_f32).astype(np.float32)
    approx = (exp_f32 / np.sum(exp_f32, axis=1, keepdims=True)).astype(np.float32)

    diff = approx.astype(np.float64) - exact
    eps = 1e-12
    metrics = {
        "kernel": "exact_softmax",
        "variant": "exp",
        "dtype": "float32",
        "shape": f"rows{ROWS}_len{LENGTH}",
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
