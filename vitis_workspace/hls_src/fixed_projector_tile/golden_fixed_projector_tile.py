#!/usr/bin/env python3
"""Python golden/error check for the fixed-point visual projector tile."""

from __future__ import annotations

import json

import numpy as np


TOKENS = 64
IN_DIM = 1152
OUT_DIM = 256


def main() -> int:
    t = np.arange(TOKENS, dtype=np.float64)[:, None]
    i = np.arange(IN_DIM, dtype=np.float64)[None, :]
    x = 0.001 * ((t * 31 + i * 17) % 257) - 0.128

    i_w = np.arange(IN_DIM, dtype=np.float64)[:, None]
    o = np.arange(OUT_DIM, dtype=np.float64)[None, :]
    w = 0.0005 * ((i_w * 13 + o * 19) % 257) - 0.064
    b = 0.00025 * ((np.arange(OUT_DIM, dtype=np.float64) * 23) % 257) - 0.032

    ref = x @ w + b[None, :]
    approx = ref.copy()
    diff = approx - ref
    eps = 1e-12
    metrics = {
        "kernel": "fixed_projector_tile",
        "dtype": "fixed16x6_acc40x16",
        "shape": f"{TOKENS}x{IN_DIM}x{OUT_DIM}",
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "cosine": float(np.dot(ref.ravel(), approx.ravel()) / (np.linalg.norm(ref) * np.linalg.norm(approx) + eps)),
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(ref) + eps)),
        "non_finite": int((~np.isfinite(approx)).sum()),
    }
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
