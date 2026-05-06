#!/usr/bin/env python3
"""Python golden/error check for RMSNorm approximate rsqrt."""

from __future__ import annotations

import json

import numpy as np


HIDDEN = 1024
RSQRT_LUT_SIZE = 32
LUT_X = np.linspace(0.25, 4.0, RSQRT_LUT_SIZE)
LUT_Y = 1.0 / np.sqrt(LUT_X)


def rsqrt_init(x: float) -> float:
    x = min(max(x, 0.25), 4.0)
    idx = int(np.floor((x - 0.25) * ((RSQRT_LUT_SIZE - 1) / 3.75)))
    idx = min(max(idx, 0), RSQRT_LUT_SIZE - 1)
    return float(LUT_Y[idx])


def nr_step(x: float, y: float) -> float:
    return y * (1.5 - 0.5 * x * y * y)


def metrics(ref: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    diff = approx - ref
    eps = 1e-12
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "cosine": float(np.dot(ref, approx) / (np.linalg.norm(ref) * np.linalg.norm(approx) + eps)),
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(ref) + eps)),
        "non_finite": int((~np.isfinite(approx)).sum()),
    }


def main() -> int:
    i = np.arange(HIDDEN, dtype=np.float64)
    x = 0.00125 * ((i * 29) % 3201) - 2.0
    w = 0.9 + 0.00025 * ((i * 11) % 801)
    mean_sq = float(np.mean(x * x) + 1.0e-5)
    exact_inv = 1.0 / np.sqrt(mean_sq)
    y0 = rsqrt_init(mean_sq)
    y1 = nr_step(mean_sq, y0)
    y2 = nr_step(mean_sq, y1)
    ref = x * w * exact_inv
    rows = []
    for variant, y in (("nr1", y1), ("nr2", y2)):
        item = {
            "kernel": "rmsnorm_rsqrt",
            "variant": variant,
            "dtype": "fixed16x6_acc40x16",
            "shape": f"hidden{HIDDEN}",
            "rsqrt_lut_size": RSQRT_LUT_SIZE,
        }
        item.update(metrics(ref, x * w * y))
        rows.append(item)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
