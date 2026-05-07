#!/usr/bin/env python3
"""Python golden/error check for exact float RMSNorm baseline."""

from __future__ import annotations

import json

import numpy as np


HIDDEN = 1024


def main() -> int:
    i = np.arange(HIDDEN, dtype=np.float64)
    x = 0.00125 * ((i * 29) % 3201) - 2.0
    w = 0.9 + 0.00025 * ((i * 11) % 801)
    x_f32 = x.astype(np.float32)
    w_f32 = w.astype(np.float32)

    ref_inv = 1.0 / np.sqrt(np.mean(x_f32.astype(np.float64) * x_f32.astype(np.float64)) + 1.0e-5)
    ref = x_f32.astype(np.float64) * w_f32.astype(np.float64) * ref_inv

    mean_sq_f32 = (np.mean(x_f32 * x_f32).astype(np.float32) + np.float32(1.0e-5)).astype(np.float32)
    inv_f32 = (np.float32(1.0) / np.sqrt(mean_sq_f32).astype(np.float32)).astype(np.float32)
    approx = (x_f32 * w_f32 * inv_f32).astype(np.float32)

    diff = approx.astype(np.float64) - ref
    eps = 1e-12
    metrics = {
        "kernel": "exact_rmsnorm",
        "variant": "sqrt",
        "dtype": "float32",
        "shape": f"hidden{HIDDEN}",
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
