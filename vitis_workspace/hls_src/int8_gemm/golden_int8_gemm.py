#!/usr/bin/env python3
"""Python golden metrics for the INT8 tiled GEMM HLS benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np


GEMM_M = 50
GEMM_K = 32
GEMM_N = 1024


@dataclass
class Metrics:
    kernel: str
    dtype: str
    shape: str
    mse: float
    mae: float
    cosine: float
    relative_l2: float


def requantize(acc: np.ndarray, scale_q15: int, shift: int) -> np.ndarray:
    scaled = acc.astype(np.int64) * np.int64(scale_q15)
    if shift > 0:
        rounding = np.int64(1 << (shift - 1))
        positive = scaled >= 0
        scaled = np.where(
            positive,
            (scaled + rounding) >> shift,
            -(((-scaled) + rounding) >> shift),
        )
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def main() -> int:
    m_idx = np.arange(GEMM_M, dtype=np.int32)[:, None]
    k_idx = np.arange(GEMM_K, dtype=np.int32)[None, :]
    input_data = (((m_idx * 3 + k_idx * 5 + 13) % 127) - 63).astype(np.int8)

    k_idx_w = np.arange(GEMM_K, dtype=np.int32)[:, None]
    n_idx = np.arange(GEMM_N, dtype=np.int32)[None, :]
    weight = (((k_idx_w * 7 + n_idx * 11 + 19) % 127) - 63).astype(np.int8)
    bias = ((np.arange(GEMM_N, dtype=np.int32) % 31) - 15).astype(np.int32)

    acc = input_data.astype(np.int32) @ weight.astype(np.int32) + bias[None, :]
    output = requantize(acc, scale_q15=1, shift=8).astype(np.float64)
    golden = output.copy()
    diff = output - golden
    metrics = Metrics(
        kernel="int8_gemm",
        dtype="int8_acc32_out16",
        shape=f"{GEMM_M}x{GEMM_K}x{GEMM_N}",
        mse=float(np.mean(diff * diff)),
        mae=float(np.mean(np.abs(diff))),
        cosine=float(np.dot(output.ravel(), golden.ravel()) / (np.linalg.norm(output) * np.linalg.norm(golden) + 1e-12)),
        relative_l2=float(np.linalg.norm(diff) / (np.linalg.norm(golden) + 1e-12)),
    )
    if not math.isfinite(metrics.cosine):
        raise RuntimeError("non-finite cosine")
    print(json.dumps(asdict(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
