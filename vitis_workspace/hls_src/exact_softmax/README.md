# Exact Softmax HLS Baseline

Module-level baseline for the LUT softmax approximation. It computes row-wise
softmax with max subtraction, float `exp`, accumulation, and division.

- pi0 mapping: VLM/action expert attention score normalization.
- Shape: `SOFTMAX_ROWS=4`, `SOFTMAX_LEN=128`.
- Data type: float32 input/output.
- Baseline role: optimization-before reference for `lut_softmax`.

This is intentionally a small HLS kernel benchmark, not a full attention block
or full pi0 deployment.
