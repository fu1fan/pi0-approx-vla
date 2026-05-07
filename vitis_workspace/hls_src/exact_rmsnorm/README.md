# Exact RMSNorm HLS Baseline

Module-level baseline for the approximate reciprocal-square-root RMSNorm kernel.
It computes mean square, `1 / sqrt(mean_square + eps)`, and the final scaled
normalization with float math.

- pi0 mapping: VLM and action expert RMSNorm.
- Shape: `RMSNORM_HIDDEN=1024`.
- Data type: float32 input/output.
- Baseline role: optimization-before reference for `rmsnorm_rsqrt`.

This benchmark validates one RMSNorm vector kernel, not a full Transformer.
