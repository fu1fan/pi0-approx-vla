# Exact GELU HLS Baseline

Module-level baseline for the PWL GELU approximation. It uses the standard tanh
GELU formula with float math.

- pi0 mapping: VLM FFN and action expert MLP activation.
- Shape: `GELU_LEN=4096`.
- Data type: float32 input/output.
- Baseline role: optimization-before reference for `gelu_pwl`.

This benchmark validates a single activation vector kernel, not a full FFN.
