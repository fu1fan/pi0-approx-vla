# pi0 Real-weight Simplification Summary

This stage is intended to repeat random-input function simplification experiments using real pi0 weights.

Current run status: skipped real-weight numeric benchmarks.

Reason: verified local pi0 tensors were not available after Stage 3. The single 13.04 GiB `model.safetensors` was not downloaded under the configured 2 GiB safety limit, so real FFN/QK/RMSNorm weights could not be loaded.

No metrics were fabricated. If a verified checkpoint is downloaded outside git tracking, re-run extraction first and then this script.

Outputs:
- `results/csv/pi0_real_weight_simplify.csv`
- `results/figures/pi0_real_weight_simplify_latency.png`
- `results/figures/pi0_real_weight_simplify_error.png`