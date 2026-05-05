# pi0-aligned Random Quantization Summary

This is a random tensor proxy benchmark. It aligns shapes and module structure with key pi0/openpi-style modules, but it does not use real pi0 weights, real activations, or end-to-end robot evaluation.

- device: `cuda`
- rows: 100
- modules: visual projector, Gemma-style q/k/v/o projections, gated VLM FFN, action expert FFN, state/action projections.
- INT8 minimum cosine: 0.998402
- INT4 minimum cosine: 0.963150
- W4A8 implemented rows: 20. W4A8 uses INT4 per-output-channel fake-quantized weights plus INT8 fake-quantized activations.
- Interpretation: results indicate module-level sensitivity trends only. They do not represent true pi0 task success rate or full-model quantization quality.

Outputs:
- `results/csv/pi0_aligned_random_quant.csv`
- `results/figures/pi0_aligned_random_quant_latency.png`
- `results/figures/pi0_aligned_random_quant_error.png`
- `results/figures/pi0_aligned_random_quant_size.png`