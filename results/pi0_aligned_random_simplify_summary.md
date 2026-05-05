# pi0-aligned Random Simplification Summary

This is a random tensor proxy benchmark using pi0-aligned shapes. It tests function replacements inside FFN, attention softmax, and RMSNorm proxy modules. It does not use real pi0 weights or real robot activations.

- device: `cuda`
- rows: 59
- max relative L2 among approximations: 1.006901
- GELU/SiLU replacements are evaluated on final FFN output, not just activation output.
- Softmax approximations subtract max, clamp where appropriate, and include KL divergence.
- RMSNorm approximations include INT8 fake quant and Newton-Raphson rsqrt variants.

Outputs:
- `results/csv/pi0_aligned_random_simplify.csv`
- `results/figures/pi0_aligned_random_simplify_latency.png`
- `results/figures/pi0_aligned_random_simplify_error.png`