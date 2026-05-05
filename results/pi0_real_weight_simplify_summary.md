# pi0 Real-weight Simplification Summary

This stage repeats function simplification benchmarks using real `lerobot/pi0_base` module weights with random tensor inputs. It reflects real parameter distributions for FFN, Q/K score generation, and RMSNorm scale, but does not measure robot task success.

- device: `cuda`
- selected tensor file: `results/pi0_module_weights/selected_modules.pt`
- rows: 38
- max FFN activation replacement relative L2: 1.019450
- best softmax approximation by KL: `base2_softmax` KL=2.289043e-09, relative_l2=0.000000
- best hardware-like softmax approximation excluding exact base-2 reformulation: `lut_softmax` KL=4.846268e-09, relative_l2=0.000043
- max RMSNorm approximation relative L2: 0.011693
- Softmax scores were generated from real Q/K projection weights before applying exact and approximate softmax variants.
- `base2_softmax` uses exact `torch.pow` in this PyTorch benchmark, so it is a mathematical reformulation baseline rather than a low-cost approximate hardware kernel.
- RMSNorm uses real extracted scale weights where available.

Outputs:
- `results/csv/pi0_real_weight_simplify.csv`
- `results/figures/pi0_real_weight_simplify_latency.png`
- `results/figures/pi0_real_weight_simplify_error.png`