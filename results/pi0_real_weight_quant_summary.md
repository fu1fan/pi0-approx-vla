# pi0 Real-weight Quantization Summary

This stage uses real pi0 module weights from `lerobot/pi0_base` with random tensor inputs. It reflects quantization error under real parameter distributions, but does not represent real robot task success rate.

- device: `cuda`
- selected tensor file: `results/pi0_module_weights/selected_modules.pt`
- rows: 70
- INT8 minimum cosine: 0.994843
- INT4 minimum cosine: 0.921080
- All selected weights were already in `[out_dim, in_dim]` layout for `torch.nn.functional.linear`; no transpose was used.

Outputs:
- `results/csv/pi0_real_weight_quant.csv`
- `results/figures/pi0_real_weight_quant_latency.png`
- `results/figures/pi0_real_weight_quant_error.png`
- `results/figures/pi0_real_weight_quant_size.png`