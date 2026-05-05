# pi0 Real-weight Quantization Summary

This stage is intended to use real pi0 module weights with random tensor inputs. It can reflect module quantization error under real parameter distributions, but still cannot represent robot task success rate.

Current run status: skipped real-weight numeric benchmarks.

Reason: full `lerobot/pi0_base` weights are a single 13.04 GiB `model.safetensors`; the download was skipped by the 2 GiB safety limit, so no local tensor keys/modules were available for extraction.

No metrics were fabricated. Re-run Stage 3 with a larger explicit download limit or a local verified checkpoint to enable this stage.

Outputs:
- `results/csv/pi0_real_weight_quant.csv`
- `results/figures/pi0_real_weight_quant_latency.png`
- `results/figures/pi0_real_weight_quant_error.png`
- `results/figures/pi0_real_weight_quant_size.png`