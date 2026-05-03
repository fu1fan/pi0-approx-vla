# Experiment Plan

## One-Week Route

Day 1: verify environment, run PyTorch linear quantization on CPU, and generate the first CSV.

Day 2: run projector quantization and compare FP32, FP16, INT8 fake quant, and INT4 weight-only fake quant.

Day 3: run softmax approximations with LUT, PWL, and Taylor variants. Record accuracy and latency tradeoffs.

Day 4: run GELU approximations with tanh, PWL, and LUT variants.

Day 5: synthesize `int8_linear` and `lut_softmax` in Vitis Unified, collect latency and resource reports.

Day 6: synthesize `fixed_projector` and `gelu_pwl` if time allows. Update result tables.

Day 7: polish plots, summarize tradeoffs, and prepare interview talking points.

## Module Mapping

| Experiment | pi0/VLA Module | Purpose |
|---|---|---|
| PyTorch Linear quant | Transformer Linear / FFN GEMM | Estimate accuracy, latency, and weight-size impact of lower precision |
| PyTorch Projector quant | VLM visual projector | Simulate visual token to language embedding projection |
| PyTorch Softmax approx | Attention score softmax | Compare nonlinear approximations after max-subtraction stabilization |
| PyTorch GELU approx | Transformer activation | Compare exact/tanh/PWL/LUT activation choices |
| HLS int8_linear | Quantized linear accelerator kernel | Generate synthesizable INT8 GEMM-like evidence |
| HLS fixed_projector | Fixed-point projector kernel | Show projector feasibility with `ap_fixed` |
| HLS lut_softmax | Softmax accelerator kernel | Show LUT nonlinear approximation in HLS |
| HLS gelu_pwl | Activation accelerator kernel | Show cheap PWL activation logic |

## Expected Metrics

PyTorch quantization experiments output MSE, MAE, max error, cosine similarity, latency mean/std, and estimated weight size.

Softmax also outputs optional KL divergence.

HLS experiments should collect latency cycles, initiation interval, estimated clock, LUT, FF, BRAM, DSP, and notes from synthesis reports.

## Minimum Deliverable

1. PyTorch Linear quantization CSV and plot
2. PyTorch Softmax approximation CSV and plot
3. HLS `int8_linear` C simulation and synthesis report
4. HLS `lut_softmax` C simulation and synthesis report
