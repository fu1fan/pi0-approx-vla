# PyTorch Experiment Summary

## Environment
- Conda env: torch
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- CUDA available: True
- GPU: NVIDIA GeForce RTX 5060 Ti
- nvidia-smi: NVIDIA GeForce RTX 5060 Ti, 16311 MiB

## Completed Experiments
- gelu_approx
- linear_quant
- projector_quant
- rmsnorm_approx
- softmax_approx
- toy_flow_matching

## CSV Outputs
- `results/csv/linear_quant.csv`
- `results/csv/projector_quant.csv`
- `results/csv/softmax_approx.csv`
- `results/csv/gelu_rmsnorm_approx.csv`
- `results/csv/toy_flow_matching.csv`

## Figures
- `results/figures/latency_compare.png`
- `results/figures/error_compare.png`
- `results/figures/cosine_compare.png`
- `results/figures/model_size_compare.png`
- `results/figures/toy_flow_matching_curve.png`

## PPT-Ready Results
- linear_quant / int8_fake_quant: cosine=0.999882, MSE=2.340e-04, latency=0.164 ms.
- projector_quant / int8_fake_quant: cosine=0.999874, MSE=2.520e-04, latency=0.294 ms.
- softmax_approx / lut_exp_softmax: cosine=1.000000, MSE=3.910e-13, latency=0.124 ms.

## Current Conclusions
- INT8 is suitable for linear_quant, projector_quant, rmsnorm_approx.
- INT4 weight-only has the largest observed error on linear_quant (batch=1,seq=128,in=4096,out=4096).
- Softmax approximation is most stable with lut_exp_softmax by KL/MSE on the current run.
- Current CSV and figures are from CUDA runs.

## Next Steps for Vitis HLS
- Prioritize INT8 linear/projector GEMM kernels first because they map directly to bandwidth and DSP savings.
- Implement LUT or PWL softmax exp next; validate with KL divergence before integration.
- Keep GELU LUT/PWL and RMSNorm reciprocal-sqrt kernels as smaller nonlinear accelerator targets.
- Use the current CUDA CSVs as the PPT latency source; rerun only if changing repeat/shape settings.
