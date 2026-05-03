# PyTorch Experiment Summary

## Environment
- Conda env: torch
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- CUDA available: False
- GPU: no cuda
- nvidia-smi: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

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
- linear_quant / int8_fake_quant: cosine=0.999886, MSE=2.319e-04, latency=1.056 ms.
- projector_quant / int8_fake_quant: cosine=0.999890, MSE=2.341e-04, latency=1.741 ms.
- softmax_approx / lut_exp_softmax: cosine=1.000021, MSE=9.347e-14, latency=4.339 ms.

## Current Conclusions
- INT8 is suitable for linear_quant, projector_quant, rmsnorm_approx.
- INT4 weight-only has the largest observed error on linear_quant (batch=1,seq=256,in=2048,out=2048).
- Softmax approximation is most stable with lut_exp_softmax by KL/MSE on the current run.
- CUDA experiments should be rerun after the NVIDIA driver is visible to PyTorch; current measurements may be CPU fallback.

## Next Steps for Vitis HLS
- Prioritize INT8 linear/projector GEMM kernels first because they map directly to bandwidth and DSP savings.
- Implement LUT or PWL softmax exp next; validate with KL divergence before integration.
- Keep GELU LUT/PWL and RMSNorm reciprocal-sqrt kernels as smaller nonlinear accelerator targets.
- Re-run CUDA latency after fixing driver visibility, then use GPU numbers for the final PPT.
