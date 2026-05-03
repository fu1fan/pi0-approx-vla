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
- scale_sweep_linear
- scale_sweep_softmax

## CSV Outputs
- `results/csv/linear_quant.csv`
- `results/csv/projector_quant.csv`
- `results/csv/softmax_approx.csv`
- `results/csv/gelu_rmsnorm_approx.csv`
- `results/csv/toy_flow_matching.csv`
- `results/csv/scale_sweep_linear.csv`
- `results/csv/scale_sweep_softmax.csv`

## Figures
- `results/figures/latency_compare.png`
- `results/figures/error_compare.png`
- `results/figures/cosine_compare.png`
- `results/figures/model_size_compare.png`
- `results/figures/toy_flow_matching_curve.png`
- `results/figures/scale_sweep_latency.png`
- `results/figures/scale_sweep_memory.png`
- `results/figures/scale_sweep_error.png`

## PPT-Ready Results
- linear_quant / int8_fake_quant: cosine=0.999882, MSE=2.340e-04, latency=0.164 ms.
- projector_quant / int8_fake_quant: cosine=0.999874, MSE=2.520e-04, latency=0.294 ms.
- softmax_approx / lut_exp_softmax: cosine=1.000000, MSE=3.910e-13, latency=0.124 ms.

## Current Conclusions
- INT8 is suitable for linear_quant, projector_quant, rmsnorm_approx.
- INT4 weight-only has the largest observed error on linear_quant (batch=1,seq=128,in=4096,out=4096).
- Softmax approximation is most stable with lut_exp_softmax by KL/MSE on the current run.
- Current CSV and figures are from CUDA runs.

## Scale Sweep Findings
- Linear scale sweep covered up to `batch=1,seq=128,in=8192,out=8192` on `cuda`.
- Model size benefit becomes more visible with scale: FP32 `256.0 MB` at the largest shape, while INT8 is 4x smaller and INT4 weight-only is 8x smaller by construction.
- PyTorch fake INT8/INT4 did not show latency speedup because quantize/dequantize is included in the measured path; FP16 reached up to `2.49x` speedup vs FP32 in this sweep.
- INT8 error stayed acceptable for module-level approximation (minimum cosine `0.999857`); INT4 weight-only is more aggressive (minimum cosine `0.986760`) and should be treated as accuracy-risky without grouping/calibration.
- Softmax scale sweep covered up to `heads=32,seq=1024`.
- LUT exp softmax is the most accurate/stable family in this run (`lut_exp_softmax`, KL `3.146e-10` on `heads=8,seq=128`).
- Exact PyTorch softmax remains faster than the Python-level approximate paths; the fastest approximate row was `taylor2_exp_softmax` at `0.069 ms`. The scale sweep is therefore strongest as an error/stability study and as motivation for HLS LUT/PWL kernels, not as a PyTorch speed claim.
- As sequence length grows, softmax approximation becomes more meaningful for hardware because the score matrix scales as O(seq^2), but PyTorch needs fused/custom kernels to convert that into latency wins.

## Scale Sweep Answers
1. INT8/INT4 model size benefits are clearly more visible at larger hidden dims; latency benefits are not shown by PyTorch fake quant because the measured path includes quantize/dequantize overhead. FP16 does show CUDA latency speedup.
2. Softmax approximation becomes more meaningful as sequence length grows because attention scores scale as O(seq^2), but PyTorch exact softmax is still faster here; fused/HLS approximate kernels are needed for latency wins.
3. Acceptable-error methods in this run: FP16, INT8 fake quant, LUT softmax, Taylor 3 softmax, LUT GELU, and approximate-rsqrt RMSNorm. INT4 weight-only and coarse PWL softmax/GELU are useful stress points but need calibration or better segmentation before high-accuracy use.
4. Best PPT results: scale_sweep_memory for 256 MB -> 64 MB/32 MB Linear storage, scale_sweep_error for INT8-vs-INT4 accuracy contrast, and scale_sweep_latency with the fake-quant caveat.
5. Vitis HLS should prioritize INT8 Linear/projector GEMM first, then LUT softmax exp for long-sequence attention.

## Scale Sweep PPT Picks
- `scale_sweep_memory.png`: largest Linear shape shows FP32 256 MB vs INT8 64 MB vs INT4 32 MB weight storage.
- `scale_sweep_error.png`: INT8 remains high-cosine across hidden dims; INT4 error grows and is the cautionary contrast.
- `scale_sweep_latency.png`: include with the caveat that fake quantization latency includes quant/dequant overhead; use it to motivate true HLS/cuBLASLt INT kernels.

## Scale Sweep HLS Priority
- First priority: INT8 Linear / projector GEMM, because accuracy is acceptable and memory bandwidth savings scale directly with hidden dimension.
- Second priority: LUT softmax exp, because it is numerically stable and the O(seq^2) attention matrix makes hardware approximation increasingly relevant at long sequence length.

## Next Steps for Vitis HLS
- Prioritize INT8 linear/projector GEMM kernels first because they map directly to bandwidth and DSP savings.
- Implement LUT or PWL softmax exp next; validate with KL divergence before integration.
- Keep GELU LUT/PWL and RMSNorm reciprocal-sqrt kernels as smaller nonlinear accelerator targets.
- Use the current CUDA CSVs as the PPT latency source; rerun only if changing repeat/shape settings.
