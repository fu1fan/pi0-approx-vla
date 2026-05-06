# PyTorch Experiment Summary

## Environment
- Conda env: `torch`
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- CUDA available: True
- GPU: NVIDIA GeForce RTX 5060 Ti
- nvidia-smi: NVIDIA GeForce RTX 5060 Ti, 16311 MiB

## Completed Experiments
- Linear / GEMM quantization
- Vision projector quantization
- Attention softmax approximation
- GELU / RMSNorm approximation
- Scale sweep for Linear and Softmax
- Random tensor + pi0-aligned shape quantization
- Random tensor + pi0-aligned function simplification
- Real pi0 weight + random input quantization
- Real pi0 weight + random input simplification
- pi0-shape toy flow step reduction
- Vitis HLS module-level kernel benchmarks: INT8 GEMM, LUT softmax, PWL GELU, RMSNorm rsqrt, and optional fixed-point visual projector tile

## CSV Outputs
- `results/csv/linear_quant.csv`
- `results/csv/projector_quant.csv`
- `results/csv/softmax_approx.csv`
- `results/csv/gelu_rmsnorm_approx.csv`
- `results/csv/toy_flow_matching.csv`
- `results/csv/scale_sweep_linear.csv`
- `results/csv/scale_sweep_softmax.csv`
- `results/csv/pi0_aligned_random_quant.csv`
- `results/csv/pi0_aligned_random_simplify.csv`
- `results/csv/pi0_real_weight_quant.csv`
- `results/csv/pi0_real_weight_simplify.csv`
- `results/csv/pi0_shape_flow_step_reduction.csv`
- `results/csv/hls_kernel_summary.csv`

## Figures
- `results/figures/latency_compare.png`
- `results/figures/error_compare.png`
- `results/figures/cosine_compare.png`
- `results/figures/model_size_compare.png`
- `results/figures/toy_flow_matching_curve.png`
- `results/figures/scale_sweep_latency.png`
- `results/figures/scale_sweep_memory.png`
- `results/figures/scale_sweep_error.png`
- `results/figures/pi0_aligned_random_quant_latency.png`
- `results/figures/pi0_aligned_random_quant_error.png`
- `results/figures/pi0_aligned_random_quant_size.png`
- `results/figures/pi0_aligned_random_simplify_latency.png`
- `results/figures/pi0_aligned_random_simplify_error.png`
- `results/figures/pi0_real_weight_quant_latency.png`
- `results/figures/pi0_real_weight_quant_error.png`
- `results/figures/pi0_real_weight_quant_size.png`
- `results/figures/pi0_real_weight_simplify_latency.png`
- `results/figures/pi0_real_weight_simplify_error.png`
- `results/figures/pi0_shape_flow_step_reduction_error.png`
- `results/figures/pi0_shape_flow_step_reduction_latency.png`

## PPT-Ready Results
- Scale sweep memory: largest Linear shape shows FP32 256 MB vs INT8 64 MB vs INT4 32 MB estimated weight storage.
- pi0 real-weight quantization: INT8 minimum cosine 0.994843 across selected real-weight modules, while INT4 weight-only minimum cosine drops to 0.921080.
- pi0-shape toy flow: reducing Euler integration from 10 to 2 steps gives about 4.46x latency speedup vs the 10-step toy baseline, with MSE 4.32e-03 relative to that baseline.
- HLS kernel verification: all five tile-level kernels passed Vitis C-sim and v++ C-synthesis on Vitis 2025.2. INT8 GEMM synthesized with 1,742,214 latency cycles, 7.300 ns estimated clock, 6,953 LUT, 4,024 FF, 19 BRAM, and 5 DSP. LUT softmax synthesized with 1,867 latency cycles, KL 1.746e-03, cosine 0.999311, and relative L2 3.77e-02 vs exact softmax. PWL GELU synthesized with 4,114 latency cycles and relative L2 3.07e-03. RMSNorm rsqrt synthesized with 2,080 latency cycles and cosine above 0.9999998 for both NR branches.

## Current Conclusions
- INT8 is the best-supported quantization target for Linear/projector/GEMM-like modules. It stays high-cosine in generic, scale-sweep, pi0-aligned random, and real-weight random-input tests.
- INT4/W4A8 gives strong storage reduction, but error is materially larger. In the real-weight run, INT4 weight-only reached max relative L2 0.416103 and minimum cosine 0.921080.
- LUT softmax is the most stable hardware-like softmax approximation in the real-weight score experiment. `base2_softmax` is nearly exact here because it uses exact `torch.pow`, so it should be treated as a reformulation baseline unless implemented with an approximate exp2 unit.
- GELU replacement should be evaluated at final FFN output. Tanh GELU is near exact; PWL GELU is moderate; identity and clipped-linear replacements are too aggressive for high-accuracy claims.
- RMSNorm approximation is comparatively safe in these proxy runs; max relative L2 in the real-weight simplification run was 0.011693.
- These results are module-level proxy benchmarks. They do not estimate real pi0 robot task success, because no real observation / activation / action dataset is used.
- The Vitis HLS results are also module-level only. They validate small kernel/tile implementability, latency, II, Fmax estimate, and resources; they do not represent a full pi0 FPGA deployment.
- PyTorch experiments remain the source for real-weight/random-input approximation trends. HLS experiments provide hardware feasibility evidence for the same module classes.

## Scale Sweep Findings
- Linear scale sweep covered up to `batch=1,seq=128,in=8192,out=8192` on CUDA.
- Model size benefit becomes more visible with scale: FP32 256.0 MB at the largest shape, INT8 4x smaller, INT4 weight-only 8x smaller.
- PyTorch fake INT8/INT4 did not show latency speedup because quantize/dequantize is included in the measured path. FP16 reached up to 2.49x speedup vs FP32 in this sweep.
- Softmax scale sweep covered up to `heads=32,seq=1024`. Approximation is more meaningful at long sequence length because the score matrix scales as O(seq^2), but PyTorch exact softmax remains a highly optimized kernel.

## pi0-aligned Benchmark Extension
- Random tensor + pi0-aligned quantization completed on CUDA: `results/csv/pi0_aligned_random_quant.csv` with 100 rows. INT8 minimum cosine was 0.998402; INT4/W4A8 showed materially larger error.
- Random tensor + pi0-aligned simplification completed on CUDA: `results/csv/pi0_aligned_random_simplify.csv` with 59 rows. Maximum relative L2 was 1.006901 for aggressive FFN activation replacement, 0.702048 for rough softmax approximations, and 0.011549 for RMSNorm approximations.
- Real pi0 checkpoint extraction completed from verified `lerobot/pi0_base`: 777 keys inspected and 31 exact-match tensors extracted into ignored local `results/pi0_module_weights/selected_modules.pt`.
- Real pi0 weight + random input quantization completed on CUDA: `results/csv/pi0_real_weight_quant.csv` with 70 rows. INT8 minimum cosine 0.994843; INT4 weight-only minimum cosine 0.921080.
- Real pi0 weight + random input simplification completed on CUDA: `results/csv/pi0_real_weight_simplify.csv` with 38 rows. LUT softmax was the best hardware-like softmax approximation by KL; RMSNorm approximations stayed low-error; aggressive activation replacement was high-error.
- pi0-shape toy flow step reduction completed on CUDA: `results/csv/pi0_shape_flow_step_reduction.csv`.

## Next Steps for Vitis HLS
- First priority validated: INT8 tiled GEMM for action projection / projector / QKV / FFN-like matrix tiles.
- Second priority validated: LUT softmax exp with max subtraction and clamp to `[-8, 0]`.
- Third priority validated: RMSNorm reciprocal-sqrt approximation with one-step and two-step Newton-Raphson branches.
- PWL GELU is a viable activation-approximation kernel; identity replacement remains too aggressive based on the PyTorch evidence.
- Flow step reduction is intentionally kept at algorithm level and is not part of the HLS kernel set.
