# Research Log

## Environment
- conda env: torch
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- CUDA: `torch.cuda.is_available() == True` when commands are executed with GPU device access.
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB

## Network / Proxy
- [network] Checked HTTP_PROXY / HTTPS_PROXY / ALL_PROXY: all empty.
- [network] No downloads performed yet; current PyTorch module experiments only need installed local packages.
- [network] `transformers` and `accelerate` are not installed, but they are not required for these module-level experiments; skipped PyPI access.

## Git
- note: workspace `.git` is mounted read-only by the execution environment, so Git metadata is stored in `/tmp/pi0-approx-vla-git` and commands use `--git-dir=/tmp/pi0-approx-vla-git --work-tree=/home/fu1fan/Develop/PROJECTS/pi0-approx-vla`.
- initial commit: `96d3ebf`
- linear benchmark commit: `cdb4f2d`
- projector benchmark commit: `8799611`
- softmax benchmark commit: `1f2144f`
- gelu/rmsnorm benchmark commit: `e244db2`
- summary commit: `b1b74e8`
- toy flow matching commit: `f1274ed`
- scale sweep benchmark commit: `b48d703`

## Experiments

### Linear Quantization
- command: `conda run -n torch python pytorch_exp/exp_linear_quant.py --device cuda --repeat 30 --warmup 5`
- result csv: `results/csv/linear_quant.csv`
- key observations: rerun on CUDA, including optional 4096 shape. INT8 fake quant kept cosine around 0.99987-0.99988; INT4 weight-only reduced weight size strongly but cosine dropped to about 0.9878-0.9899 with larger MSE.
- issues: default sandbox did not expose GPU devices, so the first pass fell back to CPU.
- fixes: reran with GPU device access using `--include-large`, repeat=30/warmup=5.

### Projector Quantization
- command: `conda run -n torch python pytorch_exp/exp_projector_quant.py --device cuda --repeat 30 --warmup 5`
- result csv: `results/csv/projector_quant.csv`
- key observations: rerun on CUDA. INT8 fake quant reached cosine about 0.99987 for the simulated VLM projector. INT4 weight-only reduced estimated weight storage from 9.0 MB FP32 to 1.125 MB but cosine dropped to about 0.98987.
- issues: default sandbox did not expose GPU devices during the first pass.
- fixes: reran with GPU device access; no NaN/Inf observed.

### Softmax Approximation
- command: `conda run -n torch python pytorch_exp/exp_softmax_approx.py --device cuda --repeat 30 --warmup 5 --include-large`
- result csv: `results/csv/softmax_approx.csv`
- key observations: rerun on CUDA for both `[8,128,128]` and optional `[16,256,256]`. LUT softmax had the lowest KL/MSE. Taylor 3 was also stable with much lower error than Taylor 2. Coarse PWL was the least accurate in this configuration.
- issues: PWL approximation has visibly higher KL divergence because the current segment layout is coarse near zero.
- fixes: softmax inputs are stabilized by subtracting max, LUT/PWL/Taylor paths clamp to `[-8, 0]`, and KL uses eps through `tensor_metrics`.

### GELU / RMSNorm Approximation
- command: `conda run -n torch python pytorch_exp/exp_gelu_rmsnorm_approx.py --device cuda --repeat 30 --warmup 5`
- result csv: `results/csv/gelu_rmsnorm_approx.csv`
- key observations: rerun on CUDA. GELU LUT had very low error, tanh GELU was also accurate, and coarse PWL GELU had higher but still small error. RMSNorm INT8 input fake quant kept very high cosine; approximate rsqrt was nearly identical to FP32 in this run.
- issues: default sandbox did not expose GPU devices during the first pass.
- fixes: RMSNorm uses eps and clamp before reciprocal sqrt; all outputs were checked finite.

### Toy Flow Matching Demo
- command: `conda run -n torch python pytorch_exp/toy_flow_matching_demo.py --device cuda --steps 400 --batch-size 512`
- result csv: `results/csv/toy_flow_matching.csv`
- result figure: `results/figures/toy_flow_matching_curve.png`
- key observations: rerun on CUDA; toy velocity MSE decreased over the 400-step run, demonstrating the conditional noisy-action-to-velocity training loop.
- issues: first run warned that Matplotlib config path was not writable.
- fixes: set `MPLCONFIGDIR=/tmp/pi0_approx_vla_matplotlib` before importing Matplotlib and reran cleanly.

### Scale Sweep Benchmarks
- command: `conda run -n torch python pytorch_exp/exp_scale_sweep.py --device cuda --repeat 30 --warmup 5`
- result csv: `results/csv/scale_sweep_linear.csv`, `results/csv/scale_sweep_softmax.csv`
- result figures: `results/figures/scale_sweep_latency.png`, `results/figures/scale_sweep_memory.png`, `results/figures/scale_sweep_error.png`
- key observations: CUDA run completed all requested Linear shapes through `[1,128,8192] -> [1,128,8192]` and all requested Softmax shapes through `heads=32,seq=1024`. INT8 Linear kept cosine above 0.99985 while giving 4x estimated weight compression; INT4 weight-only gave 8x compression but lower cosine around 0.9868 at the largest shape. LUT softmax had the best KL/MSE stability, while Taylor 3 was also accurate.
- issues: no OOM. Repeat was automatically reduced for larger cases: Linear 4096 used repeat=20, Linear 8192 used repeat=10, Softmax heads=32 seq=1024 used repeat=5. PyTorch fake INT8/INT4 latency includes quantize/dequantize overhead, so it does not represent true hardware INT kernel speed.
- fixes: logged adaptive repeat reductions and kept all original CSVs untouched.

## Problems and Fixes
- Initial sandboxed CUDA check: `nvidia-smi` could not communicate with the NVIDIA driver and PyTorch reported CUDA unavailable, so scripts safely fell back to CPU through `resolve_device`.
- Diagnosis update: the default Codex sandbox did not expose `/dev/nvidia*`, so `nvidia-smi` and PyTorch CUDA failed only inside sandboxed commands. Escalated commands can access the host GPU; all main CSVs and figures were rerun on CUDA afterward.
- Plotting first wrote PNGs but failed before `summary.md` because `nvidia-smi` returned no captured stderr in the plotting subprocess. Fixed `maybe_nvidia_smi()` to handle empty diagnostics and reran successfully.

## Final Outputs
- CSV: `results/csv/linear_quant.csv`, `results/csv/projector_quant.csv`, `results/csv/softmax_approx.csv`, `results/csv/gelu_rmsnorm_approx.csv`, `results/csv/toy_flow_matching.csv`, `results/csv/scale_sweep_linear.csv`, `results/csv/scale_sweep_softmax.csv`
- figures: `results/figures/latency_compare.png`, `results/figures/error_compare.png`, `results/figures/cosine_compare.png`, `results/figures/model_size_compare.png`, `results/figures/toy_flow_matching_curve.png`, `results/figures/scale_sweep_latency.png`, `results/figures/scale_sweep_memory.png`, `results/figures/scale_sweep_error.png`
- summary: `results/summary.md`
