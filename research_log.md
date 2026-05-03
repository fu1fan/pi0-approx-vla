# Research Log

## Environment
- conda env: torch
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- CUDA: `torch.cuda.is_available() == False`
- GPU: `nvidia-smi` failed to communicate with the NVIDIA driver; PyTorch reports `no cuda`.

## Network / Proxy
- [network] Checked HTTP_PROXY / HTTPS_PROXY / ALL_PROXY: all empty.
- [network] No downloads performed yet; current PyTorch module experiments only need installed local packages.

## Git
- note: workspace `.git` is mounted read-only by the execution environment, so Git metadata is stored in `/tmp/pi0-approx-vla-git` and commands use `--git-dir=/tmp/pi0-approx-vla-git --work-tree=/home/fu1fan/Develop/PROJECTS/pi0-approx-vla`.
- initial commit: `96d3ebf`
- linear benchmark commit: `cdb4f2d`
- projector benchmark commit: `8799611`
- softmax benchmark commit: pending
- gelu/rmsnorm benchmark commit: pending
- summary commit: pending

## Experiments

### Linear Quantization
- command: `conda run -n torch python pytorch_exp/exp_linear_quant.py --device cuda --repeat 30 --warmup 5`
- result csv: `results/csv/linear_quant.csv`
- key observations: CUDA request fell back to CPU. INT8 fake quant kept cosine around 0.99989 on tested GEMM shapes; INT4 weight-only reduced estimated weight size to 0.5x FP16 but cosine dropped to about 0.989-0.990 with much larger MSE.
- issues: optional 4096 shape skipped because CUDA driver is unavailable and CPU-only run would slow the full experiment batch.
- fixes: used default required shapes only and kept repeat=30/warmup=5.

### Projector Quantization
- command: `conda run -n torch python pytorch_exp/exp_projector_quant.py --device cuda --repeat 30 --warmup 5`
- result csv: `results/csv/projector_quant.csv`
- key observations: INT8 fake quant reached cosine about 0.99989 for the simulated VLM projector. INT4 weight-only reduced estimated weight storage from 9.0 MB FP32 to 1.125 MB but cosine dropped to about 0.98987.
- issues: CUDA request fell back to CPU because the NVIDIA driver is not visible.
- fixes: kept target projector shape and repeat=30/warmup=5; no NaN/Inf observed.

### Softmax Approximation
- command: `conda run -n torch python pytorch_exp/exp_softmax_approx.py --device cuda --repeat 30 --warmup 5 --include-large`
- result csv: `results/csv/softmax_approx.csv`
- key observations: both `[8,128,128]` and optional `[16,256,256]` completed. LUT softmax had the lowest KL/MSE. Taylor 3 was also stable with much lower error than Taylor 2. Coarse PWL was the least accurate in this configuration.
- issues: CUDA request fell back to CPU. PWL approximation has visibly higher KL divergence because the current segment layout is coarse near zero.
- fixes: softmax inputs are stabilized by subtracting max, LUT/PWL/Taylor paths clamp to `[-8, 0]`, and KL uses eps through `tensor_metrics`.

### GELU / RMSNorm Approximation
- command: pending
- result csv: `results/csv/gelu_rmsnorm_approx.csv`
- key observations: pending
- issues: pending
- fixes: pending

## Problems and Fixes
- CUDA requested but unavailable: `nvidia-smi` cannot communicate with the NVIDIA driver and PyTorch reports CUDA unavailable. Experiments are run with `--device cuda`, and scripts safely fall back to CPU through `resolve_device`.

## Final Outputs
- CSV: pending
- figures: pending
- summary: pending
