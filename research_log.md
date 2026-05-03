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
- linear benchmark commit: pending
- projector benchmark commit: pending
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
- command: pending
- result csv: `results/csv/projector_quant.csv`
- key observations: pending
- issues: pending
- fixes: pending

### Softmax Approximation
- command: pending
- result csv: `results/csv/softmax_approx.csv`
- key observations: pending
- issues: pending
- fixes: pending

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
