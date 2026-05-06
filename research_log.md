# Research Log

## Environment
- conda env: torch
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- CUDA: `torch.cuda.is_available() == True` when commands are executed with GPU device access.
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB

### Continuation Environment Check 2026-05-06
- current commit before new work: `775000020291b42fbf5958ca62979f43d0adf7c9`
- conda env: `torch`
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- CUDA availability in current execution context: `False`
- CUDA diagnostic: PyTorch reported `cudaGetDeviceCount` error 804; `nvidia-smi` reported `Driver/library version mismatch` with NVML library version 580.142.
- GPU name: unavailable from PyTorch in this run; previous verified hardware is NVIDIA GeForce RTX 5060 Ti 16GB.
- CPU: AMD Ryzen 9 7950X 16-Core Processor, 32 logical CPUs
- OS: Linux ELITE-7950X 6.17.0-22-generic x86_64 / Ubuntu 24.04 family
- action: continue with `--device auto` / CPU fallback where CUDA is unavailable; do not fabricate CUDA results.
- recovery update: after GPU stack recovery, `nvidia-smi` reports driver 580.142 and PyTorch in conda env `torch` reports `torch.cuda.is_available() == True`, GPU `NVIDIA GeForce RTX 5060 Ti`; subsequent experiments should use CUDA.

## Network / Proxy
- [network] Checked HTTP_PROXY / HTTPS_PROXY / ALL_PROXY: all empty.
- [network] No downloads performed yet; current PyTorch module experiments only need installed local packages.
- [network] `transformers` and `accelerate` are not installed, but they are not required for these module-level experiments; skipped PyPI access.
- [network] Stage 3 attempted Hugging Face mirror `https://hf-mirror.com` for `lerobot/pi0_base`. Initial sandbox DNS failed; retried outside sandbox successfully for metadata.
- [network] After explicit user approval, full `model.safetensors` download was allowed up to 20 GiB and completed from `https://hf-mirror.com`; local checkpoint cache remains ignored under `external/pi0_checkpoints/`.

## Git
- note: Git metadata has been migrated back into the project-local `.git/`; normal commands such as `git status` and `git log` now work from the repository root. Earlier commits were temporarily stored in `/tmp/pi0-approx-vla-git` because the Codex sandbox exposed `.git/` as an empty read-only directory.
- initial commit: `96d3ebf`
- linear benchmark commit: `cdb4f2d`
- projector benchmark commit: `8799611`
- softmax benchmark commit: `1f2144f`
- gelu/rmsnorm benchmark commit: `e244db2`
- summary commit: `b1b74e8`
- toy flow matching commit: `f1274ed`
- scale sweep benchmark commit: `b48d703`
- pi0-aligned random quantization commit: `780731e`
- pi0-aligned random simplification commit: `a8dfb60`
- pi0 checkpoint utilities commit: `0c0065b`
- pi0 checkpoint extraction commit: `b86da3a`
- pi0 real-weight quantization commit: `b70ff52`
- pi0 real-weight simplification commit: `cc902a2`
- pi0-shape toy flow step reduction commit: `332e762`
- real-weight final summary commit: `768b0ce`

## Experiments

### Vitis HLS Workspace and Environment Audit
- date: 2026-05-06
- outputs: `results/hls_environment.md`
- workspace scan: current `vitis_workspace/hls_src/` is absent; Vitis Unified component directories exist for `int8_gemm`, `lut_softmax`, `gelu_pwl`, and `rmsnorm_rsqrt`.
- config scan: detected component JSON/config files including `vitis-comp.json`, `hls_config.cfg`, and `compile_commands.json` in each component directory, plus `_ide/.wsdata/*.json`, `_ide/*.ini`, and `_ide/settings.json`.
- tool check: `vitis` and `v++` are available as Vitis 2025.2 / build 6295257; `vitis_hls` is not installed in `PATH`.
- issue: direct `vitis --version` reports a HOME configuration-space warning even though `df -h` shows sufficient filesystem space.
- fix / route: set `XILINX_VITIS_DATA_DIR=/tmp/vitis_data` for Vitis Unified commands; use `v++ --mode hls` for batch HLS when possible, and keep classic `run_hls.tcl` files as reproducibility templates.

### Vitis Unified Workspace Config Maintenance
- date: 2026-05-06
- outputs: `tools/backup_vitis_workspace_config.py`, `tools/validate_vitis_workspace_config.py`, `tools/update_vitis_workspace_components.py`, `results/hls_workspace_config_report.md`
- backup: active config files were copied to `vitis_workspace/config_backups/20260506_225000_pre_hls_kernel_edits/` before workspace component edits.
- JSON policy: no existing Vitis Unified JSON file was manually rewritten; the optional `fixed_projector_tile/vitis-comp.json` was created from the existing component template via Python `json`.
- validation: `tools/validate_vitis_workspace_config.py --strict` passed for active configs and component recognition checks; `python -m json.tool` passed for each active `vitis-comp.json`.
- note: the validator was adjusted to accept Vitis-style top-level `key=value` config/INI files and duplicate `_ide/.peers.ini` keys without falsely classifying the workspace as corrupt.

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

### pi0-aligned Random Quantization
- command: `conda run -n torch python pytorch_exp/exp_pi0_aligned_random_quant.py --device cuda --repeat 10 --warmup 3`
- result csv: `results/csv/pi0_aligned_random_quant.csv`
- result figures: `results/figures/pi0_aligned_random_quant_latency.png`, `results/figures/pi0_aligned_random_quant_error.png`, `results/figures/pi0_aligned_random_quant_size.png`
- summary: `results/pi0_aligned_random_quant_summary.md`
- key observations: CUDA run completed 100 rows across visual projector, Gemma-style q/k/v/o projections, VLM gated FFN, action expert FFN, and action projection modules. INT8 minimum cosine was 0.998402; INT4 weight-only minimum cosine was 0.963150; W4A8 was implemented and produced 20 rows with minimum cosine 0.961928.
- issues: no OOM. Large FFN and quantized variants used reduced repeat through script policy to keep runtime bounded.
- fixes: generated separate Stage 1 CSV/figures without replacing previous results.

### pi0-aligned Random Simplification
- command: `conda run -n torch python pytorch_exp/exp_pi0_aligned_random_simplify.py --device cuda --repeat 10 --warmup 3`
- result csv: `results/csv/pi0_aligned_random_simplify.csv`
- result figures: `results/figures/pi0_aligned_random_simplify_latency.png`, `results/figures/pi0_aligned_random_simplify_error.png`
- summary: `results/pi0_aligned_random_simplify_summary.md`
- key observations: CUDA run completed 59 rows covering FFN activation replacement, VLM/action attention softmax approximation, and RMSNorm approximation. Maximum relative L2 by subexperiment was 1.006901 for aggressive FFN activation replacement, 0.702048 for clipped/rough softmax approximations, and 0.011549 for RMSNorm approximations.
- issues: first run completed compute but failed CSV writing because softmax rows included `kl_divergence` while earlier rows did not. Non-softmax KL was later made explicit as `not_applicable` to avoid NaN-like CSV fields.
- fixes: updated common CSV writer to union all row fields, regenerated CSV/figures/summary, and verified numeric columns are finite.

### pi0 Checkpoint Download and Extraction Utilities
- commands:
  - `conda run -n torch python tools/download_pi0_checkpoint.py --repo-id lerobot/pi0_base --max-download-gb 2.0`
  - `conda run -n torch python tools/download_pi0_checkpoint.py --repo-id lerobot/pi0_base --max-download-gb 20.0`
  - `conda run -n torch python tools/inspect_pi0_checkpoint.py --checkpoint-dir external/pi0_checkpoints/lerobot_pi0_base --out results/pi0_checkpoint_keys.txt`
  - `conda run -n torch python tools/extract_pi0_module_weights.py --checkpoint-dir external/pi0_checkpoints/lerobot_pi0_base --keys-file results/pi0_checkpoint_keys.txt --out-dir results/pi0_module_weights --report results/pi0_extracted_modules.md`
- outputs: `results/pi0_checkpoint_download_status.md`, `results/pi0_checkpoint_files.txt`, `results/pi0_checkpoint_keys.txt`, `results/pi0_extracted_modules.md`
- key observations: official openpi documents `gs://openpi-assets/checkpoints/pi0_base`; the verifiable Hugging Face/LeRobot mirror `lerobot/pi0_base` lists a single `model.safetensors` of 13.04 GiB plus small metadata files. After user approval, the full safetensors file was downloaded, 777 checkpoint keys were inspected, and 31 exact-match tensors were extracted for visual projector, VLM attention, VLM FFN, action expert attention/FFN, action projection, and RMSNorm.
- issues: the first pass intentionally skipped the large weight file under a 2 GiB safety limit; real-weight experiments were blocked until explicit approval. The extracted tensor bundle is about 898 MB and is intentionally ignored by Git.
- fixes: added reusable download/inspect/extract tools, exact key selection, and `.gitignore` rules so checkpoint files and extracted `.pt` files are not committed.

### pi0 Real-weight Quantization
- command: `conda run -n torch python pytorch_exp/exp_pi0_real_weight_quant.py --device cuda --repeat 5 --warmup 2`
- result csv: `results/csv/pi0_real_weight_quant.csv`
- result figures: `results/figures/pi0_real_weight_quant_latency.png`, `results/figures/pi0_real_weight_quant_error.png`, `results/figures/pi0_real_weight_quant_size.png`
- summary: `results/pi0_real_weight_quant_summary.md`
- key observations: real-weight numeric benchmark completed on CUDA for 70 rows using extracted `lerobot/pi0_base` tensors and random inputs. INT8 minimum cosine was 0.994843 with max relative L2 0.101908. INT4 weight-only minimum cosine was 0.921080 with max relative L2 0.416103. W4A8 was similar to INT4 on the worst projection.
- issues: true PyTorch fake quant latency includes quantize/dequantize overhead and does not represent a custom INT hardware kernel. Large FFN repeats were reduced automatically to keep runtime bounded.
- fixes: used exact extracted weights in `[out_dim, in_dim]` layout with no transpose, generated numeric CSV/figures/summary, and preserved checkpoint/extracted tensors outside Git.

### pi0 Real-weight Simplification
- command: `conda run -n torch python pytorch_exp/exp_pi0_real_weight_simplify.py --device cuda --repeat 5 --warmup 2`
- result csv: `results/csv/pi0_real_weight_simplify.csv`
- result figures: `results/figures/pi0_real_weight_simplify_latency.png`, `results/figures/pi0_real_weight_simplify_error.png`
- summary: `results/pi0_real_weight_simplify_summary.md`
- key observations: real-weight simplification numeric benchmark completed on CUDA for 38 rows. FFN activation replacement is highly sensitive: identity replacement reached relative L2 1.019450, clipped linear GELU about 0.23, PWL GELU about 0.06-0.07, and tanh GELU stayed near exact. Softmax scores were generated from real Q/K weights; `base2_softmax` is nearly exact because the PyTorch implementation uses exact `torch.pow`, while LUT was the best hardware-like approximation by KL in this run. RMSNorm approximations stayed small, with max relative L2 0.011693.
- issues: `base2_softmax` should be interpreted as a mathematical reformulation baseline unless implemented with an approximate low-cost exp2 unit. The experiment still uses random inputs, not real pi0 activations.
- fixes: replaced the skipped fallback with real FFN activation, real Q/K softmax, and real RMSNorm scale benchmarks; action expert q/k/v/o tensors were added to the extraction allowlist for action-to-context attention proxy scores.

### pi0-shape Toy Flow Step Reduction
- command: `conda run -n torch python pytorch_exp/exp_pi0_shape_flow_step_reduction.py --device cuda --train-steps 1500 --batch-size 128 --eval-repeat 20`
- result csv: `results/csv/pi0_shape_flow_step_reduction.csv`
- result figures: `results/figures/pi0_shape_flow_step_reduction_error.png`, `results/figures/pi0_shape_flow_step_reduction_latency.png`
- summary: `results/pi0_shape_flow_step_reduction_summary.md`
- key observations: CUDA run completed 10/8/6/4/2 Euler-step comparison for pi0-like action chunk shape (`horizon=50`, `action_dim=32`, `cond_dim=1024`). 2-step inference was about 4.46x faster than 10-step, with MSE 4.32e-03 vs the 10-step output.
- issues: first run failed because synthetic condition dimension 1024 was smaller than flattened action dimension 1600; direct reshape was invalid. The toy velocity model remains intentionally small and does not closely reconstruct clean action, so metrics are interpreted only relative to the 10-step toy baseline.
- fixes: repeated/truncated condition features to synthesize clean actions, increased training from 500 to 1500 steps, and regenerated CSV/figures/summary.

## Problems and Fixes
- Initial sandboxed CUDA check: `nvidia-smi` could not communicate with the NVIDIA driver and PyTorch reported CUDA unavailable, so scripts safely fell back to CPU through `resolve_device`.
- Diagnosis update: the default Codex sandbox did not expose `/dev/nvidia*`, so `nvidia-smi` and PyTorch CUDA failed only inside sandboxed commands. Escalated commands can access the host GPU; all main CSVs and figures were rerun on CUDA afterward.
- Plotting first wrote PNGs but failed before `summary.md` because `nvidia-smi` returned no captured stderr in the plotting subprocess. Fixed `maybe_nvidia_smi()` to handle empty diagnostics and reran successfully.

## Final Outputs
- CSV: `results/csv/linear_quant.csv`, `results/csv/projector_quant.csv`, `results/csv/softmax_approx.csv`, `results/csv/gelu_rmsnorm_approx.csv`, `results/csv/toy_flow_matching.csv`, `results/csv/scale_sweep_linear.csv`, `results/csv/scale_sweep_softmax.csv`, `results/csv/pi0_aligned_random_quant.csv`, `results/csv/pi0_aligned_random_simplify.csv`, `results/csv/pi0_real_weight_quant.csv`, `results/csv/pi0_real_weight_simplify.csv`, `results/csv/pi0_shape_flow_step_reduction.csv`
- figures: `results/figures/latency_compare.png`, `results/figures/error_compare.png`, `results/figures/cosine_compare.png`, `results/figures/model_size_compare.png`, `results/figures/toy_flow_matching_curve.png`, `results/figures/scale_sweep_latency.png`, `results/figures/scale_sweep_memory.png`, `results/figures/scale_sweep_error.png`, `results/figures/pi0_aligned_random_quant_latency.png`, `results/figures/pi0_aligned_random_quant_error.png`, `results/figures/pi0_aligned_random_quant_size.png`, `results/figures/pi0_aligned_random_simplify_latency.png`, `results/figures/pi0_aligned_random_simplify_error.png`, `results/figures/pi0_real_weight_quant_latency.png`, `results/figures/pi0_real_weight_quant_error.png`, `results/figures/pi0_real_weight_quant_size.png`, `results/figures/pi0_real_weight_simplify_latency.png`, `results/figures/pi0_real_weight_simplify_error.png`, `results/figures/pi0_shape_flow_step_reduction_error.png`, `results/figures/pi0_shape_flow_step_reduction_latency.png`
- summary: `results/summary.md`
- final report: `results/final_pytorch_benchmark_report.md`

- [pi0_shape_flow_step_reduction] start device=cuda, train_steps=500, hidden_dim=512.

- [pi0_shape_flow_step_reduction] start device=cuda, train_steps=500, hidden_dim=512.

- [pi0_shape_flow_step_reduction] completed rows=5, csv=results/csv/pi0_shape_flow_step_reduction.csv.

- [pi0_shape_flow_step_reduction] start device=cuda, train_steps=1500, hidden_dim=512.

- [pi0_shape_flow_step_reduction] completed rows=5, csv=results/csv/pi0_shape_flow_step_reduction.csv.

- [pi0_real_weight_quant] start device=cuda, selected=results/pi0_module_weights/selected_modules.pt.

- [pi0_real_weight_quant] completed rows=70, csv=results/csv/pi0_real_weight_quant.csv.

- [pi0_real_weight_simplify] start device=cuda, selected=results/pi0_module_weights/selected_modules.pt.

- [pi0_real_weight_simplify] completed rows=38, csv=results/csv/pi0_real_weight_simplify.csv.

- [pi0_real_weight_simplify] start device=cuda, selected=results/pi0_module_weights/selected_modules.pt.

- [pi0_real_weight_simplify] completed rows=38, csv=results/csv/pi0_real_weight_simplify.csv.
