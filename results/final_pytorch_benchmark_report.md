# Final PyTorch Benchmark Report

## Scope

This project does not fully reproduce pi0, does not deploy an end-to-end VLA policy, and does not evaluate robot task success. The completed work is a set of PyTorch module-level proxy benchmarks for approximate computing in pi0/openpi-like modules.

The report separates five experiment classes:

1. random tensor + pi0-aligned shape quantization
2. random tensor + pi0-aligned shape simplification
3. real pi0 weight + random input quantization
4. real pi0 weight + random input simplification
5. pi0-shape toy flow step reduction

Because no real observation / activation / action dataset is used, these results cannot estimate real robot task success-rate degradation. They are suitable for module-level trend analysis and presentation material.

## Environment

- Conda env: `torch`
- Python: 3.13.12
- PyTorch: 2.10.0+cu130
- GPU used for runnable stages: NVIDIA GeForce RTX 5060 Ti
- CPU: AMD Ryzen 9 7950X 16-Core Processor
- OS: Ubuntu 24.04 family / Linux 6.17

## 1. Random Tensor + pi0-aligned Shape Quantization

Script: `pytorch_exp/exp_pi0_aligned_random_quant.py`

Output:

- `results/csv/pi0_aligned_random_quant.csv`
- `results/figures/pi0_aligned_random_quant_latency.png`
- `results/figures/pi0_aligned_random_quant_error.png`
- `results/figures/pi0_aligned_random_quant_size.png`
- `results/pi0_aligned_random_quant_summary.md`

Completed on CUDA with 100 rows. Modules include visual projector, Gemma-style q/k/v/o projections, gated VLM FFN, action expert FFN, and action/state projections.

Key result:

- INT8 minimum cosine: 0.998402
- INT4 weight-only minimum cosine: 0.963150
- W4A8 implemented; minimum cosine: 0.961928

Interpretation: INT8 is the most defensible quantization target from this random proxy stage. INT4/W4A8 provide stronger compression but need calibration/grouping or downstream validation.

## 2. Random Tensor + pi0-aligned Shape Simplification

Script: `pytorch_exp/exp_pi0_aligned_random_simplify.py`

Output:

- `results/csv/pi0_aligned_random_simplify.csv`
- `results/figures/pi0_aligned_random_simplify_latency.png`
- `results/figures/pi0_aligned_random_simplify_error.png`
- `results/pi0_aligned_random_simplify_summary.md`

Completed on CUDA with 59 rows. This stage tests FFN activation replacement, attention softmax approximation, and RMSNorm approximation using pi0-aligned random tensors.

Key result:

- FFN activation replacement is the most sensitive class when using aggressive replacements such as identity or clipped-linear variants.
- Softmax LUT/Taylor-style approximations are numerically more stable than coarse PWL or clipped-linear normalize.
- RMSNorm simplifications are comparatively stable in this proxy setup.

Interpretation: activation simplification should be treated carefully and evaluated after the full FFN, not only at the activation output. LUT softmax and Newton-Raphson rsqrt remain reasonable hardware-oriented candidates.

## 3. Real pi0 Weight + Random Input Quantization

Script: `pytorch_exp/exp_pi0_real_weight_quant.py`

Output:

- `results/csv/pi0_real_weight_quant.csv`
- `results/figures/pi0_real_weight_quant_latency.png`
- `results/figures/pi0_real_weight_quant_error.png`
- `results/figures/pi0_real_weight_quant_size.png`
- `results/pi0_real_weight_quant_summary.md`

Status: skipped numerically.

Reason: Stage 3 found a verifiable Hugging Face/LeRobot source (`lerobot/pi0_base`) and official openpi documents `gs://openpi-assets/checkpoints/pi0_base`, but the available HF checkpoint is a single 13.04 GiB `model.safetensors`. Under the configured 2 GiB safety limit, the full tensor file was not downloaded. No local tensor keys were available for extraction.

No real-weight quantization metrics were fabricated.

## 4. Real pi0 Weight + Random Input Simplification

Script: `pytorch_exp/exp_pi0_real_weight_simplify.py`

Output:

- `results/csv/pi0_real_weight_simplify.csv`
- `results/figures/pi0_real_weight_simplify_latency.png`
- `results/figures/pi0_real_weight_simplify_error.png`
- `results/pi0_real_weight_simplify_summary.md`

Status: skipped numerically for the same reason as Stage 3/4: verified real pi0 tensors were not locally available.

No real-weight simplification metrics were fabricated.

## 5. pi0-shape Toy Flow Step Reduction

Script: `pytorch_exp/exp_pi0_shape_flow_step_reduction.py`

Output:

- `results/csv/pi0_shape_flow_step_reduction.csv`
- `results/figures/pi0_shape_flow_step_reduction_error.png`
- `results/figures/pi0_shape_flow_step_reduction_latency.png`
- `results/pi0_shape_flow_step_reduction_summary.md`

Completed on CUDA. The toy setup uses `action_horizon=50`, `action_dim=32`, and `cond_dim=1024`.

Key result:

- 2 Euler steps vs 10-step toy baseline: about 4.46x latency speedup
- 2-step MSE vs 10-step output: 4.32e-03
- 2-step cosine vs 10-step output: 0.997542

Interpretation: this only demonstrates a latency-error tradeoff for reducing flow integration steps under a pi0-like action chunk shape. It does not imply real pi0 action quality.

## Checkpoint Handling

Scripts:

- `tools/download_pi0_checkpoint.py`
- `tools/inspect_pi0_checkpoint.py`
- `tools/extract_pi0_module_weights.py`

Outputs:

- `results/pi0_checkpoint_download_status.md`
- `results/pi0_checkpoint_files.txt`
- `results/pi0_checkpoint_keys.txt`
- `results/pi0_extracted_modules.md`

Large checkpoint files and extracted tensor files are ignored by git via `.gitignore`.

## Presentation-ready Conclusions

- Best near-term quantization target: INT8 Linear/projector/GEMM style modules. It preserves high cosine similarity in pi0-aligned random proxy tests and maps directly to memory-bandwidth savings.
- Risky but interesting compression target: INT4/W4A8. It gives stronger compression but showed much larger error in random proxy benchmarks.
- Best simplification candidates: LUT softmax, Taylor 3 softmax, and Newton-Raphson reciprocal sqrt for RMSNorm.
- Activation replacement should be evaluated at final FFN output. Aggressive replacements can produce large output error even when the activation approximation seems simple.
- Flow step reduction is a useful latency knob, but current evidence is toy-only and must not be framed as real pi0 policy quality.

## Next Steps

1. If real-weight results are required, explicitly allow downloading the 13.04 GiB verified `lerobot/pi0_base` safetensors file outside git tracking, then re-run extraction and Stage 4/5.
2. Add grouped INT4 / per-channel activation calibration for the pi0-aligned quantization stage.
3. Replace PyTorch fake quant timing with a true fused INT8/INT4 CUDA or cuBLASLt path if latency claims are needed.
4. For hardware work, prioritize INT8 GEMM and LUT softmax kernels before more speculative activation replacements.
