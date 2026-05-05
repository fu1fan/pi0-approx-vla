# pi0-approx-vla

This repository is a module-level experiment workspace for approximate computing in VLA / pi0-style systems. It does not reproduce pi0, download large robot datasets, or deploy a full VLA model. A verified `lerobot/pi0_base` checkpoint was downloaded only into ignored local cache for module-level real-weight benchmarks; checkpoint files and extracted tensor bundles are not tracked by Git.

The goal is to generate interview-ready evidence with PyTorch experiments and Vitis HLS kernels for modules that appear in VLA pipelines:

- VLM projector: visual tokens projected into language embedding space
- Transformer linear / FFN GEMM
- Attention softmax
- GELU / activation approximation

## Environment

- Conda environment: `torch`
- Vitis location: `/opt/Xilinx`
- Expected Vitis version: Vitis Unified 2025.2
- HLS projects/components are created manually by the user in Vitis Unified

Run the environment check:

```bash
bash scripts/check_env.sh
```

The latest check is written to `docs/environment_check.md`. In the current workspace check, `torch` exists, `vitis` and `v++` are available after sourcing Vitis settings, and `vitis_hls` is not found.

## PyTorch Experiments

Install missing Python dependencies into the existing `torch` conda environment:

```bash
conda activate torch
bash scripts/install_missing_deps.sh
```

Run all default experiments:

```bash
bash scripts/run_all_pytorch.sh
```

The wrapper defaults to CPU with low repeat counts for a first pass:

```bash
DEVICE=cpu REPEAT=5 WARMUP=2 bash scripts/run_all_pytorch.sh
```

Every experiment script also supports direct CLI parameters:

```bash
conda run -n torch python pytorch_exp/exp_linear_quant.py --device cpu --repeat 50 --warmup 10
```

Large optional shapes are intentionally gated behind `--include-large` because CPU runs can be slow:

- Linear `batch=1, seq=128, in_dim=4096, out_dim=4096`
- Softmax `heads=16, seq=256, seq=256`

## Outputs

- CSV results: `results/csv/`
- Figures: `results/figures/`
- HLS reports copied from Vitis: `results/hls_reports/`

## Latest PyTorch Benchmark Reports

The current PyTorch side includes earlier module microbenchmarks, scale sweeps, pi0-aligned random proxy benchmarks, and real pi0 weight + random input module benchmarks:

- Random tensor + pi0-aligned shape quantization: `results/pi0_aligned_random_quant_summary.md`
- Random tensor + pi0-aligned function simplification: `results/pi0_aligned_random_simplify_summary.md`
- Real pi0 checkpoint metadata / extraction status: `results/pi0_checkpoint_download_status.md`, `results/pi0_extracted_modules.md`
- Real pi0 weight + random input quantization: `results/pi0_real_weight_quant_summary.md`
- Real pi0 weight + random input simplification: `results/pi0_real_weight_simplify_summary.md`
- pi0-shape toy flow step reduction: `results/pi0_shape_flow_step_reduction_summary.md`
- Consolidated report: `results/final_pytorch_benchmark_report.md`

The real-weight stages use extracted module tensors from the verified `lerobot/pi0_base` safetensors checkpoint with random inputs. They reflect real parameter distributions for selected modules, but still do not measure real pi0 robot-task behavior.

## HLS / Vitis Unified

Source the Vitis environment. The user-requested path is checked by `scripts/check_env.sh`; on this machine the discovered Vitis Unified path is `/opt/Xilinx/2025.2/Vitis/settings64.sh`.

```bash
source /opt/Xilinx/2025.2/Vitis/settings64.sh
```

Recommended Vitis Unified flow:

1. Prepare a workspace-local source copy:

```bash
bash scripts/prepare_vitis_workspace.sh
```

2. Launch Vitis Unified with the workspace:

```bash
vitis -w vitis_workspace
```

3. Select `File > New Component > HLS`.
4. Set Component location to `vitis_workspace`.
5. Create one component per kernel, for example `int8_linear`.
6. On the Source Files page, add files from `vitis_workspace/hls_src/int8_linear/`, not from the repo root.
7. Set the top function listed in that kernel README.
8. Run C simulation.
9. Run synthesis.
10. Export or copy report summaries into `results/hls_reports/`.

AMD documentation says the workspace is the folder used to hold HLS component elements and design source/data/config. HLS config files can technically point to relative or absolute source paths, but for the Vitis Unified GUI the least surprising flow is to keep the files used by a component under the workspace.

If a compatible `vitis_hls` command is present, each kernel also includes a `run_hls.tcl` reference:

```bash
cd hls/int8_linear
vitis_hls -f run_hls.tcl
```

On Vitis Unified-only installs, the TCL files should be treated as source-list and settings templates, not as the primary workflow. Vitis HLS Classic project commands such as `open_project` and `open_solution` are batch-oriented in 2025.1+ and are not the preferred way to create IDE-compatible components.

## Interview Framing

Module-level optimization is used because full pi0/VLA deployment requires large checkpoints, large datasets, and system integration that are outside this project scope. The module view still maps cleanly to practical accelerator targets: projector GEMM, Transformer linear layers, attention softmax, and activation functions.

INT8, INT4 weight-only, and fixed-point formats reduce memory bandwidth, storage, and arithmetic cost. Softmax and GELU are good approximation targets because they contain expensive nonlinear operations that can be replaced by LUT or PWL approximations with measurable error/latency tradeoffs.

The FPGA side intentionally implements kernels rather than a full pi0 model. That keeps the work synthesizable, inspectable, and appropriate for Vitis HLS reports such as latency, II, LUT, FF, BRAM, and DSP usage.
