# Vitis Unified 2025.2 Notes

This repository assumes Vitis Unified 2025.2 under `/opt/Xilinx`. The user creates projects/components manually. The provided HLS files are small kernel examples, not full pi0 deployment assets.

## Suggested Flow

1. Source the Vitis environment. The requested check path `/opt/Xilinx/Vitis/2025.2/settings64.sh` may not exist on Unified installs; this machine exposes `/opt/Xilinx/2025.2/Vitis/settings64.sh`.

```bash
source /opt/Xilinx/2025.2/Vitis/settings64.sh
```

2. Launch Vitis Unified:

```bash
vitis
```

3. Prepare workspace-local source copies:

```bash
bash scripts/prepare_vitis_workspace.sh
```

4. Create a new HLS component in `vitis_workspace`.

5. Add one kernel folder from inside the workspace, for example:

```text
vitis_workspace/hls_src/int8_linear/int8_linear.cpp
vitis_workspace/hls_src/int8_linear/int8_linear.h
vitis_workspace/hls_src/int8_linear/testbench.cpp
```

6. Set the top function:

| Kernel folder | Top function |
|---|---|
| `hls/int8_linear` | `int8_linear` |
| `hls/fixed_projector` | `fixed_projector` |
| `hls/lut_softmax` | `lut_softmax` |
| `hls/gelu_pwl` | `gelu_pwl` |

7. Run C simulation and confirm pass/fail output.

8. Run synthesis.

9. Export reports or manually copy summary numbers into `docs/result_table_template.md` and `results/hls_reports/`.

## Source Location Guidance

AMD's 2025.2 documentation defines the workspace as the folder that holds the elements of HLS components or other design projects. The HLS config format can reference source and testbench files by absolute path or paths relative to the config file, and it also supports `relative_roots`. That is useful for scripted or advanced flows.

For this repository, prefer the simpler GUI flow: keep the files selected by a Vitis Unified HLS component under `vitis_workspace/hls_src/<kernel>/`. This avoids brittle outside-workspace references and makes export/import behavior easier to reason about.

## Scope

Packaging, bitstream generation, board deployment, full pi0 checkpoint loading, and full VLA model integration are intentionally out of scope.

## TCL Templates

Each HLS kernel directory includes `run_hls.tcl`. These files use the older command-oriented HLS project flow and are kept as references. AMD's 2025.2 guidance says `open_project` and `open_solution` are batch-oriented in 2025.1+; for an IDE-compatible component structure, use the Unified GUI or an `open_component`-based flow.
