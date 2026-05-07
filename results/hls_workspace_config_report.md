# Vitis Workspace Configuration Report

Date: 2026-05-07

## Backup Before Comparison Baseline Edits

- Backup directory: `vitis_workspace/config_backups/20260507_000000_pre_comparison_baselines/`
- Backup manifest: `vitis_workspace/config_backups/20260507_000000_pre_comparison_baselines/manifest.json`
- Backed up 25 active Vitis workspace configuration files before adding exact baseline components.
- File classes covered: `*.json`, `*.cfg`, `*.ini`, `*.tcl`, and `component.xml` when present.
- Build/cache/log folders and prior backups were skipped by `tools/backup_vitis_workspace_config.py`.

## Config Files Modified Or Created

| File | Reason | Method | Validation |
| --- | --- | --- | --- |
| `vitis_workspace/exact_softmax/vitis-comp.json` | HLS component shell for float exp softmax baseline. | Created by `tools/update_vitis_workspace_components.py` from the existing component template using Python `json`. | JSON parse passed; component check passed. |
| `vitis_workspace/exact_softmax/hls_config.cfg` | Vitis Unified HLS config with `syn.top=exact_softmax_kernel`. | Created by the updater; no JSON schema involved. | Config validation passed. |
| `vitis_workspace/exact_gelu/vitis-comp.json` | HLS component shell for float tanh GELU baseline. | Created by `tools/update_vitis_workspace_components.py` from the existing component template using Python `json`. | JSON parse passed; component check passed. |
| `vitis_workspace/exact_gelu/hls_config.cfg` | Vitis Unified HLS config with `syn.top=exact_gelu_kernel`. | Created by the updater; no JSON schema involved. | Config validation passed. |
| `vitis_workspace/exact_rmsnorm/vitis-comp.json` | HLS component shell for float sqrt RMSNorm baseline. | Created by `tools/update_vitis_workspace_components.py` from the existing component template using Python `json`. | JSON parse passed; component check passed. |
| `vitis_workspace/exact_rmsnorm/hls_config.cfg` | Vitis Unified HLS config with `syn.top=exact_rmsnorm_kernel`. | Created by the updater; no JSON schema involved. | Config validation passed. |

No pre-existing Vitis Unified JSON schema file was rewritten. The component updater was extended so future component configs include explicit `syn.top` entries while preserving unknown JSON fields.

## Vitis Unified Recognition

The active workspace now has internally consistent component configs for:

- `int8_gemm`
- `exact_softmax`
- `lut_softmax`
- `exact_gelu`
- `gelu_pwl`
- `exact_rmsnorm`
- `rmsnorm_rsqrt`
- `fixed_projector_tile`

Recognition here means each component has a valid `vitis-comp.json`, the referenced `hls_config.cfg` exists, and the JSON parses successfully. The validation command used was `python tools/validate_vitis_workspace_config.py --workspace vitis_workspace --report results/hls_workspace_config_report.md --strict`.

## Validation Summary

| File | Kind | Status | Detail |
| --- | --- | --- | --- |
| `_ide/.peers.ini` | ini | ok | configparser accepted file |
| `_ide/.wsdata/clang_dir_map.json` | json | ok | valid JSON |
| `_ide/.wsdata/problems_data.json` | json | ok | valid JSON |
| `_ide/settings.json` | json | ok | valid JSON |
| `_ide/version.ini` | ini | ok | configparser accepted file |
| `exact_gelu/hls_config.cfg` | cfg | ok | configparser accepted file |
| `exact_gelu/vitis-comp.json` | json | ok | valid JSON |
| `exact_rmsnorm/hls_config.cfg` | cfg | ok | configparser accepted file |
| `exact_rmsnorm/vitis-comp.json` | json | ok | valid JSON |
| `exact_softmax/hls_config.cfg` | cfg | ok | configparser accepted file |
| `exact_softmax/vitis-comp.json` | json | ok | valid JSON |
| `fixed_projector_tile/compile_commands.json` | json | ok | valid JSON |
| `fixed_projector_tile/hls_config.cfg` | cfg | ok | configparser accepted file |
| `fixed_projector_tile/vitis-comp.json` | json | ok | valid JSON |
| `gelu_pwl/compile_commands.json` | json | ok | valid JSON |
| `gelu_pwl/hls_config.cfg` | cfg | ok | configparser accepted file |
| `gelu_pwl/vitis-comp.json` | json | ok | valid JSON |
| `hls_src/exact_gelu/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `hls_src/exact_rmsnorm/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `hls_src/exact_softmax/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `hls_src/fixed_projector_tile/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `hls_src/gelu_pwl/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `hls_src/int8_gemm/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `hls_src/lut_softmax/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `hls_src/rmsnorm_rsqrt/run_hls.tcl` | tcl | ok | syntax not parsed; file discovered |
| `int8_gemm/compile_commands.json` | json | ok | valid JSON |
| `int8_gemm/hls_config.cfg` | cfg | ok | configparser accepted file |
| `int8_gemm/vitis-comp.json` | json | ok | valid JSON |
| `lut_softmax/compile_commands.json` | json | ok | valid JSON |
| `lut_softmax/hls_config.cfg` | cfg | ok | configparser accepted file |
| `lut_softmax/vitis-comp.json` | json | ok | valid JSON |
| `rmsnorm_rsqrt/compile_commands.json` | json | ok | valid JSON |
| `rmsnorm_rsqrt/hls_config.cfg` | cfg | ok | configparser accepted file |
| `rmsnorm_rsqrt/vitis-comp.json` | json | ok | valid JSON |

## Component Recognition Checks

| Component Config | Status | Detail |
| --- | --- | --- |
| `exact_gelu/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `exact_rmsnorm/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `exact_softmax/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `fixed_projector_tile/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `gelu_pwl/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `int8_gemm/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `lut_softmax/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `rmsnorm_rsqrt/vitis-comp.json` | ok | Vitis component config is internally consistent |

## Modification Policy

- Vitis Unified JSON files are parsed with Python's `json` module.
- Existing fields are preserved by the component updater; unknown schema fields are not deleted.
- This report records validation status only. Kernel source changes are tracked in the kernel commits.
