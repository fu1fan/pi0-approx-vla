# Vitis Workspace Configuration Report

Date: 2026-05-06

## Backup Before Workspace Edits

- Backup directory: `vitis_workspace/config_backups/20260506_225000_pre_hls_kernel_edits/`
- Backup manifest: `vitis_workspace/config_backups/20260506_225000_pre_hls_kernel_edits/manifest.json`
- Backed up active config files before creating the optional `fixed_projector_tile` component config.
- Backed up file classes: `*.json`, `*.cfg`, `*.ini`, `*.tcl`, and `component.xml` when present.
- Skipped build/cache/log folders and previous backups.

## Config Files Modified Or Created

| File | Reason | Method | Validation |
| --- | --- | --- | --- |
| `vitis_workspace/fixed_projector_tile/vitis-comp.json` | Optional HLS component shell for the fixed-point visual projector tile. | Created by `tools/update_vitis_workspace_components.py` from the existing Vitis component JSON template using Python `json`. | `python -m json.tool` passed; component check passed. |
| `vitis_workspace/fixed_projector_tile/hls_config.cfg` | Vitis Unified HLS config for the optional component. | Created by `tools/update_vitis_workspace_components.py`; no JSON schema involved. | Config validation passed. |
| `vitis_workspace/fixed_projector_tile/.gitignore` | Prevent component-local build/cache/log artifacts from being committed. | Created by `tools/update_vitis_workspace_components.py`. | Not a JSON/config schema file. |

No pre-existing Vitis Unified JSON file was rewritten in this stage. Unknown schema fields in existing `vitis-comp.json` files were preserved.

## Vitis Unified Recognition

The active workspace now has internally consistent component configs for:

- `int8_gemm`
- `lut_softmax`
- `gelu_pwl`
- `rmsnorm_rsqrt`
- `fixed_projector_tile`

Recognition here means each component has a valid `vitis-comp.json`, the referenced `hls_config.cfg` exists, and the JSON parses successfully. GUI recognition was not manually launched because direct `vitis` writes to the HOME configuration area unless `XILINX_VITIS_DATA_DIR` is overridden.

## Subsequent Component Config Changes

After the initial backup, the active component `hls_config.cfg` files were updated to add the correct `syn.top` entries:

| Component | Config change | Reason |
| --- | --- | --- |
| `int8_gemm/hls_config.cfg` | `syn.top=int8_gemm_kernel` | Required for Vitis C-synthesis to bind the top function. |
| `lut_softmax/hls_config.cfg` | `syn.top=lut_softmax_kernel` | Required for Vitis C-synthesis to bind the top function. |
| `gelu_pwl/hls_config.cfg` | `syn.top=gelu_pwl_kernel` | Required for Vitis C-synthesis to bind the top function. |
| `rmsnorm_rsqrt/hls_config.cfg` | `syn.top=rmsnorm_rsqrt_kernel` | Required for Vitis C-synthesis to bind the top function. |
| `fixed_projector_tile/hls_config.cfg` | `syn.top=fixed_projector_tile_kernel` | Required for Vitis C-synthesis to bind the optional projector top function. |

No existing Vitis Unified JSON schema file was rewritten during these kernel commits. The final validation command `python tools/validate_vitis_workspace_config.py --strict` passed after Vitis-generated `*_kernel/` work directories were excluded from config scans.

## Validation Summary

| File | Kind | Status | Detail |
| --- | --- | --- | --- |
| `_ide/.peers.ini` | ini | ok | configparser accepted file |
| `_ide/.wsdata/clang_dir_map.json` | json | ok | valid JSON |
| `_ide/.wsdata/problems_data.json` | json | ok | valid JSON |
| `_ide/settings.json` | json | ok | valid JSON |
| `_ide/version.ini` | ini | ok | configparser accepted file |
| `fixed_projector_tile/hls_config.cfg` | cfg | ok | configparser accepted file |
| `fixed_projector_tile/vitis-comp.json` | json | ok | valid JSON |
| `gelu_pwl/compile_commands.json` | json | ok | valid JSON |
| `gelu_pwl/hls_config.cfg` | cfg | ok | configparser accepted file |
| `gelu_pwl/vitis-comp.json` | json | ok | valid JSON |
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
| `fixed_projector_tile/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `gelu_pwl/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `int8_gemm/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `lut_softmax/vitis-comp.json` | ok | Vitis component config is internally consistent |
| `rmsnorm_rsqrt/vitis-comp.json` | ok | Vitis component config is internally consistent |

## Modification Policy

- Vitis Unified JSON files are parsed with Python's `json` module.
- Existing fields are preserved by the component updater; unknown schema fields are not deleted.
- This report records validation status only. Kernel source changes are tracked in the kernel commits.
