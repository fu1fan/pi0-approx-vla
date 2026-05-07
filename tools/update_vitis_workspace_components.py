#!/usr/bin/env python3
"""Create missing Vitis Unified HLS component config from an existing template.

The updater is intentionally conservative: by default it creates only missing
files and never rewrites an existing `vitis-comp.json`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


DEFAULT_COMPONENTS = (
    "int8_gemm",
    "lut_softmax",
    "exact_softmax",
    "gelu_pwl",
    "exact_gelu",
    "rmsnorm_rsqrt",
    "exact_rmsnorm",
    "fixed_projector_tile",
)

TOP_FUNCTIONS = {
    "int8_gemm": "int8_gemm_kernel",
    "lut_softmax": "lut_softmax_kernel",
    "exact_softmax": "exact_softmax_kernel",
    "gelu_pwl": "gelu_pwl_kernel",
    "exact_gelu": "exact_gelu_kernel",
    "rmsnorm_rsqrt": "rmsnorm_rsqrt_kernel",
    "exact_rmsnorm": "exact_rmsnorm_kernel",
    "fixed_projector_tile": "fixed_projector_tile_kernel",
}


def default_cfg(name: str) -> str:
    top = TOP_FUNCTIONS.get(name, f"{name}_kernel")
    return f"""part=xcvu9p-flga2104-2-i

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.file=main.cpp
syn.top={top}
tb.file=test.cpp
"""


def load_template(workspace: Path) -> dict:
    for path in sorted(workspace.glob("*/vitis-comp.json")):
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": "template",
        "type": "HLS",
        "configuration": {
            "componentType": "HLS",
            "configFiles": ["hls_config.cfg"],
            "work_dir": "template",
        },
        "useSysrootToolchain": False,
        "template": "empty_hls_component",
        "templateFlow": "VITIS",
    }


def component_json(template: dict, name: str) -> dict:
    data = copy.deepcopy(template)
    data["name"] = name
    data["type"] = data.get("type", "HLS")
    configuration = data.setdefault("configuration", {})
    configuration["componentType"] = configuration.get("componentType", "HLS")
    configuration["configFiles"] = configuration.get("configFiles", ["hls_config.cfg"])
    configuration["work_dir"] = name
    data["template"] = data.get("template", "empty_hls_component")
    data["templateFlow"] = data.get("templateFlow", "VITIS")
    return data


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="vitis_workspace", type=Path)
    parser.add_argument(
        "--components",
        nargs="*",
        default=list(DEFAULT_COMPONENTS),
        help="Component names to ensure.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Rewrite existing component JSON/config files using the template.",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    template = load_template(workspace)

    changed: list[str] = []
    for name in args.components:
        comp_dir = workspace / name
        comp_dir.mkdir(parents=True, exist_ok=True)

        comp_json = comp_dir / "vitis-comp.json"
        if args.refresh_existing or not comp_json.exists():
            write_json(comp_json, component_json(template, name))
            changed.append(comp_json.relative_to(workspace).as_posix())

        cfg = comp_dir / "hls_config.cfg"
        if args.refresh_existing or not cfg.exists():
            cfg.write_text(default_cfg(name), encoding="utf-8")
            changed.append(cfg.relative_to(workspace).as_posix())

        gitignore = comp_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("/build\n/export\n/.cache\n*.log\n*.jou\n*.wdb\n", encoding="utf-8")
            changed.append(gitignore.relative_to(workspace).as_posix())

    if changed:
        print("changed:")
        for path in changed:
            print(f"  {path}")
    else:
        print("no component config changes required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
