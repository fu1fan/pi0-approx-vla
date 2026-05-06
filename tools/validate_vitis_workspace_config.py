#!/usr/bin/env python3
"""Validate Vitis Unified workspace configuration files."""

from __future__ import annotations

import argparse
import configparser
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATTERNS = ("*.json", "*.cfg", "*.ini", "*.tcl", "component.xml")
SKIP_DIR_NAMES = {
    ".cache",
    ".Xil",
    "build",
    "export",
    "logs",
    "syn",
    "sim",
    "impl",
}


@dataclass
class CheckResult:
    path: str
    kind: str
    status: str
    detail: str


def iter_config_files(root: Path, include_backups: bool) -> list[Path]:
    skip = set(SKIP_DIR_NAMES)
    if not include_backups:
        skip.add("config_backups")
    files: set[Path] = set()
    for pattern in CONFIG_PATTERNS:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in skip or part.endswith("_kernel") for part in rel.parts[:-1]):
                continue
            files.add(path)
    return sorted(files)


def validate_json(path: Path) -> tuple[str, str]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report exact parser failure.
        return "invalid", str(exc)
    return "ok", "valid JSON"


def validate_cfg(path: Path) -> tuple[str, str]:
    parser = configparser.ConfigParser(strict=False)
    try:
        text = path.read_text(encoding="utf-8")
        try:
            parser.read_string(text)
        except configparser.MissingSectionHeaderError:
            parser.read_string("[root]\n" + text)
    except Exception as exc:  # noqa: BLE001
        return "invalid", str(exc)
    if path.name == "hls_config.cfg":
        missing = []
        if not text.strip().startswith("part="):
            missing.append("part")
        if not parser.has_section("hls"):
            missing.append("[hls]")
        if missing:
            return "warn", "missing " + ", ".join(missing)
    return "ok", "configparser accepted file"


def validate_ini(path: Path) -> tuple[str, str]:
    parser = configparser.ConfigParser(strict=False)
    try:
        text = path.read_text(encoding="utf-8")
        try:
            parser.read_string(text)
        except configparser.MissingSectionHeaderError:
            parser.read_string("[root]\n" + text)
    except Exception as exc:  # noqa: BLE001
        return "invalid", str(exc)
    return "ok", "configparser accepted file"


def validate_xml(path: Path) -> tuple[str, str]:
    try:
        ET.parse(path)
    except Exception as exc:  # noqa: BLE001
        return "invalid", str(exc)
    return "ok", "valid XML"


def validate_file(path: Path) -> tuple[str, str, str]:
    if path.suffix == ".json":
        status, detail = validate_json(path)
        return "json", status, detail
    if path.suffix == ".cfg":
        status, detail = validate_cfg(path)
        return "cfg", status, detail
    if path.suffix == ".ini":
        status, detail = validate_ini(path)
        return "ini", status, detail
    if path.suffix == ".xml":
        status, detail = validate_xml(path)
        return "xml", status, detail
    if path.suffix == ".tcl":
        return "tcl", "ok", "syntax not parsed; file discovered"
    return "other", "ok", "file discovered"


def check_components(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    for comp_json in sorted(root.glob("*/vitis-comp.json")):
        rel = comp_json.relative_to(root).as_posix()
        try:
            data = json.loads(comp_json.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(rel, "component", "invalid", str(exc)))
            continue
        missing = [key for key in ("name", "type", "configuration") if key not in data]
        if missing:
            results.append(
                CheckResult(rel, "component", "invalid", "missing " + ", ".join(missing))
            )
            continue
        cfg_files = data.get("configuration", {}).get("configFiles", [])
        missing_cfg = [cfg for cfg in cfg_files if not (comp_json.parent / cfg).exists()]
        if missing_cfg:
            results.append(
                CheckResult(rel, "component", "invalid", "missing configFiles " + ", ".join(missing_cfg))
            )
            continue
        results.append(CheckResult(rel, "component", "ok", "Vitis component config is internally consistent"))
    return results


def write_report(path: Path, results: list[CheckResult], components: list[CheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Vitis Workspace Configuration Report",
        "",
        "## Validation Summary",
        "",
        "| File | Kind | Status | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(f"| `{item.path}` | {item.kind} | {item.status} | {item.detail} |")
    lines.extend(
        [
            "",
            "## Component Recognition Checks",
            "",
            "| Component Config | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for item in components:
        lines.append(f"| `{item.path}` | {item.status} | {item.detail} |")
    lines.extend(
        [
            "",
            "## Modification Policy",
            "",
            "- Vitis Unified JSON files are parsed with Python's `json` module.",
            "- Existing fields are preserved by the component updater; unknown schema fields are not deleted.",
            "- This report records validation status only. Kernel source changes are tracked in the kernel commits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="vitis_workspace", type=Path)
    parser.add_argument("--report", default=None, type=Path)
    parser.add_argument("--include-backups", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if not workspace.exists():
        raise SystemExit(f"workspace does not exist: {workspace}")

    results: list[CheckResult] = []
    for path in iter_config_files(workspace, include_backups=args.include_backups):
        kind, status, detail = validate_file(path)
        results.append(CheckResult(path.relative_to(workspace).as_posix(), kind, status, detail))
    components = check_components(workspace)

    invalid = [item for item in results + components if item.status == "invalid"]
    if args.report:
        write_report(args.report, results, components)

    for item in results + components:
        print(f"{item.status}\t{item.kind}\t{item.path}\t{item.detail}")

    if invalid and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
