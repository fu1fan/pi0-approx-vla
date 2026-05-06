#!/usr/bin/env python3
"""Back up Vitis Unified workspace configuration files.

The script copies only small configuration files needed to recover or audit the
workspace metadata. Build products, logs, caches, and prior backups are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


CONFIG_PATTERNS = ("*.json", "*.cfg", "*.ini", "*.tcl", "component.xml")
SKIP_DIR_NAMES = {
    ".cache",
    ".Xil",
    "build",
    "config_backups",
    "export",
    "logs",
    "syn",
    "sim",
    "impl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_skipped(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_DIR_NAMES or part.endswith("_kernel") for part in rel.parts[:-1])


def iter_config_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in CONFIG_PATTERNS:
        for path in root.rglob(pattern):
            if path.is_file() and not is_skipped(path, root):
                files.add(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default="vitis_workspace",
        type=Path,
        help="Vitis Unified workspace path.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Backup timestamp directory name. Defaults to current local time.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help="Backup root. Defaults to <workspace>/config_backups.",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if not workspace.exists():
        raise SystemExit(f"workspace does not exist: {workspace}")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_root or workspace / "config_backups").resolve()
    backup_dir = output_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    manifest_files: list[dict[str, str]] = []
    for src in iter_config_files(workspace):
        rel = src.relative_to(workspace)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_files.append(
            {
                "path": rel.as_posix(),
                "sha256": sha256_file(src),
                "bytes": str(src.stat().st_size),
            }
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(workspace),
        "backup_dir": str(backup_dir),
        "file_count": len(manifest_files),
        "files": manifest_files,
    }
    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Backed up {len(manifest_files)} Vitis config files to {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
