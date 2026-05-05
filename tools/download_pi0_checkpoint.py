import argparse
import json
import os
from pathlib import Path


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/pi0_base")
    parser.add_argument("--local-dir", default="external/pi0_checkpoints/lerobot_pi0_base")
    parser.add_argument("--max-download-gb", type=float, default=2.0)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.endpoint)
    status_lines = [
        "# pi0 Checkpoint Download Status",
        "",
        f"- requested repo: `{args.repo_id}`",
        f"- endpoint: `{os.environ.get('HF_ENDPOINT')}`",
        "- official openpi base checkpoint documented at `gs://openpi-assets/checkpoints/pi0_base`.",
        "- Hugging Face source attempted as a verifiable LeRobot conversion/reference mirror.",
        "",
    ]

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        status_lines.append(f"- failed: huggingface_hub unavailable: `{exc}`")
        write_text(Path("results/pi0_checkpoint_download_status.md"), "\n".join(status_lines))
        return 2

    local_dir = Path(args.local_dir)
    files_path = Path("results/pi0_checkpoint_files.txt")
    try:
        api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
        info = api.model_info(args.repo_id, files_metadata=True)
        siblings = info.siblings
        file_records = []
        total_weight_bytes = 0
        for item in siblings:
            size = getattr(item, "size", None) or 0
            name = item.rfilename
            file_records.append({"name": name, "size": size})
            if name.endswith((".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".ckpt")):
                total_weight_bytes += size
        files_path.parent.mkdir(parents=True, exist_ok=True)
        files_path.write_text("\n".join(f"{rec['size']}\t{rec['name']}" for rec in file_records))
        status_lines.append(f"- listed files: {len(file_records)}")
        status_lines.append(f"- estimated weight bytes: {total_weight_bytes}")

        max_bytes = int(args.max_download_gb * 1024**3)
        if total_weight_bytes > max_bytes:
            status_lines.append(
                f"- full checkpoint skipped: estimated weights {total_weight_bytes / 1024**3:.2f} GiB exceed limit {args.max_download_gb:.2f} GiB."
            )
            allow_patterns = ["*.json", "*.md", "*.txt", "*.index.json", "config.*", "README*"]
        else:
            status_lines.append("- full checkpoint allowed by size limit.")
            allow_patterns = None

        snapshot_download(
            repo_id=args.repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        status_lines.append(f"- downloaded to: `{local_dir}`")
        status_lines.append(f"- allow_patterns: `{allow_patterns}`")
        write_text(Path("results/pi0_checkpoint_download_status.md"), "\n".join(status_lines))
        write_text(local_dir / "download_manifest.json", json.dumps({"repo_id": args.repo_id, "endpoint": os.environ.get("HF_ENDPOINT"), "files": file_records}, indent=2))
        return 0
    except Exception as exc:
        status_lines.append(f"- failed: `{type(exc).__name__}: {exc}`")
        status_lines.append("- degradation: no unverified third-party weights used; downstream real-weight stages should skip missing modules.")
        write_text(Path("results/pi0_checkpoint_download_status.md"), "\n".join(status_lines))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
