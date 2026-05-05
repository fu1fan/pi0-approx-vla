import argparse
import json
from pathlib import Path


def keys_from_safetensors(path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return [(key, f.get_tensor(key).shape) for key in f.keys()]


def keys_from_index(path):
    data = json.loads(path.read_text())
    weight_map = data.get("weight_map", {})
    return [(key, "index_only") for key in sorted(weight_map)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="external/pi0_checkpoints/lerobot_pi0_base")
    parser.add_argument("--out", default="results/pi0_checkpoint_keys.txt")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    rows = []
    notes = []
    if not ckpt.exists():
        notes.append(f"checkpoint dir missing: {ckpt}")
    else:
        for path in sorted(ckpt.rglob("*.safetensors.index.json")):
            try:
                rows.extend((str(path), key, shape) for key, shape in keys_from_index(path))
            except Exception as exc:
                notes.append(f"failed index {path}: {exc}")
        for path in sorted(ckpt.rglob("*.safetensors")):
            try:
                rows.extend((str(path), key, list(shape)) for key, shape in keys_from_safetensors(path))
            except Exception as exc:
                notes.append(f"failed safetensors {path}: {exc}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# path\tkey\tshape"]
    lines.extend(f"{path}\t{key}\t{shape}" for path, key, shape in rows)
    if notes:
        lines.append("# notes")
        lines.extend(f"# {note}" for note in notes)
    if not rows:
        lines.append("# no parameter keys found")
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
