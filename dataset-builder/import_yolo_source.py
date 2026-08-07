from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BALL_NAMES = {"golfball", "golf ball", "golf-ball", "golf_ball", "ball"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_yaml(root: Path) -> Path:
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        path = root / name
        if path.exists():
            return path
    matches = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    if not matches:
        raise FileNotFoundError(f"No YOLO data YAML found under {root}")
    return matches[0]


def ball_ids(data: dict) -> set[int]:
    names = data.get("names", {})
    if isinstance(names, list):
        items = enumerate(names)
    elif isinstance(names, dict):
        items = ((int(k), v) for k, v in names.items())
    else:
        raise ValueError("Unsupported YOLO names format")
    result = {idx for idx, name in items if str(name).strip().lower() in BALL_NAMES}
    if not result:
        raise ValueError(f"No golf-ball class found in names={names!r}")
    return result


def find_label(image: Path, root: Path) -> Path | None:
    parts = list(image.parts)
    candidates = []
    for i, part in enumerate(parts):
        if part.lower() in {"images", "image"}:
            alt = parts.copy()
            alt[i] = "labels"
            candidates.append(Path(*alt).with_suffix(".txt"))
    candidates.extend(root.rglob(image.stem + ".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def convert_label(label_path: Path | None, allowed_ids: set[int]) -> str:
    if label_path is None:
        return ""
    output = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id in allowed_ids:
            output.append("0 " + " ".join(parts[1:5]))
    return "\n".join(output) + ("\n" if output else "")


def import_source(source: Path, output: Path, source_id: str, license_name: str, seen_hashes: Path | None = None) -> dict:
    data = yaml.safe_load(find_yaml(source).read_text(encoding="utf-8"))
    allowed_ids = ball_ids(data)
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    out_images = output / "images" / "train"
    out_labels = output / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    known: set[str] = set()
    if seen_hashes and seen_hashes.exists():
        known.update(line.strip() for line in seen_hashes.read_text().splitlines() if line.strip())

    imported = duplicates = positives = negatives = 0
    records = []
    new_hashes = []
    for image in images:
        digest = sha256(image)
        if digest in known:
            duplicates += 1
            continue
        known.add(digest)
        new_hashes.append(digest)
        label = convert_label(find_label(image, source), allowed_ids)
        name = f"{source_id}__{digest[:12]}{image.suffix.lower()}"
        shutil.copy2(image, out_images / name)
        (out_labels / f"{Path(name).stem}.txt").write_text(label, encoding="utf-8")
        positive = bool(label.strip())
        positives += int(positive)
        negatives += int(not positive)
        imported += 1
        records.append({
            "image": name,
            "sha256": digest,
            "source_id": source_id,
            "license": license_name,
            "positive": positive,
            "split_policy": "train_only",
        })

    manifest = output / "third_party_manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    if seen_hashes:
        seen_hashes.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if seen_hashes.exists():
            existing = [line.strip() for line in seen_hashes.read_text().splitlines() if line.strip()]
        seen_hashes.write_text("\n".join(existing + new_hashes) + "\n", encoding="utf-8")
    return {"imported": imported, "duplicates": duplicates, "positives": positives, "negatives": negatives}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a legally licensed YOLO dataset as train-only GolfIQ data.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/v1-third-party"))
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--seen-hashes", type=Path, default=Path("datasets/v1-third-party/seen_sha256.txt"))
    args = parser.parse_args()
    print(json.dumps(import_source(args.source, args.output, args.source_id, args.license, args.seen_hashes), indent=2))


if __name__ == "__main__":
    main()
