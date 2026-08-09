from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def load_reviews(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def assign_groups(groups: list[str], seed: int = 42) -> dict[str, str]:
    rng = random.Random(seed)
    groups = sorted(set(groups))
    rng.shuffle(groups)
    n = len(groups)
    train_end = max(1, round(n * 0.70))
    val_end = max(train_end + 1, round(n * 0.85)) if n >= 3 else train_end
    mapping = {}
    for i, group in enumerate(groups):
        mapping[group] = "train" if i < train_end else "val" if i < val_end else "test"
    return mapping


def build(reviews_path: Path, output: Path, seed: int = 42) -> dict:
    rows = [r for r in load_reviews(reviews_path) if r["visibility"] != "unclear"]
    mapping = assign_groups([r["source_group"] for r in rows], seed)
    stats = defaultdict(lambda: {"images": 0, "positive": 0, "negative": 0, "groups": set()})

    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        split = mapping[row["source_group"]]
        image = Path(row["frame"])
        label = Path(row["label"])
        if not image.exists() or not label.exists():
            raise FileNotFoundError(f"Missing image or label for {row['frame']}")
        shutil.copy2(image, output / "images" / split / image.name)
        shutil.copy2(label, output / "labels" / split / label.name)
        stats[split]["images"] += 1
        stats[split]["positive" if row["visibility"] == "visible" else "negative"] += 1
        stats[split]["groups"].add(row["source_group"])

    yaml = (
        f"path: {output.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: golf_ball\n"
    )
    (output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    report = {
        split: {**values, "groups": sorted(values["groups"])}
        for split, values in stats.items()
    }
    (output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.reviews, args.output, args.seed), indent=2))
