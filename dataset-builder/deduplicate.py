from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def difference_hash(path: Path, size: int = 16) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    resized = cv2.resize(image, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_near_duplicates(paths: list[Path], max_distance: int = 8) -> list[dict]:
    hashes = [(path, difference_hash(path)) for path in paths]
    duplicates: list[dict] = []
    for index, (path, value) in enumerate(hashes):
        for other_path, other_value in hashes[index + 1 :]:
            distance = hamming(value, other_value)
            if distance <= max_distance:
                duplicates.append({"keep": str(path), "duplicate": str(other_path), "distance": distance})
    return duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("duplicate_report.json"))
    args = parser.parse_args()
    paths = sorted(p for p in args.frames.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    report = find_near_duplicates(paths, args.max_distance)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(paths), "near_duplicate_pairs": len(report)}, indent=2))


if __name__ == "__main__":
    main()
