from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the non-GPU portions of GolfIQ Dataset v1.")
    parser.add_argument("--root", type=Path, default=Path("datasets/v1-work"))
    parser.add_argument("--commons-limit", type=int, default=150)
    parser.add_argument("--synthetic-count", type=int, default=50000)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    root = args.root
    commons = root / "commons"
    synthetic = root / "synthetic"
    root.mkdir(parents=True, exist_ok=True)

    acquire_cmd = [
        sys.executable,
        "dataset-builder/acquire_commons.py",
        "--output",
        str(commons),
        "--limit-per-query",
        str(args.commons_limit),
    ]
    if not args.skip_download:
        acquire_cmd.append("--download")
    run(acquire_cmd)

    media = commons / "media"
    if media.exists() and any(media.iterdir()) and args.synthetic_count > 0:
        run([
            sys.executable,
            "dataset-builder/generate_synthetic_ball.py",
            "--backgrounds",
            str(media),
            "--output",
            str(synthetic),
            "--count",
            str(args.synthetic_count),
            "--negative-ratio",
            "0.34",
            "--flight-ratio",
            "0.72",
        ])

    status = {
        "commons_manifest": str(commons / "accepted.json"),
        "commons_attribution": str(commons / "attribution.csv"),
        "commons_rejected": str(commons / "rejected.json"),
        "synthetic_manifest": str(synthetic / "manifest.jsonl"),
        "important": [
            "Synthetic images are train-only and must never enter validation/test.",
            "Real flight frames still require human ball annotation.",
            "Distance release still requires measured carry shots.",
        ],
    }
    (root / "build_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
