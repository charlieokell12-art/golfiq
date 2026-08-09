from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="GolfIQ Kaggle training pipeline")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root containing dataset.yaml")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/golfiq"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--setup-size", type=int, default=960)
    parser.add_argument("--flight-size", type=int, default=1280)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    data = args.dataset / "dataset.yaml"
    if not data.exists():
        raise SystemExit(f"Missing {data}")

    run([
        sys.executable, "scripts/train_ball_models.py",
        "--data", str(data),
        "--output", str(args.output / "setup"),
        "--model", "yolo11s.pt",
        "--image-size", str(args.setup_size),
        "--epochs", str(args.epochs),
        "--batch", "10",
    ])

    run([
        sys.executable, "scripts/train_ball_models.py",
        "--data", str(data),
        "--output", str(args.output / "flight"),
        "--model", "yolo11s.pt",
        "--image-size", str(args.flight_size),
        "--epochs", str(args.epochs),
        "--batch", "6",
    ])

    summary = {
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "setup_image_size": args.setup_size,
        "flight_image_size": args.flight_size,
        "next_step": "Run held-out detector/tracker evaluation, then vision-engine/evaluate_release.py",
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
