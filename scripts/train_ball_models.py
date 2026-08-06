from __future__ import annotations

import argparse
from pathlib import Path


def train(data_yaml: Path, output: Path, model: str, image_size: int, epochs: int, batch: int) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the training extras: pip install '.[training]'") from exc

    output.mkdir(parents=True, exist_ok=True)
    detector = YOLO(model)
    detector.train(
        data=str(data_yaml),
        imgsz=image_size,
        epochs=epochs,
        batch=batch,
        project=str(output),
        name="ball-detector",
        patience=30,
        cos_lr=True,
        close_mosaic=15,
        cache=False,
        seed=42,
        deterministic=True,
    )
    detector.export(format="onnx", imgsz=image_size, simplify=True, dynamic=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export a GolfIQ golf-ball detector")
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=6)
    args = parser.parse_args()
    train(args.data, args.output, args.model, args.image_size, args.epochs, args.batch)


if __name__ == "__main__":
    main()
