from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BALL_COLORS = [
    (245, 245, 245),
    (225, 245, 70),
    (245, 165, 45),
]


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def motion_blur(image: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return image
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def add_ball(canvas: np.ndarray, rng: random.Random, flight: bool) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = canvas.shape[:2]
    radius = rng.randint(2 if flight else 5, 14 if flight else 28)
    margin = max(radius + 4, 8)
    cx = rng.randint(margin, max(margin, w - margin - 1))
    cy = rng.randint(margin, max(margin, h - margin - 1))
    color = BALL_COLORS[rng.randrange(len(BALL_COLORS))]

    pad = max(20, radius * 5)
    patch_size = pad * 2 + 1
    patch = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)
    pc = pad
    cv2.circle(patch, (pc, pc), radius, (*color, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(patch, (pc - max(1, radius // 4), pc - max(1, radius // 4)), max(1, radius // 4), (255, 255, 255, 90), -1, lineType=cv2.LINE_AA)

    # Very subtle dimples only when the ball is large enough to resolve them.
    if radius >= 11:
        for _ in range(max(4, radius // 2)):
            ang = rng.random() * math.tau
            rr = rng.uniform(0.2, 0.75) * radius
            dx, dy = int(math.cos(ang) * rr), int(math.sin(ang) * rr)
            cv2.circle(patch, (pc + dx, pc + dy), max(1, radius // 12), (180, 180, 180, 70), -1, lineType=cv2.LINE_AA)

    if flight:
        blur_len = rng.choice([1, 2, 3, 5, 7, 9, 11])
        patch[:, :, :3] = motion_blur(patch[:, :, :3], blur_len, rng.uniform(-75, 75))
        patch[:, :, 3] = motion_blur(patch[:, :, 3], blur_len, rng.uniform(-75, 75))

    x0, y0 = cx - pad, cy - pad
    x1, y1 = x0 + patch_size, y0 + patch_size
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = patch_size - max(0, x1 - w), patch_size - max(0, y1 - h)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x1), min(h, y1)

    crop = patch[sy0:sy1, sx0:sx1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    canvas[dy0:dy1, dx0:dx1] = (
        crop[:, :, :3].astype(np.float32) * alpha
        + canvas[dy0:dy1, dx0:dx1].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    scale_x = 1.8 if flight else 1.15
    scale_y = 1.25 if flight else 1.15
    bw = int(radius * 2 * scale_x)
    bh = int(radius * 2 * scale_y)
    return canvas, (max(0, cx - bw // 2), max(0, cy - bh // 2), min(w, bw), min(h, bh))


def camera_effects(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)
    gain = rng.uniform(0.72, 1.28)
    bias = rng.uniform(-20, 20)
    out = np.clip(out * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        sigma = rng.uniform(0.2, 1.4)
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    if rng.random() < 0.35:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 8.0), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        quality = rng.randint(45, 94)
        ok, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out


def yolo_line(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x, y, bw, bh = box
    cx = (x + bw / 2) / width
    cy = (y + bh / 2) / height
    return f"0 {cx:.8f} {cy:.8f} {bw/width:.8f} {bh/height:.8f}\n"


def generate(backgrounds: Path, output: Path, count: int, negative_ratio: float, flight_ratio: float, seed: int) -> dict:
    files = list_images(backgrounds)
    if not files:
        raise SystemExit(f"No background images found under {backgrounds}")
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = []

    positive = negative = flight_count = 0
    for i in range(count):
        source = files[rng.randrange(len(files))]
        image = cv2.imread(str(source))
        if image is None:
            continue
        h, w = image.shape[:2]
        if min(h, w) < 320:
            scale = 320 / min(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            h, w = image.shape[:2]
        target_w = rng.choice([640, 768, 960, 1280])
        scale = target_w / w
        image = cv2.resize(image, (target_w, max(360, int(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        h, w = image.shape[:2]

        is_negative = rng.random() < negative_ratio
        is_flight = (not is_negative) and rng.random() < flight_ratio
        label = ""
        if not is_negative:
            image, box = add_ball(image, rng, is_flight)
            label = yolo_line(box, w, h)
            positive += 1
            flight_count += int(is_flight)
        else:
            negative += 1

        image = camera_effects(image, rng)
        name = f"synthetic_{i:07d}.jpg"
        cv2.imwrite(str(images_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(78, 96)])
        (labels_dir / f"synthetic_{i:07d}.txt").write_text(label, encoding="utf-8")
        manifest.append({
            "image": name,
            "background_source": str(source),
            "synthetic": True,
            "negative": is_negative,
            "flight": is_flight,
            "split_policy": "train_only",
        })

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    return {"generated": len(manifest), "positive": positive, "negative": negative, "flight": flight_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train-only synthetic golf-ball images from licensed backgrounds.")
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/synthetic-v1"))
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--negative-ratio", type=float, default=0.32)
    parser.add_argument("--flight-ratio", type=float, default=0.68)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate(args.backgrounds, args.output, args.count, args.negative_ratio, args.flight_ratio, args.seed), indent=2))


if __name__ == "__main__":
    main()
