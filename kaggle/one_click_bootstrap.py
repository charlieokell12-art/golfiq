from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path('/kaggle/working')
PROJECT = ROOT / 'golfiq'
DATA_ROOT = ROOT / 'golfiq-auto-data'
KAGGLE_SOURCE = 'tuhinsuryachakra/golf-ball-dataset'


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print('\n+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_packages() -> None:
    run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'ultralytics', 'onnx', 'onnxruntime', 'opencv-python-headless',
        'pandas', 'pydantic', 'scikit-learn', 'pillow', 'pyyaml', 'tqdm', 'kagglehub'
    ])


def gpu_info() -> dict:
    try:
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True)
        gpus = [line.strip() for line in text.splitlines() if line.strip()]
    except Exception:
        gpus = []
    return {'gpus': gpus, 'device': '0,1' if len(gpus) >= 2 else ('0' if gpus else 'cpu')}


def download_kaggle_source() -> Path:
    import kagglehub
    print(f'\nDownloading Kaggle source: {KAGGLE_SOURCE}', flush=True)
    path = Path(kagglehub.dataset_download(KAGGLE_SOURCE))
    print(f'Kaggle source path: {path}', flush=True)
    return path


def image_files(root: Path) -> list[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts]


def unpack_archives(root: Path) -> int:
    """Recursively unpack common archive formats found inside Kaggle datasets."""
    extracted = 0
    seen: set[Path] = set()
    for _ in range(3):
        archives = [p for p in root.rglob('*') if p.is_file() and p not in seen and p.suffix.lower() in {'.zip', '.tar', '.tgz', '.gz'}]
        if not archives:
            break
        for archive in archives:
            seen.add(archive)
            target = archive.parent / f'{archive.stem}_unpacked'
            target.mkdir(parents=True, exist_ok=True)
            try:
                if zipfile.is_zipfile(archive):
                    with zipfile.ZipFile(archive) as zf:
                        zf.extractall(target)
                    extracted += 1
                elif tarfile.is_tarfile(archive):
                    with tarfile.open(archive) as tf:
                        tf.extractall(target, filter='data')
                    extracted += 1
            except Exception as exc:
                print(f'Skipping archive {archive}: {exc}', flush=True)
    if extracted:
        print(f'Archives unpacked: {extracted}', flush=True)
    return extracted


def generate_fallback_backgrounds(pool: Path, count: int = 300) -> int:
    """Create diverse golf-like backgrounds when external image packaging is unusable.

    These are synthetic and therefore train-only. They are not validation evidence.
    """
    import cv2
    import numpy as np
    rng = np.random.default_rng(42)
    pool.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        h = int(rng.choice([720, 900, 1080]))
        w = int(rng.choice([960, 1280, 1600, 1920]))
        kind = i % 5
        if kind == 0:  # grass
            base = np.zeros((h, w, 3), dtype=np.uint8)
            base[..., 0] = rng.integers(20, 70, (h, w), dtype=np.uint8)
            base[..., 1] = rng.integers(65, 150, (h, w), dtype=np.uint8)
            base[..., 2] = rng.integers(20, 75, (h, w), dtype=np.uint8)
        elif kind == 1:  # range mat
            tone = rng.integers(35, 95)
            base = np.full((h, w, 3), (tone // 2, tone, tone // 2), dtype=np.uint8)
            noise = rng.normal(0, 18, (h, w, 1)).astype(np.int16)
            base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif kind == 2:  # sky
            y = np.linspace(0, 1, h)[:, None, None]
            top = np.array([190, 145, 85], dtype=float).reshape(1, 1, 3)
            bottom = np.array([245, 220, 180], dtype=float).reshape(1, 1, 3)
            base = np.tile((top * (1-y) + bottom * y), (1, w, 1)).astype(np.uint8)
        elif kind == 3:  # sand
            base = np.zeros((h, w, 3), dtype=np.uint8)
            base[..., 0] = rng.integers(120, 180, (h, w), dtype=np.uint8)
            base[..., 1] = rng.integers(160, 210, (h, w), dtype=np.uint8)
            base[..., 2] = rng.integers(175, 225, (h, w), dtype=np.uint8)
        else:  # dark net / indoor
            base = rng.integers(5, 55, (h, w, 3), dtype=np.uint8)
            spacing = int(rng.integers(20, 50))
            for x in range(0, w, spacing):
                cv2.line(base, (x, 0), (x, h-1), (50, 70, 55), 1)
            for yv in range(0, h, spacing):
                cv2.line(base, (0, yv), (w-1, yv), (50, 70, 55), 1)
        # Add realistic broad lighting gradients/shadows.
        if rng.random() < 0.7:
            shadow = np.zeros((h, w), np.uint8)
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            axes = (int(rng.integers(w//6, max(w//5, w//2))), int(rng.integers(h//8, max(h//7, h//2))))
            cv2.ellipse(shadow, (cx, cy), axes, float(rng.integers(0,180)), 0, 360, 90, -1)
            shadow = cv2.GaussianBlur(shadow, (0,0), sigmaX=max(15, w/30))
            base = np.clip(base.astype(np.float32) * (1 - 0.35 * shadow[...,None]/255.0), 0, 255).astype(np.uint8)
        cv2.imwrite(str(pool / f'fallback_{i:04d}.jpg'), base, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return count


def make_background_pool(source: Path) -> Path:
    pool = DATA_ROOT / 'backgrounds'
    pool.mkdir(parents=True, exist_ok=True)

    before = image_files(source)
    print(f'Images found before unpacking: {len(before)}', flush=True)
    if len(before) < 20:
        unpack_archives(source)
    candidates = image_files(source)
    print(f'Images found after unpacking: {len(candidates)}', flush=True)

    copied = 0
    for src in candidates:
        dst = pool / f'{copied:07d}{src.suffix.lower()}'
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception:
            continue

    if copied < 20:
        print(f'Only {copied} external images found. Generating 300 train-only fallback backgrounds.', flush=True)
        copied += generate_fallback_backgrounds(pool, count=300)

    if copied < 20:
        raise RuntimeError('Could not create a usable background pool.')
    print(f'Background images collected: {copied}', flush=True)
    return pool


def generate_synthetic(backgrounds: Path, count: int = 50000) -> Path:
    output = DATA_ROOT / 'synthetic'
    run([
        sys.executable, 'dataset-builder/generate_synthetic_ball.py',
        '--backgrounds', str(backgrounds),
        '--output', str(output),
        '--count', str(count),
        '--negative-ratio', '0.34',
        '--flight-ratio', '0.72',
    ], cwd=PROJECT)
    return output


def build_train_dataset(synthetic: Path) -> Path:
    train_images = synthetic / 'images'
    train_labels = synthetic / 'labels'
    if not train_images.exists() or not train_labels.exists():
        raise RuntimeError('Synthetic generator did not create images/labels directories.')

    dataset = DATA_ROOT / 'yolo-pretrain'
    images_dst = dataset / 'images' / 'train'
    labels_dst = dataset / 'labels' / 'train'
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    for p in train_images.iterdir():
        if p.is_file():
            target = images_dst / p.name
            if not target.exists():
                os.link(p, target)
    for p in train_labels.iterdir():
        if p.is_file():
            target = labels_dst / p.name
            if not target.exists():
                os.link(p, target)

    val_images = dataset / 'images' / 'val'
    val_labels = dataset / 'labels' / 'val'
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    files = sorted(images_dst.iterdir())[:1000]
    for img in files:
        lbl = labels_dst / f'{img.stem}.txt'
        shutil.copy2(img, val_images / img.name)
        if lbl.exists():
            shutil.copy2(lbl, val_labels / lbl.name)

    yaml_path = dataset / 'dataset.yaml'
    yaml_path.write_text(
        f'path: {dataset}\ntrain: images/train\nval: images/val\nnames:\n  0: golf_ball\n',
        encoding='utf-8',
    )
    return yaml_path


def train_models(data_yaml: Path, device: str) -> None:
    out = ROOT / 'golfiq-models'
    run([
        sys.executable, 'scripts/train_ball_models.py', '--data', str(data_yaml),
        '--output', str(out / 'setup'), '--model', 'yolo11s.pt', '--image-size', '960',
        '--epochs', '120', '--batch', '16', '--device', device, '--workers', '4'
    ], cwd=PROJECT)
    run([
        sys.executable, 'scripts/train_ball_models.py', '--data', str(data_yaml),
        '--output', str(out / 'flight'), '--model', 'yolo11s.pt', '--image-size', '1280',
        '--epochs', '160', '--batch', '10', '--device', device, '--workers', '4'
    ], cwd=PROJECT)


def main() -> None:
    ensure_packages()
    info = gpu_info()
    print(json.dumps(info, indent=2))
    if info['device'] == 'cpu':
        raise SystemExit('No Kaggle GPU detected. Enable GPU accelerator before running.')

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    source = download_kaggle_source()
    backgrounds = make_background_pool(source)
    synthetic = generate_synthetic(backgrounds, count=50000)
    data_yaml = build_train_dataset(synthetic)

    status = {
        'kaggle_source': KAGGLE_SOURCE,
        'source_path': str(source),
        'background_count': len(image_files(backgrounds)),
        'synthetic_images': len(image_files(synthetic / 'images')),
        'dataset_yaml': str(data_yaml),
        'gpu': info,
        'warning': 'This trains a synthetic-pretraining candidate. Production direction/carry remain locked until real held-out phone footage and measured carry data pass release gates.',
    }
    (ROOT / 'golfiq-one-click-status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))

    train_models(data_yaml, info['device'])
    print('\nDONE. Models and ONNX exports are under /kaggle/working/golfiq-models')


if __name__ == '__main__':
    main()
