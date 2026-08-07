from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
    return Path(kagglehub.dataset_download(KAGGLE_SOURCE))


def image_files(root: Path) -> list[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts]


def make_background_pool(source: Path) -> Path:
    pool = DATA_ROOT / 'backgrounds'
    pool.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in image_files(source):
        dst = pool / f'{copied:07d}{src.suffix.lower()}'
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception:
            continue
    if copied < 20:
        raise RuntimeError(f'Only {copied} usable images were found in the Kaggle source; need at least 20 backgrounds.')
    print(f'Background images collected: {copied}')
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
    """Build a trainable YOLO dataset from synthetic train-only images.

    This is deliberately labelled pretraining, not production validation. Real held-out
    phone footage must be added before release metrics are trusted.
    """
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

    # Ultralytics requires validation paths. Keep a small synthetic validation set only
    # for training mechanics; it must never be used as a production accuracy claim.
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
    # Stage 1: faster setup detector.
    run([
        sys.executable, 'scripts/train_ball_models.py', '--data', str(data_yaml),
        '--output', str(out / 'setup'), '--model', 'yolo11s.pt', '--image-size', '960',
        '--epochs', '120', '--batch', '16', '--device', device, '--workers', '4'
    ], cwd=PROJECT)
    # Stage 2: higher-resolution early-flight detector.
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
        'warning': 'This trains a strong synthetic-pretraining candidate. Production direction/carry remain locked until real held-out phone footage and measured carry data pass release gates.',
    }
    (ROOT / 'golfiq-one-click-status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))

    train_models(data_yaml, info['device'])
    print('\nDONE. Models and ONNX exports are under /kaggle/working/golfiq-models')


if __name__ == '__main__':
    main()
