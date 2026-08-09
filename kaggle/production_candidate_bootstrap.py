from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path('/kaggle/working')
PROJECT = ROOT / 'golfiq'
INPUT_ROOT = Path('/kaggle/input')
WORK = ROOT / 'golfiq-production-candidate'
SUPPORTED_IMAGES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print('\n+', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def install() -> None:
    run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'ultralytics', 'onnx', 'onnxruntime', 'opencv-python-headless',
        'numpy', 'pandas', 'pillow', 'pyyaml', 'tqdm', 'scikit-learn'
    ])


def gpu_info() -> dict:
    try:
        text = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            text=True,
        )
        gpus = [line.strip() for line in text.splitlines() if line.strip()]
    except Exception:
        gpus = []
    return {'gpus': gpus, 'device': '0,1' if len(gpus) >= 2 else ('0' if gpus else 'cpu')}


def unpack_inputs() -> Path:
    target = WORK / 'inputs'
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for item in INPUT_ROOT.iterdir() if INPUT_ROOT.exists() else []:
        dst = target / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    seen: set[str] = set()
    for _ in range(4):
        changed = False
        for p in list(target.rglob('*')):
            if not p.is_file() or str(p) in seen:
                continue
            seen.add(str(p))
            try:
                if zipfile.is_zipfile(p):
                    out = p.parent / (p.stem + '_unpacked')
                    out.mkdir(exist_ok=True)
                    with zipfile.ZipFile(p) as zf:
                        zf.extractall(out)
                    changed = True
                elif tarfile.is_tarfile(p):
                    out = p.parent / (p.stem + '_unpacked')
                    out.mkdir(exist_ok=True)
                    with tarfile.open(p) as tf:
                        tf.extractall(out, filter='data')
                    changed = True
            except Exception as exc:
                print(f'Archive skipped: {p} ({exc})')
        if not changed:
            break
    return target


def image_paths(root: Path) -> list[Path]:
    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGES]


def candidate_label_paths(img: Path) -> list[Path]:
    candidates = [img.with_suffix('.txt')]
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part.lower() == 'images':
            q = parts.copy()
            q[i] = 'labels'
            candidates.append(Path(*q).with_suffix('.txt'))
    return candidates


def parse_yolo_label(path: Path) -> tuple[int, int]:
    positives = 0
    invalid = 0
    if not path.exists():
        return 0, 0
    for raw in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            invalid += 1
            continue
        try:
            cls = int(float(fields[0]))
            x, y, w, h = map(float, fields[1:5])
            if cls < 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                invalid += 1
            else:
                positives += 1
        except Exception:
            invalid += 1
    return positives, invalid


def source_group(path: Path) -> str:
    # Dataset builder uses stable names like video-001__frame_000123.jpg.
    stem = path.stem
    if '__' in stem:
        return stem.split('__', 1)[0]
    # Fall back to the nearest dataset directory + filename prefix. This is conservative
    # and prevents adjacent similarly named frames from leaking across splits.
    token = stem.rsplit('_', 1)[0] if '_' in stem else stem
    parent = path.parent.name
    return f'{parent}:{token}'


def discover_pairs(root: Path) -> list[dict]:
    rows = []
    for img in image_paths(root):
        label = next((p for p in candidate_label_paths(img) if p.exists()), None)
        if label is None:
            continue
        positives, invalid = parse_yolo_label(label)
        rows.append({
            'image': img,
            'label': label,
            'positive_boxes': positives,
            'invalid_lines': invalid,
            'group': source_group(img),
        })
    return rows


def audit(rows: list[dict], minimum_real_images: int) -> dict:
    images = len(rows)
    positives = sum(1 for r in rows if r['positive_boxes'] > 0)
    negatives = images - positives
    boxes = sum(r['positive_boxes'] for r in rows)
    invalid = sum(r['invalid_lines'] for r in rows)
    groups = len({r['group'] for r in rows})
    report = {
        'labelled_images': images,
        'positive_images': positives,
        'negative_images': negatives,
        'positive_boxes': boxes,
        'source_groups': groups,
        'invalid_label_lines': invalid,
        'minimum_real_images': minimum_real_images,
        'passed': images >= minimum_real_images and positives >= max(50, minimum_real_images // 3) and groups >= 3 and invalid == 0,
    }
    return report


def split_groups(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r['group']].append(r)
    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    if n < 3:
        raise RuntimeError('Need at least 3 independent source/video groups for train/val/test splitting.')
    n_test = max(1, round(n * .15))
    n_val = max(1, round(n * .15))
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    train_groups = set(groups[n_test + n_val:])
    if not train_groups:
        train_groups = {groups[-1]}
        val_groups.discard(groups[-1])
    out = {'train': [], 'val': [], 'test': []}
    for r in rows:
        split = 'test' if r['group'] in test_groups else ('val' if r['group'] in val_groups else 'train')
        out[split].append(r)
    return out


def materialize_dataset(splits: dict[str, list[dict]]) -> Path:
    dataset = WORK / 'dataset'
    if dataset.exists():
        shutil.rmtree(dataset)
    for split, rows in splits.items():
        imdir = dataset / 'images' / split
        lbdir = dataset / 'labels' / split
        imdir.mkdir(parents=True, exist_ok=True)
        lbdir.mkdir(parents=True, exist_ok=True)
        for idx, r in enumerate(rows):
            safe_group = ''.join(c if c.isalnum() or c in '-_' else '_' for c in r['group'])[:80]
            name = f'{safe_group}__{idx:07d}{r["image"].suffix.lower()}'
            dst_img = imdir / name
            dst_lbl = lbdir / Path(name).with_suffix('.txt').name
            shutil.copy2(r['image'], dst_img)
            shutil.copy2(r['label'], dst_lbl)
    yaml = dataset / 'dataset.yaml'
    yaml.write_text(
        f'path: {dataset}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: golf_ball\n',
        encoding='utf-8',
    )
    return yaml


def train_one(data_yaml: Path, project: Path, name: str, epochs: int, imgsz: int, batch: int, device: str) -> Path:
    from ultralytics import YOLO
    model = YOLO('yolo11s.pt')
    model.train(
        data=str(data_yaml), imgsz=imgsz, epochs=epochs, batch=batch, device=device,
        workers=4, project=str(project), name=name, seed=42, deterministic=True,
        patience=max(8, min(25, epochs // 3)), cos_lr=True, close_mosaic=max(3, epochs // 8),
        cache=False, amp=True, plots=True, save=True,
    )
    run_dir = project / name
    best = run_dir / 'weights' / 'best.pt'
    if not best.exists():
        raise RuntimeError(f'best.pt not found after training: {best}')
    return best


def metrics_for(weights: Path, data_yaml: Path, split: str, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, plots=True)
    box = getattr(m, 'box', None)
    out = {
        'map50': float(getattr(box, 'map50', 0.0) or 0.0),
        'map50_95': float(getattr(box, 'map', 0.0) or 0.0),
        'precision': float(getattr(box, 'mp', 0.0) or 0.0),
        'recall': float(getattr(box, 'mr', 0.0) or 0.0),
    }
    return out


def export_onnx(weights: Path, imgsz: int) -> str:
    from ultralytics import YOLO
    out = YOLO(str(weights)).export(format='onnx', imgsz=imgsz, simplify=True, dynamic=False, opset=17)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimum-real-images', type=int, default=200)
    parser.add_argument('--sanity-epochs', type=int, default=8)
    parser.add_argument('--full-epochs', type=int, default=80)
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--sanity-map50', type=float, default=.35)
    parser.add_argument('--sanity-recall', type=float, default=.25)
    args = parser.parse_args()

    install()
    WORK.mkdir(parents=True, exist_ok=True)
    info = gpu_info()
    print(json.dumps(info, indent=2))
    if info['device'] == 'cpu':
        raise SystemExit('No Kaggle GPU detected. Enable GPU accelerator first.')

    root = unpack_inputs()
    rows = discover_pairs(root)
    report = audit(rows, args.minimum_real_images)
    (WORK / 'dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nDATASET AUDIT\n' + json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(
            '\nSTOPPED BEFORE GPU TRAINING. The attached Kaggle inputs do not contain enough '
            'valid real labelled golf-ball images/source groups. Attach a YOLO-labelled dataset '
            'with images/ and labels/ directories (or a ZIP containing them), then rerun. '
            'This safeguard prevents another long synthetic-only run.'
        )

    splits = split_groups(rows)
    split_report = {k: {'images': len(v), 'groups': len({r['group'] for r in v}), 'positive_images': sum(r['positive_boxes'] > 0 for r in v)} for k, v in splits.items()}
    (WORK / 'split_report.json').write_text(json.dumps(split_report, indent=2), encoding='utf-8')
    print('\nSPLITS\n' + json.dumps(split_report, indent=2))
    data_yaml = materialize_dataset(splits)

    project = WORK / 'runs'
    sanity = train_one(data_yaml, project, 'sanity', args.sanity_epochs, min(args.image_size, 960), args.batch, info['device'])
    sanity_metrics = metrics_for(sanity, data_yaml, 'val', min(args.image_size, 960), info['device'])
    print('\nSANITY METRICS\n' + json.dumps(sanity_metrics, indent=2))
    if sanity_metrics['map50'] < args.sanity_map50 or sanity_metrics['recall'] < args.sanity_recall:
        status = {'stage': 'sanity_failed', 'audit': report, 'splits': split_report, 'sanity_metrics': sanity_metrics}
        (WORK / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        raise SystemExit('Sanity model did not pass the minimum signal gate. Full training was not started.')

    best = train_one(data_yaml, project, 'production-candidate', args.full_epochs, args.image_size, args.batch, info['device'])
    val_metrics = metrics_for(best, data_yaml, 'val', args.image_size, info['device'])
    test_metrics = metrics_for(best, data_yaml, 'test', args.image_size, info['device'])
    onnx_path = export_onnx(best, args.image_size)

    # Conservative detector gate. Carry/direction still require separate measured-shot validation.
    detector_pass = test_metrics['map50'] >= .70 and test_metrics['recall'] >= .70 and test_metrics['precision'] >= .70
    status = {
        'stage': 'complete',
        'audit': report,
        'splits': split_report,
        'sanity_metrics': sanity_metrics,
        'validation_metrics': val_metrics,
        'held_out_test_metrics': test_metrics,
        'detector_release_gate_passed': detector_pass,
        'weights': str(best),
        'onnx': onnx_path,
        'important': 'Detector metrics do not prove carry/direction accuracy. Those remain locked until measured-shot calibration passes its own held-out tests.',
        'licensing_note': 'This Kaggle trainer currently uses Ultralytics for experimentation. Review/resolve Ultralytics licensing before commercial production deployment.',
    }
    (WORK / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('\nFINAL STATUS\n' + json.dumps(status, indent=2))
    print(f'\nArtifacts: {WORK}')


if __name__ == '__main__':
    main()
