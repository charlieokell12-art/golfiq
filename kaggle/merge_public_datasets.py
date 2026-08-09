from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
BALL_NAMES = {'golfball', 'golf ball', 'golf-ball', 'golf_ball', 'golf balls'}


def run(cmd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def ensure_packages():
    run([sys.executable, '-m', 'pip', 'install', '-q', 'roboflow', 'pyyaml'])


def norm(v):
    return re.sub(r'[\s_-]+', ' ', str(v).strip().lower())


def read_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def find_yaml(root: Path):
    candidates = list(root.rglob('data.yaml')) + list(root.rglob('dataset.yaml'))
    if not candidates:
        raise RuntimeError(f'No data.yaml found in {root}')
    return candidates[0]


def class_map(root: Path):
    data = read_yaml(find_yaml(root))
    names = data.get('names', {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}
    accepted = {}
    for idx, name in names.items():
        n = norm(name)
        if n in {norm(x) for x in BALL_NAMES} or ('golf' in n and 'ball' in n):
            accepted[int(idx)] = 0
    if not accepted:
        raise RuntimeError(f'No golf-ball class found in {root}; names={names}')
    return accepted


def label_for(img: Path):
    candidates = [img.with_suffix('.txt')]
    parts = list(img.parts)
    for i, p in enumerate(parts):
        if p.lower() == 'images':
            q = parts.copy(); q[i] = 'labels'
            candidates.append(Path(*q).with_suffix('.txt'))
    return next((p for p in candidates if p.exists()), None)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def group_key(dataset: str, stem: str):
    s = re.sub(r'(?:[_-]?(?:frame|img|image)?[_-]?\d+)$', '', stem, flags=re.I)
    s = re.sub(r'[_-]\d{3,}$', '', s)
    return f'{dataset}:{(s or stem)[:100]}'


def download(ds_id: str, out: Path):
    if out.exists() and any(p.suffix.lower() in IMG_EXTS for p in out.rglob('*') if p.is_file()):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(['roboflow', 'download', '-f', 'yolov5pytorch', '-l', str(out), ds_id], check=True)
        return
    except Exception as first:
        key = os.getenv('ROBOFLOW_API_KEY', '').strip()
        if not key:
            raise RuntimeError(
                f'Roboflow public download failed for {ds_id}. Add a Kaggle secret named '
                'ROBOFLOW_API_KEY from a free Roboflow account and rerun.'
            ) from first
        from roboflow import Roboflow
        workspace, project, version = ds_id.split('/')
        rf = Roboflow(api_key=key)
        rf.workspace(workspace).project(project).version(int(version)).download('yolov5pytorch', location=str(out))


def collect(root: Path, dataset: str):
    cmap = class_map(root)
    rows = []
    for img in root.rglob('*'):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        label = label_for(img)
        if label is None:
            continue
        lines = []
        for raw in label.read_text(encoding='utf-8', errors='ignore').splitlines():
            f = raw.strip().split()
            if len(f) < 5:
                continue
            try:
                cls = int(float(f[0])); vals = list(map(float, f[1:5]))
            except Exception:
                continue
            if cls in cmap and 0 <= vals[0] <= 1 and 0 <= vals[1] <= 1 and 0 < vals[2] <= 1 and 0 < vals[3] <= 1:
                lines.append('0 ' + ' '.join(f'{v:.8f}' for v in vals))
        rows.append({'image': img, 'label_lines': lines, 'dataset': dataset, 'group': group_key(dataset, img.stem)})
    return rows


def split(rows, seed=42):
    import random
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['group']].append(row)
    keys = list(grouped)
    random.Random(seed).shuffle(keys)
    n = len(keys); n_test = max(1, round(n * .12)); n_val = max(1, round(n * .13))
    test, val = set(keys[:n_test]), set(keys[n_test:n_test + n_val])
    out = {'train': [], 'val': [], 'test': []}
    for r in rows:
        part = 'test' if r['group'] in test else ('val' if r['group'] in val else 'train')
        out[part].append(r)
    return out


def materialize(splits, out: Path):
    if out.exists(): shutil.rmtree(out)
    for part, rows in splits.items():
        (out / 'images' / part).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / part).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows):
            name = f"{r['dataset'].replace('/', '__')}__{i:07d}{r['image'].suffix.lower()}"
            shutil.copy2(r['image'], out / 'images' / part / name)
            (out / 'labels' / part / Path(name).with_suffix('.txt').name).write_text(
                '\n'.join(r['label_lines']) + ('\n' if r['label_lines'] else ''), encoding='utf-8')
    y = out / 'dataset.yaml'
    y.write_text(f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: golf_ball\n', encoding='utf-8')
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', default=str(Path(__file__).with_name('public_dataset_catalog.json')))
    ap.add_argument('--work', default='/kaggle/working/golfiq-public-merged')
    args = ap.parse_args()
    ensure_packages()
    catalog = json.loads(Path(args.catalog).read_text(encoding='utf-8'))
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    seen = set(); rows = []; stats = []
    for item in catalog:
        ds = item['id']; local = work / 'sources' / ds.replace('/', '__')
        download(ds, local)
        source_rows = collect(local, ds)
        kept = 0
        for r in source_rows:
            h = sha256(r['image'])
            if h in seen: continue
            seen.add(h); rows.append(r); kept += 1
        stats.append({'id': ds, 'license': item['license'], 'found': len(source_rows), 'kept_after_dedupe': kept})
    if len(rows) < 5000:
        raise SystemExit(f'Only {len(rows)} unique labelled images assembled; stopping.')
    splits = split(rows)
    y = materialize(splits, work / 'merged')
    report = {
        'unique_labelled_images': len(rows),
        'positive_images': sum(bool(r['label_lines']) for r in rows),
        'negative_images': sum(not r['label_lines'] for r in rows),
        'splits': {k: len(v) for k, v in splits.items()},
        'sources': stats,
        'dataset_yaml': str(y),
    }
    (work / 'attribution.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    (work / 'merge_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
