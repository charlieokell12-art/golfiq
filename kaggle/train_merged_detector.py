from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def metrics(model, data: Path, split: str, imgsz: int, device: str) -> dict:
    m = model.val(data=str(data), split=split, imgsz=imgsz, device=device, plots=True)
    b = m.box
    return {
        'map50': float(b.map50 or 0),
        'map50_95': float(b.map or 0),
        'precision': float(b.mp or 0),
        'recall': float(b.mr or 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/kaggle/working/golfiq-public-merged/merged/dataset.yaml')
    ap.add_argument('--work', default='/kaggle/working/golfiq-full-model')
    ap.add_argument('--sanity-epochs', type=int, default=8)
    ap.add_argument('--full-epochs', type=int, default=80)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'onnx', 'onnxruntime'])
    import torch
    from ultralytics import YOLO

    data = Path(args.data); work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    n = torch.cuda.device_count()
    device = '0,1' if n >= 2 else ('0' if n == 1 else 'cpu')
    if device == 'cpu':
        raise SystemExit('Enable a Kaggle GPU accelerator before training.')

    sanity = YOLO('yolo11s.pt')
    sanity.train(data=str(data), imgsz=min(args.imgsz, 960), epochs=args.sanity_epochs,
                 batch=args.batch, device=device, project=str(work), name='sanity', workers=4,
                 seed=42, deterministic=True, patience=6, amp=True, cos_lr=True, close_mosaic=3)
    sanity_weights = work / 'sanity' / 'weights' / 'best.pt'
    sanity_metrics = metrics(YOLO(str(sanity_weights)), data, 'val', min(args.imgsz, 960), device)
    print('SANITY\n' + json.dumps(sanity_metrics, indent=2))
    if sanity_metrics['map50'] < .50 or sanity_metrics['recall'] < .45:
        raise SystemExit('Sanity gate failed; full training intentionally stopped.')

    model = YOLO(str(sanity_weights))
    model.train(data=str(data), imgsz=args.imgsz, epochs=args.full_epochs,
                batch=args.batch, device=device, project=str(work), name='production', workers=4,
                seed=42, deterministic=True, patience=20, amp=True, cos_lr=True, close_mosaic=10)
    best = work / 'production' / 'weights' / 'best.pt'
    trained = YOLO(str(best))
    val = metrics(trained, data, 'val', args.imgsz, device)
    test = metrics(trained, data, 'test', args.imgsz, device)
    onnx = trained.export(format='onnx', imgsz=args.imgsz, simplify=True, dynamic=False, opset=17)

    passed = test['map50'] >= .80 and test['precision'] >= .80 and test['recall'] >= .80
    status = {
        'validation': val,
        'held_out_test': test,
        'detector_gate_passed': passed,
        'weights': str(best),
        'onnx': str(onnx),
        'note': 'This validates detector performance on the merged public datasets. Final GolfIQ release still needs a separate real-phone video holdout and measured carry/direction validation.'
    }
    (work / 'FINAL_STATUS.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('\nFINAL STATUS\n' + json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
