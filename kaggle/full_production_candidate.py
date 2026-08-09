from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import production_candidate_bootstrap as core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimum-real-images', type=int, default=200)
    parser.add_argument('--public-synthetic', type=int, default=4000)
    parser.add_argument('--sanity-epochs', type=int, default=8)
    parser.add_argument('--full-epochs', type=int, default=80)
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    core.install()
    core.WORK.mkdir(parents=True, exist_ok=True)
    info = core.gpu_info()
    print(json.dumps(info, indent=2))
    if info['device'] == 'cpu':
        raise SystemExit('No Kaggle GPU detected. Enable GPU accelerator first.')

    # 1) REAL data is the only data allowed to determine whether training may start.
    root = core.unpack_inputs()
    real_rows = core.discover_pairs(root)
    audit = core.audit(real_rows, args.minimum_real_images)
    (core.WORK / 'dataset_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('\nREAL DATA AUDIT\n' + json.dumps(audit, indent=2))
    if not audit['passed']:
        raise SystemExit(
            '\nSTOPPED BEFORE GPU TRAINING. Add a real YOLO-labelled golf-ball dataset to the '
            'Kaggle notebook Inputs. Synthetic/public augmentation is deliberately not allowed '
            'to satisfy this gate.'
        )

    splits = core.split_groups(real_rows)
    real_split_report = {
        k: {'images': len(v), 'groups': len({r['group'] for r in v}),
            'positive_images': sum(r['positive_boxes'] > 0 for r in v)}
        for k, v in splits.items()
    }

    # 2) Best-effort public augmentation. The acquisition script records attribution and
    # only admits CC0/Public Domain/CC-BY material. Generated boxes are train-only.
    aug_root = core.WORK / 'public-augmentation'
    try:
        subprocess.run([
            sys.executable, str(Path(__file__).with_name('prepare_public_augmentation.py')),
            '--project', str(core.PROJECT), '--output', str(aug_root),
            '--count', str(args.public_synthetic), '--limit-per-query', '15',
        ], check=True)
    except Exception as exc:
        print(f'Public augmentation unavailable; continuing with real training data: {exc}', flush=True)

    synthetic_rows = core.discover_pairs(aug_root / 'synthetic') if (aug_root / 'synthetic').exists() else []
    for i, row in enumerate(synthetic_rows):
        row['group'] = f'PUBLIC_SYNTHETIC_{i:08d}'
    splits['train'].extend(synthetic_rows)

    split_report = {
        'real': real_split_report,
        'public_synthetic_train_only': len(synthetic_rows),
        'final_train_images': len(splits['train']),
        'final_val_images': len(splits['val']),
        'final_test_images': len(splits['test']),
    }
    (core.WORK / 'split_report.json').write_text(json.dumps(split_report, indent=2), encoding='utf-8')
    print('\nSPLITS\n' + json.dumps(split_report, indent=2))
    data_yaml = core.materialize_dataset(splits)

    # 3) Short sanity model first. Do not burn the long GPU run unless real validation works.
    project = core.WORK / 'runs'
    sanity_size = min(args.image_size, 960)
    sanity = core.train_one(data_yaml, project, 'sanity', args.sanity_epochs, sanity_size, args.batch, info['device'])
    sanity_metrics = core.metrics_for(sanity, data_yaml, 'val', sanity_size, info['device'])
    print('\nSANITY METRICS (REAL VALIDATION ONLY)\n' + json.dumps(sanity_metrics, indent=2))
    if sanity_metrics['map50'] < .35 or sanity_metrics['recall'] < .25:
        status = {'stage': 'sanity_failed', 'audit': audit, 'splits': split_report, 'sanity_metrics': sanity_metrics}
        (core.WORK / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        raise SystemExit('Sanity validation failed; full training intentionally not started.')

    # 4) Full candidate and untouched real held-out test.
    best = core.train_one(data_yaml, project, 'production-candidate', args.full_epochs, args.image_size, args.batch, info['device'])
    val_metrics = core.metrics_for(best, data_yaml, 'val', args.image_size, info['device'])
    test_metrics = core.metrics_for(best, data_yaml, 'test', args.image_size, info['device'])
    onnx = core.export_onnx(best, args.image_size)
    detector_pass = test_metrics['map50'] >= .70 and test_metrics['recall'] >= .70 and test_metrics['precision'] >= .70

    status = {
        'stage': 'complete',
        'real_data_audit': audit,
        'splits': split_report,
        'sanity_metrics_real_val': sanity_metrics,
        'validation_metrics_real': val_metrics,
        'held_out_test_metrics_real': test_metrics,
        'detector_release_gate_passed': detector_pass,
        'weights': str(best),
        'onnx': str(onnx),
        'public_attribution': str(aug_root / 'commons' / 'attribution.csv'),
        'important': 'This validates ball detection only. Carry and direction still require measured-shot ground truth and their separate release gates.',
        'licensing_note': 'Ultralytics is used for the Kaggle experiment. Resolve its production/commercial licensing or migrate the trained architecture before commercial deployment.',
    }
    (core.WORK / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('\nFINAL STATUS\n' + json.dumps(status, indent=2))
    print('\nDONE. Download /kaggle/working/golfiq-production-candidate after the run.')


if __name__ == '__main__':
    main()
