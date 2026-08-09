from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working/golfiq')
OUTPUT = Path('/kaggle/working/golfiq-calibration-results')


def run(cmd: list[str]) -> None:
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_export() -> Path:
    candidates = []
    for base in [Path('/kaggle/input'), Path('/kaggle/working')]:
        if not base.exists():
            continue
        for path in base.rglob('*.json'):
            try:
                text = path.read_text(encoding='utf-8', errors='ignore')[:5000]
            except Exception:
                continue
            if 'golfiq-training-v85' in text or ('"samples"' in text and 'carryYards' in text):
                candidates.append(path)
    if not candidates:
        raise RuntimeError('No GolfIQ labelled training JSON found. Add the exported JSON to the notebook as a Kaggle dataset/input, then rerun.')
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    print('Using training export:', candidates[0])
    return candidates[0]


def main() -> None:
    run([sys.executable, '-m', 'pip', 'install', '-q', 'pandas', 'numpy', 'scikit-learn', 'joblib'])
    if not ROOT.exists():
        run(['git', 'clone', '--branch', 'vision-foundation', 'https://github.com/charlieokell12-art/golfiq.git', str(ROOT)])
    else:
        run(['git', '-C', str(ROOT), 'fetch', 'origin', 'vision-foundation'])
        run(['git', '-C', str(ROOT), 'checkout', 'vision-foundation'])
        run(['git', '-C', str(ROOT), 'pull', '--ff-only', 'origin', 'vision-foundation'])

    source = find_export()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        str(ROOT / 'scripts' / 'train_calibration_models.py'),
        '--input', str(source),
        '--output', str(OUTPUT),
        '--max-carry-mae', '12',
        '--min-improvement', '0.08',
    ])
    evaluation = OUTPUT / 'evaluation.json'
    if evaluation.exists():
        print('\n=== GolfIQ calibration evaluation ===')
        print(evaluation.read_text(encoding='utf-8'))
    if (OUTPUT / 'DEPLOY_APPROVED').exists():
        print('\nDEPLOYMENT GATE: APPROVED')
        print('Download the entire golfiq-calibration-results folder from Kaggle Output.')
    else:
        print('\nDEPLOYMENT GATE: REJECTED')
        print('Keep collecting labelled real shots. The current physics estimator remains safer.')


if __name__ == '__main__':
    main()
