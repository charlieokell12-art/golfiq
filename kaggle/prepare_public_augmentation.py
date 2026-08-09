from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print('\n+', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def count_images(root: Path) -> int:
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    return sum(1 for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/kaggle/working/golfiq'))
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/golfiq-production-candidate/public-augmentation'))
    parser.add_argument('--count', type=int, default=4000)
    parser.add_argument('--limit-per-query', type=int, default=15)
    args = parser.parse_args()

    commons = args.output / 'commons'
    synthetic = args.output / 'synthetic'
    commons.mkdir(parents=True, exist_ok=True)

    # Commons acquisition is best-effort. It uses thumbnail URLs, retry/backoff,
    # and records license/attribution evidence. Network failure must not destroy
    # a valid real-data training run.
    try:
        run([
            sys.executable, 'dataset-builder/acquire_commons.py',
            '--output', str(commons), '--limit-per-query', str(args.limit_per_query), '--download',
            '--query', 'golf driving range',
            '--query', 'golf ball tee',
            '--query', 'golf course fairway',
            '--query', 'golf practice net',
            '--query', 'golf course sky',
            '--query', 'golf range marker',
        ], args.project)
    except Exception as exc:
        print(f'Public-source acquisition did not complete: {exc}', flush=True)

    media = commons / 'media'
    backgrounds = count_images(media) if media.exists() else 0
    status = {'commons_backgrounds': backgrounds, 'synthetic_requested': args.count, 'synthetic_created': 0}
    if backgrounds >= 20 and args.count > 0:
        try:
            run([
                sys.executable, 'dataset-builder/generate_synthetic_ball.py',
                '--backgrounds', str(media), '--output', str(synthetic), '--count', str(args.count),
                '--negative-ratio', '0.30', '--flight-ratio', '0.78',
            ], args.project)
            status['synthetic_created'] = count_images(synthetic / 'images')
        except Exception as exc:
            status['synthetic_error'] = str(exc)
            print(f'Synthetic augmentation skipped after error: {exc}', flush=True)
    else:
        status['note'] = 'Fewer than 20 legal public backgrounds; continuing with real labelled data only.'

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
