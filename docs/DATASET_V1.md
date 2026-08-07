# GolfIQ Dataset v1

This dataset is designed for the first serious setup-ball and early-flight training run.

## Principle

Use real, legally reusable media for validation and testing. Synthetic data is train-only. Every external source must retain its page URL, license, license URL, creator/credit metadata and a content hash where available.

## Legal acquisition

`dataset-builder/acquire_commons.py` searches Wikimedia Commons and inspects the file's structured license metadata before accepting it. The strict default allows only:

- CC0 / public domain
- CC BY 2.0
- CC BY 3.0
- CC BY 4.0

ShareAlike, NonCommercial, NoDerivatives, unknown and ambiguous records are rejected by default. This is intentionally conservative.

A manually verified seed list is stored at `datasets/commons_seed_sources.csv`.

Run:

```bash
python dataset-builder/acquire_commons.py --download --output datasets/v1-work/commons
```

The downloader records accepted and rejected sources separately and generates `attribution.csv`.

## Synthetic training data

`dataset-builder/generate_synthetic_ball.py` creates labelled golf-ball examples from accepted background images. It varies:

- ball size down to tiny flight-scale targets;
- white, yellow and orange ball appearance;
- early-flight motion blur;
- exposure and contrast;
- defocus;
- sensor noise;
- JPEG compression;
- output resolution;
- negatives with no inserted ball.

Synthetic examples are automatically tagged `split_policy=train_only` and must never be used to claim validation accuracy.

Example:

```bash
python dataset-builder/generate_synthetic_ball.py \
  --backgrounds datasets/v1-work/commons/media \
  --output datasets/v1-work/synthetic \
  --count 50000
```

## One-command non-GPU build

```bash
python dataset-builder/build_dataset_v1.py \
  --root datasets/v1-work \
  --commons-limit 150 \
  --synthetic-count 50000
```

This acquires license-approved Commons media and creates the synthetic train-only portion.

## Desired first-run composition

The following are targets, not claims that the repository already contains these numbers.

| Component | Target | Use |
|---|---:|---|
| Real setup positives | 15,000+ reviewed frames | train/val/test |
| Real early-flight positives | 15,000+ reviewed frames | train/val/test |
| Real hard negatives | 20,000+ reviewed frames | train/val/test |
| Synthetic setup/flight positives | 30,000-50,000 | train only |
| Synthetic negatives | 15,000-25,000 | train only |
| Independent source videos | 100+ preferred | source-safe splitting |
| Measured carry shots | 100 minimum, 300+ preferred | carry calibration |

## Hard negatives to seek deliberately

- tees and tee fragments;
- golf gloves and white shoes;
- golf-ball logos on signs;
- flags and range markers;
- white flowers and stones;
- clouds and bright sky spots;
- birds;
- reflective club heads;
- golf carts and wheels;
- other stationary balls outside the struck-ball origin;
- compression blocks and lens flare;
- empty grass, mats and sky.

## Real flight data remains essential

Synthetic motion blur can help pre-train robustness, but it cannot substitute for genuine post-impact phone footage. Before a production release, the flight model must be evaluated on independent real videos that were never used for training or augmentation background selection.

## Recommended capture standard for owned/opt-in footage

- rear main phone camera;
- 60 fps minimum where possible;
- 120/240 fps when the native camera pipeline permits;
- no digital zoom;
- down-the-line and face-on collections stored as separate metadata;
- driving range and practice-net environments;
- target line recorded during setup;
- exact phone model and resolution saved;
- measured carry/direction added when available.

## Release rule

Do not enable production direction or carry because a training loss looks good. Use the held-out metrics and release gates in `vision-engine/evaluate_release.py` and `scripts/check_release_gates.py`.
