# GolfIQ Dataset v1

This dataset is designed for the first serious setup-ball and early-flight training run.

## Principle

Use real, legally reusable media for validation and testing. Synthetic data is train-only. Third-party labeled datasets are also treated as train-only unless their original source grouping and provenance are strong enough to prove an independent held-out split. Every external source must retain its page URL, license, license URL, creator/credit metadata and a content hash where available.

## Legal acquisition from Wikimedia Commons

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

## Existing labeled golf-ball datasets

`datasets/roboflow_sources.csv` lists public labeled object-detection projects whose project pages state Public Domain or CC BY 4.0 licensing. The current registry includes, among others:

- a roughly 25k-image Public Domain golf-ball/clubhead project;
- an approximately 18k-image CC BY 4.0 trajectory-oriented project;
- an approximately 8k-image CC BY 4.0 golf ball + club/head/shaft project;
- several smaller CC BY 4.0 ball-detection and tracker projects.

These projects can overlap or contain pre-generated augmentations. Do not add their headline image counts together blindly. Export the original/raw version where possible, then import each archive through `dataset-builder/import_yolo_source.py`, which content-hashes images and skips duplicates.

Example:

```bash
python dataset-builder/import_yolo_source.py /kaggle/input/one-export \
  --output datasets/v1-third-party \
  --source-id rf-publicdomain-25k \
  --license "Public Domain"
```

The importer maps common golf-ball class names to GolfIQ class `0`, removes unrelated annotations, hashes every image, and tags all imported third-party data `train_only`.

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
| Legally licensed third-party labeled ball images | 30,000-50,000 unique after hashing | train only |
| Real setup positives from independent reviewed sources | 5,000-15,000 | train/val/test |
| Real early-flight positives from independent reviewed videos | 5,000-15,000 | train/val/test |
| Real hard negatives | 10,000-20,000 | train/val/test |
| Synthetic setup/flight positives | 30,000-50,000 | train only |
| Synthetic negatives | 15,000-25,000 | train only |
| Independent source videos | 100+ preferred | source-safe held-out evaluation |
| Measured carry shots | 100 minimum, 300+ preferred | carry calibration |

A realistic first training corpus can therefore exceed 80k-100k examples without pretending that duplicated Roboflow augmentations are independent evidence.

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

Synthetic motion blur and third-party detection datasets can make the first model much stronger, but they cannot substitute for genuine post-impact phone footage. Before a production release, the flight model must be evaluated on independent real videos that were never used for training, pseudo-labeling, or augmentation background selection.

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
