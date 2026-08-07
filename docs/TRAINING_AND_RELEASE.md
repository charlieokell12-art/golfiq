# GolfIQ Vision: Training and Release

This repository is designed so the remaining GPU step can be run on Kaggle without changing the production architecture.

## Before Kaggle

1. Add only owned, public-domain, permissively licensed, or explicitly permitted footage to the source manifest.
2. Extract frames with `dataset-builder/extract_frames.py`.
3. Review labels in `dataset-builder/app.py`.
4. Build source-safe splits with `dataset-builder/build_dataset.py`.
5. Remove unclear and near-duplicate frames from validation/test.
6. Keep negatives: grass, sky, range markers, tees, white shoes, birds, clouds and moving highlights.

## Kaggle

Install training extras, then run:

```bash
python kaggle/run_training_pipeline.py --dataset /kaggle/input/golfiq-ball-v1 --epochs 160
```

This trains setup and flight detectors separately and exports ONNX models.

## Held-out evaluation

Do not evaluate on frames from videos represented in training. Required detector/tracker metrics:

- >= 20 independent test sources
- >= 500 positive test frames
- >= 500 negative test frames
- precision >= 0.95
- setup recall >= 0.95
- flight recall >= 0.85
- false tracks <= 0.25 per minute
- temporal continuity >= 0.75

Run `vision-engine/evaluate_release.py` with the measured metrics JSON. Direction remains disabled until every direction gate passes.

## Carry calibration

Record at least 100 shots with trusted measured carry and at least 10 independent source groups. Include phone/camera setup and club. Fit with:

```bash
python calibration/fit_carry.py measured_shots.csv
```

Carry release gates:

- grouped cross-validation MAE <= 12 yards
- 90th percentile absolute error <= 25 yards

Carry remains disabled if these thresholds fail, even when direction is enabled.

## Production behaviour

The browser runtime must receive the setup-ball location before impact and use it as the launch origin for flight tracking. This prevents high-confidence unrelated white objects from becoming the seed track. Audio, frame motion and ball departure are fused for impact timing. Low-confidence measurements are withheld rather than converted into precise-looking numbers.

## Improvement loop

After launch, opt-in corrected shots can be stored as future training candidates. Never automatically add unreviewed production predictions to the training set; that would reinforce model errors.
