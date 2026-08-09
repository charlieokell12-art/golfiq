# GolfIQ Dataset Builder

This tool converts licensed or owned golf footage into a source-traceable YOLO dataset.

## 1. Register every source

Add each video to `datasets/source_manifest.csv` before extracting frames. A usable record needs a commercial-friendly licence or explicit permission and a proof URL or permission reference.

## 2. Extract frames

Use `scripts/extract_frames.py` to sample frames into `datasets/work/frames`. Prefix frame names with a stable source group such as `video-001__frame_000123.jpg`.

## 3. Review frames

Run:

```bash
streamlit run dataset-builder/app.py
```

For each frame:

- mark the ball visible, not visible or unclear;
- draw a tight box when visible;
- mark pre-impact, impact, post-impact or unknown;
- preserve the source group;
- save negative examples with an empty YOLO label.

## 4. Build the dataset

```bash
python dataset-builder/build_dataset.py \
  --reviews datasets/work/reviews.jsonl \
  --output datasets/ball-v1
```

Whole source groups are assigned to one split only. Adjacent frames from the same video therefore cannot leak between training and evaluation.

## 5. Audit before training

```bash
python dataset-builder/audit_sources.py datasets/source_manifest.csv
```

The audit fails when a source lacks acceptable licence evidence.

## Quality rules

- Never guess that an unclear white dot is a ball.
- Include empty grass, sky, tees, range markers and moving white objects as negatives.
- Keep original video IDs and phone/device metadata.
- Do not use synthetic images in validation or testing.
- Do not release carry estimates without measured-shot calibration.
