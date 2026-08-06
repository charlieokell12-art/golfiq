from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, sample_fps: float) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(source_fps / sample_fps))
    source_id = hashlib.sha256(video_path.read_bytes()).hexdigest()[:16]

    records: list[dict[str, object]] = []
    frame_index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            filename = f"{source_id}_{frame_index:08d}.jpg"
            destination = output_dir / filename
            if not cv2.imwrite(str(destination), frame):
                raise RuntimeError(f"Could not write {destination}")
            records.append(
                {
                    "source_id": source_id,
                    "source_video": video_path.name,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / source_fps,
                    "image": filename,
                    "annotation_status": "unreviewed",
                }
            )
            saved += 1
        frame_index += 1

    capture.release()
    manifest = {
        "source_id": source_id,
        "source_video": video_path.name,
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "sample_fps": sample_fps,
        "saved_frames": saved,
        "frames": records,
    }
    (output_dir / f"{source_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-fps", type=float, default=12.0)
    args = parser.parse_args()
    manifest = extract_frames(args.video, args.output, args.sample_fps)
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
