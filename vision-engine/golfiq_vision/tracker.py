from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .types import Detection, TrackPoint


@dataclass(frozen=True)
class TrackerConfig:
    min_confidence: float = 0.35
    max_speed_px_s: float = 9000.0
    max_acceleration_px_s2: float = 90000.0
    max_gap_frames: int = 2
    min_points: int = 4


def build_track(detections: list[Detection], config: TrackerConfig = TrackerConfig()) -> list[TrackPoint]:
    candidates = sorted(
        (d for d in detections if d.confidence >= config.min_confidence),
        key=lambda d: (d.frame_index, -d.confidence),
    )
    if not candidates:
        return []

    by_frame: dict[int, list[Detection]] = {}
    for detection in candidates:
        by_frame.setdefault(detection.frame_index, []).append(detection)

    first_frame = min(by_frame)
    current = max(by_frame[first_frame], key=lambda d: d.confidence)
    track = [_to_point(current)]
    velocity: tuple[float, float] | None = None

    for frame_index in range(first_frame + 1, max(by_frame) + 1):
        if frame_index not in by_frame:
            if frame_index - track[-1].frame_index > config.max_gap_frames:
                break
            continue

        previous = track[-1]
        dt = max(1e-6, by_frame[frame_index][0].timestamp_s - previous.timestamp_s)
        predicted_x = previous.x_px + (velocity[0] * dt if velocity else 0.0)
        predicted_y = previous.y_px + (velocity[1] * dt if velocity else 0.0)

        best: Detection | None = None
        best_cost = float("inf")
        for candidate in by_frame[frame_index]:
            speed = hypot(candidate.x_px - previous.x_px, candidate.y_px - previous.y_px) / dt
            if speed > config.max_speed_px_s:
                continue
            cost = hypot(candidate.x_px - predicted_x, candidate.y_px - predicted_y)
            cost *= 1.15 - min(1.0, candidate.confidence) * 0.15
            if cost < best_cost:
                best = candidate
                best_cost = cost

        if best is None:
            continue

        new_velocity = ((best.x_px - previous.x_px) / dt, (best.y_px - previous.y_px) / dt)
        if velocity is not None:
            acceleration = hypot(new_velocity[0] - velocity[0], new_velocity[1] - velocity[1]) / dt
            if acceleration > config.max_acceleration_px_s2:
                continue
        velocity = new_velocity
        track.append(_to_point(best))

    return track if len(track) >= config.min_points else []


def _to_point(detection: Detection) -> TrackPoint:
    return TrackPoint(
        frame_index=detection.frame_index,
        timestamp_s=detection.timestamp_s,
        x_px=detection.x_px,
        y_px=detection.y_px,
        confidence=detection.confidence,
    )
