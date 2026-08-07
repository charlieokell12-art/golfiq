from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np

from .types import Detection, TrackPoint


@dataclass(frozen=True)
class TrackerConfig:
    min_confidence: float = 0.35
    max_speed_px_s: float = 9000.0
    max_acceleration_px_s2: float = 90000.0
    max_gap_frames: int = 3
    min_points: int = 4
    gate_radius_px: float = 120.0
    process_noise: float = 20.0
    measurement_noise: float = 12.0
    confidence_weight: float = 25.0
    launch_origin_x_px: float | None = None
    launch_origin_y_px: float | None = None
    max_seed_distance_px: float = 180.0


class _Kalman2D:
    """Constant-velocity Kalman filter for a very small, fast-moving target."""

    def __init__(self, x: float, y: float, config: TrackerConfig):
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4, dtype=float) * 100.0
        self.config = config

    def predict(self, dt: float) -> tuple[float, float]:
        dt = max(dt, 1e-4)
        F = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        q = self.config.process_noise
        Q = q * np.array(
            [[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2], [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]],
            dtype=float,
        )
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        return float(self.state[0]), float(self.state[1])

    def update(self, x: float, y: float) -> None:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        R = np.eye(2, dtype=float) * self.config.measurement_noise**2
        z = np.array([x, y], dtype=float)
        innovation = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P


def _choose_seed(frame: list[Detection], config: TrackerConfig) -> Detection | None:
    if config.launch_origin_x_px is None or config.launch_origin_y_px is None:
        return max(frame, key=lambda d: d.confidence, default=None)
    ranked: list[tuple[float, Detection]] = []
    for candidate in frame:
        distance = hypot(candidate.x_px - config.launch_origin_x_px, candidate.y_px - config.launch_origin_y_px)
        if distance <= config.max_seed_distance_px:
            ranked.append((distance - candidate.confidence * 20.0, candidate))
    return min(ranked, key=lambda item: item[0])[1] if ranked else None


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
    seed = _choose_seed(by_frame[first_frame], config)
    if seed is None:
        return []
    track = [_to_point(seed)]
    kf = _Kalman2D(seed.x_px, seed.y_px, config)
    previous_velocity: tuple[float, float] | None = None
    missed = 0

    for frame_index in range(first_frame + 1, max(by_frame) + 1):
        previous = track[-1]
        frame_candidates = by_frame.get(frame_index, [])
        nominal_t = frame_candidates[0].timestamp_s if frame_candidates else previous.timestamp_s + 1.0 / 120.0
        dt = max(1e-4, nominal_t - previous.timestamp_s)
        predicted_x, predicted_y = kf.predict(dt)

        best: Detection | None = None
        best_cost = float("inf")
        for candidate in frame_candidates:
            cdt = max(1e-4, candidate.timestamp_s - previous.timestamp_s)
            dx = candidate.x_px - previous.x_px
            dy = candidate.y_px - previous.y_px
            speed = hypot(dx, dy) / cdt
            if speed > config.max_speed_px_s:
                continue

            distance_from_prediction = hypot(candidate.x_px - predicted_x, candidate.y_px - predicted_y)
            dynamic_gate = config.gate_radius_px * (1.0 + 0.45 * missed)
            if distance_from_prediction > dynamic_gate:
                continue

            new_velocity = (dx / cdt, dy / cdt)
            if previous_velocity is not None:
                acceleration = hypot(new_velocity[0] - previous_velocity[0], new_velocity[1] - previous_velocity[1]) / cdt
                if acceleration > config.max_acceleration_px_s2:
                    continue

            cost = distance_from_prediction + (1.0 - min(1.0, candidate.confidence)) * config.confidence_weight
            if candidate.radius_px is not None:
                cost += max(0.0, candidate.radius_px - 30.0) * 0.5
            if cost < best_cost:
                best = candidate
                best_cost = cost

        if best is None:
            missed += 1
            if missed > config.max_gap_frames:
                break
            continue

        cdt = max(1e-4, best.timestamp_s - previous.timestamp_s)
        previous_velocity = ((best.x_px - previous.x_px) / cdt, (best.y_px - previous.y_px) / cdt)
        kf.update(best.x_px, best.y_px)
        track.append(_to_point(best))
        missed = 0

    return track if len(track) >= config.min_points else []


def track_quality(track: list[TrackPoint]) -> dict[str, float]:
    if len(track) < 2:
        return {"continuity": 0.0, "mean_confidence": 0.0, "duration_s": 0.0}
    span_frames = max(1, track[-1].frame_index - track[0].frame_index + 1)
    continuity = len(track) / span_frames
    mean_confidence = sum(p.confidence for p in track) / len(track)
    duration_s = max(0.0, track[-1].timestamp_s - track[0].timestamp_s)
    return {"continuity": float(continuity), "mean_confidence": float(mean_confidence), "duration_s": float(duration_s)}


def _to_point(detection: Detection) -> TrackPoint:
    return TrackPoint(
        frame_index=detection.frame_index,
        timestamp_s=detection.timestamp_s,
        x_px=detection.x_px,
        y_px=detection.y_px,
        confidence=detection.confidence,
    )
