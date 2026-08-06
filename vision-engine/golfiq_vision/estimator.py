from __future__ import annotations

from math import atan2, degrees, hypot

import numpy as np

from .types import CameraCalibration, CarryCalibration, ShotEstimate, TrackPoint


def estimate_shot(
    track: list[TrackPoint],
    camera: CameraCalibration | None,
    carry: CarryCalibration | None,
    club: str | None = None,
) -> ShotEstimate:
    reasons: list[str] = []
    if len(track) < 4:
        return _withheld("not_enough_track_points")
    if camera is None:
        return _withheld("camera_calibration_required")

    t = np.array([p.timestamp_s for p in track], dtype=float)
    x = np.array([p.x_px for p in track], dtype=float)
    y = np.array([p.y_px for p in track], dtype=float)
    t = t - t[0]
    if t[-1] <= 0:
        return _withheld("invalid_timestamps")

    x_fit = np.polyfit(t, x, 1)
    y_fit = np.polyfit(t, y, 2 if len(track) >= 5 else 1)
    vx_px_s = float(x_fit[-2] if len(x_fit) == 2 else x_fit[0])
    vy_px_s = float(y_fit[-2] if len(y_fit) == 3 else y_fit[0])

    vx_deg_s = vx_px_s / max(camera.pixels_per_degree_x, 1e-6)
    vy_deg_s = -vy_px_s / max(camera.pixels_per_degree_y, 1e-6)
    launch_direction = degrees(atan2(vx_deg_s, max(abs(vy_deg_s), 1e-6))) - camera.target_line_angle_deg

    direction = "straight"
    if launch_direction < -2.0:
        direction = "left"
    elif launch_direction > 2.0:
        direction = "right"

    residual = float(np.mean(np.abs(x - np.polyval(x_fit, t))))
    lateral_curve = 0.0
    if len(track) >= 5:
        curve_fit = np.polyfit(t, x, 2)
        lateral_curve = float(curve_fit[0])

    shape = "straight"
    if lateral_curve > 18:
        shape = "slice" if direction == "right" else "fade"
    elif lateral_curve < -18:
        shape = "hook" if direction == "left" else "draw"

    mean_detection_confidence = float(np.mean([p.confidence for p in track]))
    continuity = min(1.0, len(track) / max(1, track[-1].frame_index - track[0].frame_index + 1))
    fit_score = max(0.0, 1.0 - residual / 30.0)
    confidence = max(0.0, min(1.0, 0.5 * mean_detection_confidence + 0.3 * continuity + 0.2 * fit_score))

    if confidence < 0.55:
        return ShotEstimate(direction, shape, launch_direction, None, None, None, confidence, "withheld", ("low_tracking_confidence",))

    if carry is None or carry.sample_count < 100:
        reasons.append("carry_calibration_requires_100_measured_shots")
        return ShotEstimate(direction, shape, launch_direction, None, None, None, confidence, "direction_only", tuple(reasons))

    angular_speed = hypot(vx_deg_s, vy_deg_s)
    launch_angle_proxy = degrees(atan2(vy_deg_s, max(abs(vx_deg_s), 1e-6)))
    adjustment = carry.club_adjustments.get((club or "").lower(), 0.0)
    midpoint = (
        carry.intercept_yards
        + carry.speed_coefficient * angular_speed
        + carry.angle_coefficient * launch_angle_proxy
        + adjustment
    )
    spread = max(5.0, 1.64 * carry.residual_std_yards)
    return ShotEstimate(
        direction,
        shape,
        launch_direction,
        max(0.0, midpoint - spread),
        max(0.0, midpoint),
        max(0.0, midpoint + spread),
        confidence,
        "ready",
        tuple(reasons),
    )


def _withheld(reason: str) -> ShotEstimate:
    return ShotEstimate("unavailable", "unavailable", None, None, None, None, 0.0, "withheld", (reason,))
