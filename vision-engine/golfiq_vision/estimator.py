from __future__ import annotations

from math import atan2, degrees, hypot

import numpy as np

from .types import CameraCalibration, CarryCalibration, ShotEstimate, TrackPoint


def _robust_line_fit(t: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line while down-weighting isolated detector errors.

    Returns (coefficients, inlier_mask). The first fit is used to estimate residuals,
    then points beyond a robust MAD threshold are removed and the fit is repeated.
    """
    coeff = np.polyfit(t, values, 1)
    residuals = values - np.polyval(coeff, t)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    scale = max(1.0, 1.4826 * mad)
    mask = np.abs(residuals - median) <= 3.5 * scale
    if int(mask.sum()) >= max(4, len(values) // 2):
        coeff = np.polyfit(t[mask], values[mask], 1)
    else:
        mask = np.ones(len(values), dtype=bool)
    return coeff, mask


def _robust_quadratic_fit(t: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    usable = mask & np.isfinite(values)
    if int(usable.sum()) >= 5:
        return np.polyfit(t[usable], values[usable], 2)
    return np.polyfit(t[usable], values[usable], 1)


def estimate_shot(
    track: list[TrackPoint],
    camera: CameraCalibration | None,
    carry: CarryCalibration | None,
    club: str | None = None,
) -> ShotEstimate:
    reasons: list[str] = []
    if len(track) < 5:
        return _withheld("not_enough_track_points")
    if camera is None:
        return _withheld("camera_calibration_required")
    if camera.pixels_per_degree_x <= 0 or camera.pixels_per_degree_y <= 0:
        return _withheld("invalid_camera_calibration")

    ordered = sorted(track, key=lambda p: (p.timestamp_s, p.frame_index))
    t = np.array([p.timestamp_s for p in ordered], dtype=float)
    x = np.array([p.x_px for p in ordered], dtype=float)
    y = np.array([p.y_px for p in ordered], dtype=float)
    confidence_values = np.array([p.confidence for p in ordered], dtype=float)
    t = t - t[0]

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return _withheld("non_finite_track")
    if t[-1] <= 0 or np.any(np.diff(t) <= 0):
        return _withheld("invalid_timestamps")

    x_fit, x_mask = _robust_line_fit(t, x)
    y_line, y_mask = _robust_line_fit(t, y)
    joint_mask = x_mask & y_mask
    if int(joint_mask.sum()) < 4:
        return _withheld("too_many_track_outliers")

    y_fit = _robust_quadratic_fit(t, y, joint_mask)
    vx_px_s = float(x_fit[0])
    vy_px_s = float(y_line[0])

    vx_deg_s = vx_px_s / camera.pixels_per_degree_x
    vy_deg_s = -vy_px_s / camera.pixels_per_degree_y
    launch_direction = degrees(atan2(vx_deg_s, max(abs(vy_deg_s), 1e-6))) - camera.target_line_angle_deg

    # Normalize to a conventional -180..180 range in case target-line calibration wraps.
    launch_direction = ((launch_direction + 180.0) % 360.0) - 180.0

    direction = "straight"
    if launch_direction < -2.0:
        direction = "left"
    elif launch_direction > 2.0:
        direction = "right"

    x_residual = x[joint_mask] - np.polyval(x_fit, t[joint_mask])
    y_residual = y[joint_mask] - np.polyval(y_fit, t[joint_mask])
    residual = float(np.mean(np.hypot(x_residual, y_residual)))

    lateral_curve = 0.0
    if int(joint_mask.sum()) >= 5:
        curve_fit = np.polyfit(t[joint_mask], x[joint_mask], 2)
        lateral_curve = float(curve_fit[0])

    # Curvature is only trusted when the track spans enough real time. Tiny time windows
    # can produce very large quadratic coefficients from noise.
    duration_s = float(t[-1])
    shape = "straight"
    if duration_s >= 0.035:
        curve_threshold = 18.0
        if lateral_curve > curve_threshold:
            shape = "slice" if direction == "right" else "fade"
        elif lateral_curve < -curve_threshold:
            shape = "hook" if direction == "left" else "draw"

    mean_detection_confidence = float(np.mean(confidence_values[joint_mask]))
    continuity = min(1.0, len(ordered) / max(1, ordered[-1].frame_index - ordered[0].frame_index + 1))
    inlier_fraction = float(np.mean(joint_mask))
    fit_score = max(0.0, 1.0 - residual / 24.0)
    duration_score = min(1.0, duration_s / 0.06)

    confidence = max(
        0.0,
        min(
            1.0,
            0.36 * mean_detection_confidence
            + 0.22 * continuity
            + 0.20 * fit_score
            + 0.14 * inlier_fraction
            + 0.08 * duration_score,
        ),
    )

    if abs(launch_direction) > 45.0:
        return ShotEstimate("unavailable", "unavailable", None, None, None, None, confidence, "withheld", ("implausible_launch_direction",))
    if confidence < 0.62:
        return ShotEstimate(direction, shape, launch_direction, None, None, None, confidence, "withheld", ("low_tracking_confidence",))

    if carry is None or carry.sample_count < 100:
        reasons.append("carry_calibration_requires_100_measured_shots")
        return ShotEstimate(direction, shape, launch_direction, None, None, None, confidence, "direction_only", tuple(reasons))

    if carry.residual_std_yards <= 0 or carry.residual_std_yards > 30:
        reasons.append("carry_calibration_quality_insufficient")
        return ShotEstimate(direction, shape, launch_direction, None, None, None, confidence, "direction_only", tuple(reasons))

    angular_speed = hypot(vx_deg_s, vy_deg_s)
    if angular_speed <= 0 or angular_speed > 1000:
        reasons.append("angular_speed_out_of_range")
        return ShotEstimate(direction, shape, launch_direction, None, None, None, confidence, "direction_only", tuple(reasons))

    launch_angle_proxy = degrees(atan2(vy_deg_s, max(abs(vx_deg_s), 1e-6)))
    adjustment = carry.club_adjustments.get((club or "").lower(), 0.0)
    midpoint = (
        carry.intercept_yards
        + carry.speed_coefficient * angular_speed
        + carry.angle_coefficient * launch_angle_proxy
        + adjustment
    )

    # Increase uncertainty when visual evidence is weaker instead of pretending that
    # every accepted track deserves the same carry interval.
    base_spread = max(5.0, 1.64 * carry.residual_std_yards)
    confidence_penalty = 1.0 + max(0.0, 0.90 - confidence) * 1.75
    spread = base_spread * confidence_penalty

    if midpoint <= 0 or midpoint > 450:
        reasons.append("carry_prediction_out_of_range")
        return ShotEstimate(direction, shape, launch_direction, None, None, None, confidence, "direction_only", tuple(reasons))

    return ShotEstimate(
        direction,
        shape,
        launch_direction,
        max(0.0, midpoint - spread),
        midpoint,
        min(450.0, midpoint + spread),
        confidence,
        "ready",
        tuple(reasons),
    )


def _withheld(reason: str) -> ShotEstimate:
    return ShotEstimate("unavailable", "unavailable", None, None, None, None, 0.0, "withheld", (reason,))
