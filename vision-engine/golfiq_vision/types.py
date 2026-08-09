from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Detection:
    frame_index: int
    timestamp_s: float
    x_px: float
    y_px: float
    confidence: float
    radius_px: float | None = None


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    timestamp_s: float
    x_px: float
    y_px: float
    confidence: float


@dataclass(frozen=True)
class CameraCalibration:
    pixels_per_degree_x: float
    pixels_per_degree_y: float
    target_line_angle_deg: float = 0.0
    device_id: str | None = None


@dataclass(frozen=True)
class CarryCalibration:
    intercept_yards: float
    speed_coefficient: float
    angle_coefficient: float
    club_adjustments: dict[str, float]
    residual_std_yards: float
    sample_count: int


@dataclass(frozen=True)
class ShotEstimate:
    direction: Literal["left", "straight", "right", "unavailable"]
    shape: Literal["hook", "draw", "straight", "fade", "slice", "unavailable"]
    launch_direction_deg: float | None
    carry_low_yards: float | None
    carry_mid_yards: float | None
    carry_high_yards: float | None
    confidence: float
    status: Literal["ready", "direction_only", "withheld"]
    reasons: tuple[str, ...]
