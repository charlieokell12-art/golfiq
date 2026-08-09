from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    blur_score: float
    brightness_mean: float
    brightness_std: float
    clipped_dark_ratio: float
    clipped_bright_ratio: float
    usable: bool
    reasons: tuple[str, ...]


def analyse_frame(path: Path) -> FrameQuality:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    std = float(gray.std())
    dark = float(np.mean(gray <= 5))
    bright = float(np.mean(gray >= 250))
    reasons: list[str] = []
    if blur < 45:
        reasons.append("too_blurry")
    if mean < 35:
        reasons.append("too_dark")
    if mean > 225:
        reasons.append("too_bright")
    if std < 18:
        reasons.append("low_contrast")
    if dark > 0.35:
        reasons.append("excessive_shadow_clipping")
    if bright > 0.25:
        reasons.append("excessive_highlight_clipping")
    return FrameQuality(blur, mean, std, dark, bright, not reasons, tuple(reasons))


def to_dict(result: FrameQuality) -> dict:
    value = asdict(result)
    value["reasons"] = list(result.reasons)
    return value
