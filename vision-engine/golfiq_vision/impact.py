from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ImpactEvidence:
    frame_index: int
    timestamp_s: float
    ball_departure_score: float
    motion_score: float
    audio_score: float
    combined_score: float


def detect_impact(
    ball_positions: Sequence[tuple[int, float, float, float] | None],
    frame_motion: Sequence[float],
    audio_energy: Sequence[float] | None,
    fps: float,
) -> ImpactEvidence | None:
    """Fuse ball departure, image motion and optional audio transient evidence.

    ball_positions entries are (frame_index, x, y, confidence) or None.
    The output is intentionally conservative: no result if evidence is weak.
    """
    n = min(len(ball_positions), len(frame_motion))
    if n < 5 or fps <= 0:
        return None

    motion = _robust_unit(frame_motion[:n])
    audio = _robust_unit(audio_energy[:n]) if audio_energy is not None else np.zeros(n)
    departure = np.zeros(n, dtype=float)

    last_xy: tuple[float, float] | None = None
    for i, item in enumerate(ball_positions[:n]):
        if item is None:
            if i > 0 and ball_positions[i - 1] is not None:
                departure[i] = 0.65
            continue
        _, x, y, confidence = item
        if last_xy is not None:
            jump = float(np.hypot(x - last_xy[0], y - last_xy[1]))
            departure[i] = min(1.0, jump / 35.0) * min(1.0, confidence)
        last_xy = (x, y)

    combined = 0.5 * departure + 0.3 * motion + 0.2 * audio
    idx = int(np.argmax(combined))
    if combined[idx] < 0.52:
        return None

    return ImpactEvidence(
        frame_index=idx,
        timestamp_s=idx / fps,
        ball_departure_score=float(departure[idx]),
        motion_score=float(motion[idx]),
        audio_score=float(audio[idx]),
        combined_score=float(combined[idx]),
    )


def _robust_unit(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) or 1e-6
    z = (arr - median) / (1.4826 * mad)
    return np.clip((z - 1.0) / 5.0, 0.0, 1.0)
