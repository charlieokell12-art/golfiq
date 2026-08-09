from golfiq_vision.estimator import estimate_shot
from golfiq_vision.types import CameraCalibration, CarryCalibration, TrackPoint


def sample_track(dx: float = 6.0):
    return [
        TrackPoint(i, i / 120.0, 100 + dx * i, 300 - 18 * i + 0.7 * i * i, 0.92)
        for i in range(8)
    ]


def test_withholds_without_camera_calibration():
    result = estimate_shot(sample_track(), None, None)
    assert result.status == "withheld"
    assert "camera_calibration_required" in result.reasons


def test_direction_available_before_carry_calibration():
    camera = CameraCalibration(40.0, 40.0)
    result = estimate_shot(sample_track(), camera, None)
    assert result.status == "direction_only"
    assert result.direction in {"left", "straight", "right"}
    assert result.carry_mid_yards is None


def test_carry_requires_enough_measured_shots():
    camera = CameraCalibration(40.0, 40.0)
    carry = CarryCalibration(100, 2.0, 0.4, {}, 8.0, 40)
    result = estimate_shot(sample_track(), camera, carry)
    assert result.status == "direction_only"


def test_ready_with_valid_calibration():
    camera = CameraCalibration(40.0, 40.0)
    carry = CarryCalibration(100, 2.0, 0.4, {"7-iron": 3.0}, 8.0, 150)
    result = estimate_shot(sample_track(), camera, carry, "7-iron")
    assert result.status == "ready"
    assert result.carry_low_yards < result.carry_mid_yards < result.carry_high_yards
