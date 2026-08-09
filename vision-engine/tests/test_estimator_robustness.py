from golfiq_vision.estimator import estimate_shot
from golfiq_vision.types import CameraCalibration, CarryCalibration, TrackPoint


def _camera():
    return CameraCalibration(40.0, 40.0, 0.0)


def _good_track():
    return [TrackPoint(i, i / 120.0, 100 + 6 * i, 300 - 18 * i + 0.5 * i * i, 0.94) for i in range(10)]


def test_single_outlier_does_not_destroy_direction():
    track = _good_track()
    track[5] = TrackPoint(5, 5 / 120.0, 900.0, 20.0, 0.99)
    result = estimate_shot(track, _camera(), None)
    assert result.status in {"direction_only", "withheld"}
    if result.status == "direction_only":
        assert result.direction in {"left", "straight", "right"}
        assert result.launch_direction_deg is not None
        assert abs(result.launch_direction_deg) < 45


def test_invalid_camera_calibration_is_withheld():
    result = estimate_shot(_good_track(), CameraCalibration(0.0, 40.0), None)
    assert result.status == "withheld"
    assert "invalid_camera_calibration" in result.reasons


def test_bad_carry_calibration_falls_back_to_direction_only():
    carry = CarryCalibration(100, 2.0, 0.4, {}, 45.0, 200)
    result = estimate_shot(_good_track(), _camera(), carry)
    assert result.status == "direction_only"
    assert "carry_calibration_quality_insufficient" in result.reasons


def test_good_carry_calibration_returns_bounded_interval():
    carry = CarryCalibration(120, 1.4, 0.25, {}, 7.0, 200)
    result = estimate_shot(_good_track(), _camera(), carry)
    assert result.status == "ready"
    assert 0 <= result.carry_low_yards < result.carry_mid_yards < result.carry_high_yards <= 450
