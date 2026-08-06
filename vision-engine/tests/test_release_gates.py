from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SPEC = spec_from_file_location("release_gates", Path("scripts/check_release_gates.py"))
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def passing_detector():
    return {
        "test_source_groups": 25,
        "positive_test_frames": 700,
        "negative_test_frames": 800,
        "precision": 0.97,
        "setup_recall": 0.96,
        "flight_recall": 0.88,
    }


def passing_tracker():
    return {"false_tracks_per_minute": 0.1, "track_continuity": 0.82}


def passing_carry():
    return {
        "sample_count": 180,
        "source_group_count": 18,
        "group_cross_validated_mae_yards": 8.5,
        "absolute_error_p90_yards": 17.0,
    }


def test_all_gates_pass_with_strong_evidence():
    result = MODULE.evaluate(passing_detector(), passing_tracker(), passing_carry())
    assert result["direction_release_ready"] is True
    assert result["carry_release_ready"] is True


def test_direction_can_pass_while_carry_remains_locked():
    carry = passing_carry()
    carry["sample_count"] = 40
    result = MODULE.evaluate(passing_detector(), passing_tracker(), carry)
    assert result["direction_release_ready"] is True
    assert result["carry_release_ready"] is False


def test_false_tracks_block_release():
    tracker = passing_tracker()
    tracker["false_tracks_per_minute"] = 0.8
    result = MODULE.evaluate(passing_detector(), tracker, passing_carry())
    assert result["direction_release_ready"] is False
    assert result["carry_release_ready"] is False
