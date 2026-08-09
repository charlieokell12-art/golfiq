from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(detector: dict, tracker: dict, carry: dict | None) -> dict:
    failures: list[str] = []
    if detector.get("test_source_groups", 0) < 20:
        failures.append("detector_requires_20_test_source_groups")
    if detector.get("positive_test_frames", 0) < 500:
        failures.append("detector_requires_500_positive_test_frames")
    if detector.get("negative_test_frames", 0) < 500:
        failures.append("detector_requires_500_negative_test_frames")
    if detector.get("precision", 0) < 0.95:
        failures.append("detector_precision_below_0.95")
    if detector.get("setup_recall", 0) < 0.95:
        failures.append("setup_recall_below_0.95")
    if detector.get("flight_recall", 0) < 0.85:
        failures.append("flight_recall_below_0.85")
    if tracker.get("false_tracks_per_minute", 999) > 0.25:
        failures.append("false_tracks_above_0.25_per_minute")
    if tracker.get("track_continuity", 0) < 0.75:
        failures.append("track_continuity_below_0.75")
    direction_ready = not failures

    carry_failures: list[str] = []
    if carry is None:
        carry_failures.append("carry_report_missing")
    else:
        if carry.get("sample_count", 0) < 100:
            carry_failures.append("carry_requires_100_measured_shots")
        if carry.get("source_group_count", 0) < 10:
            carry_failures.append("carry_requires_10_source_groups")
        if carry.get("group_cross_validated_mae_yards", 999) > 12:
            carry_failures.append("carry_mae_above_12_yards")
        if carry.get("absolute_error_p90_yards", 999) > 25:
            carry_failures.append("carry_p90_above_25_yards")

    return {
        "direction_release_ready": direction_ready,
        "carry_release_ready": direction_ready and not carry_failures,
        "direction_failures": failures,
        "carry_failures": carry_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--carry", type=Path)
    parser.add_argument("--output", type=Path, default=Path("release_gate_report.json"))
    args = parser.parse_args()
    report = evaluate(load(args.detector), load(args.tracker), load(args.carry) if args.carry else None)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["direction_release_ready"] else 1)


if __name__ == "__main__":
    main()
