from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(metrics_path: Path, carry_metrics_path: Path | None = None) -> dict:
    detector = json.loads(metrics_path.read_text(encoding="utf-8"))
    carry = json.loads(carry_metrics_path.read_text(encoding="utf-8")) if carry_metrics_path else None

    checks = {
        "independent_sources": detector.get("independent_test_sources", 0) >= 20,
        "positive_frames": detector.get("positive_test_frames", 0) >= 500,
        "negative_frames": detector.get("negative_test_frames", 0) >= 500,
        "precision": detector.get("precision", 0.0) >= 0.95,
        "setup_recall": detector.get("setup_recall", 0.0) >= 0.95,
        "flight_recall": detector.get("flight_recall", 0.0) >= 0.85,
        "false_tracks": detector.get("false_tracks_per_minute", 999.0) <= 0.25,
        "continuity": detector.get("track_continuity", 0.0) >= 0.75,
    }
    direction_ready = all(checks.values())

    carry_checks = {
        "sample_count": False,
        "source_groups": False,
        "mae": False,
        "p90": False,
    }
    if carry:
        carry_checks = {
            "sample_count": carry.get("sample_count", 0) >= 100,
            "source_groups": carry.get("source_groups", 0) >= 10,
            "mae": carry.get("mae_yards", 999.0) <= 12.0,
            "p90": carry.get("p90_abs_error_yards", 999.0) <= 25.0,
        }
    carry_ready = direction_ready and all(carry_checks.values())

    return {
        "direction_ready": direction_ready,
        "carry_ready": carry_ready,
        "direction_checks": checks,
        "carry_checks": carry_checks,
        "shopify_mode": "full" if carry_ready else "direction_only" if direction_ready else "withheld",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("detector_metrics", type=Path)
    parser.add_argument("--carry-metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("release_report.json"))
    args = parser.parse_args()
    report = evaluate(args.detector_metrics, args.carry_metrics)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["direction_ready"] else 2)
