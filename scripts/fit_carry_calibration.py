from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = {
    "source_group",
    "angular_speed_deg_s",
    "launch_angle_proxy_deg",
    "club",
    "measured_carry_yards",
}


def fit_calibration(csv_path: Path, output: Path) -> dict:
    frame = pd.read_csv(csv_path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(frame) < 100:
        raise ValueError("At least 100 measured shots are required")
    if frame["source_group"].nunique() < 10:
        raise ValueError("At least 10 independent source groups are required")

    numeric = ["angular_speed_deg_s", "launch_angle_proxy_deg"]
    categorical = ["club"]
    pre = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("club", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline([("pre", pre), ("regressor", HuberRegressor(epsilon=1.35, max_iter=2000))])
    groups = frame["source_group"]
    folds = min(5, groups.nunique())
    predictions = cross_val_predict(
        model,
        frame[numeric + categorical],
        frame["measured_carry_yards"],
        groups=groups,
        cv=GroupKFold(n_splits=folds),
    )
    mae = float(mean_absolute_error(frame["measured_carry_yards"], predictions))
    residuals = frame["measured_carry_yards"] - predictions
    p90 = float(residuals.abs().quantile(0.90))
    model.fit(frame[numeric + categorical], frame["measured_carry_yards"])

    report = {
        "sample_count": int(len(frame)),
        "source_group_count": int(groups.nunique()),
        "group_cross_validated_mae_yards": mae,
        "absolute_error_p90_yards": p90,
        "release_ready": bool(mae <= 12 and p90 <= 25),
        "release_thresholds": {"mae_yards": 12, "p90_yards": 25},
        "note": "Model binary is intentionally not serialized here; retrain in the controlled release job and export a signed calibration artifact.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shots", type=Path)
    parser.add_argument("--output", type=Path, default=Path("calibration/carry_report.json"))
    args = parser.parse_args()
    print(json.dumps(fit_calibration(args.shots, args.output), indent=2))


if __name__ == "__main__":
    main()
