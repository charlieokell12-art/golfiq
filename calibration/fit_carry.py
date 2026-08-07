from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = {
    "source_group", "club", "angular_speed_deg_s", "launch_angle_proxy_deg", "measured_carry_yards"
}


def fit_and_evaluate(csv_path: Path, output: Path) -> dict:
    df = pd.read_csv(csv_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) < 100:
        raise ValueError("At least 100 measured shots are required")
    if df["source_group"].nunique() < 10:
        raise ValueError("At least 10 independent source groups are required")

    features = ["angular_speed_deg_s", "launch_angle_proxy_deg", "club"]
    X = df[features]
    y = df["measured_carry_yards"].astype(float)
    groups = df["source_group"]

    pre = ColumnTransformer(
        [("num", StandardScaler(), ["angular_speed_deg_s", "launch_angle_proxy_deg"]),
         ("club", OneHotEncoder(handle_unknown="ignore"), ["club"])],
        remainder="drop",
    )
    model = Pipeline([("pre", pre), ("reg", HuberRegressor(epsilon=1.35, max_iter=500))])

    folds = min(5, groups.nunique())
    cv = GroupKFold(n_splits=folds)
    predictions = np.zeros(len(df), dtype=float)
    for train_idx, test_idx in cv.split(X, y, groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])

    abs_err = np.abs(predictions - y.to_numpy())
    metrics = {
        "sample_count": int(len(df)),
        "source_groups": int(groups.nunique()),
        "mae_yards": float(mean_absolute_error(y, predictions)),
        "p50_abs_error_yards": float(np.quantile(abs_err, 0.50)),
        "p90_abs_error_yards": float(np.quantile(abs_err, 0.90)),
        "within_10_yards": float(np.mean(abs_err <= 10.0)),
        "within_20_yards": float(np.mean(abs_err <= 20.0)),
    }
    metrics["release_candidate"] = (
        metrics["mae_yards"] <= 12.0 and metrics["p90_abs_error_yards"] <= 25.0
    )

    model.fit(X, y)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "source_group": groups,
        "measured_carry_yards": y,
        "predicted_carry_yards": predictions,
        "absolute_error_yards": abs_err,
    }).to_csv(output / "carry_cv_predictions.csv", index=False)
    (output / "carry_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("calibration/output"))
    args = parser.parse_args()
    print(json.dumps(fit_and_evaluate(args.csv, args.output), indent=2))
