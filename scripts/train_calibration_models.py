from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "projectedHorizontalMetres",
    "launchSpeedMps",
    "launchAngleDeg",
    "fitRmsePx",
    "observedSpanSeconds",
    "dxNorm",
    "trackConfidence",
    "trajectoryConfidence",
    "sceneQuality",
    "impactConfidence",
]
CATEGORICAL_FEATURES = ["club", "deviceClass", "cameraCalibrationId"]
TARGETS = ["carryYards", "lateralYards"]


@dataclass
class Metrics:
    carry_mae: float
    carry_median_ae: float
    lateral_mae: float | None
    samples: int


def load_export(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload if isinstance(payload, list) else [])
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        features = sample.get("features") or {}
        truth = sample.get("truth") or sample.get("targets") or {}
        quality = sample.get("quality") or {}
        context = sample.get("context") or {}
        if truth.get("carryYards") is None:
            continue
        row: dict[str, Any] = {name: features.get(name) for name in NUMERIC_FEATURES}
        row["impactConfidence"] = quality.get("impactConfidence", features.get("impactConfidence"))
        row["club"] = context.get("club")
        row["deviceClass"] = context.get("deviceClass") or context.get("deviceModel")
        row["cameraCalibrationId"] = context.get("cameraCalibrationId")
        row["carryYards"] = float(truth["carryYards"])
        row["lateralYards"] = truth.get("lateralYards")
        row["group"] = context.get("sessionId") or context.get("videoId") or f"sample-{index // 10}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) < 30:
        raise SystemExit(f"Need at least 30 labelled shots to run evaluation; found {len(frame)}")
    return frame


def preprocess() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])


def model_candidates() -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=4.0),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=350, learning_rate=0.045, l2_regularization=2.0, random_state=42),
    }


def split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["group"]))
    return train_idx, test_idx


def evaluate(y_true: np.ndarray, carry_pred: np.ndarray, lateral_true: np.ndarray | None, lateral_pred: np.ndarray | None) -> Metrics:
    lateral_mae = None
    if lateral_true is not None and lateral_pred is not None and len(lateral_true):
        lateral_mae = float(mean_absolute_error(lateral_true, lateral_pred))
    return Metrics(
        carry_mae=float(mean_absolute_error(y_true, carry_pred)),
        carry_median_ae=float(median_absolute_error(y_true, carry_pred)),
        lateral_mae=lateral_mae,
        samples=len(y_true),
    )


def raw_physics_baseline(frame: pd.DataFrame) -> np.ndarray:
    metres = pd.to_numeric(frame["projectedHorizontalMetres"], errors="coerce").fillna(0).to_numpy()
    return metres * 1.09361


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GolfIQ carry/lateral calibration models")
    parser.add_argument("--input", type=Path, required=True, help="Shopify GolfIQ training JSON export")
    parser.add_argument("--output", type=Path, default=Path("runs/calibration"))
    parser.add_argument("--max-carry-mae", type=float, default=12.0)
    parser.add_argument("--min-improvement", type=float, default=0.08, help="Minimum fractional MAE improvement over raw physics")
    args = parser.parse_args()

    frame = load_export(args.input)
    train_idx, test_idx = split(frame)
    train, test = frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    args.output.mkdir(parents=True, exist_ok=True)
    baseline_pred = raw_physics_baseline(test)
    baseline = evaluate(test["carryYards"].to_numpy(), baseline_pred, None, None)

    leaderboard: dict[str, Any] = {"raw_physics": asdict(baseline)}
    fitted: dict[str, tuple[Pipeline, Pipeline | None]] = {}
    for name, estimator in model_candidates().items():
        carry = Pipeline([("pre", preprocess()), ("model", estimator)])
        carry.fit(train[feature_cols], train["carryYards"])
        carry_pred = carry.predict(test[feature_cols])

        lateral_pipe: Pipeline | None = None
        lateral_mae = None
        lateral_mask_train = train["lateralYards"].notna()
        lateral_mask_test = test["lateralYards"].notna()
        lateral_pred = None
        lateral_true = None
        if lateral_mask_train.sum() >= 30 and lateral_mask_test.sum() >= 5:
            lateral_estimator = model_candidates()[name]
            lateral_pipe = Pipeline([("pre", preprocess()), ("model", lateral_estimator)])
            lateral_pipe.fit(train.loc[lateral_mask_train, feature_cols], train.loc[lateral_mask_train, "lateralYards"])
            lateral_true = test.loc[lateral_mask_test, "lateralYards"].to_numpy()
            lateral_pred = lateral_pipe.predict(test.loc[lateral_mask_test, feature_cols])

        metrics = evaluate(test["carryYards"].to_numpy(), carry_pred, lateral_true, lateral_pred)
        leaderboard[name] = asdict(metrics)
        fitted[name] = (carry, lateral_pipe)

    best_name = min(fitted, key=lambda n: leaderboard[n]["carry_mae"])
    best = leaderboard[best_name]
    improvement = 1.0 - best["carry_mae"] / max(baseline.carry_mae, 1e-9)
    accepted = best["carry_mae"] <= args.max_carry_mae and improvement >= args.min_improvement

    report = {
        "schema": "golfiq-calibration-eval-v1",
        "samples": len(frame),
        "trainSamples": len(train),
        "testSamples": len(test),
        "groupSplit": True,
        "leaderboard": leaderboard,
        "winner": best_name,
        "winnerAccepted": accepted,
        "fractionalImprovementOverRawPhysics": improvement,
        "acceptance": {"maxCarryMae": args.max_carry_mae, "minImprovement": args.min_improvement},
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if accepted:
        carry_model, lateral_model = fitted[best_name]
        joblib.dump(carry_model, args.output / "carry_model.joblib")
        if lateral_model is not None:
            joblib.dump(lateral_model, args.output / "lateral_model.joblib")
        (args.output / "DEPLOY_APPROVED").write_text(best_name + "\n", encoding="utf-8")
    else:
        (args.output / "DEPLOY_REJECTED").write_text(
            "Candidate did not beat the guarded production thresholds. Keep the physics estimator.\n",
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
