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
    "projectedHorizontalMetres", "launchSpeedMps", "launchAngleDeg", "fitRmsePx",
    "observedSpanSeconds", "dxNorm", "trackConfidence", "trajectoryConfidence",
    "sceneQuality", "impactConfidence",
]
CATEGORICAL_FEATURES = ["club", "deviceClass", "cameraCalibrationId"]

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
        row = {name: features.get(name) for name in NUMERIC_FEATURES}
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
        raise SystemExit(f"Need at least 30 labelled shots to evaluate; found {len(frame)}")
    return frame

def preprocess() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])

def candidates() -> dict[str, Any]:
    return {
        "ridge_full": Ridge(alpha=4.0),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=350, learning_rate=.045, l2_regularization=2.0, random_state=42),
    }

def split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    return next(s.split(frame, groups=frame["group"]))

def evaluate(y: np.ndarray, carry: np.ndarray, lat_y=None, lat_pred=None) -> Metrics:
    lat = float(mean_absolute_error(lat_y, lat_pred)) if lat_y is not None and lat_pred is not None and len(lat_y) else None
    return Metrics(float(mean_absolute_error(y, carry)), float(median_absolute_error(y, carry)), lat, len(y))

def physics(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["projectedHorizontalMetres"], errors="coerce").fillna(0).to_numpy() * 1.09361

def fit_browser_ridge(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, Any], Metrics, int, int]:
    xtr = train[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    xte = test[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = xtr.median().fillna(0.0)
    xtr = xtr.fillna(medians); xte = xte.fillna(medians)
    means = xtr.mean(); scales = xtr.std(ddof=0).replace(0, 1.0).fillna(1.0)
    ztr = (xtr - means) / scales; zte = (xte - means) / scales
    carry = Ridge(alpha=5.0).fit(ztr, train["carryYards"])
    carry_pred = carry.predict(zte)
    lateral_payload = None; lat_true = lat_pred = None
    tm = train["lateralYards"].notna(); vm = test["lateralYards"].notna()
    lateral_train_count = int(tm.sum()); lateral_test_count = int(vm.sum())
    if lateral_train_count >= 80 and lateral_test_count >= 20:
        lat = Ridge(alpha=5.0).fit(ztr.loc[tm], train.loc[tm, "lateralYards"])
        lat_true = test.loc[vm, "lateralYards"].to_numpy(); lat_pred = lat.predict(zte.loc[vm])
        lateral_payload = {"intercept": float(lat.intercept_), "coefficients": [float(x) for x in lat.coef_]}
    payload = {
        "schema": "golfiq-browser-calibration-v1",
        "featureOrder": NUMERIC_FEATURES,
        "medians": [float(medians[k]) for k in NUMERIC_FEATURES],
        "means": [float(means[k]) for k in NUMERIC_FEATURES],
        "scales": [float(scales[k]) for k in NUMERIC_FEATURES],
        "carry": {"intercept": float(carry.intercept_), "coefficients": [float(x) for x in carry.coef_]},
        "lateral": lateral_payload,
    }
    return payload, evaluate(test["carryYards"].to_numpy(), carry_pred, lat_true, lat_pred), lateral_train_count, lateral_test_count

def main() -> None:
    p = argparse.ArgumentParser(description="Train GolfIQ calibration models")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/calibration"))
    p.add_argument("--max-carry-mae", type=float, default=12.0)
    p.add_argument("--max-lateral-mae", type=float, default=8.0)
    p.add_argument("--min-improvement", type=float, default=.08)
    p.add_argument("--min-deploy-samples", type=int, default=200)
    args = p.parse_args()
    frame = load_export(args.input); train_idx, test_idx = split(frame)
    train, test = frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES; args.output.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(test["carryYards"].to_numpy(), physics(test))
    leaderboard: dict[str, Any] = {"raw_physics": asdict(baseline)}; fitted = {}
    for name, estimator in candidates().items():
        carry = Pipeline([("pre", preprocess()), ("model", estimator)])
        carry.fit(train[cols], train["carryYards"]); pred = carry.predict(test[cols])
        leaderboard[name] = asdict(evaluate(test["carryYards"].to_numpy(), pred)); fitted[name] = carry
    browser_payload, browser_metrics, lateral_train_count, lateral_test_count = fit_browser_ridge(train, test)
    leaderboard["browser_ridge"] = asdict(browser_metrics)
    best_research = min(fitted, key=lambda n: leaderboard[n]["carry_mae"])
    browser_improvement = 1.0 - browser_metrics.carry_mae / max(baseline.carry_mae, 1e-9)
    enough_data = len(frame) >= args.min_deploy_samples and len(test) >= 40
    carry_approved = enough_data and browser_metrics.carry_mae <= args.max_carry_mae and browser_improvement >= args.min_improvement
    lateral_approved = bool(
        carry_approved and browser_payload.get("lateral") and browser_metrics.lateral_mae is not None
        and lateral_train_count >= 80 and lateral_test_count >= 20 and browser_metrics.lateral_mae <= args.max_lateral_mae
    )
    report = {
        "schema": "golfiq-calibration-eval-v3", "samples": len(frame), "trainSamples": len(train), "testSamples": len(test),
        "groupSplit": True, "leaderboard": leaderboard, "bestResearchModel": best_research,
        "deployModel": "browser_ridge", "deployApproved": carry_approved,
        "carryApproved": carry_approved, "lateralApproved": lateral_approved,
        "lateralTrainSamples": lateral_train_count, "lateralTestSamples": lateral_test_count,
        "deployImprovementOverRawPhysics": browser_improvement,
        "acceptance": {
            "minDeploySamples": args.min_deploy_samples, "minTestSamples": 40,
            "maxCarryMae": args.max_carry_mae, "maxLateralMae": args.max_lateral_mae,
            "minImprovement": args.min_improvement,
        },
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(fitted[best_research], args.output / "research_best_carry.joblib")
    if carry_approved:
        browser_payload["evaluation"] = asdict(browser_metrics)
        browser_payload["improvementOverRawPhysics"] = browser_improvement
        browser_payload["carryApproved"] = True
        browser_payload["lateralApproved"] = lateral_approved
        browser_payload["trainingSamples"] = len(frame)
        browser_payload["testSamples"] = len(test)
        if not lateral_approved:
            browser_payload["lateral"] = None
        (args.output / "deployment.json").write_text(json.dumps(browser_payload, indent=2), encoding="utf-8")
        (args.output / "DEPLOY_APPROVED").write_text("browser_ridge\n", encoding="utf-8")
    else:
        (args.output / "DEPLOY_REJECTED").write_text("Browser model did not beat guarded production thresholds; keep physics estimator.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
