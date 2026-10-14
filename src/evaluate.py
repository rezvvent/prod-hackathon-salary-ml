"""Final unbiased evaluation on the frozen hold-out test set with subgroup analysis."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.config import (
    PROCESSED_DATA_PATH,
    SPLIT_PATH,
    ARTIFACTS_DIR,
    REPORTS_DIR
)
from src.features import build_features
from src.metrics import calculate_point_metrics, calculate_fork_metrics, pinball_loss

def run_evaluation():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data & test partition
    df = pd.read_json(PROCESSED_DATA_PATH, lines=True)
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_idx = splits["test_indices"]
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_test = df_test["target"].values

    # 2. Load models
    models = {}
    for q_tag in ["q10", "q50", "q90"]:
        model_path = ARTIFACTS_DIR / f"model_{q_tag}.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_path} not found. Run training first.")
        cb = CatBoostRegressor()
        cb.load_model(str(model_path))
        models[q_tag] = cb

    # 3. Extract features
    X_test = build_features(df_test.to_dict(orient="records"), engineered=True)

    # 4. Predict
    p_lower = models["q10"].predict(X_test)
    p_mid = models["q50"].predict(X_test)
    p_upper = models["q90"].predict(X_test)

    # Enforce non-crossing constraint: lower <= mid <= upper
    p_mid = np.maximum(p_lower, p_mid)
    p_upper = np.maximum(p_mid, p_upper)

    # 5. Overall metrics
    point_metrics = calculate_point_metrics(y_test, p_mid)
    fork_metrics = calculate_fork_metrics(y_test, p_lower, p_upper)
    pinball_q10 = float(pinball_loss(y_test, p_lower, 0.10))
    pinball_q50 = float(pinball_loss(y_test, p_mid, 0.50))
    pinball_q90 = float(pinball_loss(y_test, p_upper, 0.90))

    # 6. Subgroup breakdown by experience level
    subgroups = {}
    for exp_level, group in df_test.groupby("experience"):
        g_idx = group.index
        sub_y = y_test[g_idx]
        sub_mid = p_mid[g_idx]
        sub_low = p_lower[g_idx]
        sub_high = p_upper[g_idx]
        subgroups[exp_level] = {
            "sample_size": len(group),
            "mae": round(float(np.mean(np.abs(sub_y - sub_mid))), 2),
            "wape": round(float(np.sum(np.abs(sub_y - sub_mid)) / np.sum(sub_y)), 4),
            "coverage": round(float(np.mean((sub_y >= sub_low) & (sub_y <= sub_high))), 4)
        }

    test_report = {
        "test_sample_size": len(df_test),
        "point_metrics": point_metrics,
        "fork_metrics": fork_metrics,
        "pinball_loss": {
            "q10": round(pinball_q10, 2),
            "q50": round(pinball_q50, 2),
            "q90": round(pinball_q90, 2)
        },
        "subgroup_analysis": subgroups
    }

    report_path = REPORTS_DIR / "test_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    print(f"\n=== Final Test Evaluation Complete ===")
    print(f"Test MAE (Median): {point_metrics['mae']} ₽ | WAPE: {point_metrics['wape']*100:.2f}% | R²: {point_metrics['r2']}")
    print(f"Test Interval Coverage: {fork_metrics['interval_coverage']*100:.2f}% (Nominal: 80.0%)")
    print(f"Mean Interval Width: {fork_metrics['mean_interval_width_rub']:.0f} ₽ ({fork_metrics['mean_relative_width']*100:.1f}%)")
    print(f"Subgroups:\n" + "\n".join(f"  - {k}: MAE={v['mae']} ₽, Coverage={v['coverage']*100:.1f}%" for k, v in subgroups.items()))
    print(f"Saved test report -> {report_path}")

    return test_report

if __name__ == "__main__":
    run_evaluation()
