"""Model training pipeline with Baselines, Quantile CatBoost, MLflow tracking, and Ablation analysis."""
import json
import subprocess
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import mlflow

from src.config import (
    PROCESSED_DATA_PATH,
    SPLIT_PATH,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    QUANTILES,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME
)
from src.features import build_features
from src.metrics import calculate_point_metrics, calculate_fork_metrics, pinball_loss

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "initial_commit"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def train_pipeline(iterations=500, seed=RANDOM_SEED, use_mlflow=True):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and split
    df = pd.read_json(PROCESSED_DATA_PATH, lines=True)
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]
    test_idx = splits["test_indices"]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = df_train["target"].values
    y_val = df_val["target"].values
    y_test = df_test["target"].values

    experiments_log = []

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    # ---------------------------------------------------------
    # Experiment 1: Group-Median Baseline
    # ---------------------------------------------------------
    print("--- Training Experiment 1: Group Median Baseline ---")
    medians_by_role_exp = df_train.groupby(["role", "experience"])["target"].median().to_dict()
    global_median = float(df_train["target"].median())

    def predict_baseline(row):
        return medians_by_role_exp.get((row["role"], row["experience"]), global_median)

    val_preds_baseline = df_val.apply(predict_baseline, axis=1).values
    baseline_metrics = calculate_point_metrics(y_val, val_preds_baseline)
    experiments_log.append({
        "name": "group_median_baseline",
        "description": "Hierarchical role + experience median lookup",
        "validation_metrics": baseline_metrics
    })

    if use_mlflow:
        with mlflow.start_run(run_name="baseline_group_median"):
            mlflow.log_params({"model_type": "median_baseline"})
            mlflow.log_metrics({f"val_{k}": v for k, v in baseline_metrics.items()})

    # ---------------------------------------------------------
    # Experiment 2: CatBoost Base (No Feature Engineering)
    # ---------------------------------------------------------
    print("--- Training Experiment 2: CatBoost (engineered=False) ---")
    X_train_base = build_features(df_train.to_dict(orient="records"), engineered=False)
    X_val_base = build_features(df_val.to_dict(orient="records"), engineered=False)

    cb_base = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.08,
        depth=6,
        loss_function="MAE",
        random_seed=seed,
        verbose=0
    )
    cb_base.fit(
        X_train_base, y_train,
        eval_set=(X_val_base, y_val),
        early_stopping_rounds=40
    )
    val_preds_cb_base = cb_base.predict(X_val_base)
    cb_base_metrics = calculate_point_metrics(y_val, val_preds_cb_base)
    experiments_log.append({
        "name": "catboost_base_no_fe",
        "description": "CatBoost with raw categories and skills without engineered interaction features",
        "validation_metrics": cb_base_metrics
    })

    if use_mlflow:
        with mlflow.start_run(run_name="catboost_no_fe"):
            mlflow.log_params({"model_type": "catboost", "engineered": False, "depth": 6})
            mlflow.log_metrics({f"val_{k}": v for k, v in cb_base_metrics.items()})

    # ---------------------------------------------------------
    # Experiment 3: Production Quantile Models with Full Feature Engineering
    # ---------------------------------------------------------
    print("--- Training Experiment 3: Production Quantile Models (engineered=True) ---")
    X_train_fe = build_features(df_train.to_dict(orient="records"), engineered=True)
    X_val_fe = build_features(df_val.to_dict(orient="records"), engineered=True)
    X_test_fe = build_features(df_test.to_dict(orient="records"), engineered=True)
    feature_names = list(X_train_fe.columns)

    models = {}
    val_preds_quantiles = {}

    for q in QUANTILES:
        q_tag = f"q{int(q*100)}"
        print(f"  Training Quantile {q_tag} (alpha={q})...")
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.08,
            depth=6,
            loss_function=f"Quantile:alpha={q}",
            eval_metric=f"Quantile:alpha={q}",
            random_seed=seed,
            verbose=0
        )
        train_pool = Pool(X_train_fe, y_train)
        val_pool = Pool(X_val_fe, y_val)

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40)
        models[q_tag] = model
        val_preds_quantiles[q_tag] = model.predict(X_val_fe)

        # Save model
        model_path = ARTIFACTS_DIR / f"model_{q_tag}.cbm"
        model.save_model(str(model_path))

    # Evaluate production model metrics
    val_point_metrics = calculate_point_metrics(y_val, val_preds_quantiles["q50"])
    val_fork_metrics = calculate_fork_metrics(
        y_val,
        val_preds_quantiles["q10"],
        val_preds_quantiles["q90"]
    )
    val_pinball_q10 = float(pinball_loss(y_val, val_preds_quantiles["q10"], 0.10))
    val_pinball_q50 = float(pinball_loss(y_val, val_preds_quantiles["q50"], 0.50))
    val_pinball_q90 = float(pinball_loss(y_val, val_preds_quantiles["q90"], 0.90))

    prod_metrics = {
        **val_point_metrics,
        **val_fork_metrics,
        "pinball_loss_q10": round(val_pinball_q10, 2),
        "pinball_loss_q50": round(val_pinball_q50, 2),
        "pinball_loss_q90": round(val_pinball_q90, 2)
    }

    experiments_log.append({
        "name": "catboost_quantiles_with_fe",
        "description": "Production Quantile CatBoost (Q10, Q50, Q90) with full feature engineering",
        "validation_metrics": prod_metrics
    })

    if use_mlflow:
        with mlflow.start_run(run_name="production_quantile_catboost_fe"):
            mlflow.log_params({"model_type": "catboost_quantile", "engineered": True, "depth": 6})
            mlflow.log_metrics({f"val_{k}": v for k, v in prod_metrics.items()})

    # Save experiments summary
    with open(REPORTS_DIR / "experiments.json", "w", encoding="utf-8") as f:
        json.dump(experiments_log, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------
    # Feature Importance Plot
    # ---------------------------------------------------------
    q50_model = models["q50"]
    importances = q50_model.get_feature_importance()
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="mako")
    plt.title("Топ-15 признаков по важности (CatBoost Q50 Feature Importance)", fontsize=13, fontweight="bold")
    plt.xlabel("Важность признака (Feature Importance)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Metadata for Serving & Observability
    # ---------------------------------------------------------
    metadata = {
        "model_version": "v1.0.0",
        "git_commit": get_git_commit(),
        "clean_data_sha256": compute_sha256(PROCESSED_DATA_PATH),
        "split_sha256": compute_sha256(SPLIT_PATH),
        "features": feature_names,
        "categorical_features": [],
        "quantiles": QUANTILES,
        "models": {
            f"model_{q_tag}.cbm": compute_sha256(ARTIFACTS_DIR / f"model_{q_tag}.cbm")
            for q_tag in models
        },
        "validation_results": prod_metrics,
        "support_vocabularies": {
            "roles": sorted(list(df["role"].unique())),
            "cities": sorted(list(df["city"].unique())),
            "experience": sorted(list(df["experience"].unique()))
        }
    }

    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save search index of raw training vacancies for Scenario 1 (vacancy retrieval)
    analogues_df = df_train[["id", "role", "city", "experience", "skills", "salary_from", "salary_to", "target", "description"]]
    analogues_df.to_json(ARTIFACTS_DIR / "vacancies_index.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"\n=== Training Complete ===")
    print(f"Validation MAE (Median): {val_point_metrics['mae']:.1f} ₽ (vs Baseline: {baseline_metrics['mae']:.1f} ₽)")
    print(f"Validation Interval Coverage: {val_fork_metrics['interval_coverage']*100:.1f}%")
    print(f"Metadata saved -> {ARTIFACTS_DIR / 'metadata.json'}")
    print(f"Feature importance saved -> {REPORTS_DIR / 'feature_importance.png'}")

    return metadata

if __name__ == "__main__":
    train_pipeline()
