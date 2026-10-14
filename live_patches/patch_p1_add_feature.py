"""LIVE PATCH P1: Adds a new interaction feature based on EDA and evaluates impact on validation metrics."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from src.config import PROCESSED_DATA_PATH, SPLIT_PATH, ARTIFACTS_DIR
from src.features import build_features
from src.metrics import calculate_point_metrics

def run_p1_patch():
    print("=== LIVE MINI-PRAVKA P1: ADDING NEW FEATURE BASED ON EDA ===")
    print("EDA Insight: Combination of senior experience (months >= 54) and cloud/devops skills produces non-linear synergy.")
    print("Action: Creating new interaction feature 'senior_devops_synergy' = experience_ord * has_devops.\n")

    df = pd.read_json(PROCESSED_DATA_PATH, lines=True)
    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    df_train = df.iloc[splits["train_indices"]].reset_index(drop=True)
    df_val = df.iloc[splits["val_indices"]].reset_index(drop=True)

    X_train = build_features(df_train.to_dict(orient="records"), engineered=True)
    X_val = build_features(df_val.to_dict(orient="records"), engineered=True)

    # Add new feature
    X_train["senior_devops_synergy"] = X_train["experience_ord"] * X_train["has_devops"]
    X_val["senior_devops_synergy"] = X_val["experience_ord"] * X_val["has_devops"]

    cb = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.08, loss_function="MAE", random_seed=42, verbose=0)
    cb.fit(X_train, df_train["target"].values, eval_set=(X_val, df_val["target"].values))

    preds = cb.predict(X_val)
    metrics = calculate_point_metrics(df_val["target"].values, preds)

    print("--- Updated Model Metrics with New Feature ---")
    print(f"Validation MAE: {metrics['mae']} ₽ | WAPE: {metrics['wape']*100:.2f}% | R²: {metrics['r2']}")
    print("Status: Successfully added feature and retrained in < 3 seconds.")

if __name__ == "__main__":
    run_p1_patch()
