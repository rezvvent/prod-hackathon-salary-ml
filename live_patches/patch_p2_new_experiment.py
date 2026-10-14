"""LIVE PATCH P2: Launches a new tracked experiment in MLflow with modified hyperparameters."""
import json
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.config import PROCESSED_DATA_PATH, SPLIT_PATH, MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from src.features import build_features
from src.metrics import calculate_point_metrics

def run_p2_experiment(depth=7, learning_rate=0.05, iterations=400):
    print("=== LIVE MINI-PRAVKA P2: LAUNCHING NEW TRACKED EXPERIMENT IN MLFLOW ===")
    print(f"Parameters: depth={depth}, lr={learning_rate}, iterations={iterations}\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_json(PROCESSED_DATA_PATH, lines=True)
    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    df_train = df.iloc[splits["train_indices"]].reset_index(drop=True)
    df_val = df.iloc[splits["val_indices"]].reset_index(drop=True)

    X_train = build_features(df_train.to_dict(orient="records"), engineered=True)
    X_val = build_features(df_val.to_dict(orient="records"), engineered=True)

    with mlflow.start_run(run_name=f"live_jury_exp_depth{depth}_lr{learning_rate}") as run:
        cb = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MAE",
            random_seed=42,
            verbose=0
        )
        cb.fit(X_train, df_train["target"].values, eval_set=(X_val, df_val["target"].values))

        preds = cb.predict(X_val)
        metrics = calculate_point_metrics(df_val["target"].values, preds)

        mlflow.log_params({
            "depth": depth,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "live_patch": True
        })
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        print(f"--- MLflow Run ID: {run.info.run_id} ---")
        print(f"Validation MAE: {metrics['mae']} ₽ | WAPE: {metrics['wape']*100:.2f}% | R²: {metrics['r2']}")
        print(f"Status: Logged to {MLFLOW_TRACKING_URI} under experiment '{EXPERIMENT_NAME}'.")

if __name__ == "__main__":
    run_p2_experiment()
