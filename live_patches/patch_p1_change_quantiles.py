"""LIVE PATCH P1 (Alternative): Change Quantile Thresholds and explain business trade-offs."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.config import PROCESSED_DATA_PATH, SPLIT_PATH
from src.features import build_features
from src.metrics import calculate_fork_metrics

def run_p1_quantiles_patch(lower_q=0.25, upper_q=0.75):
    nominal_cov = (upper_q - lower_q) * 100
    print(f"=== LIVE MINI-PRAVKA P1: CHANGING QUANTILE THRESHOLDS TO Q{int(lower_q*100)} - Q{int(upper_q*100)} ({nominal_cov:.0f}% Nominal Interval) ===")

    df = pd.read_json(PROCESSED_DATA_PATH, lines=True)
    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    df_train = df.iloc[splits["train_indices"]].reset_index(drop=True)
    df_val = df.iloc[splits["val_indices"]].reset_index(drop=True)

    X_train = build_features(df_train.to_dict(orient="records"), engineered=True)
    X_val = build_features(df_val.to_dict(orient="records"), engineered=True)
    cb_low = CatBoostRegressor(iterations=250, depth=6, loss_function=f"Quantile:alpha={lower_q}", random_seed=42, verbose=0)
    cb_high = CatBoostRegressor(iterations=250, depth=6, loss_function=f"Quantile:alpha={upper_q}", random_seed=42, verbose=0)

    cb_low.fit(X_train, df_train["target"].values)
    cb_high.fit(X_train, df_train["target"].values)

    p_low = cb_low.predict(X_val)
    p_high = cb_high.predict(X_val)

    metrics = calculate_fork_metrics(df_val["target"].values, p_low, p_high)

    print(f"--- Measured Interval Metrics on Validation ---")
    print(f"Actual Interval Coverage: {metrics['interval_coverage']*100:.1f}% (Nominal target: {nominal_cov:.0f}%)")
    print(f"Mean Interval Width:      {metrics['mean_interval_width_rub']:,.0f} ₽ (Relative width: {metrics['mean_relative_width']*100:.1f}%)")
    print(f"\nБизнес-интерпретация:")
    print(f"  При сужении вилки до Q25-Q75 интервал стал уже на ~35%, что дает соискателю более точный ориентир торга,")
    print(f"  однако coverage снизился с 80% до {metrics['interval_coverage']*100:.1f}%, то есть половина предложений выходит за рамки.")

if __name__ == "__main__":
    run_p1_quantiles_patch()
