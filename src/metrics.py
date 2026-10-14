"""Business and ML evaluation metrics for Point and Fork (Interval/Quantile) predictions."""
import numpy as np

def pinball_loss(y_true, y_pred, quantile):
    """
    Pinball (Quantile) loss for quantile evaluation:
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    error = np.asarray(y_true) - np.asarray(y_pred)
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

def calculate_point_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # WAPE = sum(|y - y_hat|) / sum(y)
    sum_true = np.sum(y_true)
    wape = float(np.sum(errors) / sum_true) if sum_true > 0 else 0.0

    # Median APE
    nonzero_mask = y_true > 0
    mape = float(np.median(errors[nonzero_mask] / y_true[nonzero_mask])) if np.any(nonzero_mask) else 0.0

    # R2
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape": round(wape, 4),
        "median_ape": round(mape, 4),
        "r2": round(r2, 4)
    }

def calculate_fork_metrics(y_true, y_lower, y_upper):
    """
    Evaluates the quality of predicted salary intervals [y_lower, y_upper].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)

    # Coverage: fraction of actual targets falling within [y_lower, y_upper]
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = float(np.mean(inside))

    # Width: mean and relative width of the interval
    widths = y_upper - y_lower
    mean_width = float(np.mean(widths))
    mean_rel_width = float(np.mean(widths / np.maximum(y_true, 1.0)))

    # Crossing violations (lower > upper)
    violations = float(np.mean(y_lower > y_upper))

    # Asymmetry: fraction under-predicted vs over-predicted
    under_ratio = float(np.mean(y_true < y_lower))  # candidate asked too little relative to lower bound
    over_ratio = float(np.mean(y_true > y_upper))   # candidate exceeded upper bound

    return {
        "interval_coverage": round(coverage, 4),
        "mean_interval_width_rub": round(mean_width, 2),
        "mean_relative_width": round(mean_rel_width, 4),
        "crossing_violations_rate": round(violations, 4),
        "underestimation_rate": round(under_ratio, 4),
        "overestimation_rate": round(over_ratio, 4)
    }
