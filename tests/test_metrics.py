"""Unit tests for metric computations."""
import numpy as np
import pytest
from src.metrics import calculate_point_metrics, calculate_fork_metrics, pinball_loss

def test_metrics_calculation():
    y_true = np.array([100000, 150000, 200000, 250000])
    y_pred = np.array([105000, 145000, 210000, 240000])

    res = calculate_point_metrics(y_true, y_pred)
    assert res["mae"] == 7500.0
    assert 0.0 < res["wape"] < 0.10
    assert res["r2"] > 0.90

    # Test Fork coverage
    y_low = np.array([90000, 130000, 180000, 230000])
    y_high = np.array([120000, 170000, 220000, 270000])

    fork_res = calculate_fork_metrics(y_true, y_low, y_high)
    assert fork_res["interval_coverage"] == 1.0
    assert fork_res["crossing_violations_rate"] == 0.0

    # Test Pinball Loss
    pb_50 = pinball_loss(y_true, y_pred, 0.5)
    assert pb_50 == 0.5 * res["mae"]
