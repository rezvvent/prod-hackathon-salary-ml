"""Unit tests for data validation, cleaning, and group splitting."""
import pytest
from src.config import RAW_VACANCIES_PATH, PROCESSED_DATA_PATH, SPLIT_PATH
from src.data_loader import clean_and_normalize, make_group_splits

def test_data_pipeline_execution(tmp_path):
    # Test cleaning on actual raw path
    clean_records, quality_report = clean_and_normalize()
    assert len(clean_records) > 1000
    assert quality_report["retention_rate"] > 0.90
    assert quality_report["rejected_reasons"]["extreme_outlier_salary"] >= 0

    # Test group splits
    splits = make_group_splits()
    assert splits["counts"]["train"] > splits["counts"]["val"]
    assert splits["counts"]["train"] > splits["counts"]["test"]

    # Verify no group overlap
    train_idx = set(splits["train_indices"])
    val_idx = set(splits["val_indices"])
    test_idx = set(splits["test_indices"])

    assert len(train_idx & val_idx) == 0
    assert len(train_idx & test_idx) == 0
    assert len(val_idx & test_idx) == 0
