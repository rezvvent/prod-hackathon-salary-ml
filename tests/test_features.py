"""Unit tests for feature engineering."""
import pytest
from src.features import build_features, get_feature_names

def test_feature_extraction():
    records = [
        {
            "role": "Python разработчик",
            "city": "Москва",
            "experience": "От 1 года до 3 лет",
            "skills": ["python", "sql", "docker", "fastapi"]
        },
        {
            "role": "Data Scientist / ML Engineer",
            "city": "Казань",
            "experience": "Нет опыта",
            "skills": ["python", "pytorch"]
        }
    ]

    df_fe = build_features(records, engineered=True)
    assert len(df_fe) == 2
    assert "experience_ord" in df_fe.columns
    assert "city_salary_index" in df_fe.columns
    assert "skill_python" in df_fe.columns
    assert df_fe.loc[0, "skill_python"] == 1
    assert df_fe.loc[0, "city_salary_index"] == 1.35
    assert df_fe.loc[0, "experience_ord"] == 1

    df_base = build_features(records, engineered=False)
    assert "experience_ord" not in df_base.columns
