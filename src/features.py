"""Feature engineering pipeline for training and serving inference."""
import re
import numpy as np
import pandas as pd
from src.config import TECH_SKILLS, CITY_SALARY_MULTIPLIERS, STANDARD_ROLES, TOP_CITIES

EXP_MAP = {
    "нет опыта": (0, 0),
    "от 1 года до 3 лет": (1, 24),
    "от 3 до 6 лет": (2, 54),
    "более 6 лет": (3, 84)
}

def normalize_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()

def build_features(records, engineered=True, include_raw_strings=False):
    """
    Transforms raw or API input records into a DataFrame with identical schema.
    records: list of dicts with keys: role, city, experience, skills (list or str)
    """
    rows = []
    for r in records:
        role = normalize_text(r.get("role", "developer"))
        city = normalize_text(r.get("city", "москва"))
        exp_raw = normalize_text(r.get("experience", "нет опыта"))

        # Raw skills normalization
        raw_skills = r.get("skills") or []
        if isinstance(raw_skills, str):
            raw_skills = [s.strip() for s in raw_skills.split(",")]
        normalized_skills = set(normalize_text(s) for s in raw_skills if s)

        row = {}
        if include_raw_strings:
            row["role"] = role
            row["city"] = city
            row["experience"] = exp_raw

        # One-hot indicators for standard roles (numerical float32 for ONNX compatibility)
        for std_role in STANDARD_ROLES:
            col_name = "role_" + re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "_", normalize_text(std_role))[:20].strip("_")
            row[col_name] = float(normalize_text(std_role) in role or role in normalize_text(std_role))

        # One-hot indicators for top cities
        for std_city in TOP_CITIES:
            col_name = "city_" + re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "_", normalize_text(std_city))[:15].strip("_")
            row[col_name] = float(normalize_text(std_city) in city)

        # Experience ordinal & approximate months
        ord_idx, months = EXP_MAP.get(exp_raw, (1, 24))
        row["experience_ord"] = float(ord_idx)
        row["experience_months"] = float(months)

        # Multi-hot encoded tech skills
        for s in TECH_SKILLS[:30]:
            col_name = "skill_" + re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "_", s).strip("_")
            row[col_name] = float(any(s in user_s for user_s in normalized_skills))

        if engineered:
            # Regional cost of living / salary index
            city_found = False
            for c_name, mult in CITY_SALARY_MULTIPLIERS.items():
                if normalize_text(c_name) in city:
                    row["city_salary_index"] = float(mult)
                    city_found = True
                    break
            if not city_found:
                row["city_salary_index"] = 1.0

            # Breadth and synergy features
            row["skill_count"] = float(len(normalized_skills))
            row["has_devops"] = float(any(s in normalized_skills for s in ["docker", "kubernetes", "ci/cd", "linux", "helm"]))
            row["has_ml"] = float(any(s in normalized_skills for s in ["machine learning", "pytorch", "scikit-learn", "nlp", "llm"]))
            row["has_backend"] = float(any(s in normalized_skills for s in ["fastapi", "django", "postgresql", "redis", "go", "java"]))
            row["has_cloud"] = float(any(s in normalized_skills for s in ["aws", "kubernetes", "terraform"]))

        rows.append(row)

    df = pd.DataFrame(rows).astype(np.float32)
    return df

def get_feature_names(engineered=True):
    sample = [{"role": "python", "city": "москва", "experience": "нет опыта", "skills": ["python", "sql"]}]
    df = build_features(sample, engineered=engineered)
    return list(df.columns)
