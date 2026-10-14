"""Data loading, cleaning, anomaly filtering, quality reporting, and leakage-free group splitting."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.config import (
    RAW_VACANCIES_PATH,
    PROCESSED_DATA_PATH,
    SPLIT_PATH,
    REPORTS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE
)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_and_normalize(raw_path=RAW_VACANCIES_PATH, output_path=PROCESSED_DATA_PATH):
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_count = 0
    clean_records = []
    rejected_reasons = {
        "missing_salary": 0,
        "foreign_currency": 0,
        "inverted_bounds": 0,
        "extreme_outlier_salary": 0,
        "duplicate_id": 0,
        "missing_key_fields": 0
    }

    seen_ids = set()

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_count += 1
            item = json.loads(line)

            vac_id = item.get("id")
            if not vac_id or vac_id in seen_ids:
                rejected_reasons["duplicate_id"] += 1
                continue
            seen_ids.add(vac_id)

            # Salary extraction & validation
            salary = item.get("salary")
            if not salary:
                rejected_reasons["missing_salary"] += 1
                continue

            curr = salary.get("currency", "RUR")
            if curr not in ["RUR", "RUB"]:
                rejected_reasons["foreign_currency"] += 1
                continue

            s_from = salary.get("from")
            s_to = salary.get("to")
            if s_from is None or s_to is None:
                rejected_reasons["missing_salary"] += 1
                continue

            try:
                s_from = float(s_from)
                s_to = float(s_to)
            except (ValueError, TypeError):
                rejected_reasons["missing_salary"] += 1
                continue

            if s_from > s_to or s_from <= 0:
                rejected_reasons["inverted_bounds"] += 1
                continue

            # Target is the monthly midpoint
            target_midpoint = (s_from + s_to) / 2.0

            # Reasonable market bounds: 15,000 <= salary <= 1,500,000 RUB/month
            if not (15000 <= s_from <= 1500000 and 15000 <= s_to <= 1500000):
                rejected_reasons["extreme_outlier_salary"] += 1
                continue

            # Extract fields
            role = item.get("name", "").split(" (")[0].strip()
            city = (item.get("area") or {}).get("name", "Не указан").strip()
            experience = (item.get("experience") or {}).get("name", "Нет опыта").strip()
            employer_id = (item.get("employer") or {}).get("id", f"emp_{vac_id}")

            key_skills = []
            for s in item.get("key_skills", []):
                s_name = s.get("name") if isinstance(s, dict) else str(s)
                if s_name:
                    key_skills.append(s_name.strip().lower())

            if not role or not city:
                rejected_reasons["missing_key_fields"] += 1
                continue

            clean_records.append({
                "id": vac_id,
                "group": employer_id,
                "role": role,
                "city": city,
                "experience": experience,
                "skills": key_skills,
                "salary_from": s_from,
                "salary_to": s_to,
                "target": target_midpoint,
                "description": item.get("description", "")
            })

    # Save cleaned dataset
    with open(output_path, "w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quality Report
    quality_report = {
        "raw_file": str(raw_path),
        "raw_sha256": compute_sha256(raw_path),
        "raw_records_count": raw_count,
        "clean_records_count": len(clean_records),
        "retention_rate": round(len(clean_records) / max(raw_count, 1), 4),
        "unique_employers_groups": len(set(r["group"] for r in clean_records)),
        "rejected_reasons": rejected_reasons
    }

    with open(REPORTS_DIR / "data_quality.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    print(f"Cleaned data saved: {len(clean_records)} rows ({quality_report['retention_rate']*100:.1f}% retention)")
    print(f"Data quality report saved -> {REPORTS_DIR / 'data_quality.json'}")

    return clean_records, quality_report

def make_group_splits(clean_path=PROCESSED_DATA_PATH, split_path=SPLIT_PATH, seed=RANDOM_SEED):
    df = pd.read_json(clean_path, lines=True)

    # 1. First split: Train+Val vs Test (GroupShuffleSplit by employer to prevent company-level leakage)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["group"]))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)

    # 2. Second split: Train vs Validation
    val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss2.split(df_train_val, groups=df_train_val["group"]))

    train_original_indices = train_val_idx[train_idx_rel].tolist()
    val_original_indices = train_val_idx[val_idx_rel].tolist()
    test_original_indices = test_idx.tolist()

    # Verify 0 group overlap between partitions
    train_groups = set(df.iloc[train_original_indices]["group"])
    val_groups = set(df.iloc[val_original_indices]["group"])
    test_groups = set(df.iloc[test_original_indices]["group"])

    assert len(train_groups & val_groups) == 0, "Group leakage between train and val!"
    assert len(train_groups & test_groups) == 0, "Group leakage between train and test!"
    assert len(val_groups & test_groups) == 0, "Group leakage between val and test!"

    splits = {
        "train_indices": train_original_indices,
        "val_indices": val_original_indices,
        "test_indices": test_original_indices,
        "counts": {
            "train": len(train_original_indices),
            "val": len(val_original_indices),
            "test": len(test_original_indices),
            "total": len(df)
        },
        "groups_count": {
            "train": len(train_groups),
            "val": len(val_groups),
            "test": len(test_groups)
        }
    }

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"Group split created: Train={len(train_original_indices)}, Val={len(val_original_indices)}, Test={len(test_original_indices)}")
    print(f"Split indices saved -> {split_path}")
    return splits

if __name__ == "__main__":
    clean_and_normalize()
    make_group_splits()
