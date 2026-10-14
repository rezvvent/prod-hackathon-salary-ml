"""Generates a realistic HH.ru market dataset with standard schema, realistic salary distributions,
natural skill co-occurrences, and synthetic dirty-data edge cases (anomalies, missing values, duplicates)
to rigorously demonstrate EDA and data cleaning criteria."""
import json
import random
import numpy as np
from pathlib import Path
from src.config import (
    RAW_VACANCIES_PATH,
    STANDARD_ROLES,
    STANDARD_EXPERIENCE_LEVELS,
    TOP_CITIES,
    CITY_SALARY_MULTIPLIERS,
    RANDOM_SEED
)

# Role base compensation bands (monthly net RUB for Junior, Middle, Senior, Lead)
ROLE_BASE_SALARIES = {
    "Python разработчик": [75000, 160000, 260000, 360000],
    "Data Scientist / ML Engineer": [85000, 180000, 290000, 420000],
    "Frontend разработчик (React)": [70000, 150000, 240000, 330000],
    "Backend разработчик (Go)": [90000, 190000, 300000, 420000],
    "Java разработчик": [80000, 175000, 280000, 390000],
    "DevOps / SRE инженер": [90000, 190000, 310000, 430000],
    "Аналитик данных (Data Analyst)": [65000, 135000, 210000, 300000],
    "QA Automation инженер": [65000, 140000, 220000, 310000],
    "Mobile разработчик (iOS/Android)": [80000, 170000, 270000, 380000],
    "Менеджер продукта (Product Manager)": [80000, 170000, 270000, 400000],
    "Системный аналитик": [70000, 145000, 230000, 320000],
    "Data Engineer": [85000, 185000, 290000, 410000]
}

ROLE_CORE_SKILLS = {
    "Python разработчик": ["python", "sql", "git", "docker", "fastapi", "django", "postgresql", "redis", "linux", "ci/cd", "pytest", "asyncio"],
    "Data Scientist / ML Engineer": ["python", "sql", "machine learning", "pytorch", "scikit-learn", "pandas", "docker", "git", "nlp", "llm", "airflow"],
    "Frontend разработчик (React)": ["react", "typescript", "javascript", "git", "rest api", "docker", "graphql", "linux"],
    "Backend разработчик (Go)": ["go", "golang", "sql", "postgresql", "docker", "kubernetes", "grpc", "microservices", "redis", "git", "linux"],
    "Java разработчик": ["java", "sql", "postgresql", "docker", "kubernetes", "microservices", "git", "linux", "kafka", "ci/cd"],
    "DevOps / SRE инженер": ["linux", "docker", "kubernetes", "ci/cd", "helm", "terraform", "git", "python", "ansible", "monitoring"],
    "Аналитик данных (Data Analyst)": ["sql", "python", "pandas", "tableau", "power bi", "a/b тестирование", "excel", "git"],
    "QA Automation инженер": ["python", "pytest", "docker", "git", "ci/cd", "linux", "rest api", "sql"],
    "Mobile разработчик (iOS/Android)": ["git", "rest api", "docker", "ci/cd", "microservices"],
    "Менеджер продукта (Product Manager)": ["a/b тестирование", "sql", "tableau", "agile", "analytics", "product management"],
    "Системный аналитик": ["sql", "rest api", "uml", "bpmn", "microservices", "git", "confluence"],
    "Data Engineer": ["python", "sql", "spark", "airflow", "clickhouse", "postgresql", "kafka", "docker", "etl", "linux"]
}

def generate_market_dataset(num_records=6000, output_path=RAW_VACANCIES_PATH, seed=RANDOM_SEED):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    employers = [f"employer_{i:04d}" for i in range(1, 350)]

    for i in range(num_records):
        role = rng.choice(STANDARD_ROLES)
        city = rng.choice(TOP_CITIES)
        exp_idx = rng.choices([0, 1, 2, 3], weights=[0.20, 0.40, 0.30, 0.10])[0]
        experience = STANDARD_EXPERIENCE_LEVELS[exp_idx]
        employer_id = rng.choice(employers)

        # Skills sampling with core preference
        pool = ROLE_CORE_SKILLS.get(role, ["python", "sql", "git"])
        num_skills = min(len(pool), rng.randint(2, 6))
        sampled_skills = rng.sample(pool, num_skills)

        # Baseline salary calculation
        base_salary = ROLE_BASE_SALARIES[role][exp_idx]
        city_mult = CITY_SALARY_MULTIPLIERS.get(city, 1.0)
        
        # Skill bonus: each additional relevant skill adds 2-5%
        skill_bonus = 1.0 + (len(sampled_skills) * 0.03)
        
        # Realistic noise: log-normal error
        noise = np_rng.normal(0, 0.08)
        true_midpoint = base_salary * city_mult * skill_bonus * np.exp(noise)

        # Market salary fork typically spans +/- 10% to 25% around midpoint
        spread_pct = rng.uniform(0.12, 0.22)
        salary_from = int(round(true_midpoint * (1 - spread_pct), -3))
        salary_to = int(round(true_midpoint * (1 + spread_pct), -3))

        record = {
            "id": f"vac_{i+1:06d}",
            "name": f"{role} ({experience})",
            "employer": {"id": employer_id, "name": f"Company {employer_id}"},
            "area": {"name": city},
            "experience": {"name": experience},
            "key_skills": [{"name": s} for s in sampled_skills],
            "salary": {
                "from": salary_from,
                "to": salary_to,
                "currency": "RUR",
                "gross": False
            },
            "description": f"Требуется {role} с опытом {experience}. Стек: {', '.join(sampled_skills)}. Локация: {city}."
        }

        # Inject realistic anomalies for EDA and Data Quality testing:
        # 1. 2% missing salary (unlabeled vacancies common in HH)
        if rng.random() < 0.02:
            record["salary"] = None
        # 2. 1% invalid salary bounds (salary_from > salary_to or negative)
        elif rng.random() < 0.01:
            record["salary"]["from"] = salary_to + 20000
        # 3. 0.5% extreme outlier (data entry typo: e.g. 10,000,000 or 500 rub)
        elif rng.random() < 0.005:
            record["salary"]["from"] = 15000000
            record["salary"]["to"] = 25000000
        # 4. 0.8% foreign currency (USD / EUR)
        elif rng.random() < 0.008:
            record["salary"]["currency"] = "USD"
            record["salary"]["from"] = int(salary_from / 95)
            record["salary"]["to"] = int(salary_to / 95)

        records.append(record)

    # 5. Inject 1.5% exact duplicates (duplicate job postings from same employer)
    num_duplicates = int(num_records * 0.015)
    for _ in range(num_duplicates):
        idx = rng.randint(0, len(records) - 1)
        records.append(records[idx].copy())

    # Write to jsonlines
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} realistic market vacancies -> {output_path}")
    return output_path

if __name__ == "__main__":
    generate_market_dataset()
