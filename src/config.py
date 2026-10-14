"""Global configuration and project paths for the Salary Prediction & Recommendations Case."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

RAW_VACANCIES_PATH = RAW_DATA_DIR / "hh_vacancies.jsonlines"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "salary_clean.jsonl"
SPLIT_PATH = PROCESSED_DATA_DIR / "split_indices.json"

REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Experiment & Training config
RANDOM_SEED = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10
CALIBRATION_SIZE = 0.10

QUANTILES = [0.10, 0.50, 0.90]
ALPHA = 0.20  # for 80% conformal coverage

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
EXPERIMENT_NAME = "salary_fork_prediction"

# Domain Vocabularies
STANDARD_ROLES = [
    "Python разработчик",
    "Data Scientist / ML Engineer",
    "Frontend разработчик (React)",
    "Backend разработчик (Go)",
    "Java разработчик",
    "DevOps / SRE инженер",
    "Аналитик данных (Data Analyst)",
    "QA Automation инженер",
    "Mobile разработчик (iOS/Android)",
    "Менеджер продукта (Product Manager)",
    "Системный аналитик",
    "Data Engineer"
]

STANDARD_EXPERIENCE_LEVELS = [
    "Нет опыта",
    "От 1 года до 3 лет",
    "От 3 до 6 лет",
    "Более 6 лет"
]

TOP_CITIES = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Удаленная работа (РФ)"
]

# Standard technology skills catalog for normalization & recommendation
TECH_SKILLS = [
    "python", "sql", "docker", "git", "linux", "postgresql", "fastapi",
    "django", "redis", "kubernetes", "ci/cd", "rest api", "pandas",
    "scikit-learn", "pytorch", "spark", "airflow", "clickhouse", "react",
    "typescript", "javascript", "go", "java", "kafka", "microservices",
    "mongodb", "rabbitmq", "aws", "pytest", "grpc", "graphql", "asyncio",
    "tableau", "power bi", "a/b тестирование", "etl", "machine learning",
    "computer vision", "nlp", "llm", "golang", "helm", "terraform"
]

# Regional cost of living / salary index (calibrated from cian / Rosstat empirical multipliers)
CITY_SALARY_MULTIPLIERS = {
    "Москва": 1.35,
    "Санкт-Петербург": 1.18,
    "Удаленная работа (РФ)": 1.15,
    "Новосибирск": 1.02,
    "Екатеринбург": 1.00,
    "Казань": 0.98,
    "Нижний Новгород": 0.95
}
