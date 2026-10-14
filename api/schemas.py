"""Pydantic schemas with strict validation for Resume and Salary Prediction API."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class ResumeRequest(StrictBase):
    role: str = Field(..., min_length=2, max_length=150, description="Желаемая должность / роль", example="Python разработчик")
    city: str = Field(..., min_length=2, max_length=100, description="Город или регион поиска", example="Москва")
    experience: Literal["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"] = Field(
        ..., description="Опыт работы кандидата", example="От 1 года до 3 лет"
    )
    skills: List[str] = Field(
        default_factory=list,
        max_length=50,
        description="Список ключевых навыков и технологий",
        example=["python", "sql", "git", "docker"]
    )

class SalaryForkResponse(StrictBase):
    salary_lower: float = Field(..., description="Нижняя граница вилки (10-й процентиль рынка), ₽", example=140000.0)
    salary_median: float = Field(..., description="Медианная рыночная оценка (50-й процентиль), ₽", example=180000.0)
    salary_upper: float = Field(..., description="Верхняя граница вилки (90-й процентиль рынка), ₽", example=220000.0)
    confidence_level: float = Field(0.80, description="Номинальная надежность вилки (80% интервал)")
    currency: str = "RUB"
    model_version: str

class SkillRecommendation(StrictBase):
    skill: str = Field(..., description="Рекомендуемый навык", example="fastapi")
    estimated_salary_boost_rub: float = Field(..., description="Ожидаемый прирост медианной зарплаты при освоении навыка, ₽", example=25000.0)
    new_median_salary_rub: float = Field(..., description="Новая прогнозируемая медиана, ₽", example=205000.0)
    market_frequency_pct: float = Field(..., description="Частота встречаемости навыка в высокооплачиваемых вакансиях роли", example=78.5)
    rationale: str = Field(..., description="Бизнес-обоснование рекомендации")

class VacancyItem(StrictBase):
    vacancy_id: str
    role: str
    city: str
    experience: str
    skills: List[str]
    salary_from: float
    salary_to: float
    advertised_midpoint: Optional[float] = None
    similarity_score: float

class FullEvaluationResponse(StrictBase):
    request_id: str
    salary_fork: SalaryForkResponse
    recommendations: List[SkillRecommendation]
    similar_vacancies: List[VacancyItem]

class RecalculateRequest(StrictBase):
    original_resume: ResumeRequest
    added_skills: List[str] = Field(default_factory=list, description="Новые добавленные навыки")
    updated_experience: Optional[Literal["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"]] = None

class RecalculateResponse(StrictBase):
    request_id: str
    previous_fork: SalaryForkResponse
    updated_fork: SalaryForkResponse
    salary_delta_rub: float = Field(..., description="Изменение медианной зарплаты (+/- ₽)")
    updated_recommendations: List[SkillRecommendation]
