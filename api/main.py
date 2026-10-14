"""FastAPI Production Inference Application for Salary Fork & Resume Improvement Recommendations."""
import time
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from api.schemas import (
    ResumeRequest,
    SalaryForkResponse,
    SkillRecommendation,
    VacancyItem,
    FullEvaluationResponse,
    RecalculateRequest,
    RecalculateResponse
)
from api.service import SalaryService
from api.logger import api_logger

# Prometheus Metrics
REQUESTS_TOTAL = Counter("api_requests_total", "Total HTTP requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

salary_service: SalaryService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global salary_service
    api_logger.info("Starting up Salary Prediction Service and preloading models...")
    try:
        salary_service = SalaryService()
        api_logger.info("Models and vacancy index successfully loaded into memory.")
    except Exception as e:
        api_logger.error(f"Failed to load models: {e}. Train the models first.")
        salary_service = None
    yield
    api_logger.info("Shutting down service.")

app = FastAPI(
    title="PROD Hackathon: Salary Fork & Career Recommendations Service",
    description="ML service providing market salary fork [Q10, Q50, Q90] and actionable skill recommendations to boost income.",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def audit_and_metrics_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())
    request.state.request_id = req_id
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as e:
        api_logger.error(
            f"Unhandled server error: {e}",
            extra={"request_id": req_id, "route": request.url.path, "status_code": 500}
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "request_id": req_id}
        )
    finally:
        elapsed_sec = time.perf_counter() - start_time
        latency_ms = round(elapsed_sec * 1000, 2)
        route_path = request.url.path

        REQUESTS_TOTAL.labels(endpoint=route_path, status=str(status_code)).inc()
        REQUEST_LATENCY.labels(endpoint=route_path).observe(elapsed_sec)

        api_logger.info(
            f"HTTP {request.method} {route_path} completed",
            extra={
                "request_id": req_id,
                "latency_ms": latency_ms,
                "route": route_path,
                "status_code": status_code
            }
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    errors = [{"field": " -> ".join(str(l) for l in err["loc"]), "msg": err["msg"]} for err in exc.errors()]
    api_logger.warning(
        f"Validation error: {errors}",
        extra={"request_id": req_id, "route": request.url.path, "status_code": 422}
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": errors, "request_id": req_id}
    )

def get_service() -> SalaryService:
    if salary_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not ready. Run training first."
        )
    return salary_service

@app.get("/health", tags=["Monitoring"])
def health():
    ready = salary_service is not None
    return {
        "status": "ok" if ready else "initializing",
        "model_ready": ready,
        "version": "1.0.0"
    }

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/predict", response_model=SalaryForkResponse, tags=["Inference"])
def predict_salary_fork(resume: ResumeRequest):
    """Scenario 1: Predicts the salary fork (lower, median, upper bounds) in RUB."""
    service = get_service()
    return service.predict_fork(resume)

@app.post("/api/v1/recommend", response_model=list[SkillRecommendation], tags=["Inference"])
def recommend_skills(resume: ResumeRequest):
    """Returns top high-impact skills that will increase the user's salary fork with estimated boost (+₽)."""
    service = get_service()
    return service.recommend_skills(resume)

@app.post("/api/v1/search_vacancies", response_model=list[VacancyItem], tags=["Inference"])
def search_similar_vacancies(resume: ResumeRequest, top_k: int = 5):
    """Retrieves top matching market vacancies for the given candidate profile."""
    service = get_service()
    return service.search_similar_vacancies(resume, top_k=top_k)

@app.post("/api/v1/evaluate_resume", response_model=FullEvaluationResponse, tags=["Inference"])
def evaluate_resume_all_in_one(resume: ResumeRequest, request: Request):
    """Complete Scenario 1 in a single call: Salary Fork + Skill Recommendations + Similar Vacancies."""
    service = get_service()
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return service.evaluate_full(resume, request_id=req_id)

@app.post("/api/v1/recalculate", response_model=RecalculateResponse, tags=["Inference"])
def recalculate_resume(req: RecalculateRequest, request: Request):
    """Scenario 2 (Interactive improvement): user adds skills / updates experience -> fork recalculated."""
    service = get_service()
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return service.recalculate(req, request_id=req_id)
