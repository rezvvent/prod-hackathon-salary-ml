"""Integration and contract tests for FastAPI inference endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app

def test_api_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

def test_api_metrics():
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "api_requests_total" in resp.text

def test_predict_and_contracts():
    payload = {
        "role": "Python разработчик",
        "city": "Москва",
        "experience": "От 1 года до 3 лет",
        "skills": ["python", "sql", "git"]
    }
    with TestClient(app) as client:
        # Predict fork
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["salary_lower"] <= data["salary_median"]
        assert data["salary_median"] <= data["salary_upper"]
        assert data["currency"] == "RUB"

        # Recommendations
        rec_resp = client.post("/api/v1/recommend", json=payload)
        assert rec_resp.status_code == 200
        recs = rec_resp.json()
        assert len(recs) > 0
        assert recs[0]["estimated_salary_boost_rub"] > 0

        # Full Scenario 1 evaluation
        eval_resp = client.post("/api/v1/evaluate_resume", json=payload)
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert "salary_fork" in eval_data
        assert "recommendations" in eval_data
        assert "similar_vacancies" in eval_data
        assert len(eval_data["similar_vacancies"]) > 0

        # Scenario 2 Recalculate
        recalc_payload = {
            "original_resume": payload,
            "added_skills": ["fastapi", "docker"]
        }
        recalc_resp = client.post("/api/v1/recalculate", json=recalc_payload)
        assert recalc_resp.status_code == 200
        recalc_data = recalc_resp.json()
        assert recalc_data["salary_delta_rub"] >= 0

def test_validation_error_422():
    with TestClient(app) as client:
        # Missing required role
        bad_payload = {"city": "Москва", "experience": "Нет опыта"}
        resp = client.post("/api/v1/predict", json=bad_payload)
        assert resp.status_code == 422
        assert "Validation Error" in resp.json()["detail"]
