"""Example Python client for Backend teammates to integrate the ML Service in under 5 minutes."""
import requests
import json

BASE_URL = "http://localhost:8000"

def evaluate_candidate_resume():
    endpoint = f"{BASE_URL}/api/v1/evaluate_resume"
    payload = {
        "role": "Python разработчик",
        "city": "Москва",
        "experience": "От 1 года до 3 лет",
        "skills": ["python", "sql", "git"]
    }

    print(f"1. Sending candidate resume to {endpoint}...")
    try:
        resp = requests.post(endpoint, json=payload, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()

        fork = data["salary_fork"]
        print(f"\n[Вилка дохода]: {fork['salary_lower']:,.0f} - {fork['salary_upper']:,.0f} ₽ (Медиана: {fork['salary_median']:,.0f} ₽)")

        print("\n[Топ рекомендаций по улучшению резюме]:")
        for rec in data["recommendations"]:
            print(f"  • Навык: {rec['skill'].upper()} -> Прирост: +{rec['estimated_salary_boost_rub']:,.0f} ₽ (Новая медиана: {rec['new_median_salary_rub']:,.0f} ₽)")
            print(f"    Обоснование: {rec['rationale']}")

        print(f"\n[Похожие вакансии с рынка]: {len(data['similar_vacancies'])} шт.")
        for v in data["similar_vacancies"][:2]:
            print(f"  - {v['role']} в {v['city']} | Вилка: {v['salary_from']:,.0f} - {v['salary_to']:,.0f} ₽ (Сходство: {v['similarity_score']})")

        return data
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ML service: {e}")
        return None

def recalculate_after_adding_skills():
    endpoint = f"{BASE_URL}/api/v1/recalculate"
    payload = {
        "original_resume": {
            "role": "Python разработчик",
            "city": "Москва",
            "experience": "От 1 года до 3 лет",
            "skills": ["python", "sql", "git"]
        },
        "added_skills": ["fastapi", "docker"]
    }

    print(f"\n2. Recalculating resume after candidate added FastAPI + Docker...")
    try:
        resp = requests.post(endpoint, json=payload, timeout=5.0)
        resp.raise_for_status()
        res = resp.json()

        print(f"[Старая вилка]: {res['previous_fork']['salary_median']:,.0f} ₽")
        print(f"[Новая вилка]:  {res['updated_fork']['salary_median']:,.0f} ₽")
        print(f"[Чистый прирост]: +{res['salary_delta_rub']:,.0f} ₽")
    except requests.exceptions.RequestException as e:
        print(f"Error recalculating: {e}")

if __name__ == "__main__":
    evaluate_candidate_resume()
    recalculate_after_adding_skills()
