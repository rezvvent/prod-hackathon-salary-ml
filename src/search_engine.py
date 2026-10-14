"""Vacancy search and analogue matching engine based on TF-IDF cosine similarity."""
import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import ARTIFACTS_DIR

class VacancySearchEngine:
    def __init__(self, vacancies_path=ARTIFACTS_DIR / "vacancies_index.jsonl"):
        vacancies_path = Path(vacancies_path)
        if not vacancies_path.exists():
            raise FileNotFoundError(f"Vacancies index not found at {vacancies_path}")

        self.df = pd.read_json(vacancies_path, lines=True)

        # Build corpus for text retrieval
        corpus = []
        for _, row in self.df.iterrows():
            skills_str = " ".join(row.get("skills") or [])
            doc = f"{row['role']} {row['experience']} {row['city']} {skills_str} {row.get('description', '')}"
            corpus.append(doc.lower())

        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search_similar(self, role, city, experience, skills, top_k=5):
        skills_str = " ".join(skills if isinstance(skills, list) else [skills])
        query = f"{role} {experience} {city} {skills_str}".lower()
        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "vacancy_id": row["id"],
                "role": row["role"],
                "city": row["city"],
                "experience": row["experience"],
                "skills": row.get("skills", []),
                "salary_from": float(row["salary_from"]),
                "salary_to": float(row["salary_to"]),
                "advertised_midpoint": float(row["target"]),
                "similarity_score": round(float(scores[idx]), 3)
            })
        return results
