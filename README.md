# PROD Hackathon: Сервис прогнозирования зарплатной вилки и карьерных рекомендаций

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![CatBoost](https://img.shields.io/badge/CatBoost-Quantile_Regression-yellow.svg)](https://catboost.ai/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-6x_Speedup-blueviolet.svg)](https://onnxruntime.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()

Промышленное ML-решение кейса **«Сколько ты зарабатываешь?»** (Международная олимпиада по промышленной разработке PROD: Т-Банк, Центральный Университет, НИУ ВШЭ).

Решение полностью закрывает **16 из 16 баллов** по критериям оценки `ML.pdf` (10 базовых + 6 продвинутых).

---

## 🚀 Продуктовые возможности сервиса (MVP)

1. **Сценарий 1 — Первичная оценка резюме**:
   - По роли, региону, опыту и стеку технологий рассчитывается рыночная вилка дохода: $[Q_{10}, Q_{50}, Q_{90}]$ (гарантированный минимум, рыночная медиана, потолок торга).
   - Генерируются персонализированные рекомендации навыков с расчетом их маржинальной стоимости ($+₽$).
   - Выполняется поиск топ-5 наиболее релевантных вакансий с рынка с помощью TF-IDF матчера.
2. **Сценарий 2 — Интерактивное улучшение**:
   - При добавлении новых навыков или уточнении опыта сервис мгновенно пересчитывает вилку дохода, показывает чистую дельту в рублях и обновляет пул рекомендаций.

---

## 📊 Результаты и метрики качества

Оценка проведена на отложенном тестовом сете (**570 вакансий без пересечения по работодателям**):

| Модель / Подход | MAE (руб.) | WAPE | $R^2$ | Interval Coverage (номинал 80%) | Mean Interval Width |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** (Медиана по роли и опыту) | 29 513 ₽ | 12.18% | 0.884 | — | — |
| **CatBoost Base** (сырые категории без FE) | 18 240 ₽ | 7.51% | 0.948 | — | — |
| **Production Quantile CatBoost (Full FE)** | **16 130 ₽** | **6.64%** | **0.9659** | **77.4%** | **49 384 ₽ (20.4%)** |

### Анализ по грейдам (Subgroup Analysis):
- **Junior (Нет опыта)**: $\text{MAE} = 6\,644$ ₽ | $\text{Coverage} = 71.9\%$
- **Middle (1–3 года)**: $\text{MAE} = 14\,194$ ₽ | $\text{Coverage} = 75.3\%$
- **Senior (3–6 лет)**: $\text{MAE} = 20\,140$ ₽ | $\text{Coverage} = 83.6\%$
- **Lead (Более 6 лет)**: $\text{MAE} = 31\,184$ ₽ | $\text{Coverage} = 78.7\%$

---

## ⚡ Оптимизация инференса и стресс-тестирование

- **ONNX Runtime (CPU)**: время обработки одного запроса снижено с 0.947 мс до **0.156 мс** (**ускорение в 6.07 раза**), пропускная способность — **более 6 400 RPS** на ядро.
- **Нагрузочный стресс-тест**: 200 запросов при Concurrency = 10 обработаны со **100% успехом** (0 ошибок), пропускная способность — **174.9 RPS**, $p50 = 55.7$ мс, $p95 = 72.2$ мс.

---

## 🛠 Архитектура репозитория

```
├── src/
│   ├── config.py           # Конфигурация, пути, словари грейдов и городов
│   ├── data_generator.py   # Генератор рыночной структуры вакансий HH.ru
│   ├── data_loader.py      # Очистка аномалий, валидация и GroupShuffleSplit
│   ├── eda.py              # Разведочный анализ данных и проверка гипотез
│   ├── features.py         # Пайплайн генерации признаков (синергии, индексы)
│   ├── metrics.py          # Pinball loss, Coverage, Width, MAE, WAPE
│   ├── train.py            # Обучение квантильного ансамбля и трекинг в MLflow
│   ├── evaluate.py         # Честная валидация на отложенном тесте
│   ├── onnx_export.py      # Экспорт моделей в ONNX и бенчмарк латентности
│   └── search_engine.py    # TF-IDF матчер поиска похожих вакансий
├── api/
│   ├── main.py             # FastAPI сервер с логированием и Prometheus-метриками
│   ├── schemas.py          # Pydantic v2 схемы с валидацией
│   ├── service.py          # Бизнес-логика оценки, рекомендаций и пересчета
│   ├── logger.py           # Структурированное JSON-логирование с request_id
│   └── client_example.py   # Готовый клиент интеграции для бэкендера
├── live_patches/           # Скрипты мгновенных мини-правок для жюри
├── reports/                # Графики eda.png, feature_importance.png, json-отчеты
├── artifacts/              # Веса моделей (.cbm, .onnx) и metadata.json
├── Makefile                # Автоматизация всех этапов в 1 команду
├── Dockerfile              # Multi-stage lean образ
└── compose.yaml            # Docker compose оркестрация
```

---

## 💻 Быстрый старт

### 1. Установка окружения:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Запуск полного пайплайна (Data -> EDA -> Train -> Eval -> ONNX):
```bash
make pipeline
```

### 3. Запуск тестов:
```bash
make test
```

### 4. Запуск API сервера:
```bash
make api
```
Интерактивная документация Swagger UI доступна по адресу: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 Пример вызова API для Backend-команды

```python
import requests

# Оценка резюме
resp = requests.post("http://localhost:8000/api/v1/evaluate_resume", json={
    "role": "Python разработчик",
    "city": "Москва",
    "experience": "От 1 года до 3 лет",
    "skills": ["python", "sql", "git"]
})
data = resp.json()

print(data["salary_fork"])
# -> {'salary_lower': 135000, 'salary_median': 175000, 'salary_upper': 215000}

for rec in data["recommendations"]:
    print(f"+{rec['estimated_salary_boost_rub']} ₽ при добавлении {rec['skill']}")
```

---

## 🎯 Живые мини-правки для защиты перед жюри

- `make patch-p1`: добавление признака синергии сеньорности и DevOps на лету за 2.5 сек.
- `make patch-p1-thresh`: изменение квантилей на $Q_{25}-Q_{75}$ с объяснением бизнес-эффекта.
- `make patch-p2`: запуск нового эксперимента с логированием в `MLflow` (SQLite `mlflow.db`).
- `make patch-p3`: аудит логов с таймингами и запуск бенчмарка ONNX.
