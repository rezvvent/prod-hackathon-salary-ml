"""LIVE PATCH P3: Sends sample requests to API, inspects latency logs, and triggers ONNX benchmark."""
import time
import requests
import json
from src.onnx_export import export_and_benchmark

BASE_URL = "http://localhost:8000"

def run_p3_demonstration():
    print("=== LIVE MINI-PRAVKA P3: OBSERVABILITY LOGS & INFERENCE OPTIMIZATION ===\n")

    # 1. Send live requests
    print("1. Sending sample requests to API to inspect latency and audit logs:")
    samples = [
        {"role": "Python разработчик", "city": "Москва", "experience": "От 1 года до 3 лет", "skills": ["python", "sql", "docker"]},
        {"role": "Data Scientist / ML Engineer", "city": "Санкт-Петербург", "experience": "От 3 до 6 лет", "skills": ["python", "pytorch"]},
        {"city": "Москва"}  # Invalid request -> triggers 422
    ]

    for i, s in enumerate(samples, 1):
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{BASE_URL}/api/v1/predict", json=s, timeout=5.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            req_id = resp.headers.get("X-Request-ID", "N/A")
            print(f"  [Request {i}] Status: {resp.status_code} | Latency: {elapsed_ms:.2f} ms | Request-ID: {req_id}")
            if resp.status_code == 200:
                print(f"    -> Output: {resp.json()['salary_median']:,.0f} ₽ (Fork: {resp.json()['salary_lower']:,.0f} - {resp.json()['salary_upper']:,.0f} ₽)")
            elif resp.status_code == 422:
                print(f"    -> Handled 422 Validation Error properly: {resp.json()['errors'][0]['msg']}")
        except requests.exceptions.ConnectionError:
            print(f"  [Note] API server not running at {BASE_URL}. Run 'make api' or uvicorn to see live HTTP logs.")
            break

    print("\n2. Demonstrating Inference Optimization with ONNX Runtime:")
    bench = export_and_benchmark(num_benchmark_samples=200)
    print(f"\nResult: ONNX Runtime achieves {bench['speedup_factor']}x speedup over Native CatBoost.")

if __name__ == "__main__":
    run_p3_demonstration()
