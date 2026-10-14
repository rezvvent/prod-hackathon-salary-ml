"""Asynchronous stress and load testing script for FastAPI inference benchmarking."""
import asyncio
import time
import json
from pathlib import Path
import httpx
import numpy as np

REPORT_PATH = Path("reports/load_test_report.json")

TEST_PAYLOADS = [
    {
        "role": "Python разработчик",
        "city": "Москва",
        "experience": "От 1 года до 3 лет",
        "skills": ["python", "sql", "git", "fastapi"]
    },
    {
        "role": "Data Scientist / ML Engineer",
        "city": "Санкт-Петербург",
        "experience": "От 3 до 6 лет",
        "skills": ["python", "pytorch", "scikit-learn", "docker"]
    },
    {
        "role": "Backend разработчик (Go)",
        "city": "Удаленная работа (РФ)",
        "experience": "От 1 года до 3 лет",
        "skills": ["go", "postgresql", "docker", "grpc"]
    },
    {
        "role": "Frontend разработчик (React)",
        "city": "Казань",
        "experience": "Нет опыта",
        "skills": ["react", "typescript", "git"]
    }
]

async def send_request(client, url, payload, latencies, errors):
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=10.0)
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            latencies.append(elapsed)
        else:
            errors.append(resp.status_code)
    except Exception as e:
        errors.append(str(e))

async def run_load_test(url="http://localhost:8000/api/v1/predict", total_requests=200, concurrency=10):
    print(f"=== Starting Stress Test ===")
    print(f"Target URL: {url}")
    print(f"Total Requests: {total_requests} | Concurrency: {concurrency}")

    latencies = []
    errors = []

    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        start_time = time.perf_counter()
        queue = asyncio.Queue()

        for i in range(total_requests):
            queue.put_nowait(TEST_PAYLOADS[i % len(TEST_PAYLOADS)])

        async def worker():
            while not queue.empty():
                try:
                    payload = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await send_request(client, url, payload, latencies, errors)
                queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks)

        total_duration = time.perf_counter() - start_time

    success_count = len(latencies)
    error_count = len(errors)
    throughput_rps = round(success_count / max(total_duration, 0.001), 1)

    latencies_arr = np.array(latencies)
    report = {
        "url": url,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successful_requests": success_count,
        "failed_requests": error_count,
        "total_duration_sec": round(total_duration, 3),
        "throughput_rps": throughput_rps,
        "latency_ms": {
            "mean": round(float(np.mean(latencies_arr)), 2) if success_count else 0,
            "min": round(float(np.min(latencies_arr)), 2) if success_count else 0,
            "p50": round(float(np.percentile(latencies_arr, 50)), 2) if success_count else 0,
            "p90": round(float(np.percentile(latencies_arr, 90)), 2) if success_count else 0,
            "p95": round(float(np.percentile(latencies_arr, 95)), 2) if success_count else 0,
            "p99": round(float(np.percentile(latencies_arr, 99)), 2) if success_count else 0,
            "max": round(float(np.max(latencies_arr)), 2) if success_count else 0
        }
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n--- Load Test Results ---")
    print(f"Success: {success_count}/{total_requests} ({success_count/total_requests*100:.1f}%)")
    print(f"Throughput: {throughput_rps} req/sec")
    print(f"Latency p50: {report['latency_ms']['p50']} ms | p95: {report['latency_ms']['p95']} ms | p99: {report['latency_ms']['p99']} ms")
    print(f"Report saved -> {REPORT_PATH}")
    return report

if __name__ == "__main__":
    asyncio.run(run_load_test())
