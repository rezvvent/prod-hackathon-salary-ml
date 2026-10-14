"""ONNX conversion for CatBoost Quantile models and latency/throughput benchmark."""
import time
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from catboost import CatBoostRegressor

from src.config import ARTIFACTS_DIR, REPORTS_DIR
from src.features import build_features

def export_and_benchmark(num_benchmark_samples=500):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare sample input
    sample_input = [
        {"role": "Python разработчик", "city": "Москва", "experience": "От 1 года до 3 лет", "skills": ["python", "sql", "docker", "fastapi"]},
        {"role": "Data Scientist / ML Engineer", "city": "Санкт-Петербург", "experience": "От 3 до 6 лет", "skills": ["python", "pytorch", "scikit-learn", "sql", "docker"]},
        {"role": "Frontend разработчик (React)", "city": "Удаленная работа (РФ)", "experience": "Нет опыта", "skills": ["react", "typescript", "git"]}
    ]
    df_sample = build_features(sample_input, engineered=True)
    # CatBoost expects string/numerical features
    cat_cols = ["role", "city", "experience"]

    native_models = {}
    onnx_sessions = {}

    print("--- Exporting CatBoost models to ONNX ---")
    for q_tag in ["q10", "q50", "q90"]:
        cbm_path = ARTIFACTS_DIR / f"model_{q_tag}.cbm"
        onnx_path = ARTIFACTS_DIR / f"model_{q_tag}.onnx"

        cb = CatBoostRegressor()
        cb.load_model(str(cbm_path))
        native_models[q_tag] = cb

        # Export to ONNX format natively supported by CatBoost
        cb.save_model(
            str(onnx_path),
            format="onnx",
            export_parameters={
                'onnx_doc_string': f'CatBoost salary quantile model {q_tag}',
                'onnx_graph_name': f'CatBoost_{q_tag}'
            }
        )
        print(f"  Exported {onnx_path.name}")

        # Create ONNX Runtime session
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_sessions[q_tag] = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    # 2. Benchmark Native vs ONNX latency
    print(f"\n--- Running Latency Benchmark ({num_benchmark_samples} iterations) ---")
    single_item = sample_input[:1]
    X_single = build_features(single_item, engineered=True)

    # Prepare ONNX input dict
    # Inspect ONNX input types
    input_names = [i.name for i in onnx_sessions["q50"].get_inputs()]
    
    # Benchmark Native
    native_latencies = []
    for _ in range(num_benchmark_samples):
        t0 = time.perf_counter()
        _ = native_models["q50"].predict(X_single)
        native_latencies.append((time.perf_counter() - t0) * 1000)

    # Benchmark ONNX Runtime
    input_name = onnx_sessions["q50"].get_inputs()[0].name
    X_mat = X_single.values.astype(np.float32)
    onnx_feed = {input_name: X_mat}

    onnx_latencies = []
    for _ in range(num_benchmark_samples):
        t0 = time.perf_counter()
        _ = onnx_sessions["q50"].run(None, onnx_feed)
        onnx_latencies.append((time.perf_counter() - t0) * 1000)

    bench_report = {
        "benchmark_samples": num_benchmark_samples,
        "native_catboost": {
            "mean_latency_ms": round(float(np.mean(native_latencies)), 3),
            "p50_latency_ms": round(float(np.percentile(native_latencies, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(native_latencies, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(native_latencies, 99)), 3),
            "throughput_req_per_sec": round(float(1000.0 / np.mean(native_latencies)), 1)
        },
        "onnx_runtime": {
            "mean_latency_ms": round(float(np.mean(onnx_latencies)), 3),
            "p50_latency_ms": round(float(np.percentile(onnx_latencies, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(onnx_latencies, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(onnx_latencies, 99)), 3),
            "throughput_req_per_sec": round(float(1000.0 / np.mean(onnx_latencies)), 1)
        },
        "speedup_factor": round(float(np.mean(native_latencies) / np.mean(onnx_latencies)), 2)
    }

    report_path = REPORTS_DIR / "onnx_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(bench_report, f, indent=2)

    print(f"Benchmark results:\n"
          f"  Native CatBoost: {bench_report['native_catboost']['mean_latency_ms']} ms/req ({bench_report['native_catboost']['throughput_req_per_sec']} RPS)\n"
          f"  ONNX Runtime:    {bench_report['onnx_runtime']['mean_latency_ms']} ms/req ({bench_report['onnx_runtime']['throughput_req_per_sec']} RPS)\n"
          f"  Speedup:         {bench_report['speedup_factor']}x faster with ONNX Runtime")
    print(f"Saved benchmark report -> {report_path}")
    return bench_report

if __name__ == "__main__":
    export_and_benchmark()
