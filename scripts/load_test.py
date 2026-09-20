"""
Day 27 Concurrent Load Testing & Resource Profiling Tool.
Evaluates multi-threaded concurrency levels (1, 5, 10, 25, 50) and measures
throughput (RPS), latency percentiles (P50, P95, P99), error rates, and CPU/RAM footprints.
"""

import time
import json
import statistics
import concurrent.futures
import urllib.request
import urllib.error
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import platform
import psutil
import os
from pathlib import Path

BASE_URL = os.getenv("BENCHMARK_BASE_URL", "http://127.0.0.1:8000")

CONCURRENCY_LEVELS = [1, 5, 10, 25, 50]
REQUESTS_PER_LEVEL = 100

SAMPLE_PAYLOAD = {
    "crop": "Oilseeds",
    "state": "Punjab",
    "district": "Ludhiana",
    "forecast_year": 2018,
    "yield_lag_1": 810.5,
    "yield_rolling_3yr_mean": 795.0,
    "area_lag_1": 12.0
}


def send_forecast_request(url: str, payload: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    start_time = time.perf_counter()
    headers = {"Content-Type": "application/json", "User-Agent": "Day27-LoadTest/1.0"}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    status_code = 0
    body = {}
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.getcode()
            raw = resp.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        status_code = e.code
        try:
            raw = e.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {"error": str(e)}
    except Exception as e:
        status_code = 599
        body = {"error": str(e)}

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return status_code, elapsed_ms, body


def run_concurrency_level(concurrency: int, total_requests: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/api/forecast/predict"
    latencies = []
    statuses = []
    responses = []

    process = psutil.Process()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    start_wall_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_forecast_request, url, SAMPLE_PAYLOAD) for _ in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            st, lat, body = f.result()
            statuses.append(st)
            latencies.append(lat)
            responses.append(body)

    total_wall_s = time.perf_counter() - start_wall_time
    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    cpu_after = psutil.cpu_percent(interval=None)

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    errors = sum(1 for s in statuses if s != 200)
    rps = n / total_wall_s if total_wall_s > 0 else 0.0

    p50 = statistics.median(latencies_sorted)
    p90 = latencies_sorted[int(0.90 * n) - 1]
    p95 = latencies_sorted[int(0.95 * n) - 1]
    p99 = latencies_sorted[int(0.99 * n) - 1]

    # Verify deterministic output consistency across successful predictions
    preds = [r.get("prediction") for r in responses if isinstance(r, dict) and r.get("status") == "SUCCESS"]
    strategies = [r.get("strategy") for r in responses if isinstance(r, dict) and r.get("status") == "SUCCESS"]
    versions = [r.get("model_version") for r in responses if isinstance(r, dict) and r.get("status") == "SUCCESS"]

    unique_preds = set(preds)
    unique_strategies = set(strategies)
    unique_versions = set(versions)

    is_deterministic = (len(unique_preds) <= 1) and (len(unique_strategies) <= 1)

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": n - errors,
        "failed_requests": errors,
        "error_rate_pct": round((errors / n) * 100.0, 2),
        "total_time_s": round(total_wall_s, 3),
        "throughput_rps": round(rps, 2),
        "min_ms": round(latencies_sorted[0], 2),
        "max_ms": round(latencies_sorted[-1], 2),
        "mean_ms": round(statistics.mean(latencies_sorted), 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "std_ms": round(statistics.stdev(latencies_sorted) if n > 1 else 0.0, 2),
        "cpu_util_pct": round(cpu_after, 1),
        "mem_used_mb": round(mem_after_mb, 2),
        "mem_delta_mb": round(mem_after_mb - mem_before_mb, 2),
        "deterministic_inference_verified": is_deterministic,
        "unique_predictions_count": len(unique_preds),
        "observed_prediction_value": list(unique_preds)[0] if unique_preds else None,
    }


def run_load_test_suite() -> Dict[str, Any]:
    print("=" * 80)
    print("DAY 27: CONCURRENT LOAD TESTING & DETERMINISTIC PROFILING")
    print(f"Target: {BASE_URL}/api/forecast/predict")
    print(f"Concurrency Levels: {CONCURRENCY_LEVELS}")
    print(f"Requests per Concurrency Level: {REQUESTS_PER_LEVEL}")
    print("=" * 80)
    print(f"{'Concurrency':<12} | {'Requests':<8} | {'Success':<8} | {'Errors':<6} | {'RPS':<8} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | {'Deterministic':<13}")
    print("-" * 88)

    level_results = []
    for c in CONCURRENCY_LEVELS:
        res = run_concurrency_level(c, REQUESTS_PER_LEVEL)
        level_results.append(res)
        det_str = "YES (delta=0)" if res["deterministic_inference_verified"] else "FAILED"
        print(
            f"{res['concurrency']:<12} | {res['total_requests']:<8} | {res['successful_requests']:<8} | "
            f"{res['failed_requests']:<6} | {res['throughput_rps']:<8.1f} | {res['p50_ms']:<9.2f} | "
            f"{res['p95_ms']:<9.2f} | {res['p99_ms']:<9.2f} | {det_str:<13}"
        )

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "test_parameters": {
            "concurrency_levels": CONCURRENCY_LEVELS,
            "requests_per_level": REQUESTS_PER_LEVEL,
            "target_endpoint": "/api/forecast/predict",
        },
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "concurrency_matrix": level_results,
    }

    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "concurrency_load_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print(f"Concurrency results saved to: {out_path}")
    print("=" * 80)
    return output


if __name__ == "__main__":
    run_load_test_suite()
