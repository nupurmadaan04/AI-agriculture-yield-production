import time
import json
import statistics
import urllib.request
import urllib.error
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import platform
import psutil
import os
from pathlib import Path

BASE_URL = os.getenv("BENCHMARK_BASE_URL", "http://127.0.0.1:8000")

TEST_CASES = [
    {
        "name": "Health Probe",
        "method": "GET",
        "endpoint": "/health",
        "payload": None,
        "expected_status": 200,
        "category": "HEALTH",
    },
    {
        "name": "Readiness Probe",
        "method": "GET",
        "endpoint": "/ready",
        "payload": None,
        "expected_status": 200,
        "category": "READINESS",
    },
    {
        "name": "Forecast Strategy Registry",
        "method": "GET",
        "endpoint": "/api/forecast/strategies",
        "payload": None,
        "expected_status": 200,
        "category": "STRATEGY_REGISTRY",
    },
    {
        "name": "Forecast Certification Summary",
        "method": "GET",
        "endpoint": "/api/forecast/certification",
        "payload": None,
        "expected_status": 200,
        "category": "CERTIFICATION",
    },
    {
        "name": "Forecast Coverage Matrix",
        "method": "GET",
        "endpoint": "/api/forecast/coverage",
        "payload": None,
        "expected_status": 200,
        "category": "COVERAGE",
    },
    {
        "name": "Oilseeds Forecast (ML Production Ready)",
        "method": "POST",
        "endpoint": "/api/forecast/predict",
        "payload": {
            "crop": "Oilseeds",
            "state": "Punjab",
            "district": "Ludhiana",
            "forecast_year": 2018,
            "yield_lag_1": 810.5,
            "yield_rolling_3yr_mean": 795.0,
            "area_lag_1": 12.0
        },
        "expected_status": 200,
        "category": "FORECAST_ML_PROD",
    },
    {
        "name": "Sugarcane Forecast (ML Conditional Production)",
        "method": "POST",
        "endpoint": "/api/forecast/predict",
        "payload": {
            "crop": "Sugarcane",
            "state": "Uttar Pradesh",
            "district": "Meerut",
            "forecast_year": 2018,
            "yield_lag_1": 9200.0,
            "yield_rolling_3yr_mean": 9100.0,
            "area_lag_1": 45.0
        },
        "expected_status": 200,
        "category": "FORECAST_ML_CONDITIONAL",
    },
    {
        "name": "Rice Forecast (Baseline Production)",
        "method": "POST",
        "endpoint": "/api/forecast/predict",
        "payload": {
            "crop": "Rice",
            "state": "Punjab",
            "district": "Ludhiana",
            "forecast_year": 2018,
            "yield_lag_1": 3100.0,
            "yield_rolling_3yr_mean": 3050.0,
            "area_lag_1": 250.0
        },
        "expected_status": 200,
        "category": "FORECAST_BASELINE",
    },
    {
        "name": "Unsupported Crop Rejection Guard",
        "method": "POST",
        "endpoint": "/api/forecast/predict",
        "payload": {
            "crop": "Potato",
            "state": "Punjab",
            "district": "Ludhiana",
            "forecast_year": 2018
        },
        "expected_status": 200,  # Returns structured REJECTED payload with 200 or 4xx
        "category": "FORECAST_REJECTION",
    },
]


def send_request(method: str, url: str, payload: Any = None) -> Tuple[int, float, Dict[str, Any]]:
    start_time = time.perf_counter()
    headers = {"Content-Type": "application/json", "User-Agent": "Day27-Benchmark/1.0"}
    data = json.dumps(payload).encode("utf-8") if payload is not None else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    status_code = 0
    body = {}
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
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


def run_benchmark(num_warm_requests: int = 100) -> Dict[str, Any]:
    print("=" * 70)
    print("DAY 27: PERFORMANCE & LATENCY BENCHMARK SUITE")
    print(f"Target Base URL: {BASE_URL}")
    print(f"Sample Count per Endpoint: 1 Cold + {num_warm_requests} Warm")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU Count: {psutil.cpu_count(logical=True)} logical / {psutil.cpu_count(logical=False)} physical")
    print(f"Total RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    print("=" * 70)

    results = []
    saved_request_ids = []

    for tc in TEST_CASES:
        name = tc["name"]
        method = tc["method"]
        url = f"{BASE_URL}{tc['endpoint']}"
        payload = tc["payload"]

        # 1. Cold request
        cold_status, cold_latency, cold_body = send_request(method, url, payload)
        if isinstance(cold_body, dict) and "request_id" in cold_body:
            saved_request_ids.append(cold_body["request_id"])

        # 2. Warm requests
        warm_latencies = []
        statuses = []
        errors = 0

        for _ in range(num_warm_requests):
            st, lat, b = send_request(method, url, payload)
            statuses.append(st)
            warm_latencies.append(lat)
            if st >= 400 and tc["category"] != "FORECAST_REJECTION":
                errors += 1
            if isinstance(b, dict) and "request_id" in b and len(saved_request_ids) < 5:
                saved_request_ids.append(b["request_id"])

        warm_latencies_sorted = sorted(warm_latencies)
        n = len(warm_latencies_sorted)

        min_lat = warm_latencies_sorted[0]
        max_lat = warm_latencies_sorted[-1]
        mean_lat = statistics.mean(warm_latencies_sorted)
        median_lat = statistics.median(warm_latencies_sorted)
        std_lat = statistics.stdev(warm_latencies_sorted) if n > 1 else 0.0

        p90_lat = warm_latencies_sorted[int(0.90 * n) - 1]
        p95_lat = warm_latencies_sorted[int(0.95 * n) - 1]
        p99_lat = warm_latencies_sorted[int(0.99 * n) - 1]
        total_time_s = sum(warm_latencies) / 1000.0
        rps = n / total_time_s if total_time_s > 0 else 0.0

        res = {
            "name": name,
            "category": tc["category"],
            "endpoint": tc["endpoint"],
            "method": method,
            "cold_latency_ms": round(cold_latency, 2),
            "cold_status": cold_status,
            "warm_samples": n,
            "min_ms": round(min_lat, 2),
            "max_ms": round(max_lat, 2),
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(median_lat, 2),
            "p90_ms": round(p90_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "p99_ms": round(p99_lat, 2),
            "std_ms": round(std_lat, 2),
            "rps": round(rps, 2),
            "success_rate": round(((n - errors) / n) * 100.0, 2),
            "error_count": errors,
        }
        results.append(res)

        print(
            f"[{tc['category']:<22}] Cold: {cold_latency:>6.2f}ms | "
            f"Warm P50: {median_lat:>5.2f}ms | P95: {p95_lat:>5.2f}ms | P99: {p99_lat:>5.2f}ms | "
            f"RPS: {rps:>6.1f} | Success: {res['success_rate']}%"
        )

    # Test Provenance Lookup if request_ids available
    if saved_request_ids:
        prov_req_id = saved_request_ids[0]
        prov_url = f"{BASE_URL}/api/forecast/provenance/{prov_req_id}"
        c_st, c_lat, _ = send_request("GET", prov_url)
        w_lats = []
        for _ in range(num_warm_requests):
            st, lat, _ = send_request("GET", prov_url)
            w_lats.append(lat)
        w_lats.sort()
        p_res = {
            "name": "Provenance Record Lookup",
            "category": "PROVENANCE_LOOKUP",
            "endpoint": f"/api/forecast/provenance/{prov_req_id[:8]}...",
            "method": "GET",
            "cold_latency_ms": round(c_lat, 2),
            "cold_status": c_st,
            "warm_samples": len(w_lats),
            "min_ms": round(w_lats[0], 2),
            "max_ms": round(w_lats[-1], 2),
            "mean_ms": round(statistics.mean(w_lats), 2),
            "p50_ms": round(statistics.median(w_lats), 2),
            "p90_ms": round(w_lats[int(0.90 * len(w_lats)) - 1], 2),
            "p95_ms": round(w_lats[int(0.95 * len(w_lats)) - 1], 2),
            "p99_ms": round(w_lats[int(0.99 * len(w_lats)) - 1], 2),
            "std_ms": round(statistics.stdev(w_lats), 2),
            "rps": round(len(w_lats) / (sum(w_lats) / 1000.0), 2),
            "success_rate": 100.0 if c_st == 200 else 0.0,
            "error_count": 0,
        }
        results.append(p_res)
        print(
            f"[{'PROVENANCE_LOOKUP':<22}] Cold: {c_lat:>6.2f}ms | "
            f"Warm P50: {p_res['p50_ms']:>5.2f}ms | P95: {p_res['p95_ms']:>5.2f}ms | P99: {p_res['p99_ms']:>5.2f}ms | "
            f"RPS: {p_res['rps']:>6.1f} | Success: {p_res['success_rate']}%"
        )

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "benchmark_parameters": {
            "num_warm_requests": num_warm_requests,
            "sample_strategy": "1 cold + N warm sequential requests",
        },
        "results": results,
    }

    # Save to json report
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sequential_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 70)
    print(f"Benchmark results saved to: {out_path}")
    print("=" * 70)
    return output


if __name__ == "__main__":
    run_benchmark(num_warm_requests=100)
