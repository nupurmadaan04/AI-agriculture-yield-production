# Day 28: Production Monitoring Architecture

## 1. Architectural Overview

The Day 28 Observability and Operational Intelligence layer is designed to monitor, trace, and diagnose the entire agricultural prediction lifecycle without creating a parallel execution path or compromising frozen scientific models.

```mermaid
graph TD
    User([User / Operator / API Client]) --> Nginx[Nginx Reverse Proxy]
    Nginx --> FastAPI[FastAPI Web Server]
    FastAPI --> Middleware[Request ID & Telemetry Middleware]
    
    subgraph Observability Engine [Singleton Observability Engine]
        RingBuffer[In-Memory Request Ring Buffer]
        TraceStore[In-Memory Forecast Trace Store]
        EventBuffer[Operational Event Buffer]
        PersistLog[(operational_telemetry.jsonl)]
        IntegrityChecker[Cryptographic SHA-256 Verifiers]
        AlertEvaluator[Configurable Operational Rule Engine]
    end

    Middleware -->|Non-blocking Telemetry Record| RingBuffer
    Middleware -->|Async Append| PersistLog

    FastAPI --> ForecastService[Prediction Service]
    
    subgraph Governed Forecast Execution
        Stage1[Stage 1: Input Validation] --> Stage2[Stage 2: Certification Guard]
        Stage2 --> Stage3[Stage 3: Strategy Registry Lookup]
        Stage3 --> Stage4[Stage 4: Inference / Baseline Execution]
        Stage4 --> Stage5[Stage 5: Provenance Generation]
        Stage5 --> Stage6[Stage 6: Prediction Audit Logging]
    end

    ForecastService --> Stage1
    Stage6 --> AuditCSV[(prediction_audit_log.csv)]
    Stage6 -->|Record Microsecond Stage Latencies| TraceStore

    AlertEvaluator -->|Read Telemetry & Live Checks| ObservabilityAPI[/api/observability/*]
    ObservabilityAPI --> FrontendUI[Observability Center Frontend]
```

---

## 2. Telemetry Ingestion & Storage

1. **In-Memory Ring Buffers**:
   - `_request_buffer` (capacity: 1,000 items): Fast percentile calculation (P50, P90, P95, P99) and error rate derivation.
   - `_forecast_trace_store` (keyed by `request_id`): Microsecond stage-by-stage timings and validation metadata.
   - `_event_buffer` (capacity: 500 items): Warning and error event log.
2. **Persistent Storage**:
   - `Datasets/metadata/operational_telemetry.jsonl`: Structured JSONL log for long-term request and error history.
   - `Datasets/metadata/prediction_audit_log.csv`: Authoritative CSV audit store used for deterministic provenance and trace fallback.
3. **Retention Policy**:
   - In-memory data is bounded to fixed FIFO queues preventing memory growth.
   - Filesystem rotation checks log size and truncates/rotates when exceeding 50MB.

---

## 3. Endpoints Matrix

- `GET /api/observability/summary`: High-level operational intelligence rollup.
- `GET /api/observability/health`: Live process, host CPU/RAM, and uptime telemetry.
- `GET /api/observability/metrics`: Real-time request counts, error rates, and latency percentiles.
- `GET /api/observability/forecasts`: Live strategy invocations, crop distributions, and recent operations.
- `GET /api/observability/strategies`: Registered strategies with live usage percentages.
- `GET /api/observability/trace/{request_id}`: Granular 6-stage execution pipeline trace with measured latencies.
- `GET /api/observability/models`: Live SHA-256 cryptographic verification of model artifacts.
- `GET /api/observability/dataset`: Canonical dataset metadata, row count, and schema check.
- `GET /api/observability/registry`: Strategy registry availability and certification status.
- `GET /api/observability/alerts`: Configured operational alert rules and active/resolved alerts.
- `GET /api/observability/drift`: Feature drift monitoring signals (PSI/KS) with explicit scientific notices.
- `GET /api/observability/errors`: Categorized operational errors and recent incident events.
