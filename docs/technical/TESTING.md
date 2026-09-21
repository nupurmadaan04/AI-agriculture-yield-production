# Automated Testing Strategy & Test Suites

## 1. Test Architecture Overview

The platform maintains comprehensive automated testing covering unit, contract, integration, resilience, and scientific reproducibility layers:

```
+---------------------------------------------------------------------------------------------------------+
|                                    AUTOMATED TEST SUITE MATRIX                                          |
+-----------------------------------+------------+--------------------------------------------------------+
| Test Suite File                   | Tests      | Scope & Coverage                                       |
+-----------------------------------+------------+--------------------------------------------------------+
| `test_day36_reproducibility.py`   | 7 tests    | Scientific claims, metric reproduction, hash integrity |
| `test_day35_ui_contracts.py`      | 12 tests   | WCAG 2.1 AA accessibility, terminology sanitization   |
| `test_deployment_verification.py` | 13 tests   | Container probes, Nginx proxy, restart determinism     |
| `test_end_to_end_acceptance.py`   | 16 tests   | Full forecast, explanation, and scenario workflows     |
| `test_security_acceptance.py`     | 14 tests   | OWASP headers, SQLi/XSS, non-root execution            |
| `test_failure_resilience.py`      | 12 tests   | Strategy fallbacks, corrupt inputs, graceful error     |
| `test_api_contracts.py`           | 8 tests    | Schema validation, Pydantic type safety                |
| `test_provenance_chain.py`        | 8 tests    | SHA-256 digital signature reproducibility              |
| `test_concurrency_safety.py`      | 6 tests    | Multi-threaded audit log write safety                  |
+-----------------------------------+------------+--------------------------------------------------------+
```

---

## 2. Execution Commands

```bash
# Execute Day 34-36 core verification suites (32 tests)
pytest tests/test_day36_reproducibility.py tests/test_day35_ui_contracts.py tests/test_deployment_verification.py -v

# Execute all acceptance, security, and resilience suites (96 tests)
pytest tests/test_day36_reproducibility.py tests/test_day35_ui_contracts.py tests/test_deployment_verification.py tests/test_end_to_end_acceptance.py tests/test_security_acceptance.py tests/test_failure_resilience.py tests/test_api_contracts.py tests/test_provenance_chain.py tests/test_concurrency_safety.py -v
```
