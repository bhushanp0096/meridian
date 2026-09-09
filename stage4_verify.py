"""
Stage 4 — FastAPI Serving Layer Verification Script
===================================================
Tests all HTTP endpoints exposed by api/main.py using FastAPI TestClient:
1. Validates OpenAPI schema generation and path matching.
2. Exercises all GET analysis endpoints and verifies Cache-Control & X-Cache headers.
3. Tests POST /optimize (cache hit path for precomputed scenario).
4. Tests POST /whatif simulation endpoint with custom CPM multipliers.
5. Verifies all endpoint responses conform to Stage 2 Pydantic schemas.

Outputs:
- logs/stage4_verification.json
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

APP_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))
sys.path.insert(0, str(APP_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402
from api import schemas  # noqa: E402

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "stage4_verification.json"


def run_tests() -> list[dict[str, Any]]:
    results = []

    def record(name: str, status: str, detail: str = "", duration_s: float = 0.0):
        results.append({
            "test": name,
            "status": status,
            "detail": detail,
            "duration_seconds": round(duration_s, 4),
        })
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}" + (f" — {detail}" if detail else ""))

    # 1. Test OpenAPI schema compilation
    t0 = time.time()
    try:
        openapi = app.openapi()
        expected_paths = [
            "/health",
            "/model/spec",
            "/model/info",
            "/contributions",
            "/roi-summary",
            "/response-curves",
            "/response-curves/{channel}",
            "/baseline-vs-media",
            "/accuracy",
            "/spend-summary",
            "/optimize",
            "/whatif",
        ]
        missing = [p for p in expected_paths if p not in openapi.get("paths", {})]
        assert not missing, f"Missing OpenAPI paths: {missing}"
        record(
            "api.openapi_schema",
            "PASS",
            f"All {len(expected_paths)} endpoints registered in OpenAPI",
            time.time() - t0,
        )
    except Exception as e:
        record("api.openapi_schema", "FAIL", str(e), time.time() - t0)
        return results

    # Initialize TestClient with lifespan
    with TestClient(app) as client:
        # 2. GET /health
        t0 = time.time()
        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            health_model = schemas.HealthResponse.model_validate(data)
            assert health_model.status == "healthy"
            assert health_model.model_loaded is True
            assert health_model.cache_ready is True
            record("api.get_health", "PASS", f"Status: {health_model.status}", time.time() - t0)
        except Exception as e:
            record("api.get_health", "FAIL", str(e), time.time() - t0)

        # 3. GET /model/spec
        t0 = time.time()
        try:
            resp = client.get("/model/spec")
            assert resp.status_code == 200
            assert "Cache-Control" in resp.headers
            spec_model = schemas.ModelSpecResponse.model_validate(resp.json())
            record("api.get_model_spec", "PASS", f"KPI: {spec_model.kpi_name}, {spec_model.n_geos} geos", time.time() - t0)
        except Exception as e:
            record("api.get_model_spec", "FAIL", str(e), time.time() - t0)

        # 4. GET /model/info
        t0 = time.time()
        try:
            resp = client.get("/model/info")
            assert resp.status_code == 200
            info_model = schemas.ModelInfoResponse.model_validate(resp.json())
            record("api.get_model_info", "PASS", f"Loaded={info_model.loaded}", time.time() - t0)
        except Exception as e:
            record("api.get_model_info", "FAIL", str(e), time.time() - t0)

        # 5. GET /contributions
        t0 = time.time()
        try:
            resp = client.get("/contributions")
            assert resp.status_code == 200
            assert resp.headers.get("X-Cache") == "HIT"
            items = [schemas.ChannelContribution.model_validate(x) for x in resp.json()]
            record("api.get_contributions", "PASS", f"X-Cache={resp.headers.get('X-Cache')}, {len(items)} channels", time.time() - t0)
        except Exception as e:
            record("api.get_contributions", "FAIL", str(e), time.time() - t0)

        # 6. GET /roi-summary
        t0 = time.time()
        try:
            resp = client.get("/roi-summary")
            assert resp.status_code == 200
            assert resp.headers.get("X-Cache") == "HIT"
            items = [schemas.RoiSummaryItem.model_validate(x) for x in resp.json()]
            record("api.get_roi_summary", "PASS", f"X-Cache={resp.headers.get('X-Cache')}, {len(items)} channels", time.time() - t0)
        except Exception as e:
            record("api.get_roi_summary", "FAIL", str(e), time.time() - t0)

        # 7. GET /response-curves & /response-curves/{channel}
        t0 = time.time()
        try:
            resp_all = client.get("/response-curves")
            assert resp_all.status_code == 200
            assert resp_all.headers.get("X-Cache") == "HIT"
            all_curves = resp_all.json()
            assert len(all_curves) == 7

            resp_one = client.get("/response-curves/Online_Video")
            assert resp_one.status_code == 200
            pts = [schemas.ResponseCurvePoint.model_validate(x) for x in resp_one.json()]
            assert len(pts) == 41

            # 404 check
            resp_bad = client.get("/response-curves/InvalidChannelName")
            assert resp_bad.status_code == 404

            record("api.get_response_curves", "PASS", f"All curves (7 ch) & single curve (41 pts) verified", time.time() - t0)
        except Exception as e:
            record("api.get_response_curves", "FAIL", str(e), time.time() - t0)

        # 8. GET /baseline-vs-media
        t0 = time.time()
        try:
            resp = client.get("/baseline-vs-media")
            assert resp.status_code == 200
            assert resp.headers.get("X-Cache") == "HIT"
            items = [schemas.BaselineVsMediaPoint.model_validate(x) for x in resp.json()]
            record("api.get_baseline_vs_media", "PASS", f"X-Cache={resp.headers.get('X-Cache')}, {len(items)} periods", time.time() - t0)
        except Exception as e:
            record("api.get_baseline_vs_media", "FAIL", str(e), time.time() - t0)

        # 9. GET /accuracy
        t0 = time.time()
        try:
            resp = client.get("/accuracy")
            assert resp.status_code == 200
            acc = schemas.PredictiveAccuracyResponse.model_validate(resp.json())
            record("api.get_accuracy", "PASS", f"Goodness-of-fit model validated", time.time() - t0)
        except Exception as e:
            record("api.get_accuracy", "FAIL", str(e), time.time() - t0)

        # 10. GET /spend-summary
        t0 = time.time()
        try:
            resp = client.get("/spend-summary")
            assert resp.status_code == 200
            spends = [schemas.ChannelSpendSummary.model_validate(x) for x in resp.json()]
            record("api.get_spend_summary", "PASS", f"{len(spends)} channels", time.time() - t0)
        except Exception as e:
            record("api.get_spend_summary", "FAIL", str(e), time.time() - t0)

        # 11. POST /optimize (Default scenario - Cache Hit)
        t0 = time.time()
        try:
            default_req = schemas.ScenarioRequest()
            resp = client.post("/optimize", json=default_req.model_dump())
            assert resp.status_code == 200
            assert resp.headers.get("X-Cache") == "HIT", f"Expected HIT, got {resp.headers.get('X-Cache')}"
            opt_model = schemas.OptimizationResponse.model_validate(resp.json())
            assert len(opt_model.channels) == 7
            record(
                "api.post_optimize_cache_hit",
                "PASS",
                f"X-Cache=HIT in {(time.time() - t0) * 1000:.2f}ms (instant cached serve)",
                time.time() - t0,
            )
        except Exception as e:
            record("api.post_optimize_cache_hit", "FAIL", str(e), time.time() - t0)

        # 12. POST /whatif simulation
        t0 = time.time()
        try:
            whatif_payload = {
                "scenario_name": "CPM Surge Test",
                "channel_adjustments": {
                    "Online_Video": {
                        "spend_multiplier": 1.2,
                        "cpm_multiplier": 1.25,
                    },
                    "Paid_Search": {
                        "spend_multiplier": 0.9,
                        "cpm_multiplier": 1.0,
                    }
                }
            }
            resp = client.post("/whatif", json=whatif_payload)
            assert resp.status_code == 200
            whatif_model = schemas.WhatIfResponse.model_validate(resp.json())
            assert len(whatif_model.channels) == 7
            assert whatif_model.scenario_name == "CPM Surge Test"
            record(
                "api.post_whatif",
                "PASS",
                f"Simulated {len(whatif_model.channels)} channels with CPM/spend multipliers",
                time.time() - t0,
            )
        except Exception as e:
            record("api.post_whatif", "FAIL", str(e), time.time() - t0)

    return results


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 4: API Layer Verification")
    print("=" * 60)

    start_utc = datetime.now(timezone.utc).isoformat()
    test_results = run_tests()

    n_pass = sum(1 for t in test_results if t["status"] == "PASS")
    n_fail = sum(1 for t in test_results if t["status"] == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    log_payload = {
        "stage": "Stage 4 — API Layer (FastAPI) Verification",
        "timestamp_utc": start_utc,
        "overall_status": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "tests": test_results,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }

    LOG_PATH.write_text(json.dumps(log_payload, indent=2))

    print("=" * 60)
    print(f"  Overall: {overall} ({n_pass} PASS / {n_fail} FAIL)")
    print(f"  Log → {LOG_PATH}")
    print("=" * 60)

    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
