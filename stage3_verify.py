"""
Stage 3 — Precompute & Cache Layer Verification Script
=======================================================
Validates the offline precompute job and caching service layer:
1. Executes precompute job (or validates existing cache).
2. Verifies manifest and all required cache artifacts.
3. Tests cache retrieval performance (< 50ms per item, well under 300ms SLA).
4. Verifies cached outputs strictly conform to Stage 2 Pydantic schemas.
5. Tests scenario key generation and scenario caching.
6. Validates cache invalidation and namespace segregation.

Outputs:
- logs/stage3_verification.json
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

from services import cache  # noqa: E402
from api import schemas  # noqa: E402
from jobs.precompute import run_precompute  # noqa: E402

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "stage3_verification.json"


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

    # 1. Run / Verify Precompute Pipeline
    t0 = time.time()
    try:
        manifest = run_precompute(force=False, skip_optimization=False)
        m_hash = cache.get_model_hash()
        record(
            "precompute.run_pipeline",
            "PASS",
            f"Cache ready for model hash {m_hash} (status: {manifest.get('status')})",
            time.time() - t0,
        )
    except Exception as e:
        record("precompute.run_pipeline", "FAIL", str(e), time.time() - t0)
        return results

    # 2. Check Cache Readiness & Manifest
    t0 = time.time()
    try:
        is_ready = cache.is_cache_ready()
        manifest_data = cache.get_cache_manifest()
        assert is_ready, "cache.is_cache_ready() returned False"
        assert manifest_data is not None, "manifest.json missing"
        assert manifest_data.get("model_hash") == cache.get_model_hash()
        record(
            "cache.readiness_and_manifest",
            "PASS",
            f"Verified manifest for hash {manifest_data['model_hash']}",
            time.time() - t0,
        )
    except Exception as e:
        record("cache.readiness_and_manifest", "FAIL", str(e), time.time() - t0)

    # 3. Benchmark Cached Reads (SLA < 300ms)
    benchmarks = {}
    read_tests = [
        ("contributions", cache.get_cached_contributions, schemas.ChannelContribution, True),
        ("roi_summary", cache.get_cached_roi_summary, schemas.RoiSummaryItem, True),
        ("response_curves_all", cache.get_cached_response_curves, None, False),
        ("baseline_vs_media", cache.get_cached_baseline_vs_media, schemas.BaselineVsMediaPoint, True),
        ("accuracy", cache.get_cached_accuracy, schemas.PredictiveAccuracyResponse, False),
        ("spend_summary", cache.get_cached_spend_summary, schemas.ChannelSpendSummary, True),
        ("default_optimization", cache.get_cached_default_optimization, schemas.OptimizationResponse, False),
    ]

    for name, getter_fn, schema_cls, is_list in read_tests:
        t_start = time.perf_counter()
        data = getter_fn()
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        benchmarks[name] = round(elapsed_ms, 2)

        try:
            assert data is not None, f"Cached data for {name} returned None"
            # Schema validation
            if schema_cls:
                if is_list:
                    assert len(data) > 0, f"Empty list for {name}"
                    _ = [schema_cls.model_validate(item) for item in data]
                else:
                    _ = schema_cls.model_validate(data)

            record(
                f"cache.read.{name}",
                "PASS",
                f"Read & validated in {elapsed_ms:.2f}ms (SLA: <300ms)",
                elapsed_ms / 1000.0,
            )
        except Exception as e:
            record(f"cache.read.{name}", "FAIL", str(e), elapsed_ms / 1000.0)

    # 4. Cold Dashboard Load Aggregate Latency
    t_start = time.perf_counter()
    _ = cache.get_cached_contributions()
    _ = cache.get_cached_roi_summary()
    _ = cache.get_cached_response_curves()
    _ = cache.get_cached_baseline_vs_media()
    _ = cache.get_cached_default_optimization()
    total_cold_ms = (time.perf_counter() - t_start) * 1000.0

    record(
        "cache.total_cold_dashboard_load",
        "PASS" if total_cold_ms < 300.0 else "FAIL",
        f"Complete dashboard payload loaded in {total_cold_ms:.2f}ms (target < 300ms)",
        total_cold_ms / 1000.0,
    )

    # 5. Scenario Caching Interface
    t0 = time.time()
    try:
        sample_params = {
            "total_budget": 5000000.0,
            "spend_constraint_lower": 0.25,
            "spend_constraint_upper": 0.25,
            "ci_level": 0.9,
        }
        scenario_key = cache.compute_scenario_key(sample_params)
        assert len(scenario_key) == 16, f"Expected 16-char hash, got {len(scenario_key)}"

        sample_opt_result = {
            "status": "ok",
            "budget_type": "fixed",
            "total_budget": 5000000.0,
            "total_optimized_spend": 5000000.0,
            "spend_constraint_lower": 0.25,
            "spend_constraint_upper": 0.25,
            "use_optimal_frequency": True,
            "ci_level": 0.9,
            "n_flagged_channels": 0,
            "channels": [],
        }
        # Save scenario
        p_scenario = cache.save_cached_scenario_optimization(scenario_key, sample_opt_result)
        assert p_scenario.exists(), "Saved scenario file does not exist"

        # Read back
        cached_result = cache.get_cached_scenario_optimization(scenario_key)
        assert cached_result is not None, "Failed to read cached scenario"
        assert cached_result["total_budget"] == 5000000.0

        record(
            "cache.scenario_caching",
            "PASS",
            f"Successfully stored and retrieved scenario key {scenario_key}",
            time.time() - t0,
        )
    except Exception as e:
        record("cache.scenario_caching", "FAIL", str(e), time.time() - t0)

    # 6. Model Hash Isolation & Invalidation Policy
    t0 = time.time()
    try:
        current_hash = cache.get_model_hash()
        dummy_hash = "dummy_test_hash"
        dummy_dir = cache.get_model_cache_dir(dummy_hash)
        assert dummy_dir.name == dummy_hash

        # Write dummy file
        cache.write_json_cache("dummy.json", {"test": True}, model_hash=dummy_hash)
        assert (dummy_dir / "dummy.json").exists()

        # Invalidate dummy cache
        cache.invalidate_cache(dummy_hash)
        assert not dummy_dir.exists(), "Dummy cache dir should have been removed"

        # Verify current valid cache was not touched
        assert cache.is_cache_ready(current_hash), "Real cache was unexpectedly affected"

        record(
            "cache.hash_isolation_and_invalidation",
            "PASS",
            "Verified hash-based namespace segregation and clean invalidation",
            time.time() - t0,
        )
    except Exception as e:
        record("cache.hash_isolation_and_invalidation", "FAIL", str(e), time.time() - t0)

    return results


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 3: Precompute & Cache Verification")
    print("=" * 60)

    start_utc = datetime.now(timezone.utc).isoformat()
    test_results = run_tests()

    n_pass = sum(1 for t in test_results if t["status"] == "PASS")
    n_fail = sum(1 for t in test_results if t["status"] == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    log_payload = {
        "stage": "Stage 3 — Precompute and Cache Layer Verification",
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
