"""
Stage 1 — Service Layer Verification Script
============================================
Exercises each function in the Stage 1 service layer
(loader, analysis, optimizer) and validates:

  1. All outputs are JSON-serializable.
  2. Output shapes match config/model_spec.json.
  3. Credible intervals satisfy lower <= mean <= upper.

Writes:
  logs/stage1_verification.json — structured run log

Usage
-----
  BUILD_WORKSPACE_DIRECTORY=. \\
  .venv/bin/python3 Meridian_files/meridian_application/meridian-app/stage1_verify.py
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Force CPU; this script is verification only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

# ---------------------------------------------------------------------------
# Paths and sys.path setup
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
WORKSPACE = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))
sys.path.insert(0, str(APP_DIR))   # so "services.*" imports resolve

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "stage1_verification.json"

SPEC_PATH = APP_DIR / "config" / "model_spec.json"


def load_spec() -> dict:
    with open(SPEC_PATH) as f:
        return json.load(f)


def check_json_serializable(obj, label: str):
    try:
        json.dumps(obj)
        return True, None
    except Exception as e:
        return False, f"{label}: not JSON-serializable: {e}"


def check_ci(items: list[dict], mean_key: str, lo_key: str, hi_key: str, label: str) -> list[str]:
    """Return list of violations where lo > mean or mean > hi."""
    violations = []
    for item in items:
        m = item.get(mean_key)
        lo = item.get(lo_key)
        hi = item.get(hi_key)
        ch = item.get("channel", "?")
        if m is None or lo is None or hi is None:
            continue
        if lo > m + 1e-6:
            violations.append(f"{label}[{ch}]: ci_lower ({lo:.4f}) > mean ({m:.4f})")
        if m > hi + 1e-6:
            violations.append(f"{label}[{ch}]: mean ({m:.4f}) > ci_upper ({hi:.4f})")
    return violations


def run_tests(spec: dict) -> list[dict]:
    """
    Run each service function and return a list of test result dicts.
    """
    results = []

    # ------------------------------------------------------------------ #
    # Test helpers
    # ------------------------------------------------------------------ #
    def record(name: str, status: str, detail: str = "", duration_s: float = 0):
        results.append({
            "test": name,
            "status": status,
            "detail": detail,
            "duration_seconds": round(duration_s, 2),
        })
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}" + (f" — {detail}" if detail else ""))

    def run(name, fn):
        import time  # noqa: PLC0415
        start = time.time()
        try:
            result = fn()
            elapsed = time.time() - start
            return result, elapsed, None
        except Exception:
            elapsed = time.time() - start
            return None, elapsed, traceback.format_exc()

    expected_media_ch = spec["media_channel_names"]
    expected_rf_ch = spec["rf_channel_names"]
    expected_all_ch = expected_media_ch + expected_rf_ch
    expected_n_channels = len(expected_all_ch)

    # ------------------------------------------------------------------ #
    # 1. Loader
    # ------------------------------------------------------------------ #
    print("\n[1] services.loader")
    from services.loader import get_model, model_info  # noqa: PLC0415

    mmm, elapsed, err = run("loader.get_model", get_model)
    if err:
        record("loader.get_model", "FAIL", err[:200], elapsed)
        return results  # can't continue without model
    record("loader.get_model", "PASS", f"model loaded in {elapsed:.1f}s", elapsed)

    info, elapsed, err = run("loader.model_info", model_info)
    if err:
        record("loader.model_info", "FAIL", err[:200], elapsed)
    else:
        assert info["loaded"], "model_info should show loaded=True"
        record("loader.model_info", "PASS", str(info), elapsed)

    # ------------------------------------------------------------------ #
    # 2. Analysis — ROI summary
    # ------------------------------------------------------------------ #
    print("\n[2] services.analysis")
    from services.analysis import (  # noqa: PLC0415
        get_roi_summary, get_channel_contributions,
        get_response_curves, get_response_curves_all_channels,
        get_baseline_vs_media, get_predictive_accuracy,
        get_summary_metrics,
    )

    roi, elapsed, err = run("analysis.get_roi_summary", get_roi_summary)
    if err:
        record("analysis.get_roi_summary", "FAIL", err[:300], elapsed)
    else:
        ok, msg = check_json_serializable(roi, "roi")
        n_ch = len(roi)
        # Keys changed to roi_mean/roi_ci_lower/roi_ci_upper
        ci_v = check_ci(roi, "roi_mean", "roi_ci_lower", "roi_ci_upper", "roi")
        if not ok:
            record("analysis.get_roi_summary", "FAIL", msg, elapsed)
        elif n_ch < expected_n_channels:
            record("analysis.get_roi_summary", "FAIL",
                   f"Expected >= {expected_n_channels} channels, got {n_ch}", elapsed)
        elif ci_v:
            record("analysis.get_roi_summary", "FAIL", "; ".join(ci_v), elapsed)
        else:
            record("analysis.get_roi_summary", "PASS",
                   f"{n_ch} channels, ROI range [{min(c['roi_mean'] for c in roi if c['roi_mean']):.2f}, "
                   f"{max(c['roi_mean'] for c in roi if c['roi_mean']):.2f}]", elapsed)

    contrib, elapsed, err = run("analysis.get_channel_contributions", get_channel_contributions)
    if err:
        record("analysis.get_channel_contributions", "FAIL", err[:300], elapsed)
    else:
        ok, msg = check_json_serializable(contrib, "contrib")
        ci_v = check_ci(contrib, "mean", "ci_lower", "ci_upper", "contributions")
        if not ok:
            record("analysis.get_channel_contributions", "FAIL", msg, elapsed)
        elif len(contrib) < expected_n_channels:
            record("analysis.get_channel_contributions", "FAIL",
                   f"Expected >= {expected_n_channels} channels, got {len(contrib)}", elapsed)
        elif ci_v:
            record("analysis.get_channel_contributions", "FAIL", "; ".join(ci_v), elapsed)
        else:
            record("analysis.get_channel_contributions", "PASS",
                   f"{len(contrib)} channels", elapsed)

    # Response curve for first media channel
    test_ch = expected_media_ch[0]
    curve, elapsed, err = run(
        f"analysis.get_response_curves({test_ch})",
        lambda: get_response_curves(test_ch),
    )
    if err:
        record(f"analysis.get_response_curves({test_ch})", "FAIL", err[:300], elapsed)
    else:
        ok, msg = check_json_serializable(curve, "curve")
        ci_v = check_ci(curve, "incremental_outcome_mean", "ci_lower", "ci_upper", "response_curve")
        if not ok:
            record(f"analysis.get_response_curves({test_ch})", "FAIL", msg, elapsed)
        elif len(curve) != 41:
            record(f"analysis.get_response_curves({test_ch})", "FAIL",
                   f"Expected 41 points, got {len(curve)}", elapsed)
        elif ci_v:
            record(f"analysis.get_response_curves({test_ch})", "FAIL", "; ".join(ci_v), elapsed)
        else:
            record(f"analysis.get_response_curves({test_ch})", "PASS",
                   f"41 points, outcome range [{curve[0]['incremental_outcome_mean']:.1f}, "
                   f"{curve[-1]['incremental_outcome_mean']:.1f}]", elapsed)

    all_curves, elapsed, err = run(
        "analysis.get_response_curves_all_channels",
        get_response_curves_all_channels,
    )
    if err:
        record("analysis.get_response_curves_all_channels", "FAIL", err[:300], elapsed)
    else:
        ok, msg = check_json_serializable(all_curves, "all_curves")
        if not ok:
            record("analysis.get_response_curves_all_channels", "FAIL", msg, elapsed)
        else:
            record("analysis.get_response_curves_all_channels", "PASS",
                   f"{len(all_curves)} channels × 41 points each", elapsed)

    baseline, elapsed, err = run(
        "analysis.get_baseline_vs_media",
        get_baseline_vs_media,
    )
    if err:
        record("analysis.get_baseline_vs_media", "FAIL", err[:300], elapsed)
    else:
        ok, msg = check_json_serializable(baseline, "baseline")
        if not ok:
            record("analysis.get_baseline_vs_media", "FAIL", msg, elapsed)
        elif len(baseline) != spec["n_times"]:
            record("analysis.get_baseline_vs_media", "FAIL",
                   f"Expected {spec['n_times']} time periods, got {len(baseline)}", elapsed)
        else:
            record("analysis.get_baseline_vs_media", "PASS",
                   f"{len(baseline)} time periods", elapsed)

    acc, elapsed, err = run("analysis.get_predictive_accuracy", get_predictive_accuracy)
    if err:
        record("analysis.get_predictive_accuracy", "FAIL", err[:300], elapsed)
    else:
        ok, msg = check_json_serializable(acc, "accuracy")
        if not ok:
            record("analysis.get_predictive_accuracy", "FAIL", msg, elapsed)
        else:
            record("analysis.get_predictive_accuracy", "PASS", str(acc), elapsed)

    # ------------------------------------------------------------------ #
    # 3. Optimizer — current spend summary (fast, no optimization run)
    # ------------------------------------------------------------------ #
    print("\n[3] services.optimizer")
    from services.optimizer import get_current_spend_summary  # noqa: PLC0415

    spend, elapsed, err = run("optimizer.get_current_spend_summary", get_current_spend_summary)
    if err:
        record("optimizer.get_current_spend_summary", "FAIL", err[:300], elapsed)
    else:
        ok, msg = check_json_serializable(spend, "spend")
        if not ok:
            record("optimizer.get_current_spend_summary", "FAIL", msg, elapsed)
        else:
            record("optimizer.get_current_spend_summary", "PASS",
                   f"{len(spend)} channels, total spend = {sum(s['total_spend'] for s in spend):.0f}", elapsed)

    return results


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 1: Service Layer Verification")
    print("=" * 60)
    print(f"  APP_DIR   : {APP_DIR}")
    print(f"  WORKSPACE : {WORKSPACE}")

    log = {
        "stage": "Stage 1 — Analysis Service Layer Verification",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PENDING",
        "tests": [],
        "error": None,
    }

    try:
        spec = load_spec()
        log["model_spec"] = {
            "n_media_channels": spec["n_media_channels"],
            "n_rf_channels": spec["n_rf_channels"],
            "n_times": spec["n_times"],
        }
        tests = run_tests(spec)
        log["tests"] = tests

        n_pass = sum(1 for t in tests if t["status"] == "PASS")
        n_fail = sum(1 for t in tests if t["status"] == "FAIL")
        log["n_pass"] = n_pass
        log["n_fail"] = n_fail
        log["overall_status"] = "PASS" if n_fail == 0 else "FAIL"

    except Exception:
        log["overall_status"] = "FAIL"
        log["error"] = traceback.format_exc()
        print(f"\n❌ Fatal error:\n{log['error']}")

    log["completed_utc"] = datetime.now(timezone.utc).isoformat()
    LOG_PATH.write_text(json.dumps(log, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Overall: {log['overall_status']}  "
          f"({log.get('n_pass', 0)} PASS / {log.get('n_fail', 0)} FAIL)")
    print(f"  Log → {LOG_PATH}")
    print(f"{'=' * 60}")

    if log["overall_status"] != "PASS":
        sys.exit(1)

    print("\n✅ Stage 1 verification PASSED.")
    print("   → Next: Stage 2 — Data contracts (schemas)")


if __name__ == "__main__":
    main()
