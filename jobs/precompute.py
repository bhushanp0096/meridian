"""
jobs/precompute.py
==================
Offline / Batch job to precompute expensive posterior calculations and populate
the cache for the Meridian Serving Application.

Calculations Precomputed:
1. Full channel contributions (summary_metrics)
2. Channel ROI & Marginal ROI summary
3. Full response & saturation curves across spend grid (0% to 200%)
4. Baseline vs Media-driven decomposition over time
5. Goodness-of-fit / predictive accuracy metrics
6. Historical channel spend summary
7. Default fixed-budget optimization run
8. Optional: Standalone OptimizationGrid object for on-demand re-optimizations

Usage:
------
    # Standard full precompute (analysis views + default optimization)
    python jobs/precompute.py

    # Force recompute even if cache already exists
    python jobs/precompute.py --force

    # Fast run (skip optimizer run)
    python jobs/precompute.py --skip-optimization

    # Also build and save full OptimizationGrid object (heavy)
    python jobs/precompute.py --include-grid
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# Ensure CPU execution for deterministic memory footprint
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from services.loader import MODEL_PATH, get_model, model_info  # noqa: E402
from services.analysis import (  # noqa: E402
    get_channel_contributions,
    get_roi_summary,
    get_response_curves_all_channels,
    get_baseline_vs_media,
    get_predictive_accuracy,
)
from services.optimizer import (  # noqa: E402
    get_current_spend_summary,
    run_fixed_budget_optimization,
    build_optimization_grid,
)
from api import schemas  # noqa: E402
from services.cache import (  # noqa: E402
    get_model_hash,
    get_model_cache_dir,
    is_cache_ready,
    write_json_cache,
    save_cached_scenario_optimization,
    save_cached_optimization_grid,
    compute_scenario_key,
    FILE_MANIFEST,
    FILE_CONTRIBUTIONS,
    FILE_ROI_SUMMARY,
    FILE_RESPONSE_CURVES,
    FILE_BASELINE_VS_MEDIA,
    FILE_ACCURACY,
    FILE_SPEND_SUMMARY,
    FILE_DEFAULT_OPTIMIZATION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jobs.precompute")


def run_precompute(
    force: bool = False,
    include_grid: bool = False,
    skip_optimization: bool = False,
) -> dict[str, Any]:
    """
    Execute precompute pipeline and return execution summary.
    """
    start_time = time.time()
    m_hash = get_model_hash()
    cache_dir = get_model_cache_dir(m_hash)

    logger.info("Starting precompute job for model hash: %s", m_hash)
    logger.info("Target cache directory: %s", cache_dir)

    if not force and is_cache_ready(m_hash):
        logger.info("Cache is already ready and valid for hash %s. Use --force to recompute.", m_hash)
        manifest = write_json_cache(
            FILE_MANIFEST,
            {
                "model_hash": m_hash,
                "model_file": str(MODEL_PATH),
                "status": "cached",
                "verified_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"status": "skipped", "model_hash": m_hash, "cache_dir": str(cache_dir)}

    # Ensure model is loaded once
    logger.info("Verifying model access...")
    get_model()
    info = model_info()
    logger.info("Model verified: %d geos, %d media ch, %d RF ch",
                info["n_geos"], info["n_media_channels"], info["n_rf_channels"])

    durations: dict[str, float] = {}
    artifacts: dict[str, Any] = {}

    # 1. Contributions
    logger.info("[1/7] Precomputing channel contributions...")
    t0 = time.time()
    contrib = get_channel_contributions()
    p_contrib = write_json_cache(FILE_CONTRIBUTIONS, contrib, model_hash=m_hash)
    durations["contributions"] = round(time.time() - t0, 2)
    artifacts["contributions"] = {"channels": len(contrib), "bytes": p_contrib.stat().st_size}

    # 2. ROI Summary
    logger.info("[2/7] Precomputing ROI summary...")
    t0 = time.time()
    roi = get_roi_summary()
    p_roi = write_json_cache(FILE_ROI_SUMMARY, roi, model_hash=m_hash)
    durations["roi_summary"] = round(time.time() - t0, 2)
    artifacts["roi_summary"] = {"channels": len(roi), "bytes": p_roi.stat().st_size}

    # 3. Response Curves (All channels)
    logger.info("[3/7] Precomputing response curves across spend grid...")
    t0 = time.time()
    curves = get_response_curves_all_channels()
    p_curves = write_json_cache(FILE_RESPONSE_CURVES, curves, model_hash=m_hash)
    durations["response_curves"] = round(time.time() - t0, 2)
    total_points = sum(len(pts) for pts in curves.values())
    artifacts["response_curves"] = {"channels": len(curves), "points": total_points, "bytes": p_curves.stat().st_size}

    # 4. Baseline vs Media Decomposition
    logger.info("[4/7] Precomputing baseline vs media decomposition over time...")
    t0 = time.time()
    baseline = get_baseline_vs_media()
    p_base = write_json_cache(FILE_BASELINE_VS_MEDIA, baseline, model_hash=m_hash)
    durations["baseline_vs_media"] = round(time.time() - t0, 2)
    artifacts["baseline_vs_media"] = {"periods": len(baseline), "bytes": p_base.stat().st_size}

    # 5. Predictive Accuracy
    logger.info("[5/7] Precomputing predictive accuracy metrics...")
    t0 = time.time()
    accuracy = get_predictive_accuracy()
    p_acc = write_json_cache(FILE_ACCURACY, accuracy, model_hash=m_hash)
    durations["predictive_accuracy"] = round(time.time() - t0, 2)
    artifacts["predictive_accuracy"] = {"metrics": list(accuracy.keys()), "bytes": p_acc.stat().st_size}

    # 6. Current Spend Summary
    logger.info("[6/7] Extracting historical channel spend...")
    t0 = time.time()
    spend = get_current_spend_summary()
    p_spend = write_json_cache(FILE_SPEND_SUMMARY, spend, model_hash=m_hash)
    durations["spend_summary"] = round(time.time() - t0, 2)
    artifacts["spend_summary"] = {"channels": len(spend), "bytes": p_spend.stat().st_size}

    # 7. Default Optimization Run
    if not skip_optimization:
        logger.info("[7/7] Running default fixed-budget optimization...")
        t0 = time.time()
        default_opt = run_fixed_budget_optimization()
        p_opt = write_json_cache(FILE_DEFAULT_OPTIMIZATION, default_opt, model_hash=m_hash)
        
        # Also index under default scenario cache key
        default_scenario_params = schemas.ScenarioRequest().model_dump()
        scenario_key = compute_scenario_key(default_scenario_params)
        save_cached_scenario_optimization(scenario_key, default_opt)

        durations["default_optimization"] = round(time.time() - t0, 2)
        artifacts["default_optimization"] = {
            "total_budget": default_opt.get("total_budget"),
            "channels_optimized": len(default_opt.get("channels", [])),
            "bytes": p_opt.stat().st_size,
            "scenario_key": scenario_key,
        }
    else:
        logger.info("[7/7] Skipped default optimization (--skip-optimization requested).")

    # Optional: Standalone OptimizationGrid
    if include_grid:
        logger.info("[Bonus] Building standalone OptimizationGrid...")
        t0 = time.time()
        grid = build_optimization_grid()
        p_grid = save_cached_optimization_grid(grid, model_hash=m_hash)
        durations["optimization_grid"] = round(time.time() - t0, 2)
        artifacts["optimization_grid"] = {"bytes": p_grid.stat().st_size}

    total_duration = round(time.time() - start_time, 2)

    # Manifest metadata
    manifest_data = {
        "model_hash": m_hash,
        "model_file": str(MODEL_PATH),
        "model_file_size_bytes": MODEL_PATH.stat().st_size,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_duration_seconds": total_duration,
        "durations": durations,
        "artifacts": artifacts,
        "status": "complete",
    }
    write_json_cache(FILE_MANIFEST, manifest_data, model_hash=m_hash)

    logger.info("Precompute completed in %.2fs. All artifacts saved to %s", total_duration, cache_dir)
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Precompute cache for Meridian Serving App.")
    parser.add_argument("--force", action="store_true", help="Force recompute even if cache exists.")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip the default budget optimization step.")
    parser.add_argument("--include-grid", action="store_true", help="Also build and save the full OptimizationGrid object.")
    args = parser.parse_args()

    res = run_precompute(
        force=args.force,
        include_grid=args.include_grid,
        skip_optimization=args.skip_optimization,
    )
    logger.info("Precompute result summary: %s", res.get("status"))


if __name__ == "__main__":
    main()
