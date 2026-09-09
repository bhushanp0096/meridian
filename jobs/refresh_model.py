"""
jobs/refresh_model.py
=====================
Unified CLI Orchestrator for replacing fitted Google Meridian model weights (.binpb).

Automates:
1. Model Validation & Smoke Test
2. Multi-Metric Channel & Coordinate Discovery
3. Diff & Drift Analysis against existing model specification
4. Automatic and User-Overridden Taxonomy Classification (RF, Clicks, Impressions, Organic, Spend)
5. Generation of config/model_spec.json
6. Cache Precomputation (jobs/precompute.py)
7. End-to-End Verification (stage5_verify.py)

Usage:
------
    # Inspect current or default candidate model (dry-run, no changes):
    python jobs/refresh_model.py --inspect

    # Inspect a new candidate model file:
    python jobs/refresh_model.py --inspect --model-path /path/to/new_model.binpb

    # Refresh model and rebuild cache:
    python jobs/refresh_model.py --model-path model_weights/meridian_model.binpb

    # Refresh with custom metric overrides:
    python jobs/refresh_model.py --metric-types "TikTok=impressions,Affiliate=clicks"

    # Refresh with custom category overrides:
    python jobs/refresh_model.py --category-mapping "TikTok=Social,Affiliate=Performance"
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# Ensure CPU execution for deterministic execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from extract_model_spec import extract_spec, build_channel_taxonomy, CONFIG_DIR  # noqa: E402
from utils.common import load_model_spec, compute_file_sha256  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jobs.refresh_model")

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REFRESH_LOG_PATH = LOG_DIR / "refresh_model.json"
SPEC_PATH = CONFIG_DIR / "model_spec.json"


def find_default_model_path() -> Path:
    candidates = [
        APP_DIR / "model_weights" / "meridian_model.binpb",
        APP_DIR / "model_weights" / "saved_mmm.binpb",
        APP_DIR / "model" / "meridian_model.binpb",
        APP_DIR / "model" / "saved_mmm.binpb",
        APP_DIR.parents[1] / "model_build" / "meridian_model.binpb",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


def parse_key_value_pairs(kv_str: str | None) -> dict[str, str]:
    """Parse string formatted as 'Key1=Val1,Key2=Val2' into dict."""
    if not kv_str:
        return {}
    res = {}
    for part in kv_str.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            res[k.strip()] = v.strip()
    return res


def compute_spec_diff(old_spec: dict[str, Any], new_spec: dict[str, Any]) -> dict[str, Any]:
    """Compute difference between old and new model specifications."""
    old_paid = set(old_spec.get("media_channel_names", [])) | set(old_spec.get("rf_channel_names", []))
    new_paid = set(new_spec.get("media_channel_names", [])) | set(new_spec.get("rf_channel_names", []))

    old_all = old_paid | set(old_spec.get("organic_media_channel_names", []))
    new_all = new_paid | set(new_spec.get("organic_media_channel_names", []))

    added = sorted(list(new_all - old_all))
    removed = sorted(list(old_all - new_all))
    retained = sorted(list(new_all & old_all))

    old_tax = old_spec.get("channel_taxonomy", {})
    new_tax = new_spec.get("channel_taxonomy", {})

    metric_changes = {}
    for ch in retained:
        old_m = old_tax.get(ch, {}).get("execution_metric")
        new_m = new_tax.get(ch, {}).get("execution_metric")
        if old_m and new_m and old_m != new_m:
            metric_changes[ch] = {"old": old_m, "new": new_m}

    return {
        "added_channels": added,
        "removed_channels": removed,
        "retained_channels": retained,
        "metric_changes": metric_changes,
        "date_range": {
            "old": f"{old_spec.get('date_start')} → {old_spec.get('date_end')}",
            "new": f"{new_spec.get('date_start')} → {new_spec.get('date_end')}",
        },
        "time_periods": {
            "old": old_spec.get("n_times"),
            "new": new_spec.get("n_times"),
        },
        "geos": {
            "old": old_spec.get("n_geos"),
            "new": new_spec.get("n_geos"),
        },
    }


def print_diff_report(diff: dict[str, Any], new_spec: dict[str, Any]):
    print("\n" + "=" * 70)
    print("  MODEL TAXONOMY & DRIFT REPORT")
    print("=" * 70)

    print(f"  Geos            : {diff['geos']['old']} → {diff['geos']['new']}")
    print(f"  Time periods    : {diff['time_periods']['old']} → {diff['time_periods']['new']} wks")
    print(f"  Date Range      : {diff['date_range']['old']} → {diff['date_range']['new']}")

    if diff["added_channels"]:
        print(f"\n  ➕ Added Channels ({len(diff['added_channels'])}):")
        for ch in diff["added_channels"]:
            tax = new_spec.get("channel_taxonomy", {}).get(ch, {})
            print(f"     + {ch:<18} [{tax.get('execution_metric', 'spend')}] ({tax.get('category', 'General')})")
    else:
        print("\n  ➕ Added Channels   : None")

    if diff["removed_channels"]:
        print(f"\n  ➖ Removed Channels ({len(diff['removed_channels'])}):")
        for ch in diff["removed_channels"]:
            print(f"     - {ch}")
    else:
        print("  ➖ Removed Channels : None")

    if diff["metric_changes"]:
        print(f"\n  🔄 Metric Changes ({len(diff['metric_changes'])}):")
        for ch, change in diff["metric_changes"].items():
            print(f"     ~ {ch:<18} {change['old']} → {change['new']}")

    print(f"\n  📋 Channel Taxonomy Breakdown ({len(new_spec.get('channel_taxonomy', {}))} channels):")
    for ch, tax in new_spec.get("channel_taxonomy", {}).items():
        unit_str = f", Cost: {tax.get('cost_unit')}" if tax.get('cost_unit') else ""
        print(f"     • {ch:<18} Metric: {tax.get('execution_metric'):<20} Cat: {tax.get('category')}{unit_str}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Meridian Model Refresh & Taxonomy CLI")
    parser.add_argument("--model-path", type=str, default=None, help="Path to new meridian_model.binpb")
    parser.add_argument("--inspect", action="store_true", help="Inspect and display diff without modifying files")
    parser.add_argument("--metric-types", type=str, default=None, help="Comma-separated overrides, e.g. 'TikTok=impressions,Affiliate=clicks'")
    parser.add_argument("--category-mapping", type=str, default=None, help="Comma-separated overrides, e.g. 'TikTok=Social,Linear_TV=Traditional'")
    parser.add_argument("--force", action="store_true", help="Force precompute cache rebuild")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip precomputing cache")
    parser.add_argument("--skip-verify", action="store_true", help="Skip running stage5_verify suite")

    args = parser.parse_args()

    model_path = Path(args.model_path).resolve() if args.model_path else find_default_model_path()
    logger.info("Target model path: %s", model_path)

    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)

    # 1. Validation & Smoke Test
    logger.info("Step 1: Validating model binary integrity...")
    try:
        from meridian.schema.serde import meridian_serde
        t0 = time.time()
        mmm = meridian_serde.load_meridian(str(model_path))
        logger.info("Model loaded successfully in %.2fs", time.time() - t0)
    except Exception as exc:
        logger.error("Model load validation failed: %s", exc)
        sys.exit(1)

    # 2. Extract Coordinates & Build Multi-Metric Taxonomy
    logger.info("Step 2: Extracting dimensions and discovering channels...")
    spec = extract_spec(mmm)

    # Apply CLI overrides
    cli_metrics = parse_key_value_pairs(args.metric_types)
    cli_cats = parse_key_value_pairs(args.category_mapping)

    if cli_metrics or cli_cats:
        for ch, metric in cli_metrics.items():
            if ch in spec.get("channel_taxonomy", {}):
                spec["channel_taxonomy"][ch]["execution_metric"] = metric
                if metric == "clicks":
                    spec["channel_taxonomy"][ch]["cost_unit"] = "CPC"
                elif metric == "impressions":
                    spec["channel_taxonomy"][ch]["cost_unit"] = "CPM"
                elif metric == "reach_and_frequency":
                    spec["channel_taxonomy"][ch]["cost_unit"] = "CPR"
        for ch, cat in cli_cats.items():
            if ch in spec.get("channel_taxonomy", {}):
                spec["channel_taxonomy"][ch]["category"] = cat
                if cat not in spec["categories"]:
                    spec["categories"].append(cat)

    # 3. Diff Analysis
    old_spec = {}
    if SPEC_PATH.exists():
        try:
            old_spec = json.loads(SPEC_PATH.read_text())
        except Exception:
            pass

    diff = compute_spec_diff(old_spec, spec)
    print_diff_report(diff, spec)

    if args.inspect:
        logger.info("Inspect mode: No changes made to files or cache. Exiting.")
        return

    # 4. Write config/model_spec.json
    logger.info("Step 3: Writing updated model specification to %s", SPEC_PATH)
    SPEC_PATH.write_text(json.dumps(spec, indent=2))

    # Log action
    log_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_sha256": compute_file_sha256(model_path),
        "diff": diff,
        "spec": spec,
    }
    REFRESH_LOG_PATH.write_text(json.dumps(log_data, indent=2))

    # 5. Run Precompute Job
    if not args.skip_precompute:
        logger.info("Step 4: Running cache precomputation job...")
        from jobs.precompute import run_precompute
        precomp_res = run_precompute(force=args.force)
        logger.info("Precompute status: %s (duration: %.2fs)", precomp_res.get("status"), precomp_res.get("duration_seconds", 0))

    # 6. Verification
    if not args.skip_verify:
        logger.info("Step 5: Running end-to-end verification suite...")
        verify_script = APP_DIR / "stage5_verify.py"
        res = subprocess.run([sys.executable, str(verify_script)], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            logger.error("Verification suite encountered failures!")
            print(res.stderr, file=sys.stderr)
            sys.exit(res.returncode)
        logger.info("All verification assertions passed.")

    logger.info("Model refresh completed successfully!")


if __name__ == "__main__":
    main()
