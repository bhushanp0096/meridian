"""
Stage 0 — Model Spec Extractor
================================
Loads the fitted Meridian model and extracts all metadata needed to drive
the serving application (channel names, geo names, date range, KPI info).

Writes:
  config/model_spec.json  — the single source of truth for frontend labels
  logs/stage0_spec_extract.json — structured run log

Usage
-----
  .venv/bin/python3 Meridian_files/meridian_application/meridian-app/extract_model_spec.py
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Force CPU — spec extraction does not need GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))
candidates = [
    APP_DIR / "model_weights" / "meridian_model.binpb",
    APP_DIR / "model_weights" / "saved_mmm.binpb",
    APP_DIR / "model" / "meridian_model.binpb",
    APP_DIR / "model" / "saved_mmm.binpb",
    WORKSPACE / "Meridian_files" / "model_build" / "meridian_model.binpb",
]
MODEL_PATH = next((p for p in candidates if p.exists()), candidates[0])
CONFIG_DIR = APP_DIR / "config"
LOG_DIR = APP_DIR / "logs"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SPEC_PATH = CONFIG_DIR / "model_spec.json"
LOG_PATH = LOG_DIR / "stage0_spec_extract.json"


def extract_spec(mmm) -> dict:
    """
    Extract metadata from a loaded Meridian model object.
    Returns a plain-Python dict (fully JSON-serialisable).
    """
    spec = {}

    # --- Geo ---
    try:
        geo_names = list(mmm.input_data.geo.values.tolist())
    except Exception:
        geo_names = [f"geo_{i}" for i in range(mmm.n_geos)]
    spec["n_geos"] = int(mmm.n_geos)
    spec["geo_names"] = geo_names

    # --- Time ---
    try:
        times = mmm.input_data.time.values.tolist()
        spec["n_times"] = len(times)
        # Convert numpy datetime64 / string to ISO string
        spec["date_start"] = str(times[0])
        spec["date_end"] = str(times[-1])
    except Exception:
        spec["n_times"] = None
        spec["date_start"] = None
        spec["date_end"] = None

    spec["time_granularity"] = "weekly"  # Known from mmm_model.py

    # --- Media channels ---
    try:
        media_channel_names = list(mmm.input_data.media_channel.values.tolist())
    except Exception:
        media_channel_names = [f"media_{i}" for i in range(mmm.n_media_channels)]
    spec["n_media_channels"] = int(mmm.n_media_channels)
    spec["media_channel_names"] = media_channel_names

    # --- RF channels ---
    try:
        rf_channel_names = list(mmm.input_data.rf_channel.values.tolist())
    except Exception:
        rf_channel_names = [f"rf_{i}" for i in range(mmm.n_rf_channels)]
    spec["n_rf_channels"] = int(mmm.n_rf_channels)
    spec["rf_channel_names"] = rf_channel_names

    # --- Organic media channels ---
    try:
        organic_channel_names = list(mmm.input_data.organic_media_channel.values.tolist())
        spec["n_organic_media_channels"] = len(organic_channel_names)
        spec["organic_media_channel_names"] = organic_channel_names
    except Exception:
        spec["n_organic_media_channels"] = 0
        spec["organic_media_channel_names"] = []

    # --- Controls ---
    try:
        control_names = list(mmm.input_data.control_variable.values.tolist())
        spec["n_controls"] = len(control_names)
        spec["control_names"] = control_names
    except Exception:
        spec["n_controls"] = 0
        spec["control_names"] = []

    # --- KPI (from known model config; mmm_model.py is source of truth) ---
    spec["kpi_name"] = "Salon_Bookings"
    spec["kpi_type"] = "non_revenue"
    spec["kpi_units"] = "bookings"
    spec["revenue_per_kpi_col"] = "Avg_Revenue_Per_Booking"

    # --- Posterior draws info ---
    try:
        # Use .sizes (not .dims) to avoid xarray FutureWarning in xarray ≥ 2024
        n_chains = int(mmm.inference_data.posterior.sizes.get("chain", 0))
        n_draws = int(mmm.inference_data.posterior.sizes.get("draw", 0))
        spec["mcmc_n_chains"] = n_chains
        spec["mcmc_n_draws"] = n_draws
    except Exception:
        spec["mcmc_n_chains"] = None
        spec["mcmc_n_draws"] = None

    # --- ModelSpec metadata ---
    spec["max_lag"] = 8
    spec["hill_before_adstock"] = False
    spec["media_effects_dist"] = "log_normal"
    spec["enable_aks"] = True

    # --- Channel Taxonomy & Multi-Metric Classification ---
    channel_cfg_path = CONFIG_DIR / "channel_config.json"
    cfg_overrides = {}
    if channel_cfg_path.exists():
        try:
            cfg_overrides = json.loads(channel_cfg_path.read_text())
        except Exception:
            pass

    channel_taxonomy, categories = build_channel_taxonomy(
        media_channels=spec["media_channel_names"],
        rf_channels=spec["rf_channel_names"],
        organic_channels=spec["organic_media_channel_names"],
        config_overrides=cfg_overrides,
    )
    spec["channel_taxonomy"] = channel_taxonomy
    spec["categories"] = categories

    return spec


DEFAULT_PALETTE = [
    "#4F46E5",  # Indigo
    "#06B6D4",  # Cyan
    "#EC4899",  # Pink
    "#F59E0B",  # Amber
    "#10B981",  # Emerald
    "#8B5CF6",  # Purple
    "#3B82F6",  # Sky Blue
    "#64748B",  # Slate
    "#E11D48",  # Rose
    "#D97706",  # Ochre
    "#059669",  # Mint
    "#7C3AED",  # Violet
]

KNOWN_COLORS = {
    "Online_Video": "#4F46E5",
    "Display": "#06B6D4",
    "Paid_Social": "#EC4899",
    "Paid_Search": "#F59E0B",
    "Affiliate": "#10B981",
    "CTV": "#8B5CF6",
    "Linear_TV": "#3B82F6",
    "Email_Opens": "#64748B",
}


def build_channel_taxonomy(
    media_channels: list[str],
    rf_channels: list[str],
    organic_channels: list[str],
    config_overrides: dict | None = None,
) -> tuple[dict, list[str]]:
    """
    Build multi-metric taxonomy for all media channels.
    Classifies channels into execution metrics:
      - reach_and_frequency: CPR
      - clicks: CPC
      - impressions: CPM
      - organic: None
      - spend: Spend
    """
    overrides = config_overrides or {}
    taxonomy = {}
    categories = []

    def _add_cat(cat):
        if cat and cat not in categories:
            categories.append(cat)

    color_idx = 0

    # 1. Standard media channels
    for ch in media_channels:
        ch_ovr = overrides.get(ch, {})
        color = ch_ovr.get("color") or KNOWN_COLORS.get(ch) or DEFAULT_PALETTE[color_idx % len(DEFAULT_PALETTE)]
        color_idx += 1

        ch_lower = ch.lower()
        if any(k in ch_lower for k in ["_click", "click", "search", "affiliate", "cpc", "sem"]):
            def_metric = "clicks"
            def_unit = "CPC"
            def_cat = "Performance & Intent"
        elif any(k in ch_lower for k in ["_imp", "imp", "video", "display", "social", "cpm", "yt", "fb", "ig", "meta", "tiktok"]):
            def_metric = "impressions"
            def_unit = "CPM"
            def_cat = "Digital Video & Display"
        else:
            def_metric = "spend"
            def_unit = "Spend"
            def_cat = "Other Media"

        category = ch_ovr.get("category", def_cat)
        _add_cat(category)

        taxonomy[ch] = {
            "execution_metric": ch_ovr.get("execution_metric", def_metric),
            "cost_unit": ch_ovr.get("cost_unit", def_unit),
            "category": category,
            "color": color,
        }

    # 2. RF channels
    for ch in rf_channels:
        ch_ovr = overrides.get(ch, {})
        color = ch_ovr.get("color") or KNOWN_COLORS.get(ch) or DEFAULT_PALETTE[color_idx % len(DEFAULT_PALETTE)]
        color_idx += 1
        category = ch_ovr.get("category", "Television & Streaming")
        _add_cat(category)

        taxonomy[ch] = {
            "execution_metric": ch_ovr.get("execution_metric", "reach_and_frequency"),
            "cost_unit": ch_ovr.get("cost_unit", "CPR"),
            "category": category,
            "color": color,
            "has_optimal_frequency": True,
        }

    # 3. Organic channels
    for ch in organic_channels:
        ch_ovr = overrides.get(ch, {})
        color = ch_ovr.get("color") or KNOWN_COLORS.get(ch) or DEFAULT_PALETTE[color_idx % len(DEFAULT_PALETTE)]
        color_idx += 1
        category = ch_ovr.get("category", "Retention & CRM")
        _add_cat(category)

        taxonomy[ch] = {
            "execution_metric": ch_ovr.get("execution_metric", "organic"),
            "cost_unit": ch_ovr.get("cost_unit", None),
            "category": category,
            "color": color,
        }

    return taxonomy, categories


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 0: Spec Extraction")
    print("=" * 60)

    log = {
        "stage": "Stage 0 — Spec Extraction",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "spec_path": str(SPEC_PATH),
        "status": "PENDING",
        "spec": {},
        "error": None,
    }

    if not MODEL_PATH.exists():
        log["status"] = "FAIL"
        log["error"] = f"Model not found: {MODEL_PATH}"
        LOG_PATH.write_text(json.dumps(log, indent=2))
        print(f"❌ {log['error']}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"[spec] Loading model from: {MODEL_PATH} ...")
        from meridian.schema.serde import meridian_serde  # noqa: PLC0415

        mmm = meridian_serde.load_meridian(str(MODEL_PATH))
        print("[spec] Model loaded. Extracting metadata ...")

        spec = extract_spec(mmm)

        SPEC_PATH.write_text(json.dumps(spec, indent=2))
        print(f"[spec] ✅ model_spec.json written → {SPEC_PATH}")

        # Pretty-print key fields
        print(f"\n  KPI              : {spec['kpi_name']} ({spec['kpi_type']})")
        print(f"  Geos ({spec['n_geos']})      : {spec['geo_names']}")
        print(f"  Media channels   : {spec['media_channel_names']}")
        print(f"  RF channels      : {spec['rf_channel_names']}")
        print(f"  Organic channels : {spec['organic_media_channel_names']}")
        print(f"  Controls         : {spec['control_names']}")
        print(f"  Date range       : {spec['date_start']} → {spec['date_end']}")
        print(f"  Time periods     : {spec['n_times']} ({spec['time_granularity']})")
        print(f"  MCMC             : {spec['mcmc_n_chains']} chains × {spec['mcmc_n_draws']} draws")

        log["status"] = "PASS"
        log["spec"] = spec

    except Exception as exc:
        log["status"] = "FAIL"
        log["error"] = traceback.format_exc()
        print(f"❌ FAIL — {exc}", file=sys.stderr)

    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\n[spec] Log written → {LOG_PATH}")

    if log["status"] != "PASS":
        sys.exit(1)

    print("\n✅ Spec extraction PASSED.")


if __name__ == "__main__":
    main()
