"""
services/cache.py
=================
Cache read/write service for precomputed Meridian model outputs and scenarios.

Design Rules & Invalidation Policy:
-----------------------------------
1. Cache Namespace:
   Every cache artifact is isolated under `cache/<model_hash>/`, where `model_hash`
   is the first 16 characters of the SHA-256 digest of `saved_mmm.binpb`.
   Any refit of the model automatically creates a new namespace, preventing stale
   numbers from ever being served.

2. Storage Format:
   - Analysis outputs & scenarios: Clean, human-readable JSON files.
   - OptimizationGrid: Pickled Python dataclass (`optimization_grid.pkl`).

3. Request Path:
   - Default dashboard views (contributions, ROI, response curves, baseline decomposition)
     are read directly from flat JSON files with ~0ms disk I/O.
   - Optimizer requests first check `scenarios/<scenario_hash>.json`. If a match is found,
     results return instantly.
   - Novel scenarios run either with the precomputed `OptimizationGrid` or live via optimizer.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Any

from utils.common import (
    compute_file_sha256,
    compute_dict_hash,
    get_cache_root_dir,
)
from services.loader import MODEL_PATH

logger = logging.getLogger(__name__)

# Cached model hash in-memory singleton
_MODEL_HASH: str | None = None

# Core precomputed file names
FILE_MANIFEST = "manifest.json"
FILE_CONTRIBUTIONS = "contributions.json"
FILE_ROI_SUMMARY = "roi_summary.json"
FILE_RESPONSE_CURVES = "response_curves.json"
FILE_BASELINE_VS_MEDIA = "baseline_vs_media.json"
FILE_ACCURACY = "predictive_accuracy.json"
FILE_SPEND_SUMMARY = "spend_summary.json"
FILE_DEFAULT_OPTIMIZATION = "default_optimization.json"
FILE_OPTIMIZATION_GRID = "optimization_grid.pkl"
DIR_SCENARIOS = "scenarios"


def get_model_hash(force_recompute: bool = False) -> str:
    """
    Return the 16-character SHA-256 hash of the Meridian model file.
    Cached in memory after first call.
    """
    global _MODEL_HASH
    if _MODEL_HASH is not None and not force_recompute:
        return _MODEL_HASH

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    full_sha = compute_file_sha256(MODEL_PATH)
    _MODEL_HASH = full_sha[:16]
    logger.info("Model hash for %s: %s", MODEL_PATH.name, _MODEL_HASH)
    return _MODEL_HASH


def get_model_cache_dir(model_hash: str | None = None) -> Path:
    """Return the absolute path to the directory for this model's cache namespace."""
    m_hash = model_hash or get_model_hash()
    cache_dir = get_cache_root_dir() / m_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_cache_ready(model_hash: str | None = None) -> bool:
    """
    Verify whether the required precomputed analysis artifacts are present and non-empty.
    """
    try:
        cdir = get_model_cache_dir(model_hash)
    except Exception:
        return False

    required_files = [
        FILE_MANIFEST,
        FILE_CONTRIBUTIONS,
        FILE_ROI_SUMMARY,
        FILE_RESPONSE_CURVES,
        FILE_BASELINE_VS_MEDIA,
        FILE_SPEND_SUMMARY,
    ]

    for fname in required_files:
        p = cdir / fname
        if not p.exists() or p.stat().st_size == 0:
            return False
    return True


def get_cache_manifest(model_hash: str | None = None) -> dict[str, Any] | None:
    """Read manifest.json from the cache namespace."""
    return read_json_cache(FILE_MANIFEST, model_hash=model_hash)


def write_json_cache(filename: str, data: Any, model_hash: str | None = None) -> Path:
    """Write data as indented JSON to the model's cache namespace."""
    cdir = get_model_cache_dir(model_hash)
    target = cdir / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.debug("Wrote cache artifact: %s (%d bytes)", target.name, target.stat().st_size)
    return target


def read_json_cache(filename: str, model_hash: str | None = None) -> Any | None:
    """Read a JSON cache file from the model's cache namespace. Returns None if missing."""
    try:
        cdir = get_model_cache_dir(model_hash)
        target = cdir / filename
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read cache file %s: %s", filename, exc)
        return None


# ---------------------------------------------------------------------------
# Specific Analysis Getters & Setters
# ---------------------------------------------------------------------------

def get_cached_contributions() -> list[dict[str, Any]] | None:
    return read_json_cache(FILE_CONTRIBUTIONS)


def get_cached_roi_summary() -> list[dict[str, Any]] | None:
    return read_json_cache(FILE_ROI_SUMMARY)


def get_cached_response_curves(channel: str | None = None) -> Any | None:
    data = read_json_cache(FILE_RESPONSE_CURVES)
    if data is None:
        return None
    if channel:
        return data.get(channel)
    return data


def get_cached_baseline_vs_media() -> list[dict[str, Any]] | None:
    return read_json_cache(FILE_BASELINE_VS_MEDIA)


def get_cached_accuracy() -> dict[str, Any] | None:
    return read_json_cache(FILE_ACCURACY)


def get_cached_spend_summary() -> list[dict[str, Any]] | None:
    return read_json_cache(FILE_SPEND_SUMMARY)


def get_cached_default_optimization() -> dict[str, Any] | None:
    return read_json_cache(FILE_DEFAULT_OPTIMIZATION)


# ---------------------------------------------------------------------------
# Scenario & Optimization Cache
# ---------------------------------------------------------------------------

def compute_scenario_key(scenario_params: dict[str, Any]) -> str:
    """Generate a deterministic 16-character hash key for scenario parameters."""
    return compute_dict_hash(scenario_params)


def get_cached_scenario_optimization(scenario_key: str) -> dict[str, Any] | None:
    """Check for a previously saved optimization run for this scenario."""
    return read_json_cache(f"{DIR_SCENARIOS}/{scenario_key}.json")


def save_cached_scenario_optimization(scenario_key: str, result: dict[str, Any]) -> Path:
    """Save an optimization result under its scenario key."""
    return write_json_cache(f"{DIR_SCENARIOS}/{scenario_key}.json", result)


# ---------------------------------------------------------------------------
# OptimizationGrid (Meridian dataclass) Serialization
# ---------------------------------------------------------------------------

def save_cached_optimization_grid(grid: Any, model_hash: str | None = None) -> Path:
    """Serialize the Meridian OptimizationGrid object."""
    cdir = get_model_cache_dir(model_hash)
    target = cdir / FILE_OPTIMIZATION_GRID
    with open(target, "wb") as f:
        pickle.dump(grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved optimization grid to %s (%d bytes)", target.name, target.stat().st_size)
    return target


def get_cached_optimization_grid(model_hash: str | None = None) -> Any | None:
    """Load the cached Meridian OptimizationGrid object. Returns None if missing."""
    try:
        cdir = get_model_cache_dir(model_hash)
        target = cdir / FILE_OPTIMIZATION_GRID
        if not target.exists():
            return None
        with open(target, "rb") as f:
            grid = pickle.load(f)
        logger.debug("Loaded optimization grid from cache.")
        return grid
    except Exception as exc:
        logger.warning("Failed to load cached optimization grid: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Cache Invalidation / Cleanup
# ---------------------------------------------------------------------------

def invalidate_cache(model_hash: str | None = None) -> bool:
    """
    Remove all cached files for the given model hash.
    If model_hash is None, invalidates current model's cache.
    """
    import shutil  # noqa: PLC0415
    try:
        cdir = get_model_cache_dir(model_hash)
        if cdir.exists():
            shutil.rmtree(cdir)
            logger.info("Invalidated cache at %s", cdir)
        return True
    except Exception as exc:
        logger.error("Failed to invalidate cache: %s", exc)
        return False
