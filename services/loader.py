"""
services/loader.py
==================
Singleton accessor for the fitted Meridian model.

The Meridian model object is large (hundreds of MB in memory once the posterior
is loaded).  It must be loaded **once** at process start, never per-request.

Usage
-----
    from services.loader import get_model
    mmm = get_model()  # fast after first call

Environment variables
---------------------
    MERIDIAN_MODEL_PATH  : absolute or workspace-relative path to .binpb
                           Default: Meridian_files/model_build/meridian_model.binpb
    BUILD_WORKSPACE_DIRECTORY : workspace root used to resolve relative paths
                           Default: "."
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def _resolve_model_path() -> Path:
    """Resolve model path from environment or relative to app location."""
    if "MERIDIAN_MODEL_PATH" in os.environ:
        return Path(os.environ["MERIDIAN_MODEL_PATH"])

    app_dir = Path(__file__).resolve().parent.parent
    candidates = [
        app_dir / "model_weights" / "meridian_model.binpb",
        app_dir / "model_weights" / "saved_mmm.binpb",
        app_dir / "model" / "meridian_model.binpb",
        app_dir / "model" / "saved_mmm.binpb",
        app_dir.parents[1] / "model_build" / "meridian_model.binpb",
        Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")) / "Meridian_files" / "model_build" / "meridian_model.binpb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0]


MODEL_PATH: Path = _resolve_model_path()

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_mmm = None


def get_model():
    """
    Return the singleton loaded Meridian model object.

    Loads from disk on first call; returns the cached object on all
    subsequent calls.  Thread-safe enough for single-worker FastAPI; for
    multi-worker setups pre-load before forking.

    Returns
    -------
    meridian.model.model.Meridian
        The fully loaded posterior-fitted model.

    Raises
    ------
    FileNotFoundError
        If the .binpb model file does not exist at MODEL_PATH.
    RuntimeError
        If the model fails to load (version mismatch, corrupt file, etc.).
    """
    global _mmm
    if _mmm is not None:
        return _mmm

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Meridian model not found at: {MODEL_PATH}\n"
            f"Set MERIDIAN_MODEL_PATH env var to the correct path."
        )

    logger.info("Loading Meridian model from %s ...", MODEL_PATH)

    try:
        from meridian.schema.serde import meridian_serde  # noqa: PLC0415
        _mmm = meridian_serde.load_meridian(str(MODEL_PATH))
        logger.info(
            "Model loaded: %d geos, %d media ch, %d RF ch",
            _mmm.n_geos, _mmm.n_media_channels, _mmm.n_rf_channels,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load Meridian model: {exc}") from exc

    return _mmm


def reset_model():
    """
    Force reload on next call to get_model().

    Only intended for testing — do not call in production.
    """
    global _mmm
    _mmm = None
    logger.warning("Meridian model singleton reset — will reload on next get_model() call.")


def model_info() -> dict:
    """
    Return a dict of basic model metadata (does NOT load the model if not
    already loaded — returns None for all fields instead).
    """
    if _mmm is None:
        return {"loaded": False, "model_path": str(MODEL_PATH)}
    return {
        "loaded": True,
        "model_path": str(MODEL_PATH),
        "n_geos": int(_mmm.n_geos),
        "n_media_channels": int(_mmm.n_media_channels),
        "n_rf_channels": int(_mmm.n_rf_channels),
    }
