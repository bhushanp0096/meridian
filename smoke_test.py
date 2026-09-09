"""
Stage 0 — Smoke Test
====================
Validates that the fitted Meridian model artifact is loadable in the
current virtual environment without version-mismatch errors.

Outputs
-------
- Prints model dimensions to stdout.
- Writes a structured JSON result to logs/stage0_smoke_test.json.

Usage
-----
  .venv/bin/python3 Meridian_files/meridian_application/meridian-app/smoke_test.py
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Force CPU — smoke test does not need GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

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
LOG_DIR = APP_DIR / "logs"
LOG_PATH = LOG_DIR / "stage0_smoke_test.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_smoke_test() -> dict:
    """Load the model and return a structured result dict."""
    result = {
        "stage": "Stage 0 — Smoke Test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_size_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None,
        "status": "PENDING",
        "dimensions": {},
        "error": None,
    }

    if not MODEL_PATH.exists():
        result["status"] = "FAIL"
        result["error"] = f"Model file not found: {MODEL_PATH}"
        return result

    try:
        print(f"[smoke_test] Loading model from: {MODEL_PATH}")
        from meridian.schema.serde import meridian_serde  # noqa: PLC0415

        mmm = meridian_serde.load_meridian(str(MODEL_PATH))

        dims = {
            "n_geos": int(mmm.n_geos),
            "n_media_channels": int(mmm.n_media_channels),
            "n_rf_channels": int(mmm.n_rf_channels),
        }

        # Attempt to pull more dimensions if available
        try:
            dims["n_times"] = int(mmm.n_times)
        except Exception:
            pass
        try:
            dims["n_organic_media_channels"] = int(mmm.n_organic_media_channels)
        except Exception:
            pass
        try:
            dims["n_controls"] = int(mmm.n_controls)
        except Exception:
            pass

        result["dimensions"] = dims
        result["status"] = "PASS"
        print(f"[smoke_test] ✅ Model loaded successfully.")
        for k, v in dims.items():
            print(f"             {k} = {v}")

    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        print(f"[smoke_test] ❌ FAIL — {exc}", file=sys.stderr)

    return result


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 0: Smoke Test")
    print("=" * 60)

    result = run_smoke_test()

    LOG_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n[smoke_test] Log written → {LOG_PATH}")

    if result["status"] != "PASS":
        sys.exit(1)

    print("\n✅ Smoke test PASSED — model is loadable.")


if __name__ == "__main__":
    main()
