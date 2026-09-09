"""
Stage 0 — Pin Serving Requirements
====================================
Captures the exact library versions from the current virtual environment
that are relevant to serving the Meridian model.

Uses importlib.metadata (works with uv-managed venvs that have no pip binary).

Writes:
  requirements-serving.txt — pinned deps for the Docker serving image
  logs/stage0_pin_requirements.json — structured run log

Usage
-----
  .venv/bin/python3 Meridian_files/meridian_application/meridian-app/pin_requirements.py
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import packages_distributions, version, PackageNotFoundError

APP_DIR = Path(__file__).parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "stage0_pin_requirements.json"
REQ_PATH = APP_DIR / "requirements-serving.txt"

# Key packages to pin for the serving image (install name, not import name)
KEY_PACKAGES = [
    "google-meridian",
    "tensorflow",
    "tensorflow-probability",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-prng",
    "numpy",
    "scipy",
    "xarray",
    "arviz",
    "pandas",
    "protobuf",
    "ml-dtypes",
    "opt-einsum",
    "etils",
    "immutabledict",
    "fastapi",
    "uvicorn",
    "pydantic",
    "streamlit",
    "httpx",
    "structlog",
    "aiofiles",
]


def get_pinned_packages() -> list[str]:
    """
    Use importlib.metadata to resolve installed versions.
    Works with uv-managed venvs that have no pip binary installed.
    """
    pinned = []
    for pkg in KEY_PACKAGES:
        try:
            v = version(pkg)
            pinned.append(f"{pkg}=={v}")
        except PackageNotFoundError:
            pass  # not installed — skip
    return pinned


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 0: Pin Requirements")
    print("=" * 60)

    log = {
        "stage": "Stage 0 — Pin Requirements",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING",
        "requirements_path": str(REQ_PATH),
        "python_bin": sys.executable,
        "python_version": sys.version,
        "pinned_packages": [],
        "error": None,
    }

    print(f"[pin] Python : {sys.executable}")
    print(f"[pin] Version: {sys.version}")

    try:
        pinned = get_pinned_packages()

        header = (
            f"# Meridian Serving — Pinned Requirements\n"
            f"# Generated  : {datetime.now(timezone.utc).isoformat()}\n"
            f"# Python     : {sys.executable}\n"
            f"# Py version : {sys.version}\n"
            f"# Purpose    : Pin exact versions for Docker serving image\n"
            f"#\n"
            f"# WARNING: meridian_model.binpb is NOT forward-compatible across\n"
            f"#          Meridian/TF/JAX versions. The serving container MUST\n"
            f"#          match the training environment exactly.\n\n"
        )
        REQ_PATH.write_text(header + "\n".join(pinned) + "\n")

        print(f"\n[pin] ✅ requirements-serving.txt written → {REQ_PATH}")
        print(f"[pin]    Pinned {len(pinned)} packages:")
        for p in pinned:
            print(f"           {p}")

        log["status"] = "PASS"
        log["pinned_packages"] = pinned

    except Exception as exc:
        log["status"] = "FAIL"
        log["error"] = traceback.format_exc()
        print(f"❌ FAIL — {exc}", file=sys.stderr)

    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\n[pin] Log written → {LOG_PATH}")

    if log["status"] != "PASS":
        sys.exit(1)

    print("\n✅ Requirements pinning PASSED.")


if __name__ == "__main__":
    main()
