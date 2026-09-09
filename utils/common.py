"""
utils/common.py
===============
Reusable utility functions for the Meridian Serving Application.
"""

import json
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = APP_DIR / "config"
DEFAULT_SPEC_PATH = CONFIG_DIR / "model_spec.json"


def get_app_dir() -> Path:
    """Return the absolute path to the meridian-app directory."""
    return APP_DIR


def get_model_spec_path() -> Path:
    """Return the path to model_spec.json."""
    return DEFAULT_SPEC_PATH


def load_model_spec(spec_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load model specification JSON from config.

    Parameters
    ----------
    spec_path : Path | str | None
        Optional custom path. Defaults to meridian-app/config/model_spec.json.

    Returns
    -------
    dict[str, Any]
        Parsed JSON dictionary.
    """
    target = Path(spec_path) if spec_path else DEFAULT_SPEC_PATH
    if not target.exists():
        raise FileNotFoundError(f"Model spec file not found at: {target}")

    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def is_json_serializable(obj: Any) -> tuple[bool, str | None]:
    """
    Check if a Python object is strictly JSON serializable.

    Returns (True, None) if serializable, or (False, error_message).
    """
    try:
        json.dumps(obj)
        return True, None
    except (TypeError, OverflowError, ValueError) as exc:
        return False, str(exc)


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """
    Compute hex SHA-256 digest of a file.

    Parameters
    ----------
    file_path : Path | str
        Path to the target file.
    chunk_size : int
        Chunk size in bytes for reading.

    Returns
    -------
    str
        Full 64-character lowercase hexadecimal digest.
    """
    import hashlib  # noqa: PLC0415
    target = Path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {target}")

    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_hash(data: dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 hash of a dictionary by sorting keys.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to hash.

    Returns
    -------
    str
        First 16 characters of the hexadecimal digest.
    """
    import hashlib  # noqa: PLC0415
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def get_cache_root_dir() -> Path:
    """
    Return the root cache directory.
    Configurable via MERIDIAN_CACHE_DIR environment variable.
    """
    import os  # noqa: PLC0415
    env_dir = os.environ.get("MERIDIAN_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return APP_DIR / "cache"

