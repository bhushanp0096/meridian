"""
frontend/api_client.py
======================
HTTP client for communicating with the Meridian FastAPI serving backend.

Supports:
- Remote HTTP requests via httpx.Client (with configurable base_url)
- In-process fallback using FastAPI TestClient if external server is offline
- Streamlit caching via st.cache_data for instant interactive updates
- Conversion of raw API dicts to pandas DataFrames
"""

import os
import logging
from typing import Any, Optional
import httpx
import pandas as pd

logger = logging.getLogger("frontend.api_client")

# Configurable backend URL (defaults to standard FastAPI port 8000)
DEFAULT_API_URL = os.environ.get("MERIDIAN_API_URL", "http://127.0.0.1:8000")


class MeridianApiClient:
    """Client for the Meridian MMM FastAPI service."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._test_client = None

    def _get_fallback_client(self):
        """Lazy-initialize an in-process FastAPI TestClient fallback."""
        if self._test_client is None:
            try:
                from fastapi.testclient import TestClient
                from api.main import app
                self._test_client = TestClient(app)
                logger.info("Initialized in-process TestClient fallback for API calls.")
            except Exception as exc:
                logger.warning(f"Could not initialize TestClient fallback: {exc}")
        return self._test_client

    def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any] | list[Any]:
        """
        Execute an HTTP request against the API server.
        Falls back to in-process TestClient if connection is refused.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as net_err:
            logger.debug(f"Direct connection to {url} failed ({net_err}). Attempting in-process fallback...")
            fallback = self._get_fallback_client()
            if fallback is not None:
                resp = fallback.request(method, endpoint, **kwargs)
                if resp.status_code >= 400:
                    raise RuntimeError(f"API {endpoint} returned {resp.status_code}: {resp.text}")
                return resp.json()
            raise ConnectionError(
                f"Cannot connect to Meridian API at {self.base_url} and in-process fallback is unavailable. "
                f"Please ensure FastAPI is running via 'uvicorn api.main:app --port 8000'."
            ) from net_err

    # -----------------------------------------------------------------------
    # System & Metadata
    # -----------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        """GET /health"""
        return self._request("GET", "/health")

    def get_model_spec(self) -> dict[str, Any]:
        """GET /model/spec"""
        return self._request("GET", "/model/spec")

    def get_model_info(self) -> dict[str, Any]:
        """GET /model/info"""
        return self._request("GET", "/model/info")

    # -----------------------------------------------------------------------
    # Analysis Endpoints
    # -----------------------------------------------------------------------

    def get_contributions(self) -> list[dict[str, Any]]:
        """GET /contributions"""
        return self._request("GET", "/contributions")

    def get_contributions_df(self) -> pd.DataFrame:
        """Return channel contributions as a pandas DataFrame."""
        data = self.get_contributions()
        return pd.DataFrame(data)

    def get_roi_summary(self) -> list[dict[str, Any]]:
        """GET /roi-summary"""
        return self._request("GET", "/roi-summary")

    def get_roi_summary_df(self) -> pd.DataFrame:
        """Return ROI summary table as a pandas DataFrame."""
        data = self.get_roi_summary()
        return pd.DataFrame(data)

    def get_response_curves(self, channel: Optional[str] = None) -> dict[str, list[dict]] | list[dict]:
        """GET /response-curves or GET /response-curves/{channel}"""
        if channel:
            return self._request("GET", f"/response-curves/{channel}")
        return self._request("GET", "/response-curves")

    def get_response_curves_df(self, channel: Optional[str] = None) -> pd.DataFrame:
        """Return response curves as a flattened pandas DataFrame."""
        data = self.get_response_curves(channel=channel)
        if channel:
            df = pd.DataFrame(data)
            return df
        # Flatten dictionary of channel -> list of points
        rows = []
        for ch, pts in data.items():
            for p in pts:
                rows.append(p)
        return pd.DataFrame(rows)

    def get_baseline_vs_media(self) -> list[dict[str, Any]]:
        """GET /baseline-vs-media"""
        return self._request("GET", "/baseline-vs-media")

    def get_baseline_vs_media_df(self) -> pd.DataFrame:
        """Return baseline vs media time series as a pandas DataFrame."""
        data = self.get_baseline_vs_media()
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_accuracy(self) -> dict[str, Any]:
        """GET /accuracy"""
        return self._request("GET", "/accuracy")

    def get_spend_summary(self) -> list[dict[str, Any]]:
        """GET /spend-summary"""
        return self._request("GET", "/spend-summary")

    def get_spend_summary_df(self) -> pd.DataFrame:
        """Return historical spend summary as a DataFrame."""
        data = self.get_spend_summary()
        return pd.DataFrame(data)

    # -----------------------------------------------------------------------
    # Optimizer & Simulation
    # -----------------------------------------------------------------------

    def post_optimize(self, scenario_request: dict[str, Any]) -> dict[str, Any]:
        """POST /optimize"""
        return self._request("POST", "/optimize", json=scenario_request)

    def post_whatif(self, whatif_request: dict[str, Any]) -> dict[str, Any]:
        """POST /whatif"""
        return self._request("POST", "/whatif", json=whatif_request)


# Singleton instance for easy import
api_client = MeridianApiClient()
