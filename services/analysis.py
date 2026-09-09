"""
services/analysis.py
=====================
Thin wrappers around Meridian's Analyzer that expose plain-Python,
JSON-serializable outputs.

Design rules
------------
- This is the ONLY module that imports Meridian analysis classes.
- Every public function returns only dicts / lists of Python primitives
  (str, int, float, bool, None).  No numpy arrays, no xarray, no JAX tensors.
- Conversion happens at the edge of each function with `.numpy().tolist()`
  / `float()` / `str()`.
- The `Analyzer` is instantiated once per loaded model and cached as a module-
  level singleton alongside the model singleton in loader.py.

Available Meridian Analyzer methods used here:
  summary_metrics()        → xr.Dataset  (roi, mroi, cpik, incremental_outcome …)
  baseline_summary_metrics()→ xr.Dataset  (baseline breakdown)
  response_curves()        → xr.Dataset  (spend_multiplier × channel)
  expected_vs_actual_data()→ xr.Dataset  (modeled vs actuals over time)
  predictive_accuracy()    → xr.Dataset  (R², MAPE, wMAPE)
  marginal_roi()           → jax.Array   (per-channel mROI distribution)
  roi()                    → jax.Array   (per-channel ROI distribution)
  incremental_outcome()    → jax.Array   (per-channel incremental KPI)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level Analyzer singleton (created lazily alongside the model)
# ---------------------------------------------------------------------------
_analyzer = None


def _get_analyzer():
    """Return the cached Analyzer, creating it if needed."""
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    from services.loader import get_model           # noqa: PLC0415
    from meridian.analysis.analyzer import Analyzer  # noqa: PLC0415

    mmm = get_model()
    _analyzer = Analyzer(mmm)
    logger.info("Analyzer initialized.")
    return _analyzer


def reset_analyzer():
    """Force re-creation on next call.  Test use only."""
    global _analyzer
    _analyzer = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float(x) -> float:
    """Safely convert any scalar (numpy, jax, python) to Python float."""
    try:
        return float(x)
    except Exception:
        return None


def _jax_to_list(arr) -> list:
    """Convert a JAX/numpy array to a nested Python list."""
    try:
        import numpy as np  # noqa: PLC0415
        return np.array(arr).tolist()
    except Exception:
        return []


def _ci(arr, ci_level: float = 0.9):
    """
    Return (mean, lower, upper) credible interval from a 1-D or 2-D
    posterior draw array.  Assumes the last axis is the draw axis.
    For a (n_channels, n_draws) array returns per-channel stats.
    """
    import numpy as np  # noqa: PLC0415
    a = np.array(arr)
    alpha = (1.0 - ci_level) / 2.0
    mean = a.mean(axis=-1)
    lower = np.quantile(a, alpha, axis=-1)
    upper = np.quantile(a, 1.0 - alpha, axis=-1)
    return mean, lower, upper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_roi_summary(ci_level: float = 0.9, use_kpi: bool = False) -> list[dict]:
    """
    Return posterior ROI summary per media channel using summary_metrics().

    summary_metrics() already computes mean/ci_lo/ci_hi per channel.
    Channel coord includes all 7 media+RF channels plus 'All Channels'.

    Each dict has keys:
      channel, roi_mean, roi_ci_lower, roi_ci_upper,
      mroi_mean, mroi_ci_lower, mroi_ci_upper, ci_level
    """
    an = _get_analyzer()

    sm = an.summary_metrics(
        aggregate_geos=True,
        aggregate_times=True,
        use_kpi=use_kpi,
        confidence_level=ci_level,
    )

    # summary_metrics coords: channel × metric × distribution
    # metric = ['mean', 'median', 'ci_lo', 'ci_hi']
    # distribution = ['prior', 'posterior']
    channels = [str(c) for c in sm.coords["channel"].values if str(c) != "All Channels"]

    result = []
    for ch in channels:
        def _get(var, metric, dist="posterior"):
            try:
                val = sm[var].sel(channel=ch, metric=metric, distribution=dist)
                return float(val.values)
            except Exception:
                return None

        entry = {
            "channel": ch,
            "roi_mean": _get("roi", "mean"),
            "roi_ci_lower": _get("roi", "ci_lo"),
            "roi_ci_upper": _get("roi", "ci_hi"),
            "mroi_mean": _get("mroi", "mean"),
            "mroi_ci_lower": _get("mroi", "ci_lo"),
            "mroi_ci_upper": _get("mroi", "ci_hi"),
            "incremental_outcome_mean": _get("incremental_outcome", "mean"),
            "incremental_outcome_ci_lower": _get("incremental_outcome", "ci_lo"),
            "incremental_outcome_ci_upper": _get("incremental_outcome", "ci_hi"),
            "pct_of_contribution": _get("pct_of_contribution", "mean"),
            "spend_mean": _get("spend", "mean"),
            "cpik_mean": _get("cpik", "mean"),
            "ci_level": ci_level,
        }
        result.append(entry)

    logger.debug("get_roi_summary: %d channels", len(result))
    return result


def get_channel_contributions(
    ci_level: float = 0.9,
    use_kpi: bool = False,
    include_non_paid: bool = True,
) -> list[dict]:
    """
    Return posterior incremental outcome (contribution) per channel
    using summary_metrics().

    Each dict has keys:
      channel, mean, ci_lower, ci_upper, ci_level, pct_of_total_mean

    Parameters
    ----------
    ci_level : float
        Credible interval width (default 0.9).
    use_kpi : bool
        If False (default), uses revenue scale.
    include_non_paid : bool
        Include organic / non-paid channels (default True).
        Note: Meridian summary_metrics excludes organics by default;
        set include_non_paid_channels=True to include.
    """
    an = _get_analyzer()

    sm = an.summary_metrics(
        aggregate_geos=True,
        aggregate_times=True,
        use_kpi=use_kpi,
        confidence_level=ci_level,
        include_non_paid_channels=include_non_paid,
    )

    channels = [str(c) for c in sm.coords["channel"].values if str(c) != "All Channels"]

    result = []
    for ch in channels:
        def _get(var, metric, dist="posterior", _ch=ch):
            try:
                val = sm[var].sel(channel=_ch, metric=metric, distribution=dist)
                return float(val.values)
            except Exception:
                return None

        mean_val = _get("incremental_outcome", "mean")
        entry = {
            "channel": ch,
            "mean": mean_val,
            "ci_lower": _get("incremental_outcome", "ci_lo"),
            "ci_upper": _get("incremental_outcome", "ci_hi"),
            "pct_of_total_mean": _get("pct_of_contribution", "mean"),
            "ci_level": ci_level,
        }
        result.append(entry)

    logger.debug("get_channel_contributions: %d channels", len(result))
    return result




def get_response_curves(
    channel: str,
    spend_multipliers: list[float] | None = None,
    ci_level: float = 0.9,
    use_kpi: bool = False,
) -> list[dict]:
    """
    Return saturation/response curve data for a single channel.

    Each dict has keys:
      channel, spend_multiplier, spend_pct_of_current,
      incremental_outcome_mean, ci_lower, ci_upper, ci_level

    Parameters
    ----------
    channel : str
        Channel name (must be in media_channel_names or rf_channel_names).
    spend_multipliers : list[float] | None
        Spend multipliers relative to historical (e.g., 0.0 to 2.0).
        Default: 41 points from 0.0 to 2.0 in 0.05 steps.
    ci_level : float
        Credible interval width.
    use_kpi : bool
        If False (default), uses revenue scale.
    """
    import numpy as np  # noqa: PLC0415
    if spend_multipliers is None:
        spend_multipliers = [round(i * 0.05, 2) for i in range(41)]  # 0.0 to 2.0

    an = _get_analyzer()

    # response_curves() → xr.Dataset with dims (channel, spend_multiplier, metric)
    ds = an.response_curves(
        spend_multipliers=spend_multipliers,
        use_posterior=True,
        use_kpi=use_kpi,
        confidence_level=ci_level,
    )

    alpha = (1.0 - ci_level) / 2.0

    # Dataset variables: typically 'incremental_outcome' with coords
    # channel × spend_multiplier × sample (or already summarized)
    # We need to locate the right channel in the dataset coords.
    try:
        ch_da = ds["incremental_outcome"].sel(channel=channel)
    except Exception:
        # Try matching by position if channel coord name differs
        all_channels = list(ds.coords.get("channel", ds.coords.get("media_and_rf_channel", [])).values)
        if channel in all_channels:
            idx = all_channels.index(channel)
            ch_da = ds["incremental_outcome"].isel(channel=idx)
        else:
            raise ValueError(
                f"Channel '{channel}' not found in response_curves output. "
                f"Available: {all_channels}"
            )

    result = []
    ch_arr = np.array(ch_da)  # shape: (n_spend_multipliers, n_draws) or (n_spend_multipliers,)

    for j, mult in enumerate(spend_multipliers):
        if ch_arr.ndim == 2:
            draws = ch_arr[j]
            mean_val = float(draws.mean())
            lo = float(np.quantile(draws, alpha))
            hi = float(np.quantile(draws, 1.0 - alpha))
        else:
            # Already summarized — just use the value
            mean_val = float(ch_arr[j])
            lo, hi = mean_val, mean_val

        result.append({
            "channel": channel,
            "spend_multiplier": mult,
            "spend_pct_of_current": mult * 100.0,
            "incremental_outcome_mean": mean_val,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_level": ci_level,
        })

    logger.debug("get_response_curves(%s): %d points", channel, len(result))
    return result


def get_response_curves_all_channels(
    spend_multipliers: list[float] | None = None,
    ci_level: float = 0.9,
    use_kpi: bool = False,
) -> dict[str, list[dict]]:
    """
    Return response curves for ALL media + RF channels.

    Returns
    -------
    dict mapping channel_name → list of curve point dicts.
    """
    import numpy as np  # noqa: PLC0415
    if spend_multipliers is None:
        spend_multipliers = [round(i * 0.05, 2) for i in range(41)]

    an = _get_analyzer()
    mmm = an.model_context  # noqa: F841 — kept for potential future use
    alpha = (1.0 - ci_level) / 2.0

    ds = an.response_curves(
        spend_multipliers=spend_multipliers,
        use_posterior=True,
        use_kpi=use_kpi,
        confidence_level=ci_level,
    )

    # Detect channel coordinate name
    ch_coord = None
    for coord_name in ["channel", "media_and_rf_channel"]:
        if coord_name in ds.coords:
            ch_coord = coord_name
            break

    if ch_coord is None:
        raise RuntimeError("Cannot locate channel coordinate in response_curves dataset.")

    all_channel_names = list(ds.coords[ch_coord].values)
    result = {}

    for ch_name in all_channel_names:
        ch_da = ds["incremental_outcome"].sel({ch_coord: ch_name})
        ch_arr = np.array(ch_da)

        points = []
        for j, mult in enumerate(spend_multipliers):
            if ch_arr.ndim == 2:
                draws = ch_arr[j]
                mean_val = float(draws.mean())
                lo = float(np.quantile(draws, alpha))
                hi = float(np.quantile(draws, 1.0 - alpha))
            else:
                mean_val = float(ch_arr[j])
                lo, hi = mean_val, mean_val

            points.append({
                "channel": str(ch_name),
                "spend_multiplier": mult,
                "spend_pct_of_current": mult * 100.0,
                "incremental_outcome_mean": mean_val,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_level": ci_level,
            })
        result[str(ch_name)] = points

    logger.debug("get_response_curves_all_channels: %d channels", len(result))
    return result


def get_baseline_vs_media(
    ci_level: float = 0.9,
    use_kpi: bool = False,
) -> list[dict]:
    """
    Return the baseline vs. media-driven decomposition over time.

    Each dict has keys:
      date (ISO str), baseline_mean, media_mean, total_mean,
      baseline_ci_lower, baseline_ci_upper, ci_level

    Uses expected_vs_actual_data() aggregated across geos.
    """
    import numpy as np  # noqa: PLC0415
    an = _get_analyzer()

    ds = an.expected_vs_actual_data(
        aggregate_geos=True,
        aggregate_times=False,
        use_kpi=use_kpi,
        confidence_level=ci_level,
    )

    # Dataset variables: 'expected', 'baseline', 'actual' (if available)
    # Coords: 'time' (or 'date')
    time_coord = None
    for c in ["time", "date"]:
        if c in ds.coords:
            time_coord = c
            break

    times = [str(t) for t in ds.coords[time_coord].values] if time_coord else []
    alpha = (1.0 - ci_level) / 2.0

    result = []
    for j, date_str in enumerate(times):
        row: dict[str, Any] = {"date": date_str, "ci_level": ci_level}

        for var_name, out_key in [
            ("baseline", "baseline"),
            ("expected", "media"),
        ]:
            if var_name in ds:
                arr = np.array(ds[var_name].isel({time_coord: j}))
                if arr.ndim == 0:
                    row[f"{out_key}_mean"] = float(arr)
                    row[f"{out_key}_ci_lower"] = float(arr)
                    row[f"{out_key}_ci_upper"] = float(arr)
                else:
                    row[f"{out_key}_mean"] = float(arr.mean())
                    row[f"{out_key}_ci_lower"] = float(np.quantile(arr, alpha))
                    row[f"{out_key}_ci_upper"] = float(np.quantile(arr, 1.0 - alpha))
            else:
                row[f"{out_key}_mean"] = None
                row[f"{out_key}_ci_lower"] = None
                row[f"{out_key}_ci_upper"] = None

        if "actual" in ds:
            row["actual"] = float(np.array(ds["actual"].isel({time_coord: j})).mean())
        else:
            row["actual"] = None

        result.append(row)

    logger.debug("get_baseline_vs_media: %d time periods", len(result))
    return result


def get_predictive_accuracy(
    use_kpi: bool = False,
) -> dict:
    """
    Return goodness-of-fit metrics: R², MAPE, wMAPE.

    Returns a dict with keys: r_squared, mape, wmape.
    """
    an = _get_analyzer()
    ds = an.predictive_accuracy(use_kpi=use_kpi)

    result = {}
    for metric in ["r_squared", "mape", "wmape"]:
        if metric in ds:
            val = float(ds[metric].values.mean())
            result[metric] = val
        else:
            result[metric] = None

    logger.debug("get_predictive_accuracy: %s", result)
    return result


def get_summary_metrics(
    ci_level: float = 0.9,
    use_kpi: bool = False,
    include_non_paid: bool = False,
) -> dict:
    """
    Return the full Meridian summary metrics dataset as a plain dict.

    Keys map to xr.Dataset variable names: roi, mroi, cpik,
    incremental_outcome, effectiveness, etc. Each variable is a dict of
    {channel → {mean, ci_lower, ci_upper}}.
    """
    import numpy as np  # noqa: PLC0415
    an = _get_analyzer()

    ds = an.summary_metrics(
        aggregate_geos=True,
        aggregate_times=True,
        use_kpi=use_kpi,
        confidence_level=ci_level,
        include_non_paid_channels=include_non_paid,
    )

    alpha = (1.0 - ci_level) / 2.0
    result = {}

    for var_name in ds.data_vars:
        da = ds[var_name]
        # Detect channel coordinate
        ch_coord = next(
            (c for c in da.coords if "channel" in c), None
        )
        if ch_coord is None:
            # Scalar metric
            result[var_name] = float(np.array(da).mean())
            continue

        channel_vals = [str(c) for c in da.coords[ch_coord].values]
        per_channel = {}
        for i, ch in enumerate(channel_vals):
            arr = np.array(da.isel({ch_coord: i}))
            # arr may be (n_draws,) or scalar
            if arr.ndim == 0:
                per_channel[ch] = {
                    "mean": float(arr),
                    "ci_lower": float(arr),
                    "ci_upper": float(arr),
                }
            else:
                per_channel[ch] = {
                    "mean": float(arr.mean()),
                    "ci_lower": float(np.quantile(arr, alpha)),
                    "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
                }
        result[var_name] = per_channel

    result["_ci_level"] = ci_level
    logger.debug("get_summary_metrics: %d variables", len(result) - 1)
    return result
