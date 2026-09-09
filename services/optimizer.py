"""
services/optimizer.py
======================
Thin wrappers around Meridian's BudgetOptimizer.

Design rules
------------
- All outputs are plain Python dicts / lists of primitives.
- The BudgetOptimizer is instantiated once and cached alongside the model.
- Expensive operations (optimization runs) are expected to be called from the
  precompute job (Stage 3) for the default scenario, and from the API directly
  for novel/on-demand scenarios.

Meridian BudgetOptimizer API used:
  optimizer.optimize()                → OptimizationResults
  optimizer.create_optimization_grid()→ OptimizationGrid (precompute cache)
  optimizer.create_optimization_tensors() → DataTensors (cost_override scenarios)

OptimizationResults fields:
  .optimized_data    : xr.Dataset  (spend, roi, mroi, cpik, incremental_outcome)
  .nonoptimized_data : xr.Dataset  (same, under historical budget)
  .spend_ratio       : the final budget/historical ratio
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level BudgetOptimizer singleton
# ---------------------------------------------------------------------------
_budget_optimizer = None


def _get_optimizer():
    """Return the cached BudgetOptimizer, creating it if needed."""
    global _budget_optimizer
    if _budget_optimizer is not None:
        return _budget_optimizer

    from services.loader import get_model               # noqa: PLC0415
    from meridian.analysis.optimizer import BudgetOptimizer  # noqa: PLC0415

    mmm = get_model()
    _budget_optimizer = BudgetOptimizer(mmm)
    logger.info("BudgetOptimizer initialized.")
    return _budget_optimizer


def reset_optimizer():
    """Force re-creation on next call.  Test use only."""
    global _budget_optimizer
    _budget_optimizer = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_optimization_results(
    opt_results,
    ci_level: float = 0.9,
) -> list[dict]:
    """
    Convert an OptimizationResults object to a list of per-channel dicts.

    Each dict has:
      channel, current_spend, optimized_spend, spend_delta, spend_delta_pct,
      roi_mean, roi_ci_lower, roi_ci_upper,
      mroi_mean, mroi_ci_lower, mroi_ci_upper,
      incremental_outcome_mean, incremental_outcome_ci_lower, incremental_outcome_ci_upper,
      incremental_outcome_delta_mean,
      confidence_flagged   (True if lift CI crosses zero)
      ci_level
    """
    import numpy as np  # noqa: PLC0415

    alpha = (1.0 - ci_level) / 2.0
    optimized_ds = opt_results.optimized_data
    nonopt_ds = opt_results.nonoptimized_data

    # Detect channel coordinate
    ch_coord = None
    for c in optimized_ds.coords:
        if "channel" in c:
            ch_coord = c
            break
    if ch_coord is None:
        raise RuntimeError("Cannot locate channel coordinate in OptimizationResults.")

    channels = [str(c) for c in optimized_ds.coords[ch_coord].values]
    result = []

    for i, ch in enumerate(channels):
        row: dict[str, Any] = {"channel": ch, "ci_level": ci_level}

        # --- Spend ---
        def _get_scalar(ds, var, i_ch):
            if var in ds:
                arr = np.array(ds[var].isel({ch_coord: i_ch}))
                return float(arr.mean() if arr.ndim > 0 else arr)
            return None

        curr_spend = _get_scalar(nonopt_ds, "spend", i)
        opt_spend = _get_scalar(optimized_ds, "spend", i)

        row["current_spend"] = curr_spend
        row["optimized_spend"] = opt_spend
        row["spend_delta"] = (
            round(opt_spend - curr_spend, 4) if (opt_spend is not None and curr_spend is not None) else None
        )
        row["spend_delta_pct"] = (
            round((opt_spend - curr_spend) / curr_spend * 100, 2)
            if (opt_spend is not None and curr_spend is not None and curr_spend != 0)
            else None
        )

        # --- ROI ---
        for var, key in [("roi", "roi"), ("mroi", "mroi")]:
            if var in optimized_ds:
                arr = np.array(optimized_ds[var].isel({ch_coord: i}))
                if arr.ndim > 0:
                    row[f"{key}_mean"] = float(arr.mean())
                    row[f"{key}_ci_lower"] = float(np.quantile(arr, alpha))
                    row[f"{key}_ci_upper"] = float(np.quantile(arr, 1.0 - alpha))
                else:
                    row[f"{key}_mean"] = float(arr)
                    row[f"{key}_ci_lower"] = float(arr)
                    row[f"{key}_ci_upper"] = float(arr)
            else:
                row[f"{key}_mean"] = None
                row[f"{key}_ci_lower"] = None
                row[f"{key}_ci_upper"] = None

        # --- Incremental outcome delta (optimized vs non-optimized) ---
        opt_inc = None
        curr_inc = None
        if "incremental_outcome" in optimized_ds:
            arr_opt = np.array(optimized_ds["incremental_outcome"].isel({ch_coord: i}))
            arr_cur = np.array(nonopt_ds["incremental_outcome"].isel({ch_coord: i})) if "incremental_outcome" in nonopt_ds else arr_opt

            if arr_opt.ndim > 0:
                row["incremental_outcome_mean"] = float(arr_opt.mean())
                row["incremental_outcome_ci_lower"] = float(np.quantile(arr_opt, alpha))
                row["incremental_outcome_ci_upper"] = float(np.quantile(arr_opt, 1.0 - alpha))
                opt_inc = arr_opt
                curr_inc = arr_cur

                # Lift (optimized - current)
                if arr_cur is not None:
                    lift = arr_opt - arr_cur
                    row["lift_mean"] = float(lift.mean())
                    row["lift_ci_lower"] = float(np.quantile(lift, alpha))
                    row["lift_ci_upper"] = float(np.quantile(lift, 1.0 - alpha))
                    # confidence_flagged: CI crosses zero → uncertain lift direction
                    row["confidence_flagged"] = bool(
                        row["lift_ci_lower"] <= 0 <= row["lift_ci_upper"]
                    )
                else:
                    row["lift_mean"] = None
                    row["lift_ci_lower"] = None
                    row["lift_ci_upper"] = None
                    row["confidence_flagged"] = False
            else:
                row["incremental_outcome_mean"] = float(arr_opt)
                row["incremental_outcome_ci_lower"] = float(arr_opt)
                row["incremental_outcome_ci_upper"] = float(arr_opt)
                row["lift_mean"] = None
                row["lift_ci_lower"] = None
                row["lift_ci_upper"] = None
                row["confidence_flagged"] = False
        else:
            for k in ["incremental_outcome_mean", "incremental_outcome_ci_lower",
                      "incremental_outcome_ci_upper", "lift_mean", "lift_ci_lower",
                      "lift_ci_upper"]:
                row[k] = None
            row["confidence_flagged"] = False

        result.append(row)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_fixed_budget_optimization(
    total_budget: float | None = None,
    spend_constraint_lower: float = 0.3,
    spend_constraint_upper: float = 0.3,
    start_date: str | None = None,
    end_date: str | None = None,
    use_optimal_frequency: bool = True,
    use_kpi: bool = False,
    ci_level: float = 0.9,
    optimization_grid=None,
) -> dict:
    """
    Run a fixed-budget optimization and return structured results.

    Parameters
    ----------
    total_budget : float | None
        Total budget to allocate across all channels. If None, uses
        historical total spend.
    spend_constraint_lower : float
        Maximum fractional DECREASE in any channel's spend (default 0.3 = 30%).
    spend_constraint_upper : float
        Maximum fractional INCREASE in any channel's spend (default 0.3 = 30%).
    start_date / end_date : str | None
        Optional date range filter (ISO format: "YYYY-MM-DD").
    use_optimal_frequency : bool
        If True, optimize frequency for RF channels (default True).
    use_kpi : bool
        If False (default), optimize on revenue scale.
    ci_level : float
        Credible interval width for result uncertainty (default 0.9).
    optimization_grid : OptimizationGrid | None
        Pre-computed grid to reuse (from precompute cache). If None,
        the optimizer recomputes from scratch (slow).

    Returns
    -------
    dict with keys:
      status, budget_type, total_budget, start_date, end_date,
      n_flagged_channels, channels (list of per-channel dicts)
    """
    opt = _get_optimizer()
    logger.info(
        "run_fixed_budget_optimization: budget=%.2f, constraints=[%.2f, %.2f]",
        total_budget or -1, spend_constraint_lower, spend_constraint_upper,
    )

    kwargs: dict[str, Any] = dict(
        fixed_budget=True,
        budget=total_budget,
        spend_constraint_lower=spend_constraint_lower,
        spend_constraint_upper=spend_constraint_upper,
        use_optimal_frequency=use_optimal_frequency,
        use_kpi=use_kpi,
        confidence_level=ci_level,
        use_posterior=True,
    )
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date
    if optimization_grid is not None:
        kwargs["optimization_grid"] = optimization_grid

    opt_results = opt.optimize(**kwargs)
    channels = _extract_optimization_results(opt_results, ci_level=ci_level)

    n_flagged = sum(1 for c in channels if c.get("confidence_flagged"))
    total_optimized = sum(c["optimized_spend"] or 0 for c in channels)

    return {
        "status": "ok",
        "budget_type": "fixed",
        "total_budget": total_budget,
        "total_optimized_spend": round(total_optimized, 2),
        "start_date": start_date,
        "end_date": end_date,
        "spend_constraint_lower": spend_constraint_lower,
        "spend_constraint_upper": spend_constraint_upper,
        "use_optimal_frequency": use_optimal_frequency,
        "ci_level": ci_level,
        "n_flagged_channels": n_flagged,
        "channels": channels,
    }


def run_flexible_budget_optimization(
    target_roi: float | None = None,
    target_mroi: float | None = None,
    spend_constraint_lower: float = 1.0,
    spend_constraint_upper: float = 1.0,
    start_date: str | None = None,
    end_date: str | None = None,
    use_optimal_frequency: bool = True,
    use_kpi: bool = False,
    ci_level: float = 0.9,
) -> dict:
    """
    Run a flexible-budget optimization (target ROI / mROI).

    Parameters
    ----------
    target_roi : float | None
        Target overall ROI.
    target_mroi : float | None
        Target marginal ROI (used as stopping criterion).
    spend_constraint_lower / upper : float
        Per-channel spend bounds as fraction of historical (default 1.0 = no bounds).
    start_date / end_date : str | None
        Optional date range filter.
    use_optimal_frequency : bool
        Optimize frequency for RF channels (default True).
    use_kpi : bool
        If False (default), use revenue scale.
    ci_level : float
        Credible interval width (default 0.9).

    Returns
    -------
    dict with same schema as run_fixed_budget_optimization.
    """
    opt = _get_optimizer()
    logger.info(
        "run_flexible_budget_optimization: target_roi=%s, target_mroi=%s",
        target_roi, target_mroi,
    )

    kwargs: dict[str, Any] = dict(
        fixed_budget=False,
        spend_constraint_lower=spend_constraint_lower,
        spend_constraint_upper=spend_constraint_upper,
        use_optimal_frequency=use_optimal_frequency,
        use_kpi=use_kpi,
        confidence_level=ci_level,
        use_posterior=True,
    )
    if target_roi is not None:
        kwargs["target_roi"] = target_roi
    if target_mroi is not None:
        kwargs["target_mroi"] = target_mroi
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date

    opt_results = opt.optimize(**kwargs)
    channels = _extract_optimization_results(opt_results, ci_level=ci_level)

    n_flagged = sum(1 for c in channels if c.get("confidence_flagged"))
    total_optimized = sum(c["optimized_spend"] or 0 for c in channels)

    return {
        "status": "ok",
        "budget_type": "flexible",
        "total_budget": None,
        "total_optimized_spend": round(total_optimized, 2),
        "target_roi": target_roi,
        "target_mroi": target_mroi,
        "start_date": start_date,
        "end_date": end_date,
        "spend_constraint_lower": spend_constraint_lower,
        "spend_constraint_upper": spend_constraint_upper,
        "use_optimal_frequency": use_optimal_frequency,
        "ci_level": ci_level,
        "n_flagged_channels": n_flagged,
        "channels": channels,
    }


def build_optimization_grid(
    budget: float | None = None,
    spend_constraint_lower: float = 1.0,
    spend_constraint_upper: float = 1.0,
    start_date: str | None = None,
    end_date: str | None = None,
    use_optimal_frequency: bool = True,
    use_kpi: bool = False,
):
    """
    Build and return an OptimizationGrid for use as a precomputed cache.

    The returned object is a Meridian OptimizationGrid — it is NOT JSON-
    serializable.  It should be held in memory by the precompute job and
    passed to run_fixed_budget_optimization() via the `optimization_grid`
    argument to skip the expensive recomputation.

    Parameters
    ----------
    budget : float | None
        Total budget for the grid. If None, uses historical spend.
    spend_constraint_lower / upper : float
        Grid span per channel (default 1.0 = ±100%).
    start_date / end_date : str | None
        Date range filter.
    use_optimal_frequency : bool
        Optimize RF frequency (default True).
    use_kpi : bool
        Use raw KPI scale (default False → revenue scale).

    Returns
    -------
    OptimizationGrid (Meridian internal object)
    """
    opt = _get_optimizer()
    logger.info(
        "build_optimization_grid: budget=%s, constraints=[%.2f, %.2f]",
        budget, spend_constraint_lower, spend_constraint_upper,
    )

    kwargs: dict[str, Any] = dict(
        budget=budget,
        spend_constraint_lower=spend_constraint_lower,
        spend_constraint_upper=spend_constraint_upper,
        use_optimal_frequency=use_optimal_frequency,
        use_kpi=use_kpi,
        use_posterior=True,
    )
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date

    return opt.create_optimization_grid(**kwargs)


def get_current_spend_summary() -> list[dict]:
    """
    Return historical (non-optimized) spend per channel from the model.

    Each dict has: channel, total_spend
    Useful to populate default budget inputs in the frontend.
    """
    from services.analysis import _get_analyzer  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    an = _get_analyzer()
    # get_aggregated_spend returns xr.DataArray indexed by channel
    spend_da = an.get_aggregated_spend(aggregate_times=True)

    # Detect the channel coordinate name
    ch_coord = None
    for c in spend_da.coords:
        if "channel" in c:
            ch_coord = c
            break

    if ch_coord:
        channels = [str(c) for c in spend_da.coords[ch_coord].values]
    else:
        channels = [f"ch_{i}" for i in range(len(spend_da))]

    spends = np.array(spend_da).tolist()

    return [
        {"channel": ch, "total_spend": float(sp)}
        for ch, sp in zip(channels, spends)
    ]
