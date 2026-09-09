"""
api/schemas.py
==============
Data contracts (Pydantic models) for the Meridian Serving Application.

This module defines all request and response models crossing the API boundary:
- Analysis outputs (contributions, ROI summary, response curves, baseline decomposition)
- Optimizer outputs (fixed and flexible budget scenarios, current spend)
- Scenario and What-If planning requests
- Model metadata and system health

Rules:
- Strictly typed: NO `Any` or untyped fields.
- Backward compatibility: This is the shared contract between backend services,
  FastAPI endpoints, and frontend clients.
"""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Common / Base Configuration
# ---------------------------------------------------------------------------

class StrictBaseModel(BaseModel):
    """Base model enforcing clear serialization and field configuration."""
    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Model Spec & System Health Contracts
# ---------------------------------------------------------------------------

class ChannelTaxonomyItem(StrictBaseModel):
    """Taxonomy and execution metadata for a channel."""
    execution_metric: str = Field(..., description="Execution metric type (e.g. impressions, clicks, reach_and_frequency, organic, spend)")
    cost_unit: str | None = Field(default=None, description="Cost unit (CPM, CPC, CPR, Spend)")
    category: str = Field(..., description="Business category grouping")
    color: str = Field(..., description="Hex color token")
    has_optimal_frequency: bool | None = Field(default=None, description="Whether RF frequency optimization applies")


class ModelSpecResponse(StrictBaseModel):
    """Schema matching config/model_spec.json."""
    n_geos: int = Field(..., description="Number of geographical units")
    geo_names: list[str] = Field(..., description="List of geographical names")
    n_times: int = Field(..., description="Number of time intervals")
    date_start: str = Field(..., description="Start date of dataset (ISO format)")
    date_end: str = Field(..., description="End date of dataset (ISO format)")
    time_granularity: str = Field(..., description="Time interval granularity (e.g., weekly)")
    n_media_channels: int = Field(..., description="Number of standard media channels")
    media_channel_names: list[str] = Field(..., description="Names of standard media channels")
    n_rf_channels: int = Field(..., description="Number of reach & frequency channels")
    rf_channel_names: list[str] = Field(..., description="Names of reach & frequency channels")
    n_organic_media_channels: int = Field(..., description="Number of non-paid / organic channels")
    organic_media_channel_names: list[str] = Field(..., description="Names of organic channels")
    n_controls: int = Field(..., description="Number of control variables")
    control_names: list[str] = Field(..., description="Names of control variables")
    kpi_name: str = Field(..., description="Target KPI variable name")
    kpi_type: str = Field(..., description="KPI type (e.g. non_revenue, revenue)")
    kpi_units: str = Field(..., description="Measurement units for KPI")
    revenue_per_kpi_col: str = Field(..., description="Column holding revenue conversion factor")
    mcmc_n_chains: int = Field(..., description="Number of MCMC chains used in sampling")
    mcmc_n_draws: int = Field(..., description="Number of posterior draws per chain")
    max_lag: int = Field(..., description="Maximum adstock lag in time units")
    hill_before_adstock: bool = Field(..., description="Whether Hill curve applied before Adstock")
    media_effects_dist: str = Field(..., description="Distribution used for media effects prior")
    enable_aks: bool = Field(..., description="Whether Auto-Knot Selection was enabled")
    channel_taxonomy: dict[str, ChannelTaxonomyItem] | None = Field(
        default=None, description="Per-channel taxonomy: execution metric, cost unit, category, color"
    )
    categories: list[str] | None = Field(default=None, description="List of channel category names")


class ModelInfoResponse(StrictBaseModel):
    """Basic model status returned from loader."""
    loaded: bool = Field(..., description="Whether model .binpb is currently loaded in memory")
    model_path: str = Field(..., description="Path to .binpb model file")
    n_geos: int | None = Field(default=None, description="Number of geos in loaded model")
    n_media_channels: int | None = Field(default=None, description="Number of media channels")
    n_rf_channels: int | None = Field(default=None, description="Number of reach/frequency channels")


class HealthResponse(StrictBaseModel):
    """System health check endpoint response."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Service health state")
    model_loaded: bool = Field(..., description="Model loaded status")
    cache_ready: bool = Field(default=False, description="Whether precomputed cache is populated")
    version: str = Field(default="1.0.0", description="API version")


# ---------------------------------------------------------------------------
# Analysis Layer Contracts (Stage 1 services/analysis.py)
# ---------------------------------------------------------------------------

class ChannelContribution(StrictBaseModel):
    """Posterior incremental outcome (contribution) per channel."""
    channel: str = Field(..., description="Channel name")
    mean: float | None = Field(default=None, description="Posterior mean incremental outcome")
    ci_lower: float | None = Field(default=None, description="Lower credible bound")
    ci_upper: float | None = Field(default=None, description="Upper credible bound")
    pct_of_total_mean: float | None = Field(default=None, description="Share of total contribution (percentage)")
    ci_level: float = Field(default=0.9, description="Credible interval width (e.g. 0.90 for 90% CI)")
    is_organic: bool = Field(default=False, description="Whether channel is non-paid / organic")
    execution_metric: str | None = Field(default=None, description="Execution metric type")
    category: str | None = Field(default=None, description="Channel category")


class RoiSummaryItem(StrictBaseModel):
    """Posterior ROI and performance summary per channel."""
    channel: str = Field(..., description="Channel name")
    roi_mean: float | None = Field(default=None, description="Posterior mean ROI")
    roi_ci_lower: float | None = Field(default=None, description="Lower bound of ROI CI")
    roi_ci_upper: float | None = Field(default=None, description="Upper bound of ROI CI")
    mroi_mean: float | None = Field(default=None, description="Posterior mean marginal ROI")
    mroi_ci_lower: float | None = Field(default=None, description="Lower bound of marginal ROI CI")
    mroi_ci_upper: float | None = Field(default=None, description="Upper bound of marginal ROI CI")
    incremental_outcome_mean: float | None = Field(default=None, description="Mean incremental outcome")
    incremental_outcome_ci_lower: float | None = Field(default=None, description="Lower bound of outcome CI")
    incremental_outcome_ci_upper: float | None = Field(default=None, description="Upper bound of outcome CI")
    pct_of_contribution: float | None = Field(default=None, description="Mean percentage contribution")
    spend_mean: float | None = Field(default=None, description="Mean historical spend")
    cpik_mean: float | None = Field(default=None, description="Cost per incremental KPI unit")
    ci_level: float = Field(default=0.9, description="Credible interval width")
    execution_metric: str | None = Field(
        default=None, description="Execution metric type (reach_and_frequency, clicks, impressions, organic, spend)"
    )
    cost_unit: str | None = Field(default=None, description="Cost unit (e.g. CPR, CPC, CPM, Spend)")
    category: str | None = Field(default=None, description="Channel category")


class ResponseCurvePoint(StrictBaseModel):
    """Single point along a channel's saturation / response curve."""
    channel: str = Field(..., description="Channel name")
    spend_multiplier: float = Field(..., description="Multiplier applied to historical spend")
    spend_pct_of_current: float = Field(..., description="Spend expressed as percentage of historical spend")
    incremental_outcome_mean: float = Field(..., description="Mean incremental outcome at this spend")
    ci_lower: float = Field(..., description="Lower credible bound at this spend")
    ci_upper: float = Field(..., description="Upper credible bound at this spend")
    ci_level: float = Field(default=0.9, description="Credible interval width")


class BaselineVsMediaPoint(StrictBaseModel):
    """Time-series decomposition point (baseline vs media-driven outcome)."""
    date: str = Field(..., description="Date identifier (ISO string YYYY-MM-DD)")
    ci_level: float = Field(default=0.9, description="Credible interval width")
    baseline_mean: float | None = Field(default=None, description="Posterior mean baseline outcome")
    baseline_ci_lower: float | None = Field(default=None, description="Lower bound of baseline outcome CI")
    baseline_ci_upper: float | None = Field(default=None, description="Upper bound of baseline outcome CI")
    media_mean: float | None = Field(default=None, description="Posterior mean media outcome")
    media_ci_lower: float | None = Field(default=None, description="Lower bound of media outcome CI")
    media_ci_upper: float | None = Field(default=None, description="Upper bound of media outcome CI")
    actual: float | None = Field(default=None, description="Actual observed outcome if available")


class PredictiveAccuracyResponse(StrictBaseModel):
    """Goodness-of-fit / accuracy metrics."""
    r_squared: float | None = Field(default=None, description="R-squared metric")
    mape: float | None = Field(default=None, description="Mean Absolute Percentage Error")
    wmape: float | None = Field(default=None, description="Weighted Mean Absolute Percentage Error")


class MetricStat(StrictBaseModel):
    """Statistical summary for a single metric variable and channel."""
    mean: float = Field(..., description="Mean estimate")
    ci_lower: float = Field(..., description="Lower credible interval bound")
    ci_upper: float = Field(..., description="Upper credible interval bound")


class SummaryMetricsResponse(StrictBaseModel):
    """Full summary metrics dataset keyed by variable and channel."""
    ci_level: float = Field(default=0.9, description="Credible interval width")
    metrics: dict[str, dict[str, MetricStat] | float] = Field(
        ..., description="Map of metric variable name to channel stats or overall scalar"
    )


# ---------------------------------------------------------------------------
# Optimizer Layer Contracts (Stage 1 services/optimizer.py)
# ---------------------------------------------------------------------------

class ChannelSpendSummary(StrictBaseModel):
    """Historical spend summary for a single channel."""
    channel: str = Field(..., description="Channel name")
    total_spend: float = Field(..., description="Total historical spend across model period")


class ChannelOptimizationResult(StrictBaseModel):
    """Per-channel optimization outcome details."""
    channel: str = Field(..., description="Channel name")
    ci_level: float = Field(default=0.9, description="Credible interval width")
    current_spend: float | None = Field(default=None, description="Historical spend")
    optimized_spend: float | None = Field(default=None, description="Allocated optimal spend")
    spend_delta: float | None = Field(default=None, description="Change in spend (optimized - current)")
    spend_delta_pct: float | None = Field(default=None, description="Percentage change in spend")
    roi_mean: float | None = Field(default=None, description="Posterior mean ROI under optimal spend")
    roi_ci_lower: float | None = Field(default=None, description="Lower bound of optimal ROI CI")
    roi_ci_upper: float | None = Field(default=None, description="Upper bound of optimal ROI CI")
    mroi_mean: float | None = Field(default=None, description="Posterior mean marginal ROI under optimal spend")
    mroi_ci_lower: float | None = Field(default=None, description="Lower bound of optimal mROI CI")
    mroi_ci_upper: float | None = Field(default=None, description="Upper bound of optimal mROI CI")
    incremental_outcome_mean: float | None = Field(default=None, description="Expected incremental outcome")
    incremental_outcome_ci_lower: float | None = Field(default=None, description="Lower bound of outcome CI")
    incremental_outcome_ci_upper: float | None = Field(default=None, description="Upper bound of outcome CI")
    lift_mean: float | None = Field(default=None, description="Incremental lift over current allocation")
    lift_ci_lower: float | None = Field(default=None, description="Lower bound of lift CI")
    lift_ci_upper: float | None = Field(default=None, description="Upper bound of lift CI")
    confidence_flagged: bool = Field(
        default=False,
        description="True if lift credible interval crosses zero (uncertain direction)",
    )


class OptimizationResponse(StrictBaseModel):
    """Complete budget optimization response."""
    status: str = Field(default="ok", description="Execution status ('ok' or error detail)")
    budget_type: Literal["fixed", "flexible"] = Field(..., description="Optimization type")
    total_budget: float | None = Field(default=None, description="Total budget target if fixed")
    total_optimized_spend: float | None = Field(default=None, description="Sum of optimized channel spend")
    target_roi: float | None = Field(default=None, description="Target ROI if flexible")
    target_mroi: float | None = Field(default=None, description="Target marginal ROI if flexible")
    start_date: str | None = Field(default=None, description="Start date filter applied")
    end_date: str | None = Field(default=None, description="End date filter applied")
    spend_constraint_lower: float = Field(default=0.3, description="Lower spend bound constraint")
    spend_constraint_upper: float = Field(default=0.3, description="Upper spend bound constraint")
    use_optimal_frequency: bool = Field(default=True, description="Whether RF frequency was optimized")
    ci_level: float = Field(default=0.9, description="Credible interval width")
    n_flagged_channels: int = Field(default=0, description="Count of channels with confidence flags")
    channels: list[ChannelOptimizationResult] = Field(..., description="Per-channel optimization results")


# ---------------------------------------------------------------------------
# Scenario & What-If Request Contracts
# ---------------------------------------------------------------------------

class ScenarioRequest(StrictBaseModel):
    """
    Budget optimization scenario request contract.
    Sent when a user adjusts budget sliders or sets constraints in the UI.
    """
    total_budget: float | None = Field(
        default=None,
        description="Total budget to allocate. If None, uses historical spend.",
    )
    spend_constraint_lower: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum fractional spend decrease per channel (default 0.3 = 30%).",
    )
    spend_constraint_upper: float = Field(
        default=0.3,
        ge=0.0,
        description="Maximum fractional spend increase per channel (default 0.3 = 30%).",
    )
    channel_constraints: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Explicit (min_spend, max_spend) bounds per channel name.",
    )
    cost_overrides: dict[str, float] | None = Field(
        default=None,
        description="Channel name to cost-per-unit override multiplier (e.g. {'Online_Video': 1.25}).",
    )
    target_roi: float | None = Field(
        default=None,
        description="Target ROI for flexible budget optimization.",
    )
    target_mroi: float | None = Field(
        default=None,
        description="Target marginal ROI for flexible budget optimization.",
    )
    start_date: str | None = Field(
        default=None,
        description="Optional start date filter (ISO format YYYY-MM-DD).",
    )
    end_date: str | None = Field(
        default=None,
        description="Optional end date filter (ISO format YYYY-MM-DD).",
    )
    use_optimal_frequency: bool = Field(
        default=True,
        description="Optimize RF channel frequencies.",
    )
    use_kpi: bool = Field(
        default=False,
        description="Optimize on KPI count scale rather than revenue scale.",
    )
    ci_level: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Credible interval width (0.0 to 1.0).",
    )


class WhatIfChannelAdjustment(StrictBaseModel):
    """Specific channel adjustment parameters for what-if simulation."""
    spend: float | None = Field(default=None, description="Proposed absolute spend for this channel")
    spend_multiplier: float | None = Field(
        default=None, description="Spend multiplier relative to historical spend (e.g. 1.2 = +20%)"
    )
    cpm_multiplier: float = Field(
        default=1.0, description="Cost-per-unit multiplier (e.g. 2.0 = CPM doubles)"
    )
    cost_multiplier: float | None = Field(
        default=None, description="Cost-per-unit multiplier (alias for cpm_multiplier across any execution metric)"
    )


class WhatIfRequest(StrictBaseModel):
    """
    Simulation request contract for custom CPM or spend adjustments.
    """
    scenario_name: str = Field(default="Custom Scenario", description="Display name for this simulation")
    channel_adjustments: dict[str, WhatIfChannelAdjustment] = Field(
        default_factory=dict,
        description="Map of channel name to adjustment parameters",
    )
    use_kpi: bool = Field(default=False, description="Use raw KPI scale instead of revenue")
    ci_level: float = Field(default=0.9, gt=0.0, lt=1.0, description="Credible interval width")


class WhatIfChannelResult(StrictBaseModel):
    """Outcome for a single channel under a what-if scenario."""
    channel: str = Field(..., description="Channel name")
    baseline_spend: float = Field(..., description="Historical spend")
    scenario_spend: float = Field(..., description="Simulated scenario spend")
    spend_delta: float = Field(..., description="Difference in spend")
    spend_delta_pct: float = Field(..., description="Percentage difference in spend")
    cpm_multiplier: float = Field(default=1.0, description="Cost multiplier applied (CPM)")
    cost_multiplier: float = Field(default=1.0, description="Cost multiplier applied (metric-aware)")
    execution_metric: str | None = Field(default=None, description="Channel execution metric")
    cost_unit: str | None = Field(default=None, description="Channel cost unit")
    category: str | None = Field(default=None, description="Channel category")
    expected_outcome_mean: float = Field(..., description="Expected outcome posterior mean")
    ci_lower: float = Field(..., description="Lower bound of outcome CI")
    ci_upper: float = Field(..., description="Upper bound of outcome CI")


class WhatIfResponse(StrictBaseModel):
    """Response contract for what-if simulation."""
    status: str = Field(default="ok", description="Status string")
    scenario_name: str = Field(..., description="Name of scenario simulated")
    total_baseline_spend: float = Field(..., description="Total baseline spend across channels")
    total_scenario_spend: float = Field(..., description="Total simulated spend across channels")
    total_outcome_mean: float = Field(..., description="Total outcome posterior mean")
    total_outcome_ci_lower: float = Field(..., description="Total outcome lower credible bound")
    total_outcome_ci_upper: float = Field(..., description="Total outcome upper credible bound")
    ci_level: float = Field(default=0.9, description="Credible interval width")
    channels: list[WhatIfChannelResult] = Field(..., description="Per-channel simulated outcomes")
