"""
api/main.py
===========
FastAPI HTTP Serving Application for Google Meridian MMM.

Exposes:
- GET  /health                    System health, model loaded status, cache ready
- GET  /model/spec                Model metadata and specifications
- GET  /model/info                Model runtime object dimensions
- GET  /contributions             Cached channel contributions with credible intervals
- GET  /roi-summary               Cached ROI, mROI, and spend metrics
- GET  /response-curves           Cached response curves for all channels
- GET  /response-curves/{channel} Cached response curve for a single channel
- GET  /baseline-vs-media         Cached time-series baseline decomposition
- GET  /accuracy                  Predictive accuracy metrics (R2, MAPE, wMAPE)
- GET  /spend-summary             Historical spend summary per channel
- POST /optimize                  Budget optimization (cache-first, live fallback)
- POST /whatif                    What-if simulation with CPM and spend adjustments
- GET  /docs                      Interactive OpenAPI documentation (Swagger UI)
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure CPU execution for serving
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

from utils.common import load_model_spec  # noqa: E402
from services import loader, analysis, optimizer, cache  # noqa: E402
from api import schemas  # noqa: E402

# Structured logger for API events
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api.serving")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load Meridian model singleton and verify cache readiness once at startup.
    Never load per-request.
    """
    logger.info("Initializing Meridian Serving Application...")
    try:
        loader.get_model()
        m_hash = cache.get_model_hash()
        cache_ready = cache.is_cache_ready(m_hash)
        logger.info("Model loaded successfully. Model hash: %s | Cache ready: %s", m_hash, cache_ready)
    except Exception as exc:
        logger.error("Failed to load model during startup: %s", exc)

    yield

    logger.info("Shutting down Meridian Serving Application.")


app = FastAPI(
    title="Google Meridian MMM Serving API",
    description="Interactive HTTP API serving posterior analysis, response curves, and budget optimization.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_CONTROL_HEADER = "public, max-age=3600"


# ---------------------------------------------------------------------------
# Health & Metadata Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=schemas.HealthResponse,
    tags=["System"],
    summary="Health check",
)
def get_health() -> schemas.HealthResponse:
    """Return model loaded status, cache readiness, and service health."""
    info = loader.model_info()
    model_loaded = info.get("loaded", False)
    cache_ready = cache.is_cache_ready()

    health_status = "healthy" if (model_loaded and cache_ready) else ("degraded" if model_loaded else "unhealthy")

    return schemas.HealthResponse(
        status=health_status,
        model_loaded=model_loaded,
        cache_ready=cache_ready,
        version="1.0.0",
    )


@app.get(
    "/model/spec",
    response_model=schemas.ModelSpecResponse,
    tags=["Metadata"],
    summary="Model specification",
)
def get_model_spec(response: Response) -> schemas.ModelSpecResponse:
    """Return the frozen model specifications extracted from the training artifact."""
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER
    raw_spec = load_model_spec()
    return schemas.ModelSpecResponse.model_validate(raw_spec)


@app.get(
    "/model/info",
    response_model=schemas.ModelInfoResponse,
    tags=["Metadata"],
    summary="Loaded model runtime info",
)
def get_model_info() -> schemas.ModelInfoResponse:
    """Return dimensions and path of the loaded model object."""
    info = loader.model_info()
    return schemas.ModelInfoResponse.model_validate(info)


# ---------------------------------------------------------------------------
# Analysis Endpoints (Cached)
# ---------------------------------------------------------------------------

@app.get(
    "/contributions",
    response_model=list[schemas.ChannelContribution],
    tags=["Analysis"],
    summary="Channel incremental contributions",
)
def get_contributions(response: Response) -> list[schemas.ChannelContribution]:
    """Return posterior mean and credible intervals of channel contributions."""
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER

    cached_data = cache.get_cached_contributions()
    if cached_data is not None:
        response.headers["X-Cache"] = "HIT"
        return [schemas.ChannelContribution.model_validate(item) for item in cached_data]

    response.headers["X-Cache"] = "MISS"
    raw_data = analysis.get_channel_contributions()
    cache.write_json_cache(cache.FILE_CONTRIBUTIONS, raw_data)
    return [schemas.ChannelContribution.model_validate(item) for item in raw_data]


@app.get(
    "/roi-summary",
    response_model=list[schemas.RoiSummaryItem],
    tags=["Analysis"],
    summary="ROI summary table",
)
def get_roi_summary(response: Response) -> list[schemas.RoiSummaryItem]:
    """Return posterior mean ROI, marginal ROI, spend, and credible bounds."""
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER

    cached_data = cache.get_cached_roi_summary()
    if cached_data is not None:
        response.headers["X-Cache"] = "HIT"
        return [schemas.RoiSummaryItem.model_validate(item) for item in cached_data]

    response.headers["X-Cache"] = "MISS"
    raw_data = analysis.get_roi_summary()
    cache.write_json_cache(cache.FILE_ROI_SUMMARY, raw_data)
    return [schemas.RoiSummaryItem.model_validate(item) for item in raw_data]


@app.get(
    "/response-curves",
    response_model=dict[str, list[schemas.ResponseCurvePoint]],
    tags=["Analysis"],
    summary="All channel response curves",
)
def get_all_response_curves(response: Response) -> dict[str, list[schemas.ResponseCurvePoint]]:
    """Return spend vs incremental outcome saturation curves for all channels."""
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER

    cached_data = cache.get_cached_response_curves()
    if cached_data is not None:
        response.headers["X-Cache"] = "HIT"
        return {
            ch: [schemas.ResponseCurvePoint.model_validate(p) for p in pts]
            for ch, pts in cached_data.items()
        }

    response.headers["X-Cache"] = "MISS"
    raw_data = analysis.get_response_curves_all_channels()
    cache.write_json_cache(cache.FILE_RESPONSE_CURVES, raw_data)
    return {
        ch: [schemas.ResponseCurvePoint.model_validate(p) for p in pts]
        for ch, pts in raw_data.items()
    }


@app.get(
    "/response-curves/{channel}",
    response_model=list[schemas.ResponseCurvePoint],
    tags=["Analysis"],
    summary="Single channel response curve",
)
def get_single_response_curve(channel: str, response: Response) -> list[schemas.ResponseCurvePoint]:
    """Return spend vs incremental outcome saturation curve points for one channel."""
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER

    cached_points = cache.get_cached_response_curves(channel=channel)
    if cached_points is not None:
        response.headers["X-Cache"] = "HIT"
        return [schemas.ResponseCurvePoint.model_validate(p) for p in cached_points]

    try:
        raw_points = analysis.get_response_curves(channel=channel)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Channel '{channel}' not found: {exc}")

    response.headers["X-Cache"] = "MISS"
    return [schemas.ResponseCurvePoint.model_validate(p) for p in raw_points]


@app.get(
    "/baseline-vs-media",
    response_model=list[schemas.BaselineVsMediaPoint],
    tags=["Analysis"],
    summary="Baseline vs media timeline",
)
def get_baseline_vs_media(response: Response) -> list[schemas.BaselineVsMediaPoint]:
    """Return time-series decomposition into baseline outcome vs media-driven outcome."""
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER

    cached_data = cache.get_cached_baseline_vs_media()
    if cached_data is not None:
        response.headers["X-Cache"] = "HIT"
        return [schemas.BaselineVsMediaPoint.model_validate(item) for item in cached_data]

    response.headers["X-Cache"] = "MISS"
    raw_data = analysis.get_baseline_vs_media()
    cache.write_json_cache(cache.FILE_BASELINE_VS_MEDIA, raw_data)
    return [schemas.BaselineVsMediaPoint.model_validate(item) for item in raw_data]


@app.get(
    "/accuracy",
    response_model=schemas.PredictiveAccuracyResponse,
    tags=["Analysis"],
    summary="Goodness-of-fit metrics",
)
def get_accuracy() -> schemas.PredictiveAccuracyResponse:
    """Return R-squared, MAPE, and wMAPE model accuracy metrics."""
    cached_data = cache.get_cached_accuracy()
    if cached_data is not None:
        return schemas.PredictiveAccuracyResponse.model_validate(cached_data)

    raw_data = analysis.get_predictive_accuracy()
    cache.write_json_cache(cache.FILE_ACCURACY, raw_data)
    return schemas.PredictiveAccuracyResponse.model_validate(raw_data)


@app.get(
    "/spend-summary",
    response_model=list[schemas.ChannelSpendSummary],
    tags=["Optimizer"],
    summary="Historical spend summary",
)
def get_spend_summary() -> list[schemas.ChannelSpendSummary]:
    """Return aggregated historical spend per channel for frontend slider initial state."""
    cached_data = cache.get_cached_spend_summary()
    if cached_data is not None:
        return [schemas.ChannelSpendSummary.model_validate(item) for item in cached_data]

    raw_data = optimizer.get_current_spend_summary()
    cache.write_json_cache(cache.FILE_SPEND_SUMMARY, raw_data)
    return [schemas.ChannelSpendSummary.model_validate(item) for item in raw_data]


# ---------------------------------------------------------------------------
# Optimizer & Simulation Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/optimize",
    response_model=schemas.OptimizationResponse,
    tags=["Optimizer"],
    summary="Run or retrieve budget optimization",
)
def post_optimize(
    scenario: schemas.ScenarioRequest,
    response: Response,
) -> schemas.OptimizationResponse:
    """
    Solve for optimal budget allocation across channels.
    Serves from deterministic cache when scenario parameters match; computes live otherwise.
    """
    scenario_dict = scenario.model_dump()
    scenario_key = cache.compute_scenario_key(scenario_dict)

    # 1. Check cache hit (scenario key or default scenario cache)
    cached_result = cache.get_cached_scenario_optimization(scenario_key)
    if cached_result is None and scenario == schemas.ScenarioRequest():
        cached_result = cache.get_cached_default_optimization()
        if cached_result is not None:
            cache.save_cached_scenario_optimization(scenario_key, cached_result)

    if cached_result is not None:
        logger.info(
            json.dumps({
                "event": "optimize",
                "cache_hit": True,
                "scenario_key": scenario_key,
                "total_budget": scenario.total_budget,
            })
        )
        response.headers["X-Cache"] = "HIT"
        return schemas.OptimizationResponse.model_validate(cached_result)

    # 2. Cache miss: Compute live
    logger.info(
        json.dumps({
            "event": "optimize",
            "cache_hit": False,
            "scenario_key": scenario_key,
            "total_budget": scenario.total_budget,
            "constraints": [scenario.spend_constraint_lower, scenario.spend_constraint_upper],
        })
    )

    try:
        if scenario.target_roi is not None or scenario.target_mroi is not None:
            raw_result = optimizer.run_flexible_budget_optimization(
                target_roi=scenario.target_roi,
                target_mroi=scenario.target_mroi,
                spend_constraint_lower=scenario.spend_constraint_lower,
                spend_constraint_upper=scenario.spend_constraint_upper,
                start_date=scenario.start_date,
                end_date=scenario.end_date,
                use_optimal_frequency=scenario.use_optimal_frequency,
                use_kpi=scenario.use_kpi,
                ci_level=scenario.ci_level,
            )
        else:
            raw_result = optimizer.run_fixed_budget_optimization(
                total_budget=scenario.total_budget,
                spend_constraint_lower=scenario.spend_constraint_lower,
                spend_constraint_upper=scenario.spend_constraint_upper,
                start_date=scenario.start_date,
                end_date=scenario.end_date,
                use_optimal_frequency=scenario.use_optimal_frequency,
                use_kpi=scenario.use_kpi,
                ci_level=scenario.ci_level,
            )

        # Cache the result for future identical requests
        cache.save_cached_scenario_optimization(scenario_key, raw_result)
        response.headers["X-Cache"] = "MISS"
        return schemas.OptimizationResponse.model_validate(raw_result)

    except Exception as exc:
        logger.error("Optimization failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Budget optimization failed: {exc}",
        )


@app.post(
    "/whatif",
    response_model=schemas.WhatIfResponse,
    tags=["Optimizer"],
    summary="Simulate what-if scenario",
)
def post_whatif(request: schemas.WhatIfRequest) -> schemas.WhatIfResponse:
    """
    Run what-if scenario simulation with custom spend and CPM cost multipliers.
    """
    logger.info(
        json.dumps({
            "event": "whatif",
            "scenario_name": request.scenario_name,
            "channels_adjusted": list(request.channel_adjustments.keys()),
        })
    )

    # Read historical spend baseline
    spends = cache.get_cached_spend_summary() or optimizer.get_current_spend_summary()
    spend_map = {s["channel"]: s["total_spend"] for s in spends}

    # Read response curves for fast, accurate response estimation
    all_curves = cache.get_cached_response_curves() or analysis.get_response_curves_all_channels()

    # Read channel taxonomy for metric and cost unit awareness
    spec = {}
    try:
        spec = load_model_spec()
    except Exception:
        pass
    taxonomy = spec.get("channel_taxonomy", {})

    channel_results: list[schemas.WhatIfChannelResult] = []
    total_baseline_spend = 0.0
    total_scenario_spend = 0.0
    total_outcome_mean = 0.0
    total_outcome_lo = 0.0
    total_outcome_hi = 0.0

    for ch, base_spend in spend_map.items():
        adj = request.channel_adjustments.get(ch)
        if adj and adj.cost_multiplier is not None:
            cost_mult = adj.cost_multiplier
        elif adj and adj.cpm_multiplier is not None:
            cost_mult = adj.cpm_multiplier
        else:
            cost_mult = 1.0
        cpm_mult = cost_mult

        tax = taxonomy.get(ch, {})
        metric = tax.get("execution_metric", "spend")
        unit = tax.get("cost_unit", "Spend")
        category = tax.get("category")

        if adj and adj.spend is not None:
            scen_spend = adj.spend
        elif adj and adj.spend_multiplier is not None:
            scen_spend = base_spend * adj.spend_multiplier
        else:
            scen_spend = base_spend

        delta = scen_spend - base_spend
        delta_pct = (delta / base_spend * 100.0) if base_spend > 0 else 0.0

        # Estimate outcome from response curve adjusted for cost multiplier (CPM/CPC/CPR)
        effective_mult = (scen_spend / base_spend) / cost_mult if (base_spend > 0 and cost_mult > 0) else 1.0

        curve = all_curves.get(ch, [])
        if curve:
            # Interpolate nearest curve point
            closest_pt = min(curve, key=lambda pt: abs(pt["spend_multiplier"] - effective_mult))
            out_mean = closest_pt["incremental_outcome_mean"]
            out_lo = closest_pt["ci_lower"]
            out_hi = closest_pt["ci_upper"]
        else:
            out_mean, out_lo, out_hi = 0.0, 0.0, 0.0

        total_baseline_spend += base_spend
        total_scenario_spend += scen_spend
        total_outcome_mean += out_mean
        total_outcome_lo += out_lo
        total_outcome_hi += out_hi

        channel_results.append(
            schemas.WhatIfChannelResult(
                channel=ch,
                baseline_spend=round(base_spend, 2),
                scenario_spend=round(scen_spend, 2),
                spend_delta=round(delta, 2),
                spend_delta_pct=round(delta_pct, 2),
                cpm_multiplier=cpm_mult,
                cost_multiplier=cost_mult,
                execution_metric=metric,
                cost_unit=unit,
                category=category,
                expected_outcome_mean=round(out_mean, 2),
                ci_lower=round(out_lo, 2),
                ci_upper=round(out_hi, 2),
            )
        )

    return schemas.WhatIfResponse(
        status="ok",
        scenario_name=request.scenario_name,
        total_baseline_spend=round(total_baseline_spend, 2),
        total_scenario_spend=round(total_scenario_spend, 2),
        total_outcome_mean=round(total_outcome_mean, 2),
        total_outcome_ci_lower=round(total_outcome_lo, 2),
        total_outcome_ci_upper=round(total_outcome_hi, 2),
        ci_level=request.ci_level,
        channels=channel_results,
    )
