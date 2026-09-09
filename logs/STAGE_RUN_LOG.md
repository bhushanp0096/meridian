# Stage 0 Run Log — Environment Validation & Artifact Inventory

**Stage**: 0 — Environment Validation & Artifact Inventory  
**Status**: ✅ PASSED  
**Run date**: 2026-09-09  
**Project**: Meridian Serving Application  
**Build location**: `Meridian_files/meridian_application/meridian-app/`

---

## Task Summary

| Task | Script | Status | Notes |
|---|---|---|---|
| Pin Requirements | `pin_requirements.py` | ✅ PASS | 14 packages pinned |
| Smoke Test | `smoke_test.py` | ✅ PASS | Model loaded, all dims verified |
| Spec Extraction | `extract_model_spec.py` | ✅ PASS | `model_spec.json` written |

---

## Environment

| Item | Value |
|---|---|
| Python | 3.12.13 (Clang 22.1.3) |
| Python binary | `.venv/bin/python3` |
| Venv manager | `uv` (no pip binary; used `importlib.metadata`) |
| OS | Linux (WSL2 Ubuntu) |

---

## Pinned Serving Requirements

| Package | Version |
|---|---|
| google-meridian | 2.0.0 |
| tensorflow | 2.21.0 |
| jax | 0.11.1 |
| jaxlib | 0.11.1 |
| jax-cuda12-plugin | 0.11.1 |
| numpy | 2.3.5 |
| scipy | 1.18.1 |
| xarray | 2026.7.0 |
| arviz | 0.19.0 |
| pandas | 2.3.3 |
| protobuf | 7.36.1 |
| ml-dtypes | 0.6.0 |
| opt-einsum | 3.4.0 |
| immutabledict | 4.3.1 |

> ⚠️ `meridian_model.binpb` is **NOT** forward-compatible across Meridian/TF/JAX versions. The serving container must match these versions exactly.

---

## Smoke Test Results

| Dimension | Value |
|---|---|
| `n_geos` | 6 |
| `n_media_channels` | 5 |
| `n_rf_channels` | 2 |
| `n_times` | 104 |
| `n_organic_media_channels` | 1 |
| `n_controls` | 3 |

**Model file**: `Meridian_files/model_build/meridian_model.binpb` (2.4 MB)  
**Warning observed**: `arviz: UserWarning: The group trace is not defined in the InferenceData scheme` — **non-critical**, cosmetic only, does not affect model loading.

---

## Extracted Model Spec (`config/model_spec.json`)

| Field | Value |
|---|---|
| KPI | `Salon_Bookings` (non_revenue) |
| Revenue per KPI | `Avg_Revenue_Per_Booking` |
| Geos (6) | Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, Pune |
| Media channels (5) | Online_Video, Display, Paid_Social, Paid_Search, Affiliate |
| RF channels (2) | CTV, Linear_TV |
| Organic channels (1) | Email_Opens |
| Controls (3) | Total_Markdowns, Competitor_Spend, Inflation_Rate |
| Date range | 2024-01-01 → 2025-12-22 |
| Time periods | 104 weeks |
| Time granularity | weekly |
| MCMC | 2 chains × 400 draws |
| `max_lag` | 8 weeks |
| `enable_aks` | True |
| `media_effects_dist` | log_normal |

---

## Deliverables

| File | Path | Status |
|---|---|---|
| Smoke test script | `meridian-app/smoke_test.py` | ✅ Created |
| Spec extractor | `meridian-app/extract_model_spec.py` | ✅ Created |
| Req. pinner | `meridian-app/pin_requirements.py` | ✅ Created |
| Stage orchestrator | `meridian-app/stage0_run.py` | ✅ Created |
| Pinned requirements | `meridian-app/requirements-serving.txt` | ✅ Written |
| Model spec | `meridian-app/config/model_spec.json` | ✅ Written |
| Smoke test log | `meridian-app/logs/stage0_smoke_test.json` | ✅ Written |
| Spec extract log | `meridian-app/logs/stage0_spec_extract.json` | ✅ Written |
| Pin requirements log | `meridian-app/logs/stage0_pin_requirements.json` | ✅ Written |

---

## Notes & Observations

1. **arviz FutureWarning** (`Dataset.dims` deprecation): fixed in `extract_model_spec.py` to use `.sizes` instead.
2. **No pip in venv**: uv-managed venv has no pip binary; switched `pin_requirements.py` to use `importlib.metadata.version()`.
3. **fastapi/uvicorn/pydantic/streamlit/httpx not yet installed** in venv — expected; these are serving dependencies that will be added in Stage 4/5.
4. **GPU not used** for Stage 0 — all scripts force CPU via `JAX_PLATFORMS=cpu`.

---

## Next Stage

**Stage 1 — Analysis Service Layer**  
Build `services/loader.py`, `services/analysis.py`, `services/optimizer.py` — the thin Python wrappers around Meridian's Analyzer and BudgetOptimizer that produce JSON-serializable outputs.

---
*Auto-maintained by AI Agent | Stage 0 completed: 2026-09-09*

---

## Stage 1 Run Log — Analysis Service Layer

**Stage**: 1 — Analysis Service Layer  
**Status**: ✅ PASSED (9/9 tests)  
**Run date**: 2026-09-09

### Test Results

| Test | Status | Detail |
|---|---|---|
| `loader.get_model` | ✅ PASS | Model loaded in ~12s |
| `loader.model_info` | ✅ PASS | 6 geos, 5 media ch, 2 RF ch |
| `analysis.get_roi_summary` | ✅ PASS | 7 channels, ROI range [1.30, 977.28] |
| `analysis.get_channel_contributions` | ✅ PASS | 8 channels (incl. Email_Opens organic) |
| `analysis.get_response_curves(Online_Video)` | ✅ PASS | 41 points, outcome [0.0, 838958.9] |
| `analysis.get_response_curves_all_channels` | ✅ PASS | 7 channels × 41 points |
| `analysis.get_baseline_vs_media` | ✅ PASS | 104 time periods |
| `analysis.get_predictive_accuracy` | ✅ PASS | R²/MAPE/wMAPE returned (None = not computed yet) |
| `optimizer.get_current_spend_summary` | ✅ PASS | 7 channels, total spend = 4,185,397 |

### Key Discoveries from Meridian 2.0.0 API

| Discovery | Implication |
|---|---|
| `roi()` returns shape `(n_chains, n_draws, n_channels)` not `(n_channels, n_draws)` | Switched to `summary_metrics()` for ROI/contribution — already pre-computes CI stats |
| `Analyzer` has no `.meridian` attr | Use `.model_context` to access `input_data`, `n_geos`, etc. |
| `summary_metrics` coords: `channel × metric × distribution` | `metric = ['mean', 'median', 'ci_lo', 'ci_hi']`, `distribution = ['prior', 'posterior']` |
| Channel coord in `summary_metrics` includes `'All Channels'` aggregate | Filter with `if str(c) != 'All Channels'` |
| `get_aggregated_spend` channel coord varies | Use dynamic coord name detection (`for c in da.coords: if 'channel' in c`) |

### Deliverables

| File | Status |
|---|---|
| `services/loader.py` | ✅ Created |
| `services/analysis.py` | ✅ Created & verified |
| `services/optimizer.py` | ✅ Created & verified |
| `stage1_verify.py` | ✅ Created |
| `logs/stage1_verification.json` | ✅ Written |

---
*Auto-maintained by AI Agent | Stage 1 completed: 2026-09-09*

---

## Stage 2 Run Log — Data Contracts (Schemas)

**Stage**: 2 — Data Contracts (Schemas)  
**Status**: ✅ PASSED (12/12 tests)  
**Run date**: 2026-09-09  
**Deliverable**: `api/schemas.py`, `utils/common.py`, `stage2_verify.py`

### Test Results

| Test | Status | Detail |
|---|---|---|
| `schemas.strict_typing` | ✅ PASS | Verified 18 models, 0 untyped/Any fields |
| `schemas.ModelSpecResponse` | ✅ PASS | Validated against `config/model_spec.json` (5 media + 2 RF channels) |
| `schemas.ModelInfoResponse` | ✅ PASS | Validated against `loader.model_info()` (Loaded=True, 6 geos) |
| `schemas.RoiSummaryItem` | ✅ PASS | Validated 7 channel ROI items from `analysis.get_roi_summary()` |
| `schemas.ChannelContribution` | ✅ PASS | Validated 8 contribution items from `analysis.get_channel_contributions()` |
| `schemas.ResponseCurvePoint` | ✅ PASS | Validated 41 points for single curve and 7 channels for all curves |
| `schemas.BaselineVsMediaPoint` | ✅ PASS | Validated 104 time periods from `analysis.get_baseline_vs_media()` |
| `schemas.PredictiveAccuracyResponse` | ✅ PASS | Validated `analysis.get_predictive_accuracy()` (`r_squared`, `mape`, `wmape`) |
| `schemas.ChannelSpendSummary` | ✅ PASS | Validated 7 channels from `optimizer.get_current_spend_summary()` |
| `schemas.ScenarioRequest` | ✅ PASS | Validated defaults, custom parameters, and bound constraints |
| `schemas.WhatIfContracts` | ✅ PASS | Validated `WhatIfRequest` and `WhatIfResponse` simulation contracts |
| `schemas.OptimizationResponse` | ✅ PASS | Validated complete optimization result payload with uncertainty metrics |

### Key Contract Specifications

| Model | Purpose | Key Attributes |
|---|---|---|
| `ModelSpecResponse` | System metadata | `media_channel_names`, `rf_channel_names`, `geo_names`, `kpi_name` |
| `RoiSummaryItem` | Channel ROI table | `roi_mean`, `roi_ci_lower`, `roi_ci_upper`, `mroi_mean`, `incremental_outcome_mean` |
| `ChannelContribution` | Incremental contribution | `channel`, `mean`, `ci_lower`, `ci_upper`, `pct_of_total_mean` |
| `ResponseCurvePoint` | Saturation curve grid | `spend_multiplier`, `spend_pct_of_current`, `incremental_outcome_mean`, `ci_lower`, `ci_upper` |
| `BaselineVsMediaPoint` | Timeline decomposition | `date`, `baseline_mean`, `media_mean`, `actual`, CI bounds |
| `OptimizationResponse` | Budget allocation results | `total_optimized_spend`, `channels` (`current_spend`, `optimized_spend`, `lift_mean`, `confidence_flagged`) |
| `ScenarioRequest` | Frontend budget slider contract | `total_budget`, `spend_constraint_lower`, `spend_constraint_upper`, `channel_constraints`, `cost_overrides` |
| `WhatIfRequest` / `WhatIfResponse` | Scenario simulation contract | `channel_adjustments` (custom spend/CPM multipliers), projected outcome CI |

### Deliverables

| File | Status | Notes |
|---|---|---|
| `utils/__init__.py` | ✅ Created | Reusable utility exports |
| `utils/common.py` | ✅ Created | Modular functions for spec loading & validation without duplication |
| `api/schemas.py` | ✅ Created | 18 frozen Pydantic contracts (0 `Any` fields) |
| `stage2_verify.py` | ✅ Created | Automated contract verification suite |
| `logs/stage2_verification.json` | ✅ Written | Structured verification log |

---
*Auto-maintained by AI Agent | Stage 2 completed: 2026-09-09*

---

## Stage 3 Run Log — Precompute and Cache Layer

**Stage**: 3 — Precompute & Cache Layer  
**Status**: ✅ PASSED (12/12 tests)  
**Run date**: 2026-09-09  
**Model Hash**: `2565b33608e6c218`  
**Cache Location**: `meridian-app/cache/2565b33608e6c218/`

### Benchmark & Performance Results

| Operation / Endpoint | Latency (ms) | Target SLA | Status |
|---|---|---|---|
| `cache.read.contributions` | 0.10 ms | < 300 ms | ✅ PASS (>3000x faster) |
| `cache.read.roi_summary` | 0.12 ms | < 300 ms | ✅ PASS (>2500x faster) |
| `cache.read.response_curves_all` | 0.92 ms | < 300 ms | ✅ PASS (>300x faster) |
| `cache.read.baseline_vs_media` | 0.48 ms | < 300 ms | ✅ PASS (>600x faster) |
| `cache.read.accuracy` | 0.25 ms | < 300 ms | ✅ PASS |
| `cache.read.spend_summary` | 0.14 ms | < 300 ms | ✅ PASS |
| `cache.read.default_optimization` | 0.29 ms | < 300 ms | ✅ PASS (>1000x faster) |
| **Total Cold Dashboard Load** | **1.43 ms** | **< 300 ms** | ✅ **PASS (>200x faster than SLA)** |
| `cache.scenario_caching` | 0.40 ms | < 300 ms | ✅ PASS |
| `cache.hash_isolation_and_invalidation` | 0.90 ms | N/A | ✅ PASS |

### Precomputed Cache Artifacts Inventory

| Artifact File | Size | Contents |
|---|---|---|
| `manifest.json` | 1,178 B | Cache metadata, timestamps, item counts & generation durations |
| `contributions.json` | 1,632 B | 8 channels (5 media, 2 RF, 1 organic) |
| `roi_summary.json` | 3,910 B | 7 channels (ROI, mROI, spend, CPIK, credible bounds) |
| `response_curves.json` | 74,971 B | 7 channels × 41 spend points (0% to 200% historical spend) |
| `baseline_vs_media.json` | 34,704 B | 104 weekly time periods (baseline vs media-driven breakdown) |
| `predictive_accuracy.json` | 56 B | Goodness-of-fit indicators ($R^2$, MAPE, wMAPE) |
| `spend_summary.json` | 495 B | 7 channel historical spend aggregates |
| `default_optimization.json` | 5,678 B | Default allocation, lift estimates, confidence flags |
| `scenarios/a7ccfc1727e2e0b2.json` | 5,678 B | Precomputed default scenario keyed by deterministic parameter hash |

### Key Design & Invalidation Policy Highlights

1. **Deterministic Cache Namespace**:
   - Cache keyed to `sha256(saved_mmm.binpb)[:16]` (`2565b33608e6c218`).
   - When a new model is refit and saved, it automatically generates a new hash and directory namespace, guaranteeing zero stale numbers.
2. **Deterministic Scenario Caching**:
   - `ScenarioRequest` parameters are normalized and hashed into 16-character keys for instant cache hits upon repeated budget planner adjustments.
3. **Execution Off the Request Path**:
   - Heavy posterior arithmetic (128s offline) is moved completely to `jobs/precompute.py`.
   - API endpoints can serve default dashboard and precomputed scenarios in ~1 ms cold.

### Deliverables

| File | Status | Description |
|---|---|---|
| `services/cache.py` | ✅ Created & Verified | Read/write interface, model-hash isolation, scenario keying |
| `jobs/precompute.py` | ✅ Created & Verified | Offline batch precomputation script |
| `stage3_verify.py` | ✅ Created & Verified | End-to-end cache verification & latency benchmark suite |
| `logs/stage3_verification.json` | ✅ Written | Structured run results log |

---
*Auto-maintained by AI Agent | Stage 3 completed: 2026-09-09*

---

## Stage 4 Run Log — API Layer (FastAPI)

**Stage**: 4 — API Layer (FastAPI)  
**Status**: ✅ PASSED (12/12 tests)  
**Run date**: 2026-09-09  
**Application Entry**: `api/main.py`  
**OpenAPI Docs**: `/docs` (Swagger UI) & `/redoc`

### Test & Endpoint Verification Results

| Endpoint / Operation | HTTP Method | Response Status | Cache Header | Latency | Status |
|---|---|---|---|---|---|
| `OpenAPI Schema Generation` | Meta | 200 OK | — | 81.6 ms | ✅ PASS (all 12 routes registered) |
| `/health` | `GET` | 200 OK | — | 3.4 ms | ✅ PASS (`healthy`, `model_loaded=True`, `cache_ready=True`) |
| `/model/spec` | `GET` | 200 OK | `Cache-Control: public, max-age=3600` | 2.7 ms | ✅ PASS (5 media + 2 RF channels, 6 geos) |
| `/model/info` | `GET` | 200 OK | — | 2.2 ms | ✅ PASS (`Loaded=True`) |
| `/contributions` | `GET` | 200 OK | `X-Cache: HIT` | 2.2 ms | ✅ PASS (8 channels with posterior CI) |
| `/roi-summary` | `GET` | 200 OK | `X-Cache: HIT` | 3.5 ms | ✅ PASS (7 channels ROI/mROI/spend) |
| `/response-curves` | `GET` | 200 OK | `X-Cache: HIT` | — | ✅ PASS (all 7 channels curves) |
| `/response-curves/{channel}` | `GET` | 200 OK / 404 | `X-Cache: HIT` | — | ✅ PASS (41 points for Online_Video; 404 on invalid) |
| `/baseline-vs-media` | `GET` | 200 OK | `X-Cache: HIT` | 3.5 ms | ✅ PASS (104 weekly periods decomposition) |
| `/accuracy` | `GET` | 200 OK | — | 2.0 ms | ✅ PASS ($R^2$, MAPE, wMAPE contract) |
| `/spend-summary` | `GET` | 200 OK | — | 2.7 ms | ✅ PASS (7 channels historical spend) |
| `/optimize` | `POST` | 200 OK | `X-Cache: HIT` | **3.4 ms** | ✅ **PASS (instant serve from precomputed cache)** |
| `/whatif` | `POST` | 200 OK | — | 5.0 ms | ✅ PASS (7 channels simulation with CPM & spend deltas) |

### Key Architecture & Implementation Details

1. **Process Lifespan Management**:
   - Uses FastAPI's `lifespan` context manager to load the Meridian MMM model singleton once during app startup (`@loader.get_model()`).
   - Verifies cache namespace readiness immediately upon process start.
2. **HTTP Caching & Headers**:
   - `Cache-Control: public, max-age=3600` added to all deterministic `GET` analysis routes.
   - `X-Cache: HIT` / `X-Cache: MISS` header indicates cache provenance.
3. **Structured Request Logging**:
   - `/optimize` logs JSON event records with `cache_hit`, `scenario_key`, constraints, and budgets for observability.
   - `/whatif` logs scenario parameters and adjustments.
4. **CORS Enabled**:
   - Cross-Origin Resource Sharing enabled for modern frontend clients (Streamlit, React, Next.js).

### Deliverables

| File | Status | Description |
|---|---|---|
| `api/main.py` | ✅ Created & Verified | Full FastAPI HTTP serving application |
| `stage4_verify.py` | ✅ Created & Verified | End-to-end endpoint verification suite using TestClient |
| `logs/stage4_verification.json` | ✅ Written | Structured verification log |
| `requirements-serving.txt` | ✅ Updated | Added `fastapi==0.141.1`, `uvicorn==0.52.4`, `httpx==0.28.1` |

---
*Auto-maintained by AI Agent | Stage 4 completed: 2026-09-09*

---

## Stage 5 Run Log — Frontend MVP (Streamlit Only)

**Stage**: 5 — Frontend MVP (Streamlit Only)  
**Status**: ✅ PASSED (12/12 tests)  
**Run date**: 2026-09-09  
**Application Entry**: `frontend/app.py`  
**Stack**: Streamlit 1.63.0 + Plotly 7.0.0 (Python only)

### Test & Component Verification Results

| Test / Component | Status | Detail | Latency |
|---|---|---|---|
| `client.metadata_and_spec` | ✅ PASS | Health check, 6 geos, 7 paid channels verified | 438.2 ms |
| `analysis.roi_summary_ci_integrity` | ✅ PASS | 7 channels verified; `roi_ci_lower <= roi_mean <= roi_ci_upper` invariant held | 30.4 ms |
| `analysis.contributions_ci_integrity` | ✅ PASS | 8 channels (incl organic) verified with strict credible interval bounds | 22.8 ms |
| `analysis.response_curves_ci_integrity` | ✅ PASS | 7 channels × 41 curve points verified with valid uncertainty ribbons | 32.8 ms |
| `analysis.timeline_decomposition` | ✅ PASS | 104 weekly points; baseline and media decomposition bounds verified | 25.8 ms |
| `analysis.spend_summary` | ✅ PASS | 7 channels; Total historical spend = ₹4,185,397 | 11.3 ms |
| `optimizer.post_optimize_execution` | ✅ PASS | Budget allocation solved; lift credible intervals & confidence flags verified | 12.4 ms |
| `optimizer.post_whatif_execution` | ✅ PASS | Simulated 7 channels; Projected outcome: 152.5M [72.1M – 237.9M] | 14.6 ms |
| `plotly.roi_error_bars_generation` | ✅ PASS | Verified asymmetric error bars trace compiled for all channels | 64.1 ms |
| `plotly.response_curve_ribbon_generation` | ✅ PASS | Verified shaded uncertainty ribbon (`tonexty`) and current spend marker | 17.3 ms |
| `plotly.timeline_decomposition_generation` | ✅ PASS | Verified time-series baseline vs. media traces compiled | 13.1 ms |
| `frontend.modules_import_integrity` | ✅ PASS | All 8 frontend modules imported and compiled cleanly with zero errors | < 1 ms |

### Key Architecture & UX Implementation Details

1. **Strict Posterior Uncertainty Semantics (Task 3)**:
   - Every single chart rendering posterior metrics displays credible intervals:
     - Channel ROI & mROI: Asymmetric error bars (`roi_ci_lower` to `roi_ci_upper`).
     - Channel Contribution: Asymmetric error bars on incremental bookings.
     - Saturation Curves: Semi-transparent shaded ribbon (`ci_lower` to `ci_upper`) with historical spend ($1.0\times$) diamond marker.
     - Timeline Decomposition: Weekly baseline and media uncertainty bands across all 104 weeks.
     - Budget Planner: Net lift and per-channel lift error bars.
2. **Dual-Mode API Client**:
   - `MeridianApiClient` automatically attempts connection to the external FastAPI server (`http://127.0.0.1:8000`), and transparently falls back to an in-process `fastapi.testclient.TestClient(app)` if the external process is offline, guaranteeing seamless execution.
   - Streamlit caching via `@st.cache_data(ttl=300)` ensures sub-10ms page transitions.
3. **Interactive Budget Optimizer**:
   - Offers Fixed Budget (target spend) and Flexible Budget (target ROI/mROI) allocation modes.
   - Sliders for lower/upper spend deviation constraints ($\pm 5\%$ to $\pm 100\%$) with per-channel override capabilities.
   - Grouped before-and-after spend comparison chart and expected lift chart.
   - Confidence-gating alerts flag channels where the lift CI crosses zero.
   - One-click CSV export of optimized budget allocations.
4. **What-If Scenario Simulator**:
   - Custom spend multipliers and CPM cost-per-unit inflation multipliers per channel.
   - Instant response interpolation from precomputed saturation profiles.
5. **Modern Mix Studio Styling**:
   - Custom CSS styling (`frontend/styles.py`) with metric cards, typography, status badges, and cohesive channel color palettes.

### Deliverables

| File | Status | Description |
|---|---|---|
| `frontend/styles.py` | ✅ Created | Theme tokens, typography, channel palettes, and custom CSS |
| `frontend/api_client.py` | ✅ Created & Verified | HTTP client with automatic fallback and data transformation |
| `frontend/components/kpi_cards.py` | ✅ Created & Verified | Executive summary KPI metric cards |
| `frontend/components/channel_performance.py` | ✅ Created & Verified | ROI and Contribution charts with 90% CI error bars + data table |
| `frontend/components/curve_explorer.py` | ✅ Created & Verified | Saturation curves with shaded uncertainty bands and markers |
| `frontend/components/budget_planner.py` | ✅ Created & Verified | Budget planner with spend sliders, constraints, and lift charts |
| `frontend/components/timeline_view.py` | ✅ Created & Verified | 104-week baseline vs. media decomposition with error bands |
| `frontend/components/__init__.py` | ✅ Created | Component module package |
| `frontend/app.py` | ✅ Created & Verified | Main Streamlit dashboard application |
| `stage5_verify.py` | ✅ Created & Verified | Automated end-to-end verification suite (12/12 passed) |
| `logs/stage5_verification.json` | ✅ Written | Structured test results log |
| `requirements-serving.txt` | ✅ Updated | Added `streamlit==1.63.0`, `plotly==7.0.0` |

---
*Auto-maintained by AI Agent | Stage 5 completed: 2026-09-09*
