# Meridian Marketing Mix Modeling (MMM) Serving Platform

High-performance serving architecture and API layer for Google Meridian Marketing Mix Models.

## Features

- **FastAPI Serving Engine**: Sub-5ms response times leveraging precomputed posterior inference and scenario simulation.
- **Scenario Simulator & Optimizer**: Real-time `/optimize` (budget reallocation) and `/whatif` spend simulation across media & RF channels.
- **Two-Tier Precomputation & Caching**: Precomputed posterior summaries, response curves, and marginal ROI curves with instant cache retrieval.
- **Model Decoupled**: Can serve cached artifacts and scenario approximations without loading heavy probabilistic model binaries into memory at runtime.

## Architecture

```
meridian-app/
├── api/                  # FastAPI app and route definitions
│   ├── main.py          # App lifespan, routes, CORS & cache headers
│   └── schemas.py       # Pydantic request/response models
├── services/             # Core business logic
│   ├── loader.py        # Model binary loader & cache manager
│   ├── optimizer.py     # Budget optimization & what-if simulator
│   └── reporter.py      # Summary metrics and decomposition reports
├── jobs/                 # Batch processing & cache generation
│   └── precompute.py    # Offline posterior calculation & cache warmer
├── cache/                # Model-keyed precomputed JSON artifacts
├── config/               # Model specification & channel configurations
├── logs/                 # Verification logs & stage execution records
├── requirements-serving.txt # Production serving dependencies
└── stage*.py             # Automated end-to-end verification suites
```

## Quickstart

### 1. Installation

```bash
pip install -r requirements-serving.txt
```

### 2. Run the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 3. Key Endpoints

- `GET /health` - Health check & cache status
- `GET /model/spec` - Channel names, geo breakdown, dates, and KPI
- `GET /contributions` - Channel incremental contributions with credible intervals
- `GET /roi-summary` - Historical ROI, mROI, and spend by channel
- `GET /response-curves` - Saturation response curves
- `POST /optimize` - Optimal budget allocation
- `POST /whatif` - What-if scenario simulation
