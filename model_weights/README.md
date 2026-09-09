# Model Weights Directory

This directory stores fitted Google Meridian MMM binary model weights (`.binpb`) for the serving application.

### Expected Files
- `meridian_model.binpb` (or `saved_mmm.binpb`): Fitted Meridian serialized model artifact containing model context, posterior draws, and inference data.

### Loading Hierarchy
All serving scripts and backend services (`services/loader.py`, `smoke_test.py`, `extract_model_spec.py`, `jobs/precompute.py`, `api/main.py`) check this folder first before any fallback directories.
