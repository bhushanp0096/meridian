"""
Stage 2 — Schema & Data Contracts Verification Script
======================================================
Validates that all Stage 1 service outputs and request contracts
conform strictly to the Pydantic schemas in api/schemas.py.

Verification checks:
1. No `Any` or untyped fields in api/schemas.py models.
2. Model spec schema validates against config/model_spec.json.
3. Model info schema validates against loader.model_info().
4. ROI summary schema validates against analysis.get_roi_summary().
5. Contribution schema validates against analysis.get_channel_contributions().
6. Response curve schemas validate against analysis.get_response_curves().
7. Baseline vs media schema validates against analysis.get_baseline_vs_media().
8. Accuracy schema validates against analysis.get_predictive_accuracy().
9. Spend summary schema validates against optimizer.get_current_spend_summary().
10. ScenarioRequest & WhatIfRequest validate correctly on valid and invalid inputs.
11. OptimizationResponse validates correctly against expected structure.

Outputs:
- logs/stage2_verification.json
"""

import os
import sys
import json
import time
import inspect
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# Force CPU mode for verification
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

APP_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))
sys.path.insert(0, str(APP_DIR))

from utils.common import load_model_spec, is_json_serializable  # noqa: E402
from api import schemas  # noqa: E402

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "stage2_verification.json"


def check_no_any_types() -> tuple[bool, str]:
    """Check that no Pydantic model in api.schemas contains Any or untyped fields."""
    untyped_or_any = []
    model_classes = [
        cls for name, cls in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(cls, schemas.BaseModel) and cls is not schemas.BaseModel and cls is not schemas.StrictBaseModel
    ]

    for model in model_classes:
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            if annotation is None:
                untyped_or_any.append(f"{model.__name__}.{field_name} has no type annotation")
            elif annotation is Any:
                untyped_or_any.append(f"{model.__name__}.{field_name} uses Any type")

    if untyped_or_any:
        return False, "; ".join(untyped_or_any)
    return True, f"Verified {len(model_classes)} models, 0 untyped/Any fields"


def run_tests() -> list[dict[str, Any]]:
    results = []

    def record(name: str, status: str, detail: str = "", duration_s: float = 0.0):
        results.append({
            "test": name,
            "status": status,
            "detail": detail,
            "duration_seconds": round(duration_s, 3),
        })
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}" + (f" — {detail}" if detail else ""))

    # 1. Inspect schemas for Any / untyped fields
    t0 = time.time()
    ok, msg = check_no_any_types()
    record("schemas.strict_typing", "PASS" if ok else "FAIL", msg, time.time() - t0)

    # 2. Model Spec validation
    t0 = time.time()
    try:
        raw_spec = load_model_spec()
        spec_model = schemas.ModelSpecResponse.model_validate(raw_spec)
        record(
            "schemas.ModelSpecResponse",
            "PASS",
            f"Validated {len(spec_model.media_channel_names)} media + {len(spec_model.rf_channel_names)} RF channels",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.ModelSpecResponse", "FAIL", str(e), time.time() - t0)

    # 3. Model Info validation
    t0 = time.time()
    try:
        from services.loader import model_info, get_model  # noqa: PLC0415
        get_model()  # ensure loaded
        info = model_info()
        info_model = schemas.ModelInfoResponse.model_validate(info)
        record(
            "schemas.ModelInfoResponse",
            "PASS",
            f"Loaded={info_model.loaded}, {info_model.n_geos} geos",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.ModelInfoResponse", "FAIL", str(e), time.time() - t0)

    # 4. ROI Summary validation
    t0 = time.time()
    try:
        from services.analysis import get_roi_summary  # noqa: PLC0415
        raw_roi = get_roi_summary()
        validated_roi = [schemas.RoiSummaryItem.model_validate(item) for item in raw_roi]
        record(
            "schemas.RoiSummaryItem",
            "PASS",
            f"Validated {len(validated_roi)} channel ROI items",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.RoiSummaryItem", "FAIL", str(e), time.time() - t0)

    # 5. Channel Contributions validation
    t0 = time.time()
    try:
        from services.analysis import get_channel_contributions  # noqa: PLC0415
        raw_contrib = get_channel_contributions()
        validated_contrib = [schemas.ChannelContribution.model_validate(item) for item in raw_contrib]
        record(
            "schemas.ChannelContribution",
            "PASS",
            f"Validated {len(validated_contrib)} contribution items",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.ChannelContribution", "FAIL", str(e), time.time() - t0)

    # 6. Response Curves validation
    t0 = time.time()
    try:
        from services.analysis import get_response_curves, get_response_curves_all_channels  # noqa: PLC0415
        raw_curve = get_response_curves("Online_Video")
        validated_curve = [schemas.ResponseCurvePoint.model_validate(item) for item in raw_curve]
        
        all_curves = get_response_curves_all_channels()
        validated_all = {
            ch: [schemas.ResponseCurvePoint.model_validate(p) for p in pts]
            for ch, pts in all_curves.items()
        }
        record(
            "schemas.ResponseCurvePoint",
            "PASS",
            f"Single curve: {len(validated_curve)} pts; All curves: {len(validated_all)} channels",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.ResponseCurvePoint", "FAIL", str(e), time.time() - t0)

    # 7. Baseline vs Media validation
    t0 = time.time()
    try:
        from services.analysis import get_baseline_vs_media  # noqa: PLC0415
        raw_baseline = get_baseline_vs_media()
        validated_baseline = [schemas.BaselineVsMediaPoint.model_validate(item) for item in raw_baseline]
        record(
            "schemas.BaselineVsMediaPoint",
            "PASS",
            f"Validated {len(validated_baseline)} time periods",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.BaselineVsMediaPoint", "FAIL", str(e), time.time() - t0)

    # 8. Predictive Accuracy validation
    t0 = time.time()
    try:
        from services.analysis import get_predictive_accuracy  # noqa: PLC0415
        raw_acc = get_predictive_accuracy()
        validated_acc = schemas.PredictiveAccuracyResponse.model_validate(raw_acc)
        record(
            "schemas.PredictiveAccuracyResponse",
            "PASS",
            f"r_squared={validated_acc.r_squared}, mape={validated_acc.mape}",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.PredictiveAccuracyResponse", "FAIL", str(e), time.time() - t0)

    # 9. Optimizer Spend Summary validation
    t0 = time.time()
    try:
        from services.optimizer import get_current_spend_summary  # noqa: PLC0415
        raw_spends = get_current_spend_summary()
        validated_spends = [schemas.ChannelSpendSummary.model_validate(item) for item in raw_spends]
        record(
            "schemas.ChannelSpendSummary",
            "PASS",
            f"Validated {len(validated_spends)} channel spend items",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.ChannelSpendSummary", "FAIL", str(e), time.time() - t0)

    # 10. ScenarioRequest validation & constraint checking
    t0 = time.time()
    try:
        # Valid default
        req_default = schemas.ScenarioRequest()
        assert req_default.spend_constraint_lower == 0.3

        # Valid custom payload
        custom_payload = {
            "total_budget": 5000000.0,
            "spend_constraint_lower": 0.2,
            "spend_constraint_upper": 0.5,
            "channel_constraints": {
                "Online_Video": (10000.0, 500000.0)
            },
            "cost_overrides": {
                "Display": 1.15
            },
            "ci_level": 0.95
        }
        req_custom = schemas.ScenarioRequest.model_validate(custom_payload)
        assert req_custom.total_budget == 5000000.0
        assert req_custom.cost_overrides["Display"] == 1.15

        # Invalid constraints should raise ValidationError
        invalid_caught = False
        try:
            schemas.ScenarioRequest(spend_constraint_lower=-0.5)
        except Exception:
            invalid_caught = True

        assert invalid_caught, "Validation should have caught negative constraint lower bound"

        record(
            "schemas.ScenarioRequest",
            "PASS",
            "Validated default, custom parameters, and bound constraints",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.ScenarioRequest", "FAIL", str(e), time.time() - t0)

    # 11. WhatIf contracts validation
    t0 = time.time()
    try:
        whatif_req = schemas.WhatIfRequest(
            scenario_name="Holiday CPM Spike",
            channel_adjustments={
                "Online_Video": schemas.WhatIfChannelAdjustment(cpm_multiplier=1.25, spend_multiplier=1.1)
            }
        )
        assert whatif_req.channel_adjustments["Online_Video"].cpm_multiplier == 1.25

        whatif_resp = schemas.WhatIfResponse(
            scenario_name=whatif_req.scenario_name,
            total_baseline_spend=4000000.0,
            total_scenario_spend=4400000.0,
            total_outcome_mean=120000.0,
            total_outcome_ci_lower=105000.0,
            total_outcome_ci_upper=135000.0,
            channels=[
                schemas.WhatIfChannelResult(
                    channel="Online_Video",
                    baseline_spend=500000.0,
                    scenario_spend=550000.0,
                    spend_delta=50000.0,
                    spend_delta_pct=10.0,
                    cpm_multiplier=1.25,
                    expected_outcome_mean=25000.0,
                    ci_lower=22000.0,
                    ci_upper=28000.0,
                )
            ]
        )
        assert len(whatif_resp.channels) == 1
        record(
            "schemas.WhatIfContracts",
            "PASS",
            "Validated WhatIfRequest and WhatIfResponse schema contracts",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.WhatIfContracts", "FAIL", str(e), time.time() - t0)

    # 12. Full OptimizationResponse schema validation
    t0 = time.time()
    try:
        sample_opt_result = {
            "status": "ok",
            "budget_type": "fixed",
            "total_budget": 4185397.0,
            "total_optimized_spend": 4185397.0,
            "spend_constraint_lower": 0.3,
            "spend_constraint_upper": 0.3,
            "use_optimal_frequency": True,
            "ci_level": 0.9,
            "n_flagged_channels": 0,
            "channels": [
                {
                    "channel": "Online_Video",
                    "ci_level": 0.9,
                    "current_spend": 600000.0,
                    "optimized_spend": 700000.0,
                    "spend_delta": 100000.0,
                    "spend_delta_pct": 16.67,
                    "roi_mean": 2.5,
                    "roi_ci_lower": 2.1,
                    "roi_ci_upper": 2.9,
                    "mroi_mean": 1.8,
                    "mroi_ci_lower": 1.4,
                    "mroi_ci_upper": 2.2,
                    "incremental_outcome_mean": 1750000.0,
                    "incremental_outcome_ci_lower": 1470000.0,
                    "incremental_outcome_ci_upper": 2030000.0,
                    "lift_mean": 250000.0,
                    "lift_ci_lower": 100000.0,
                    "lift_ci_upper": 400000.0,
                    "confidence_flagged": False,
                }
            ]
        }
        opt_resp = schemas.OptimizationResponse.model_validate(sample_opt_result)
        assert opt_resp.channels[0].channel == "Online_Video"
        record(
            "schemas.OptimizationResponse",
            "PASS",
            "Validated full OptimizationResponse contract with per-channel uncertainty",
            time.time() - t0,
        )
    except Exception as e:
        record("schemas.OptimizationResponse", "FAIL", str(e), time.time() - t0)

    return results


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 2: Schema Contracts Verification")
    print("=" * 60)

    start_utc = datetime.now(timezone.utc).isoformat()
    test_results = run_tests()

    n_pass = sum(1 for t in test_results if t["status"] == "PASS")
    n_fail = sum(1 for t in test_results if t["status"] == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    log_payload = {
        "stage": "Stage 2 — Data Contracts (Schemas) Verification",
        "timestamp_utc": start_utc,
        "overall_status": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "tests": test_results,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }

    LOG_PATH.write_text(json.dumps(log_payload, indent=2))

    print("=" * 60)
    print(f"  Overall: {overall} ({n_pass} PASS / {n_fail} FAIL)")
    print(f"  Log → {LOG_PATH}")
    print("=" * 60)

    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
