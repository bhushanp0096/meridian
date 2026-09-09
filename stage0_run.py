"""
Stage 0 — Full Orchestrator
==============================
Runs all Stage 0 tasks in sequence:
  1. Pin serving requirements
  2. Smoke test (model loads cleanly)
  3. Extract model_spec.json

Writes:
  logs/stage0_run_log.json  — combined structured run log for Stage 0

Usage
-----
  .venv/bin/python3 Meridian_files/meridian_application/meridian-app/stage0_run.py
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))
VENV_PYTHON = WORKSPACE / ".venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

TASKS = [
    ("pin_requirements",  APP_DIR / "pin_requirements.py"),
    ("smoke_test",        APP_DIR / "smoke_test.py"),
    ("spec_extraction",   APP_DIR / "extract_model_spec.py"),
]


def run_task(name: str, script: Path) -> dict:
    """Run a single Stage 0 task script as a subprocess."""
    print(f"\n{'─' * 60}")
    print(f"  TASK: {name}")
    print(f"  Script: {script}")
    print(f"{'─' * 60}")

    start = datetime.now(timezone.utc)
    result = subprocess.run(
        [PYTHON, str(script)],
        capture_output=False,   # let stdout/stderr pass through
        text=True,
        cwd=str(WORKSPACE),
    )
    end = datetime.now(timezone.utc)
    duration_s = (end - start).total_seconds()

    status = "PASS" if result.returncode == 0 else "FAIL"
    return {
        "task": name,
        "script": str(script),
        "status": status,
        "return_code": result.returncode,
        "duration_seconds": round(duration_s, 2),
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
    }


def load_sub_log(path: Path) -> dict:
    """Load a sub-task log if present."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def main():
    print("=" * 60)
    print("  Meridian Serving App — Stage 0: Full Run")
    print("=" * 60)
    print(f"  Workspace : {WORKSPACE}")
    print(f"  Python    : {PYTHON}")
    print(f"  Started   : {datetime.now(timezone.utc).isoformat()}")

    stage_log = {
        "stage": "Stage 0 — Environment Validation & Artifact Inventory",
        "stage_started_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WORKSPACE),
        "python_bin": PYTHON,
        "tasks": [],
        "overall_status": "PENDING",
        "stage_completed_utc": None,
    }

    all_passed = True
    for name, script in TASKS:
        task_result = run_task(name, script)
        stage_log["tasks"].append(task_result)
        if task_result["status"] != "PASS":
            all_passed = False
            print(f"\n❌ Task '{name}' FAILED (exit code {task_result['return_code']})")
            print("   Aborting Stage 0 — fix the above error before proceeding.")
            break

    # Attach sub-logs for rich traceability
    stage_log["sub_logs"] = {
        "pin_requirements": load_sub_log(LOG_DIR / "stage0_pin_requirements.json"),
        "smoke_test":       load_sub_log(LOG_DIR / "stage0_smoke_test.json"),
        "spec_extraction":  load_sub_log(LOG_DIR / "stage0_spec_extract.json"),
    }

    stage_log["overall_status"] = "PASS" if all_passed else "FAIL"
    stage_log["stage_completed_utc"] = datetime.now(timezone.utc).isoformat()

    combined_log_path = LOG_DIR / "stage0_run_log.json"
    combined_log_path.write_text(json.dumps(stage_log, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Stage 0 overall status: {stage_log['overall_status']}")
    print(f"  Combined log → {combined_log_path}")
    print(f"{'=' * 60}")

    if not all_passed:
        sys.exit(1)

    print("\n✅ Stage 0 COMPLETE — environment validated, model spec extracted.")
    print("   → Next: Stage 1 — Analysis service layer")


if __name__ == "__main__":
    main()
