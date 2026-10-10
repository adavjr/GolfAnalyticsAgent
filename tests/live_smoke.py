"""Explicit, credentialed smoke test: writes 90 mock shots and makes paid API calls.

Run from the repository root: .venv/bin/python -m tests.live_smoke is not required;
use .venv/bin/python tests/live_smoke.py after installing the package.
Credentials and raw service errors are never printed or saved.
"""
import json
import logging
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from golf_agent.agent import ToolRuntime, run_agent
from golf_agent.data import read_csv
from golf_agent.ml import Predictor
from golf_agent.store import AstraStore


def main():
    output = ROOT / "artifacts" / "live_test"
    output.mkdir(parents=True, exist_ok=True)
    report = {"checks": {}, "completed": False}
    stage = "configuration"
    started = time.monotonic()
    try:
        needed = ["OPENAI_API_KEY", "OPENAI_MODEL", "ASTRA_DB_API_ENDPOINT",
                  "ASTRA_DB_APPLICATION_TOKEN"]
        missing = [key for key in needed if not os.environ.get(key)]
        if missing:
            report["missing_variables"] = missing
            raise ValueError("Missing configuration")
        report["checks"][stage] = "passed"
        stage = "astra_ingestion"
        store = AstraStore(create=True)
        fixture = read_csv(ROOT / "data" / "trackman_mock.csv").groupby("Player_ID").tail(30)
        result = store.ingest(fixture)
        report["checks"][stage] = result
        print(f"Astra ingestion passed: {result['upserted']} mock shots", flush=True)
        stage = "astra_query"
        history = store.history("Player A", 30)
        assert len(history) == 30 and history.Player_ID.eq("Player A").all()
        report["checks"][stage] = {"shots": len(history)}
        print("Astra history query passed", flush=True)
        stage = "openai_agent"
        calls = []
        class RecordingRuntime(ToolRuntime):
            def dispatch(self, name, arguments):
                result = super().dispatch(name, arguments)
                calls.append({"name": name, "result": result})
                print(f"Tool completed: {name}", flush=True)
                return result
        runtime = RecordingRuntime(store, Predictor(ROOT / "artifacts"), output)
        brief = run_agent(
            "Analyze Player A's latest 30 driver shots. Tell me whether a stiffer shaft "
            "would reduce spin, classify the latest shot and generate a spin/carry chart.",
            runtime, os.environ["OPENAI_MODEL"])
        required = {"query_player_history", "predict_swing_tendency", "generate_performance_chart"}
        assert required <= {call["name"] for call in calls}, "Missing required tool calls"
        assert brief["player_id"] == "Player A"
        assert brief["equipment_assessment"]["can_estimate_shaft_effect"] is False
        charts = {call["result"]["path"] for call in calls if call["name"] == "generate_performance_chart"}
        assert brief["charts"] and set(brief["charts"]) <= charts, "Unsupported chart references"
        assert all(Path(path).is_file() for path in brief["charts"])
        (output / "brief.json").write_text(json.dumps(brief, indent=2))
        (output / "tool_trace.json").write_text(json.dumps(calls, indent=2))
        report["checks"][stage] = {"tools_completed": [call["name"] for call in calls]}
        report["completed"] = True
        print("Live agent analysis passed; brief and charts saved", flush=True)
    except Exception as exc:
        report["failure"] = {"stage": stage, "type": type(exc).__name__}
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            report["failure"]["http_status"] = status
        # Record only structured error codes, never arbitrary messages or URLs.
        code = getattr(exc, "code", None)
        if isinstance(code, str) and len(code) < 80 and code.replace("_", "").isalnum():
            report["failure"]["code"] = code
        print(json.dumps(report["failure"]), flush=True)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (output / "report.json").write_text(json.dumps(report, indent=2))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
