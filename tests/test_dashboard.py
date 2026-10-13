"""Smoke tests: each dashboard page renders without exceptions on fixture data."""
import json
import os
import sys

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.monitoring import EventLogger  # noqa: E402

DASHBOARD = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")


@pytest.fixture(autouse=True)
def dashboard_on_path(monkeypatch):
    monkeypatch.syspath_prepend(DASHBOARD)


def run_page(name, **state):
    at = AppTest.from_file(os.path.join(DASHBOARD, "pages", f"{name}.py"), default_timeout=30)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def event_log(tmp_path):
    log = EventLogger(str(tmp_path / "events.jsonl"))
    log.log("interpret", field_name="age", rule="At least 18", source="t5",
            fallback_reason=None, t5_confidence=0.98, total_latency_ms=700.0,
            confidence_threshold=0.7)
    log.log("interpret", field_name="age", rule="Over 18", source="llm",
            fallback_reason="low_confidence", t5_confidence=0.4, t5_output={"minimum": 18},
            llm_output={"exclusiveMinimum": 18}, llm_latency_ms=1300.0,
            llm_usage={"cost": 0.00004}, total_latency_ms=2000.0, confidence_threshold=0.7)
    log.log("training_example", rule="Over 18", output={"exclusiveMinimum": 18},
            source="llm")
    return log.path


class TestBatchPage:

    def test_recorded_reports(self):
        at = run_page("batch", reports_dir="docs/experiments")
        assert at.title[0].value == "Batch evaluation"
        assert len(at.get("plotly_chart")) >= 2

    def test_reports_with_cases(self, tmp_path):
        cases = [{"input": f"rule {i}", "expected": '{"minimum":1}', "actual": "{}",
                  "constraint_type": "minimum", "semantic_match": i % 2 == 0,
                  "exact_match": i % 2 == 0, "schema_valid": True, "latency_ms": 10.0 + i,
                  "source": "t5"} for i in range(6)]
        for name in ("X", "Y"):
            (tmp_path / f"expA_2026-01-01_{name}.json").write_text(json.dumps({
                "metadata": {"dataset_sha256": "s"}, "condition": name, "total_tests": 6,
                "semantic_match_rate": 0.5, "by_constraint_type": {}, "failures": [],
                "cases": cases}))
        at = run_page("batch", reports_dir=str(tmp_path))
        assert len(at.dataframe) >= 1

    def test_empty_folder(self, tmp_path):
        at = run_page("batch", reports_dir=str(tmp_path))
        assert at.info


class TestLivePage:

    def test_with_events(self, event_log):
        at = run_page("live", event_log=event_log)
        values = {m.label: m.value for m in at.metric}
        assert values["Rules interpreted"] == "2"
        assert values["Fallback rate"] == "50.0%"
        assert values["LLM spend"] == "$0.00004"
        assert len(at.dataframe) == 1  # failures stream

    def test_no_events(self, tmp_path):
        at = run_page("live", event_log=str(tmp_path / "missing.jsonl"))
        assert at.info


class TestTrainingPage:

    def test_with_events(self, event_log, tmp_path):
        at = run_page("training", event_log=event_log,
                      collected_path=str(tmp_path / "none.json"))
        values = {m.label: m.value for m in at.metric}
        assert values["Collected (event log)"] == "1"


def test_app_entry_point_runs_default_page():
    at = AppTest.from_file(os.path.join(DASHBOARD, "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Batch evaluation"
