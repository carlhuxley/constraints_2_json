"""Smoke tests: each dashboard page renders without exceptions on fixture data."""
import json
import os
import subprocess
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
    at = AppTest.from_file(os.path.join(DASHBOARD, "views", f"{name}.py"), default_timeout=30)
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


@pytest.mark.parametrize("page,title", [
    ("views/live.py", "Live monitoring"),
    ("views/training.py", "Training data collection"),
])
def test_navigating_from_the_app_to_each_page(page, title, event_log, monkeypatch):
    monkeypatch.setenv("C2J_EVENT_LOG", event_log)
    at = AppTest.from_file(os.path.join(DASHBOARD, "app.py"), default_timeout=30)
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == title


def test_no_legacy_pages_folder():
    """A dashboard/pages/ folder makes Streamlit run pages without app.py (legacy
    multipage mode), so `src` imports fail on every page but the first."""
    assert not os.path.exists(os.path.join(DASHBOARD, "pages"))


@pytest.mark.parametrize("page", ["batch", "live", "training"])
def test_page_imports_work_without_repo_root_on_path(page, tmp_path):
    """Streamlit only puts dashboard/ on sys.path; each page must find src itself."""
    repo = os.path.dirname(DASHBOARD)
    code = (
        "import runpy, sys\n"
        f"sys.path = [{DASHBOARD!r}] + [p for p in sys.path if p not in ('', {repo!r})]\n"
        f"runpy.run_path({os.path.join(DASHBOARD, 'views', page + '.py')!r})\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert "ModuleNotFoundError" not in result.stderr, result.stderr


class TestStreamSwitch:

    @pytest.fixture
    def simulated_stream(self, monkeypatch, event_log, tmp_path):
        """Point 'Simulated' at a fixture log, not the real logs/ folder."""
        from src import monitoring
        paths = (event_log, str(tmp_path / "sim_collected.json"))
        monkeypatch.setitem(monitoring.STREAMS, "Simulated", paths)
        monkeypatch.delenv("C2J_EVENT_LOG", raising=False)
        monkeypatch.delenv("C2J_COLLECTED", raising=False)
        return paths

    def app(self):
        return AppTest.from_file(os.path.join(DASHBOARD, "app.py"), default_timeout=30)

    def test_defaults_to_real(self, simulated_stream):
        from src.monitoring import STREAMS
        at = self.app()
        at.run()
        assert at.radio(key="stream").value == "Real"
        assert at.session_state.event_log == STREAMS["Real"][0]
        assert at.session_state.collected_path == STREAMS["Real"][1]

    def test_switching_to_simulated_points_live_page_at_its_log(self, simulated_stream):
        at = self.app()
        at.run()
        at.radio(key="stream").set_value("Simulated").run()
        assert (at.session_state.event_log, at.session_state.collected_path) == simulated_stream

        at.switch_page("views/live.py")
        at.run()
        assert not at.exception, at.exception
        assert {m.label: m.value for m in at.metric}["Rules interpreted"] == "2"

    def test_switching_back_restores_real_paths(self, simulated_stream):
        from src.monitoring import STREAMS
        at = self.app()
        at.run()
        at.radio(key="stream").set_value("Simulated").run()
        at.radio(key="stream").set_value("Real").run()
        assert at.session_state.event_log == STREAMS["Real"][0]

    def test_env_var_starts_on_simulated(self, simulated_stream, monkeypatch):
        monkeypatch.setenv("C2J_EVENT_LOG", simulated_stream[0])
        at = self.app()
        at.run()
        assert at.radio(key="stream").value == "Simulated"
