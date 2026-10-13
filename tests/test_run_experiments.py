"""Tests for scripts/run_experiments.py using fake interpreters (no model or API calls)."""
import json

import pytest

from scripts import run_experiments as rx
from src.model_evaluator import EvalTestCase


CASES = [
    EvalTestCase(input="At least 18", expected_output={"minimum": 18},
                 field_type="integer", constraint_type="minimum"),
    EvalTestCase(input="At most 50 chars", expected_output={"maxLength": 50},
                 field_type="string", constraint_type="maxLength"),
]


class FakeHybrid:
    """Answers minimum rules 'from T5' and everything else 'from the LLM'."""

    def __init__(self):
        self.last_source = None
        self.seen_types = []

    def interpret(self, field_name, field_type, business_rule):
        self.seen_types.append(field_type)
        if field_type == "integer":
            self.last_source = "t5"
            return {"minimum": 18}
        self.last_source = "llm"
        return {"maxLength": 99}


class TestRunCondition:

    def test_field_type_set_per_case(self):
        hybrid = FakeHybrid()
        adapter = rx.FieldTypeAdapter(hybrid.interpret, source_fn=lambda: hybrid.last_source)

        rx.run_condition("Hybrid", adapter, CASES)

        assert hybrid.seen_types == ["integer", "string"]

    def test_records_routing_and_fallback_rate(self):
        hybrid = FakeHybrid()
        adapter = rx.FieldTypeAdapter(hybrid.interpret, source_fn=lambda: hybrid.last_source)

        report = rx.build_report(rx.run_condition("Hybrid", adapter, CASES))

        assert report["routing"] == {"llm": 1, "t5": 1}
        assert report["fallback_rate"] == 0.5
        assert report["semantic_match_rate"] == 0.5

    def test_plain_interpreter_has_no_routing(self):
        class FakeT5:
            def interpret(self, business_rule):
                return {"minimum": 18}

        report = rx.build_report(rx.run_condition("T5-orig", FakeT5(), CASES))

        assert "fallback_rate" not in report
        assert report["by_constraint_type"]["minimum"]["semantic_match_rate"] == 1.0
        assert report["by_constraint_type"]["maxLength"]["semantic_match_rate"] == 0.0
        assert report["p99_latency_ms"] >= report["median_latency_ms"]

    def test_report_has_one_case_row_per_test(self):
        hybrid = FakeHybrid()
        adapter = rx.FieldTypeAdapter(hybrid.interpret, source_fn=lambda: hybrid.last_source)

        report = rx.build_report(rx.run_condition("Hybrid", adapter, CASES))

        first, second = report["cases"]
        assert first["input"] == "At least 18"
        assert first["semantic_match"] is True and first["source"] == "t5"
        assert second["constraint_type"] == "maxLength"
        assert second["semantic_match"] is False and second["source"] == "llm"
        assert second["actual"] == '{"maxLength":99}'
        assert all(c["latency_ms"] >= 0 for c in report["cases"])


class TestWriteReports:

    def test_shared_condition_written_for_each_experiment(self, tmp_path):
        reports = {
            name: {"total_tests": 2, "semantic_match_rate": 1.0, "failures": []}
            for name in ("B0", "T5-orig", "Hybrid", "T5-retrained")
        }
        metadata = {"date": "2026-10-11", "git_sha": "abc"}

        written = rx.write_reports(["A", "B"], reports, metadata, str(tmp_path))

        names = sorted(p.split("/")[-1] for p in written)
        assert "expA_2026-10-11_T5-orig.json" in names
        assert "expB_2026-10-11_T5-orig.json" in names
        summary = json.loads((tmp_path / "expB_2026-10-11_summary.json").read_text())
        assert set(summary["conditions"]) == {"T5-orig", "T5-retrained"}
        assert "failures" not in summary["conditions"]["T5-orig"]


class TestMain:

    def test_missing_dataset_fails_cleanly(self, tmp_path):
        assert rx.main(["--dataset", str(tmp_path / "nope.json")]) == 1

    def test_skips_unavailable_conditions_and_runs_the_rest(self, tmp_path, monkeypatch):
        dataset = tmp_path / "cases.json"
        dataset.write_text(json.dumps({"examples": [
            {"input": "At least 18", "output": {"minimum": 18}, "field_type": "integer"},
        ]}))

        class FakeT5:
            def interpret(self, business_rule):
                return {"minimum": 18}

        def fake_build(condition, args):
            if condition == "T5-retrained":
                raise FileNotFoundError("not trained yet")
            return FakeT5()

        monkeypatch.setattr(rx, "build_interpreter", fake_build)
        out = tmp_path / "out"

        code = rx.main(["--experiment", "B", "--dataset", str(dataset),
                        "--output-dir", str(out)])

        assert code == 0
        files = sorted(p.name for p in out.iterdir())
        assert any(f.endswith("_T5-orig.json") for f in files)
        assert not any(f.endswith("_T5-retrained.json") for f in files)

    def test_unknown_condition_rejected(self):
        with pytest.raises(SystemExit):
            rx.main(["--experiment", "B", "--conditions", "B0"])


class TestLlmProvenance:

    def test_report_counts_which_model_and_provider_served_llm_calls(self):
        class FakeClient:
            call_log = []

        client = FakeClient()

        def fake_llm(field_name, field_type, rule):
            client.call_log.append({"requested_model": "deepseek/deepseek-chat",
                                    "model": "deepseek/deepseek-chat-v3-0324",
                                    "provider": "DeepInfra" if "18" in rule else "Novita",
                                    "id": "gen"})
            return {"minimum": 18}

        adapter = rx.FieldTypeAdapter(fake_llm, llm_client=client)
        report = rx.build_report(rx.run_condition("B0", adapter, CASES))

        assert report["llm_served_by"] == {
            "deepseek/deepseek-chat-v3-0324 via DeepInfra": 1,
            "deepseek/deepseek-chat-v3-0324 via Novita": 1,
        }

    def test_no_llm_no_provenance(self):
        class FakeT5:
            def interpret(self, business_rule):
                return {}

        report = rx.build_report(rx.run_condition("T5-orig", FakeT5(), CASES))
        assert "llm_served_by" not in report


class TestLlmErrors:

    def test_failed_llm_call_aborts_the_condition(self):
        """A failed call (bad key, outage) must stop the run, not be scored as an empty answer."""
        class FakeClient:
            call_log = []
            error_log = []

        client = FakeClient()

        def failing_llm(field_name, field_type, rule):
            client.error_log.append("401 Client Error: Unauthorized")
            return {}

        adapter = rx.FieldTypeAdapter(failing_llm, llm_client=client)
        with pytest.raises(rx.LlmCallError, match="B0.*401"):
            rx.run_condition("B0", adapter, CASES)

    def test_main_writes_no_reports_when_llm_fails(self, tmp_path, monkeypatch):
        dataset = tmp_path / "cases.json"
        dataset.write_text(json.dumps({"examples": [
            {"input": "At least 18", "output": {"minimum": 18}, "field_type": "integer"},
        ]}))

        class FakeClient:
            call_log = []
            error_log = []

        def fake_build(condition, args):
            client = FakeClient()

            def failing_llm(field_name, field_type, rule):
                client.error_log.append("401 Client Error: Unauthorized")
                return {}

            return rx.FieldTypeAdapter(failing_llm, llm_client=client)

        monkeypatch.setattr(rx, "build_interpreter", fake_build)
        out = tmp_path / "out"

        code = rx.main(["--experiment", "A", "--conditions", "B0",
                        "--dataset", str(dataset), "--output-dir", str(out)])

        assert code == 1
        assert not out.exists()
