"""Tests for the dashboard data layer (src/monitoring_data.py)."""
import json

import pytest

from src import monitoring_data as md
from src.monitoring import EventLogger

EXP = "docs/experiments"


def write_report(path, condition, cases, by_type=None, sha="abc"):
    """A report in the current format, with per-case rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metadata": {"dataset_sha256": sha},
        "condition": condition,
        "total_tests": len(cases),
        "semantic_match_rate": sum(c["semantic_match"] for c in cases) / len(cases),
        "by_constraint_type": by_type or {},
        "failures": [],
        "cases": cases,
    }))
    return str(path)


def case(rule, ok, ctype="minimum"):
    return {"input": rule, "expected": '{"minimum":1}', "actual": "{}" if not ok else '{"minimum":1}',
            "constraint_type": ctype, "semantic_match": ok, "exact_match": ok,
            "schema_valid": True, "latency_ms": 5.0, "source": "t5"}


class TestFindAndLoad:

    def test_skips_summaries_and_calibration_files(self):
        paths = md.find_reports(EXP)
        assert paths
        assert not any(p.endswith("_summary.json") or "confidence_" in p for p in paths)

    def test_labels_runs_in_subfolders(self):
        labels = {md.load_report(p, EXP)["label"] for p in md.find_reports(EXP)}
        assert "B0" in labels and "B0 (expA_b0_repeats/run2)" in labels
        assert "Hybrid (expA_hybrid_gate)" in labels

    def test_old_report_has_no_cases(self):
        report = md.load_report(f"{EXP}/expA_2026-10-11_B0.json", EXP)
        assert report["cases"] is None
        assert report["experiment"] == "A"
        assert report["rates"]["Semantic match"] == pytest.approx(0.9706, abs=1e-4)


class TestSemanticFailures:

    def test_old_report_failures_are_rescored(self):
        # B0 run 1: 46 exact-match failures, of which 3 are semantic failures
        report = md.load_report(f"{EXP}/expA_2026-10-11_B0.json", EXP)
        assert len(report["failures"]) > 3
        assert len(md.semantic_failures(report)) == 3

    def test_uses_case_rows_when_present(self, tmp_path):
        path = write_report(tmp_path / "r.json", "X", [case("a", True), case("b", False)])
        (failure,) = md.semantic_failures(md.load_report(path))
        assert failure["input"] == "b" and failure["source"] == "t5"


class TestCompare:

    def test_reproduces_recorded_gate_comparison(self):
        ungated = md.load_report(f"{EXP}/expA_2026-10-11_Hybrid.json", EXP)
        gated = md.load_report(f"{EXP}/expA_hybrid_gate/expA_2026-10-11_Hybrid.json", EXP)

        result = md.compare(ungated, gated)

        assert result["same_dataset"]
        assert (len(result["fixed"]), len(result["broken"])) == (12, 0)
        assert result["p_value"] == pytest.approx(0.000488, abs=1e-6)

    def test_reproduces_recorded_experiment_b(self):
        orig = md.load_report(f"{EXP}/expB_2026-10-11_T5-orig.json", EXP)
        retrained = md.load_report(f"{EXP}/expB_2026-10-11_T5-retrained.json", EXP)

        result = md.compare(orig, retrained)

        assert (len(result["fixed"]), len(result["broken"])) == (10, 5)
        assert result["p_value"] == pytest.approx(0.3018, abs=1e-4)
        exclusive = next(r for r in result["by_type"]
                         if r["constraint_type"] == "contrastive/exclusiveMinimum")
        assert exclusive["delta"] == pytest.approx(0.25)

    def test_flags_different_datasets(self, tmp_path):
        a = md.load_report(write_report(tmp_path / "a.json", "A", [case("x", True)], sha="1"))
        b = md.load_report(write_report(tmp_path / "b.json", "B", [case("x", True)], sha="2"))
        assert not md.compare(a, b)["same_dataset"]

    @pytest.mark.parametrize("fixed,broken,p", [(0, 0, 1.0), (5, 5, 1.0), (9, 1, 0.021484375)])
    def test_mcnemar_exact(self, fixed, broken, p):
        assert md.mcnemar_exact(fixed, broken) == pytest.approx(p)


class TestReadEvents:

    def test_incremental_reads(self, tmp_path):
        log = EventLogger(str(tmp_path / "ev.jsonl"))
        log.log("interpret", rule="a")
        events, offset = md.read_events(log.path)
        assert [e["rule"] for e in events] == ["a"]

        log.log("interpret", rule="b")
        events, offset = md.read_events(log.path, offset)
        assert [e["rule"] for e in events] == ["b"]
        assert md.read_events(log.path, offset)[0] == []

    def test_partial_last_line_waits_for_next_read(self, tmp_path):
        path = tmp_path / "ev.jsonl"
        path.write_text('{"rule": "a"}\n{"rule": "b"')
        events, offset = md.read_events(str(path))
        assert [e["rule"] for e in events] == ["a"]

        with open(path, "a") as f:
            f.write('}\n')
        assert [e["rule"] for e in md.read_events(str(path), offset)[0]] == ["b"]

    def test_missing_file_and_truncation(self, tmp_path):
        assert md.read_events(str(tmp_path / "none.jsonl")) == ([], 0)
        path = tmp_path / "ev.jsonl"
        path.write_text('{"rule": "a"}\n')
        assert md.read_events(str(path), offset=999)[0] == [{"rule": "a"}]


def interpret_event(source, reason=None, latency=10.0, cost=None, llm_error=None, run="r1"):
    return {
        "type": "interpret", "run_id": run, "source": source, "fallback_reason": reason,
        "total_latency_ms": latency, "llm_error": llm_error,
        "llm_latency_ms": 1000.0 if reason else None,
        "llm_usage": {"cost": cost} if cost is not None else None,
    }


class TestSummarize:

    EVENTS = [
        interpret_event("t5"),
        interpret_event("t5"),
        interpret_event("llm", "low_confidence", 1200.0, cost=0.0001),
        interpret_event("llm", "invalid", 1300.0, cost=0.0003, run="r2"),
        interpret_event("t5", "low_confidence", 1100.0, llm_error="503"),
        {"type": "training_example", "source": "t5"},
    ]

    def test_counts_and_rates(self):
        s = md.summarize(self.EVENTS)
        assert s["calls"] == 5
        assert s["t5_answered"] == 2 and s["llm_answered"] == 2
        assert s["t5_kept_after_llm_failure"] == 1
        assert s["fallback_rate"] == pytest.approx(0.6)
        assert s["t5_success_rate"] == pytest.approx(0.4)
        assert s["llm_error_rate"] == pytest.approx(1 / 3)
        assert s["fallback_reasons"] == {"low_confidence": 2, "invalid": 1}
        assert s["runs"] == 2

    def test_cost_projection(self):
        s = md.summarize(self.EVENTS, rules_per_day=10_000)
        assert s["spend"] == pytest.approx(0.0004)
        assert s["cost_per_rule"] == pytest.approx(0.00008)
        assert s["projected_monthly_cost"] == pytest.approx(0.00008 * 10_000 * 30)

    def test_total_cost_preferred_over_interpret_call_usage(self):
        event = interpret_event("llm", "invalid", cost=0.0001)
        event["llm_total_cost"] = 0.0003
        assert md.summarize([event])["spend"] == pytest.approx(0.0003)

    def test_no_cost_data(self):
        s = md.summarize([interpret_event("t5")])
        assert s["spend"] is None and s["projected_monthly_cost"] is None

    def test_empty(self):
        s = md.summarize([])
        assert s["calls"] == 0 and s["fallback_rate"] is None

    def test_latency_by_source(self):
        s = md.summarize(self.EVENTS)
        assert s["latency_by_source"]["t5"]["count"] == 3
        assert s["latency_by_source"]["llm"]["median_ms"] == pytest.approx(1250.0)

    def test_failure_stream_newest_first(self):
        stream = md.failure_stream(self.EVENTS)
        assert [e["fallback_reason"] for e in stream] == ["low_confidence", "invalid",
                                                           "low_confidence"]
        assert stream[0]["source"] == "t5"


class TestTrainingSummary:

    def test_counts_sources_and_flags_known_rules(self, tmp_path):
        collected = tmp_path / "collected.json"
        collected.write_text(json.dumps({"total_examples": 7, "sources": {"t5": 5, "llm": 2}}))
        events = [
            {"type": "training_example", "rule": "Must be positive", "source": "t5"},
            {"type": "training_example", "rule": "A brand new rule", "source": "llm_variation"},
            {"type": "interpret", "source": "t5"},
        ]
        s = md.training_summary(events, str(collected),
                                known_inputs={md.normalise_rule("Must be positive!")})

        assert s["events"] == 2
        assert s["by_source"] == {"t5": 1, "llm_variation": 1}
        assert s["collected_file_total"] == 7
        assert s["duplicates_of_training_data"] == 1
        assert s["recent"][0]["rule"] == "A brand new rule"  # newest first

    def test_known_inputs_excludes_holdout(self):
        known = md.known_training_inputs()
        holdout = json.load(open("training_data/holdout_financial_test.json"))["examples"]
        held_out_only = {md.normalise_rule(e["input"]) for e in holdout} - known
        assert held_out_only  # held-out rules are not treated as training data
        assert md.normalise_rule("Balance must not be negative") in known
