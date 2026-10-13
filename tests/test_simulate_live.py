"""Tests for scripts/simulate_live.py (no model loading, no sleeping)."""
import json
import random
from unittest.mock import Mock, patch

import pytest

from scripts import simulate_live as sim
from src.llm_interpreter import _build_prompt
from src.monitoring_data import read_events, summarize

RULES = json.load(open(sim.DEFAULT_RULES))["examples"]


def replay(answers=None, error_rate=0.0):
    return sim.ReplayLLMClient(answers or {"Over 18": {"exclusiveMinimum": 18}},
                               median_latency_ms=1400, cost_per_call=0.00008,
                               error_rate=error_rate, rng=random.Random(0),
                               sleep=lambda s: None)


class TestRecordedAnswers:

    def test_uses_b0_answers_where_they_differed_and_expected_elsewhere(self):
        answers = sim.recorded_llm_answers(sim.DEFAULT_B0_REPORT, RULES)
        assert len(answers) == len(RULES)
        differing = sum(answers[r["input"]] != r["output"] for r in RULES)
        b0 = json.load(open(sim.DEFAULT_B0_REPORT))
        assert differing == len(b0["failures"])


class TestReplayClient:

    def test_answers_the_rule_in_the_prompt(self):
        llm = replay()
        answer = llm.complete(_build_prompt("age", "integer", "Over 18"))
        assert json.loads(answer) == {"exclusiveMinimum": 18}
        (call,) = llm.call_log
        assert call["provider"] == "simulated"
        assert 0.00008 * 0.85 <= call["usage"]["cost"] <= 0.00008 * 1.15

    def test_simulated_errors(self):
        llm = replay(error_rate=1.0)
        assert llm.complete(_build_prompt("age", "integer", "Over 18")) == ""
        assert llm.error_log and not llm.call_log

    def test_paraphrase_requests_get_nothing(self):
        llm = replay()
        assert llm.complete("Generate 3 alternative phrasings of this business rule") == "[]"
        assert not llm.call_log


class TestMain:

    def test_refuses_to_write_under_docs(self):
        with pytest.raises(SystemExit):
            sim.main(["--event-log", "docs/experiments/sim.jsonl"])

    def test_streams_events_through_the_real_hybrid(self, tmp_path):
        t5 = Mock()
        t5.interpret.side_effect = lambda rule: {"minimum": 1}
        confidences = iter([0.95, 0.4] * len(RULES))
        type(t5).last_confidence = property(lambda self: next(confidences))
        paths = {k: str(tmp_path / f"{k}.json") for k in ("events", "collected", "failures")}

        with patch("src.t5_interpreter.T5Interpreter", return_value=t5), \
                patch.object(sim.time, "sleep"):
            code = sim.main(["--event-log", paths["events"], "--collected", paths["collected"],
                             "--failures", paths["failures"], "--delay", "0",
                             "--llm-error-rate", "0"])

        assert code == 0
        events, _ = read_events(paths["events"])
        s = summarize(events)
        assert s["calls"] == len(RULES)
        assert s["fallback_reasons"].get("low_confidence")  # the gate fired
        assert s["spend"] > 0
        assert json.load(open(paths["collected"]))["total_examples"] == len(RULES)
