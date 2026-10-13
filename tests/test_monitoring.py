"""Tests for the monitoring event log and the hybrid interpreter's events."""
import json
from unittest.mock import Mock

import pytest

from src.hybrid_interpreter import HybridInterpreter
from src.monitoring import EventLogger


@pytest.fixture(autouse=True)
def isolate_default_log_files(tmp_path, monkeypatch):
    """Keep t5_failures.json / training_data_collected.json writes out of the repo."""
    monkeypatch.chdir(tmp_path)


def read_events(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class FakeLLMClient:
    """Answers like OpenRouterClient and keeps the same call_log / error_log."""

    def __init__(self, answer='{"exclusiveMinimum": 18}', fail=False):
        self.answer = answer
        self.fail = fail
        self.call_log = []
        self.error_log = []

    def complete(self, prompt, temperature=0.0):
        if self.fail:
            self.error_log.append("503 Service Unavailable")
            return ""
        if "alternative phrasings" in prompt:
            return '["Older than 18", "Above 18"]'
        self.call_log.append({
            "model": "deepseek/deepseek-chat-v3",
            "provider": "DeepInfra",
            "usage": {"prompt_tokens": 100, "completion_tokens": 6, "cost": 0.00003},
            "latency_ms": 1200.0,
        })
        return self.answer


def make_t5(output, confidence=0.95, error=None):
    t5 = Mock()
    if error:
        t5.interpret.side_effect = error
    else:
        t5.interpret.return_value = output
    t5.last_confidence = confidence
    return t5


def make_hybrid(tmp_path, t5=None, llm=None, **kwargs):
    log = EventLogger(str(tmp_path / "logs" / "events.jsonl"), run_id="run1")
    hybrid = HybridInterpreter(
        t5_interpreter=t5, llm_client=llm, log_failures=False, event_logger=log, **kwargs
    )
    return hybrid, log.path


class TestEventLogger:

    def test_appends_one_line_per_event_and_creates_dir(self, tmp_path):
        log = EventLogger(str(tmp_path / "new" / "events.jsonl"))
        log.log("interpret", rule="a")
        log.log("interpret", rule="b")

        events = read_events(log.path)
        assert [e["rule"] for e in events] == ["a", "b"]
        assert all(e["type"] == "interpret" and e["run_id"] == log.run_id for e in events)

    def test_new_logger_appends_to_existing_file(self, tmp_path):
        path = str(tmp_path / "events.jsonl")
        EventLogger(path).log("interpret", rule="first run")
        EventLogger(path).log("interpret", rule="second run")

        events = read_events(path)
        assert len(events) == 2
        assert events[0]["run_id"] != events[1]["run_id"]


class TestInterpretEvents:

    def test_t5_success(self, tmp_path):
        hybrid, path = make_hybrid(tmp_path, t5=make_t5({"minimum": 18}, 0.97), llm=FakeLLMClient())
        hybrid.interpret("age", "integer", "At least 18")

        (event,) = read_events(path)
        assert event["source"] == "t5"
        assert event["fallback_reason"] is None
        assert event["t5_confidence"] == 0.97
        assert event["output"] == {"minimum": 18}
        assert event["llm_output"] is None
        assert event["t5_latency_ms"] >= 0 and event["total_latency_ms"] >= 0

    def test_invalid_t5_output_falls_back_with_llm_details(self, tmp_path):
        hybrid, path = make_hybrid(tmp_path, t5=make_t5({"maxLength": 5}), llm=FakeLLMClient())
        hybrid.interpret("age", "integer", "Over 18")

        (event,) = read_events(path)
        assert event["source"] == "llm"
        assert event["fallback_reason"] == "invalid"
        assert event["validation_errors"]
        assert event["llm_output"] == {"exclusiveMinimum": 18}
        assert event["llm_provider"] == "DeepInfra"
        assert event["llm_usage"]["cost"] == 0.00003
        assert event["llm_error"] is None

    @pytest.mark.parametrize("t5,reason", [
        (make_t5({}), "empty"),
        (make_t5(None, error=RuntimeError("boom")), "exception"),
    ])
    def test_fallback_reasons(self, tmp_path, t5, reason):
        hybrid, path = make_hybrid(tmp_path, t5=t5, llm=FakeLLMClient())
        hybrid.interpret("age", "integer", "Over 18")

        (event,) = read_events(path)
        assert event["fallback_reason"] == reason
        assert event["source"] == "llm"

    def test_low_confidence(self, tmp_path):
        hybrid, path = make_hybrid(
            tmp_path, t5=make_t5({"minimum": 18}, 0.4), llm=FakeLLMClient(),
            confidence_threshold=0.7,
        )
        hybrid.interpret("age", "integer", "Over 18")

        (event,) = read_events(path)
        assert event["fallback_reason"] == "low_confidence"
        assert event["t5_confidence"] == 0.4
        assert event["confidence_threshold"] == 0.7
        assert event["source"] == "llm"

    def test_llm_failure_is_recorded(self, tmp_path):
        hybrid, path = make_hybrid(tmp_path, t5=make_t5({}), llm=FakeLLMClient(fail=True))
        result = hybrid.interpret("age", "integer", "Over 18")

        (event,) = read_events(path)
        assert result == {}
        assert event["source"] == "none"
        assert event["llm_error"] == "503 Service Unavailable"

    def test_no_logger_writes_nothing(self, tmp_path):
        hybrid = HybridInterpreter(t5_interpreter=make_t5({"minimum": 1}), log_failures=False)
        assert hybrid.interpret("n", "integer", "At least 1") == {"minimum": 1}
        assert not list(tmp_path.rglob("*.jsonl"))

    def test_broken_logger_does_not_break_interpretation(self, tmp_path):
        broken = Mock()
        broken.log.side_effect = OSError("disk full")
        hybrid = HybridInterpreter(
            t5_interpreter=make_t5({"minimum": 1}), log_failures=False, event_logger=broken
        )
        assert hybrid.interpret("n", "integer", "At least 1") == {"minimum": 1}


class TestTrainingExampleEvents:

    def test_llm_answer_and_variations_are_tagged(self, tmp_path):
        hybrid, path = make_hybrid(
            tmp_path, t5=make_t5({}), llm=FakeLLMClient(),
            collect_training_data=True, training_log_path=str(tmp_path / "collected.json"),
        )
        hybrid.interpret("age", "integer", "Over 18")

        examples = [e for e in read_events(path) if e["type"] == "training_example"]
        assert [e["source"] for e in examples] == ["llm", "llm_variation", "llm_variation"]
        assert examples[1]["rule"] == "Older than 18"

    def test_paraphrase_calls_count_towards_the_rules_cost(self, tmp_path):
        llm = FakeLLMClient()
        original = llm.complete

        def complete(prompt, temperature=0.0):
            answer = original(prompt, temperature)
            if "alternative phrasings" in prompt:  # a billed call, like the real client
                llm.call_log.append({"usage": {"cost": 0.00002}})
            return answer
        llm.complete = complete

        hybrid, path = make_hybrid(
            tmp_path, t5=make_t5({}), llm=llm,
            collect_training_data=True, training_log_path=str(tmp_path / "collected.json"),
        )
        hybrid.interpret("age", "integer", "Over 18")

        (event,) = [e for e in read_events(path) if e["type"] == "interpret"]
        assert event["llm_call_count"] == 2
        assert event["llm_total_cost"] == pytest.approx(0.00005)
        assert event["llm_usage"]["cost"] == 0.00003  # the interpret call itself

    def test_no_events_when_collection_disabled(self, tmp_path):
        hybrid, path = make_hybrid(tmp_path, t5=make_t5({"minimum": 1}))
        hybrid.interpret("n", "integer", "At least 1")
        assert [e["type"] for e in read_events(path)] == ["interpret"]


class TestLogFilesAccumulate:
    """Each run used to overwrite the files with only its own records."""

    def test_training_data_accumulates_across_runs(self, tmp_path):
        path = str(tmp_path / "collected.json")
        for rule in ("At least 1", "At least 2"):
            HybridInterpreter(
                t5_interpreter=make_t5({"minimum": int(rule[-1])}), log_failures=False,
                collect_training_data=True, training_log_path=path,
            ).interpret("n", "integer", rule)

        data = json.load(open(path))
        assert [e["input"] for e in data["examples"]] == ["At least 1", "At least 2"]
        assert data["total_examples"] == 2
        assert data["sources"] == {"t5": 2, "llm": 0}

    def test_failures_accumulate_across_runs(self, tmp_path):
        path = str(tmp_path / "failures.json")
        for rule in ("Rule one", "Rule two"):
            HybridInterpreter(
                t5_interpreter=make_t5({}), failure_log_path=path,
            ).interpret("f", "string", rule)

        data = json.load(open(path))
        assert [f["business_rule"] for f in data["failures"]] == ["Rule one", "Rule two"]
        assert data["stats"]["total_calls"] == 1  # stats describe the latest run


class TestCliFlags:

    def _run(self, tmp_path, *extra):
        from unittest.mock import patch
        from src import main as cli

        with patch.object(cli, "create_hybrid_interpreter") as create, \
                patch.object(cli, "enrich_schema", return_value={}):
            code = cli.main(["-s", "schema.json", "-d", "dict.csv", "--llm", "none",
                             "--adapter", "some/adapter", "-o", str(tmp_path / "out.json"),
                             *extra])
        assert code == 0
        return create.call_args.kwargs

    def test_event_log_and_threshold_reach_the_interpreter(self, tmp_path):
        kwargs = self._run(tmp_path, "--confidence-threshold", "0.7",
                           "--event-log", str(tmp_path / "ev.jsonl"))

        assert kwargs["confidence_threshold"] == 0.7
        assert kwargs["event_logger"].path == str(tmp_path / "ev.jsonl")

    def test_no_event_log(self, tmp_path):
        kwargs = self._run(tmp_path, "--no-event-log")

        assert kwargs["event_logger"] is None
        assert kwargs["confidence_threshold"] is None
