"""Simulate a live stream for the monitoring dashboard from existing data.

Replays the held-out rules through the real HybridInterpreter: the real v2 T5
adapter answers and is scored for confidence, the real 0.7 gate decides fallbacks,
and the real code writes the events. Only the LLM is replayed:

- its answers are the recorded B0 answers (docs/experiments/expA_2026-10-11_B0.json,
  read-only; a rule B0 matched exactly is answered with the expected constraint);
- its latency, cost and occasional errors are synthetic, set by the options below
  (defaults: B0's recorded 1.4 s median latency, ~$0.00008 per call as seen in real
  OpenRouter runs, 3% errors).

Everything is written to separate simulation files under logs/, never to the CLI's
real event log or the tracked training-data files.

Usage (from the repo root), with the dashboard running in another terminal
(`streamlit run dashboard/app.py`, then choose Stream: Simulated in the sidebar):
    python -m scripts.simulate_live --reset
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hybrid_interpreter import HybridInterpreter  # noqa: E402
from src.monitoring import (  # noqa: E402
    SIMULATED_COLLECTED,
    SIMULATED_EVENT_LOG,
    SIMULATED_FAILURES,
    EventLogger,
)

DEFAULT_RULES = "training_data/holdout_financial_test.json"
DEFAULT_ADAPTER = "adapters/adapter_financial_v2/final_adapter"
DEFAULT_B0_REPORT = "docs/experiments/expA_2026-10-11_B0.json"


def recorded_llm_answers(report_path: str, rules: list) -> Dict[str, dict]:
    """B0's answer for every rule: its failures list holds the answers that differed
    from the expected constraint; every other rule was answered exactly as expected."""
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    answers = {r["input"]: r["output"] for r in rules}
    for failure in report.get("failures", []):
        answers[failure["input"]] = json.loads(failure["actual"])
    return answers


class ReplayLLMClient:
    """Stands in for OpenRouterClient: recorded answers, synthetic latency/cost/errors."""

    def __init__(self, answers: Dict[str, dict], median_latency_ms: float,
                 cost_per_call: float, error_rate: float, rng: random.Random,
                 sleep=None):
        self.answers = answers
        self.median_latency_ms = median_latency_ms
        self.cost_per_call = cost_per_call
        self.error_rate = error_rate
        self.rng = rng
        self.sleep = sleep or (lambda seconds: time.sleep(seconds))
        self.call_log: list = []
        self.error_log: list = []

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        if "alternative phrasings" in prompt:
            return "[]"  # paraphrase generation is not replayed
        # sigma 0.35 puts p99 near B0's recorded ~3.2 s for a 1.4 s median
        latency_ms = self.rng.lognormvariate(math.log(self.median_latency_ms), 0.35)
        self.sleep(latency_ms / 1000)
        if self.rng.random() < self.error_rate:
            self.error_log.append("Simulated 503 Service Unavailable")
            return ""
        match = re.search(r"^Rule: (.*)$", prompt, re.MULTILINE)
        answer = self.answers.get(match.group(1).strip()) if match else None
        self.call_log.append({
            "requested_model": "replay",
            "model": "replay of B0 (deepseek/deepseek-chat)",
            "provider": "simulated",
            "usage": {"prompt_tokens": 180, "completion_tokens": 27,
                      "cost": self.cost_per_call * self.rng.uniform(0.85, 1.15)},
            "latency_ms": latency_ms,
        })
        return json.dumps(answer or {})


def reset_files(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate a live monitoring stream")
    parser.add_argument("--rules", default=DEFAULT_RULES)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--b0-report", default=DEFAULT_B0_REPORT)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--event-log", default=SIMULATED_EVENT_LOG)
    parser.add_argument("--collected", default=SIMULATED_COLLECTED)
    parser.add_argument("--failures", default=SIMULATED_FAILURES)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Extra seconds between rules, to make the stream watchable")
    parser.add_argument("--llm-latency-ms", type=float, default=1400.0)
    parser.add_argument("--llm-cost", type=float, default=0.00008)
    parser.add_argument("--llm-error-rate", type=float, default=0.03)
    parser.add_argument("--rounds", type=int, default=1,
                        help="Passes over the rules (each in a new shuffled order)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true",
                        help="Delete earlier simulation files first")
    args = parser.parse_args(argv)

    for path in (args.event_log, args.collected, args.failures):
        if os.path.abspath(path).startswith(os.path.abspath("docs")):
            parser.error(f"refusing to write simulation output under docs/: {path}")
    if args.reset:
        reset_files(args.event_log, args.collected, args.failures)

    with open(args.rules, encoding="utf-8") as f:
        rules = json.load(f)["examples"]
    rng = random.Random(args.seed)
    llm = ReplayLLMClient(recorded_llm_answers(args.b0_report, rules), args.llm_latency_ms,
                          args.llm_cost, args.llm_error_rate, rng)

    from src.t5_interpreter import T5Interpreter

    print(f"Loading {args.adapter} ...")
    t5 = T5Interpreter(adapter_path=args.adapter)
    for round_no in range(1, args.rounds + 1):
        # One logger per round, so each round shows as its own run
        hybrid = HybridInterpreter(
            t5_interpreter=t5,
            llm_client=llm,
            failure_log_path=args.failures,
            collect_training_data=True,
            training_log_path=args.collected,
            confidence_threshold=args.confidence_threshold,
            event_logger=EventLogger(args.event_log),
        )
        order = rules[:]
        rng.shuffle(order)
        print(f"Round {round_no}: streaming {len(order)} rules to {args.event_log} "
              f"(Ctrl+C to stop)")
        for i, rule in enumerate(order, 1):
            hybrid.interpret(rule["id"], rule["field_type"], rule["input"])
            print(f"  {i:3}/{len(order)} {hybrid.last_source:5} {rule['input'][:60]}")
            time.sleep(args.delay)
        stats = hybrid.get_stats()
        print(f"Round {round_no} done: T5 {stats['t5_success_rate']}, "
              f"fallback {stats['fallback_rate']}, LLM failures {stats['llm_failures']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
