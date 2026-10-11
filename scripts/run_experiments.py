#!/usr/bin/env python3
"""
Run the S22 methodology experiments (see docs/s22-methodology-notes.md §3.4 and §4).

Experiment A - headline hypothesis: B0 (prompt-only LLM) vs Hybrid, with T5-orig for context.
Experiment B - EDA-driven retrain:  T5-orig vs T5-retrained.

Each condition is evaluated once on the held-out test set and the reports are written to
docs/experiments/exp<A|B>_<date>_<condition>.json, plus a summary per experiment.

Usage:
    python -m scripts.run_experiments --experiment A
    python -m scripts.run_experiments --experiment all \\
        --retrained-adapter adapters/adapter_financial_v2/final_adapter
    # Pipeline smoke test (NOT a held-out set - do not report these numbers):
    python -m scripts.run_experiments --experiment B \\
        --dataset training_data/financial_domain_small.json --output-dir /tmp/exp_smoke
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from statistics import median
from typing import Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.model_evaluator import (
    EvalTestCase,
    EvaluationResults,
    EvaluationSuite,
    generate_report,
    load_evaluation_dataset,
)


DEFAULT_DATASET = "training_data/holdout_financial_test.json"
# T5-orig is the 509-example baseline rebuilt with the same script and text format as
# T5-retrained, so Experiment B varies only the training data. The shipped adapter
# (adapters/adapter_financial) predates src/t5_text.py and is not a fair baseline.
DEFAULT_ORIG_ADAPTER = "adapters/adapter_financial_v1_rebuild/final_adapter"
DEFAULT_RETRAINED_ADAPTER = "adapters/adapter_financial_v2/final_adapter"
DEFAULT_LLM_MODEL = "deepseek/deepseek-chat"
DEFAULT_OUTPUT_DIR = "docs/experiments"

EXPERIMENTS = {
    "A": ["B0", "T5-orig", "Hybrid"],
    "B": ["T5-orig", "T5-retrained"],
}

# Field name passed to field-aware interpreters. The held-out cases carry no field name,
# and the LLM prompt only uses it as context, so a neutral placeholder keeps B0 and Hybrid
# on identical inputs.
EVAL_FIELD_NAME = "field"


class LlmCallError(RuntimeError):
    """An LLM call failed (bad key, outage, timeout), so the condition's scores are invalid."""


class FieldTypeAdapter:
    """
    Adapt a field-aware interpreter to EvaluationSuite's interpret(business_rule) protocol.

    The LLM and Hybrid interpreters need the field type, which varies per test case, so
    run_condition() sets `field_type` before each case.
    """

    def __init__(
        self,
        interpret_fn: Callable[[str, str, str], dict],
        source_fn: Optional[Callable[[], Optional[str]]] = None,
        llm_client=None,
    ):
        self.interpret_fn = interpret_fn
        self.source_fn = source_fn  # reports which engine answered the last call (Hybrid)
        self.llm_client = llm_client  # its call_log records who served each LLM call
        self.field_type = "string"

    def interpret(self, business_rule: str) -> dict:
        return self.interpret_fn(EVAL_FIELD_NAME, self.field_type, business_rule)


class ConditionRun:
    """The results of one condition plus any per-case routing information."""

    def __init__(
        self,
        name: str,
        results: EvaluationResults,
        sources: List[Optional[str]],
        llm_calls: Optional[List[dict]] = None,
    ):
        self.name = name
        self.results = results
        self.sources = sources  # per-case "t5" / "llm" / "none" for Hybrid; None otherwise
        self.llm_calls = llm_calls or []  # OpenRouterClient.call_log entries


def run_condition(name: str, interpreter, test_cases: List[EvalTestCase]) -> ConditionRun:
    """
    Evaluate one condition, one case at a time so field_type can be set per case.

    Running per case does not change the measured latency: EvaluationSuite times each
    interpret() call individually either way.
    """
    suite = EvaluationSuite(interpreter=interpreter)
    results = EvaluationResults()
    sources: List[Optional[str]] = []
    client = getattr(interpreter, "llm_client", None)
    errors = getattr(client, "error_log", [])

    for case in test_cases:
        adapter = interpreter if isinstance(interpreter, FieldTypeAdapter) else None
        if adapter:
            adapter.field_type = case.field_type
        errors_before = len(errors)
        results.test_results.extend(suite.run([case]).test_results)
        # OpenRouterClient swallows failures and returns "", which would otherwise be
        # scored as an empty answer - stop at the first one instead
        if len(errors) > errors_before:
            raise LlmCallError(f"{name}: LLM call failed on {case.input!r}: {errors[-1]}")
        sources.append(adapter.source_fn() if adapter and adapter.source_fn else None)

    return ConditionRun(name, results, sources, list(getattr(client, "call_log", [])))


def build_report(run: ConditionRun) -> dict:
    """Extend generate_report() with the extra metrics §3.4 asks for."""
    report = generate_report(run.results)
    latencies = sorted(r.latency_ms for r in run.results.test_results)

    report["condition"] = run.name
    report["median_latency_ms"] = median(latencies) if latencies else 0.0
    report["p99_latency_ms"] = (
        latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)] if latencies else 0.0
    )
    report["by_constraint_type"] = {
        ctype: {
            "total_tests": group.total_tests,
            "semantic_match_rate": group.semantic_match_rate,
            "exact_match_rate": group.exact_match_rate,
            "schema_valid_rate": group.schema_valid_rate,
        }
        for ctype, group in sorted(run.results.group_by_constraint_type().items())
    }

    routed = [s for s in run.sources if s is not None]
    if routed:
        report["routing"] = {source: routed.count(source) for source in sorted(set(routed))}
        report["fallback_rate"] = routed.count("llm") / len(routed)

    if run.llm_calls:
        served = [f"{c['model']} via {c['provider']}" for c in run.llm_calls]
        report["llm_served_by"] = {s: served.count(s) for s in sorted(set(served))}

    return report


def build_interpreter(condition: str, args):
    """Construct the interpreter for a condition. Heavy imports stay lazy."""
    if condition == "B0":
        from src.llm_interpreter import OpenRouterClient, interpret_business_rule

        client = OpenRouterClient(args.llm_model)
        if not client.api_key:
            raise RuntimeError("B0 needs OPENROUTER_API_KEY (env or .env)")
        return FieldTypeAdapter(
            lambda name, ftype, rule: interpret_business_rule(name, ftype, rule, client),
            llm_client=client,
        )

    if condition in ("T5-orig", "T5-retrained"):
        from src.t5_interpreter import T5Interpreter

        path = args.orig_adapter if condition == "T5-orig" else args.retrained_adapter
        _require_adapter(path, condition)
        return T5Interpreter(adapter_path=path)

    if condition == "Hybrid":
        from src.hybrid_interpreter import HybridInterpreter
        from src.llm_interpreter import OpenRouterClient
        from src.t5_interpreter import T5Interpreter

        _require_adapter(args.hybrid_adapter, condition)
        client = OpenRouterClient(args.llm_model)
        if not client.api_key:
            raise RuntimeError("Hybrid needs OPENROUTER_API_KEY (env or .env) for its fallback")
        hybrid = HybridInterpreter(
            t5_interpreter=T5Interpreter(adapter_path=args.hybrid_adapter),
            llm_client=client,
            # Keep evaluation side-effect free: no t5_failures.json / training-data writes.
            log_failures=False,
            collect_training_data=False,
            confidence_threshold=args.confidence_threshold,
        )
        return FieldTypeAdapter(
            hybrid.interpret, source_fn=lambda: hybrid.last_source, llm_client=client
        )

    raise ValueError(f"Unknown condition: {condition}")


def _require_adapter(path: str, condition: str) -> None:
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(
            f"{condition}: no adapter at {path}. "
            "For T5-retrained, train it first (docs/s22-methodology-notes.md §4, Experiment B)."
        )


def run_metadata(dataset_path: str, args) -> dict:
    """Provenance recorded with every report (§3.4 procedure step 7)."""
    with open(dataset_path, "rb") as f:
        dataset_sha256 = hashlib.sha256(f.read()).hexdigest()
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        git_sha, dirty = "unknown", None

    return {
        "date": date.today().isoformat(),
        "git_sha": git_sha,
        "git_dirty": dirty,
        "dataset": dataset_path,
        "dataset_sha256": dataset_sha256,
        "llm_model": args.llm_model,
        "orig_adapter": args.orig_adapter,
        "retrained_adapter": args.retrained_adapter,
        "hybrid_adapter": args.hybrid_adapter,
        "confidence_threshold": args.confidence_threshold,
    }


def write_reports(
    experiments: List[str],
    reports: Dict[str, dict],
    metadata: dict,
    output_dir: str,
) -> List[str]:
    """Write exp<X>_<date>_<condition>.json and exp<X>_<date>_summary.json."""
    os.makedirs(output_dir, exist_ok=True)
    written = []
    stamp = metadata["date"]

    for exp in experiments:
        summary = {"experiment": exp, "metadata": metadata, "conditions": {}}
        for condition in EXPERIMENTS[exp]:
            if condition not in reports:
                continue
            path = os.path.join(output_dir, f"exp{exp}_{stamp}_{condition}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"metadata": metadata, **reports[condition]}, f, indent=2)
            written.append(path)
            summary["conditions"][condition] = {
                k: v for k, v in reports[condition].items()
                if k not in ("failures", "by_constraint_type")
            }
        path = os.path.join(output_dir, f"exp{exp}_{stamp}_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        written.append(path)

    return written


def print_table(reports: Dict[str, dict]) -> None:
    print(f"\n{'Condition':<14}{'N':>5}{'JSON':>8}{'Valid':>8}{'Exact':>8}"
          f"{'Semantic':>10}{'Med ms':>10}{'p99 ms':>10}{'Fallback':>10}")
    for name, r in reports.items():
        fallback = f"{r['fallback_rate']:.1%}" if "fallback_rate" in r else "-"
        print(f"{name:<14}{r['total_tests']:>5}{r['json_correctness_rate']:>8.1%}"
              f"{r['schema_valid_rate']:>8.1%}{r['exact_match_rate']:>8.1%}"
              f"{r['semantic_match_rate']:>10.1%}{r['median_latency_ms']:>10.1f}"
              f"{r['p99_latency_ms']:>10.1f}{fallback:>10}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run S22 experiments A and/or B")
    parser.add_argument("--experiment", choices=["A", "B", "all"], default="all")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"Held-out test set (default: {DEFAULT_DATASET})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--orig-adapter", default=DEFAULT_ORIG_ADAPTER)
    parser.add_argument("--retrained-adapter", default=DEFAULT_RETRAINED_ADAPTER)
    parser.add_argument("--hybrid-adapter",
                        help="Adapter used inside Hybrid (default: --retrained-adapter, per §3.4)")
    parser.add_argument("--confidence-threshold", type=float,
                        help="Hybrid sends T5 answers whose least certain token is below this "
                             "probability to the LLM (default: off). Fix it before seeing test results.")
    parser.add_argument("--conditions", nargs="+",
                        help="Run only these conditions (e.g. T5-orig) - for partial runs")
    args = parser.parse_args(argv)
    args.hybrid_adapter = args.hybrid_adapter or args.retrained_adapter

    experiments = ["A", "B"] if args.experiment == "all" else [args.experiment]
    conditions = list(dict.fromkeys(c for exp in experiments for c in EXPERIMENTS[exp]))
    if args.conditions:
        unknown = set(args.conditions) - set(conditions)
        if unknown:
            parser.error(f"conditions not in experiment {args.experiment}: {sorted(unknown)}")
        conditions = [c for c in conditions if c in args.conditions]

    if not os.path.exists(args.dataset):
        print(f"Error: dataset not found: {args.dataset}\n"
              "Author the held-out set first (docs/s22-methodology-notes.md §3.4).",
              file=sys.stderr)
        return 1
    if os.path.abspath(args.dataset) != os.path.abspath(DEFAULT_DATASET):
        print(f"Warning: {args.dataset} is not the held-out test set; "
              "results are for pipeline checks only.", file=sys.stderr)

    test_cases = load_evaluation_dataset(args.dataset)
    metadata = run_metadata(args.dataset, args)
    print(f"Loaded {len(test_cases)} cases from {args.dataset} (git {metadata['git_sha'][:8]})")

    reports: Dict[str, dict] = {}
    for condition in conditions:
        print(f"Running {condition}...")
        try:
            interpreter = build_interpreter(condition, args)
        except (RuntimeError, FileNotFoundError, ImportError) as e:
            print(f"  Skipped: {e}", file=sys.stderr)
            continue
        try:
            reports[condition] = build_report(run_condition(condition, interpreter, test_cases))
        except LlmCallError as e:
            print(f"Error: {e}\nNo reports written.", file=sys.stderr)
            return 1

    if not reports:
        print("Error: no condition could run.", file=sys.stderr)
        return 1

    print_table(reports)
    for path in write_reports(experiments, reports, metadata, args.output_dir):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
