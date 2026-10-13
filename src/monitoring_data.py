"""
Data layer for the monitoring dashboard (dashboard/): pure functions, no Streamlit.

Batch side: load experiment reports (docs/experiments/ or a new run's output dir) and
compare conditions. Reports written before per-case rows were added hold only
aggregates and exact-match failures; those are handled read-only, never rewritten.

Live side: read the append-only event log written by HybridInterpreter
(src/monitoring.py) and summarise it.
"""
import glob
import json
import os
import re
from math import comb
from statistics import median
from typing import Dict, List, Optional, Tuple

from .model_evaluator import SemanticMatchMetric

RATE_KEYS = {
    "JSON-correct": "json_correctness_rate",
    "Schema-valid": "schema_valid_rate",
    "Exact match": "exact_match_rate",
    "Semantic match": "semantic_match_rate",
}

# docs/next_steps.md cost estimate at 10k rules/day (dollars/month), for context only
ESTIMATED_MONTHLY_COST_100PCT_LLM = (50, 200)


# --- Batch evaluation reports ------------------------------------------------

def find_reports(root: str) -> List[str]:
    """Per-condition report files under root (summaries and calibration files skipped)."""
    paths = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    return sorted(
        p for p in paths
        if not p.endswith("_summary.json") and not os.path.basename(p).startswith("confidence_")
    )


def load_report(path: str, root: Optional[str] = None) -> dict:
    """Read one condition report into a uniform shape. The file is only read."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    condition = raw.get("condition") or os.path.splitext(os.path.basename(path))[0]
    folder = os.path.relpath(os.path.dirname(path), root) if root else ""
    match = re.match(r"exp([A-Z])_", os.path.basename(path))
    return {
        "path": path,
        "label": condition if folder in ("", ".") else f"{condition} ({folder})",
        "condition": condition,
        "experiment": match.group(1) if match else None,
        "metadata": raw.get("metadata", {}),
        "total_tests": raw.get("total_tests", 0),
        "rates": {name: raw.get(key) for name, key in RATE_KEYS.items()},
        "median_latency_ms": raw.get("median_latency_ms"),
        "p99_latency_ms": raw.get("p99_latency_ms"),
        "average_latency_ms": raw.get("average_latency_ms"),
        "fallback_rate": raw.get("fallback_rate"),
        "routing": raw.get("routing"),
        "by_constraint_type": raw.get("by_constraint_type", {}),
        "cases": raw.get("cases"),  # None for reports written before per-case rows
        "failures": raw.get("failures", []),
    }


def semantic_failures(report: dict) -> List[dict]:
    """Rules the condition got semantically wrong.

    From per-case rows when present; otherwise rescored from the exact-match
    `failures` list (every semantic failure is also an exact-match failure).
    """
    if report["cases"] is not None:
        return [
            {k: c.get(k) for k in ("input", "expected", "actual", "constraint_type",
                                   "source", "latency_ms")}
            for c in report["cases"] if not c["semantic_match"]
        ]
    metric = SemanticMatchMetric()
    return [
        {"input": f["input"], "expected": f["expected"], "actual": f["actual"],
         "constraint_type": None, "source": None, "latency_ms": None}
        for f in report["failures"]
        if metric.measure(f["actual"], f["expected"]) < 1.0
    ]


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(report_a: dict, report_b: dict) -> dict:
    """Paired comparison of B against A on the same test set (semantic match)."""
    wrong_a = {f["input"]: f for f in semantic_failures(report_a)}
    wrong_b = {f["input"]: f for f in semantic_failures(report_b)}
    fixed = sorted(set(wrong_a) - set(wrong_b))
    broken = sorted(set(wrong_b) - set(wrong_a))

    types_a, types_b = report_a["by_constraint_type"], report_b["by_constraint_type"]
    by_type = [
        {
            "constraint_type": t,
            "total_tests": (types_a.get(t) or types_b.get(t))["total_tests"],
            "a": types_a[t]["semantic_match_rate"] if t in types_a else None,
            "b": types_b[t]["semantic_match_rate"] if t in types_b else None,
        }
        for t in sorted(set(types_a) | set(types_b))
    ]
    for row in by_type:
        row["delta"] = (row["b"] - row["a"]) if None not in (row["a"], row["b"]) else None

    sha_a = report_a["metadata"].get("dataset_sha256")
    sha_b = report_b["metadata"].get("dataset_sha256")
    return {
        "same_dataset": sha_a is not None and sha_a == sha_b,
        "fixed": [{"input": i, "a_actual": wrong_a[i]["actual"],
                   "expected": wrong_a[i]["expected"]} for i in fixed],
        "broken": [{"input": i, "b_actual": wrong_b[i]["actual"],
                    "expected": wrong_b[i]["expected"]} for i in broken],
        "still_wrong": len(set(wrong_a) & set(wrong_b)),
        "p_value": mcnemar_exact(len(fixed), len(broken)),
        "by_type": by_type,
    }


# --- Live event log -----------------------------------------------------------

def read_events(path: str, offset: int = 0) -> Tuple[List[dict], int]:
    """Events appended since byte `offset`, and the offset to resume from.

    A last line still being written (no newline yet) is left for the next read.
    """
    if not os.path.exists(path):
        return [], 0
    if os.path.getsize(path) < offset:  # file was truncated or replaced
        offset = 0
    events = []
    with open(path, "rb") as f:
        f.seek(offset)
        for line in f:
            if not line.endswith(b"\n"):
                break
            offset += len(line)
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events, offset


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def _event_cost(event: dict) -> Optional[float]:
    """All LLM spend for one rule: llm_total_cost covers every call (paraphrase
    generation too); older events only carry the interpret call's usage."""
    if event.get("llm_total_cost") is not None:
        return event["llm_total_cost"]
    usage = event.get("llm_usage")
    return usage.get("cost") if isinstance(usage, dict) else None


def summarize(events: List[dict], rules_per_day: int = 10_000) -> dict:
    """Headline numbers for the interpret events in `events`."""
    calls = [e for e in events if e.get("type") == "interpret"]
    n = len(calls)

    def count(pred):
        return sum(1 for e in calls if pred(e))

    reasons: Dict[str, int] = {}
    for e in calls:
        if e.get("fallback_reason"):
            reasons[e["fallback_reason"]] = reasons.get(e["fallback_reason"], 0) + 1

    latency_by_source = {}
    for source in ("t5", "llm", "none"):
        values = [e["total_latency_ms"] for e in calls
                  if e.get("source") == source and e.get("total_latency_ms") is not None]
        if values:
            latency_by_source[source] = {
                "count": len(values), "median_ms": median(values),
                "p99_ms": _percentile(values, 0.99),
            }

    llm_calls = [e for e in calls if e.get("llm_latency_ms") is not None]
    costs = [_event_cost(e) for e in calls if _event_cost(e) is not None]
    spend = sum(costs) if costs else None
    cost_per_rule = spend / n if spend is not None and n else None

    return {
        "calls": n,
        "t5_answered": count(lambda e: e.get("source") == "t5" and not e.get("fallback_reason")),
        "llm_answered": count(lambda e: e.get("source") == "llm"),
        "t5_kept_after_llm_failure": count(
            lambda e: e.get("source") == "t5" and e.get("fallback_reason")),
        "unanswered": count(lambda e: e.get("source") == "none"),
        "fallbacks": count(lambda e: e.get("fallback_reason")),
        "fallback_rate": count(lambda e: e.get("fallback_reason")) / n if n else None,
        "t5_success_rate": (count(lambda e: e.get("source") == "t5"
                                  and not e.get("fallback_reason")) / n if n else None),
        "llm_calls": len(llm_calls),
        "llm_errors": count(lambda e: e.get("llm_error")),
        "llm_error_rate": (count(lambda e: e.get("llm_error")) / len(llm_calls)
                           if llm_calls else None),
        "fallback_reasons": reasons,
        "latency_by_source": latency_by_source,
        "spend": spend,
        "cost_per_rule": cost_per_rule,
        "projected_monthly_cost": (cost_per_rule * rules_per_day * 30
                                   if cost_per_rule is not None else None),
        "runs": len({e.get("run_id") for e in calls}),
    }


def failure_stream(events: List[dict], limit: int = 50) -> List[dict]:
    """Newest-first interpret events where T5 was not used as-is."""
    rows = [e for e in events
            if e.get("type") == "interpret"
            and (e.get("fallback_reason") or e.get("source") == "none")]
    return list(reversed(rows))[:limit]


# --- Training data collection ---------------------------------------------------

def normalise_rule(text: str) -> str:
    return re.sub(r"[^a-z0-9<>= ]", "", text.lower()).strip()


def known_training_inputs(pattern: str = "training_data/*.json") -> set:
    """Normalised inputs of the curated training sets (the held-out test set excluded)."""
    known = set()
    for path in glob.glob(pattern):
        if "holdout" in os.path.basename(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rows = data.get("examples", []) if isinstance(data, dict) else data
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and isinstance(row.get("input"), str):
                known.add(normalise_rule(row["input"]))
    return known


def training_summary(events: List[dict], collected_path: Optional[str] = None,
                     known_inputs: Optional[set] = None) -> dict:
    """What the collection process has gathered, from events and the collected file."""
    examples = [e for e in events if e.get("type") == "training_example"]
    by_source: Dict[str, int] = {}
    for e in examples:
        by_source[e.get("source")] = by_source.get(e.get("source"), 0) + 1

    collected = {}
    if collected_path and os.path.exists(collected_path):
        try:
            with open(collected_path, encoding="utf-8") as f:
                collected = json.load(f)
        except (OSError, json.JSONDecodeError):
            collected = {}

    known = known_inputs or set()
    recent = [
        {"ts": e.get("ts"), "rule": e.get("rule"), "output": e.get("output"),
         "source": e.get("source"),
         "already_in_training_data": normalise_rule(e.get("rule") or "") in known}
        for e in reversed(examples)
    ]
    return {
        "events": len(examples),
        "by_source": by_source,
        "collected_file_total": collected.get("total_examples"),
        "collected_file_sources": collected.get("sources"),
        "duplicates_of_training_data": sum(r["already_in_training_data"] for r in recent),
        "recent": recent,
    }
