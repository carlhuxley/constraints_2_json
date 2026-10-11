"""Does T5's confidence separate right answers from wrong ones?

Scores the adapter's own validation split (the 10% train_adapter_simple.py held back,
so the adapter never trained on it) and reports, for each candidate threshold, how many
T5 errors a confidence gate would send to the LLM and at what fallback rate. Use it to
fix --confidence-threshold for run_experiments.py *before* Hybrid sees the held-out set.

Usage (from the repo root):
    python scripts/confidence_calibration.py
    python scripts/confidence_calibration.py --dataset training_data/financial_domain.json \\
        --adapter adapters/adapter_financial_v1_rebuild/final_adapter
"""
import argparse
import json
import os
import sys
from datetime import date
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_evaluator import SemanticMatchMetric  # noqa: E402

DEFAULT_DATASET = "training_data/financial_domain_combined.json"
DEFAULT_ADAPTER = "adapters/adapter_financial_v2/final_adapter"
DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def auroc(confidences: List[float], correct: List[bool]) -> Optional[float]:
    """P(a random correct answer outscores a random wrong one); ties count half.
    0.5 = confidence carries no signal, 1.0 = perfect separation."""
    right = [c for c, ok in zip(confidences, correct) if ok]
    wrong = [c for c, ok in zip(confidences, correct) if not ok]
    if not right or not wrong:
        return None
    wins = sum((r > w) + 0.5 * (r == w) for r in right for w in wrong)
    return wins / (len(right) * len(wrong))


def threshold_table(confidences: List[float], correct: List[bool],
                    thresholds: List[float]) -> List[dict]:
    """For each threshold: fallback rate, share of T5 errors caught, and the
    accuracy ceiling if the LLM got every fallback right."""
    n = len(correct)
    errors = n - sum(correct)
    rows = []
    for t in thresholds:
        flagged = [c < t for c in confidences]
        caught = sum(f and not ok for f, ok in zip(flagged, correct))
        needless = sum(f and ok for f, ok in zip(flagged, correct))
        rows.append({
            "threshold": t,
            "fallback_rate": sum(flagged) / n,
            "errors_caught": caught,
            "errors_caught_rate": caught / errors if errors else None,
            "correct_sent_to_llm": needless,
            "accuracy_ceiling": (sum(correct) + caught) / n,
        })
    return rows


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Training file the adapter was trained on (its split is reused)")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", help="JSON report path "
                        "(default: docs/experiments/confidence_<date>_<adapter>.json)")
    args = parser.parse_args(argv)

    from src.t5_interpreter import T5Interpreter
    from src.train_adapter_simple import load_dataset, split_examples

    _, val_examples = split_examples(load_dataset(args.dataset))
    t5 = T5Interpreter(adapter_path=args.adapter)
    metric = SemanticMatchMetric()

    cases = []
    for ex in val_examples:
        output = t5.interpret(ex["input"])
        ok = metric.measure(json.dumps(output), json.dumps(ex["output"])) == 1.0
        cases.append({
            "input": ex["input"],
            "expected": ex["output"],
            "actual": output,
            "correct": ok,
            "confidence": t5.last_confidence,
        })

    confidences = [c["confidence"] for c in cases]
    correct = [c["correct"] for c in cases]
    report = {
        "date": date.today().isoformat(),
        "dataset": args.dataset,
        "adapter": args.adapter,
        "validation_examples": len(cases),
        "semantic_match_rate": sum(correct) / len(cases),
        "auroc": auroc(confidences, correct),
        "thresholds": threshold_table(confidences, correct, args.thresholds),
        "cases": sorted(cases, key=lambda c: c["confidence"]),
    }

    adapter_name = os.path.basename(os.path.dirname(os.path.normpath(args.adapter)))
    output = args.output or f"docs/experiments/confidence_{report['date']}_{adapter_name}.json"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"{len(cases)} validation examples, semantic match {report['semantic_match_rate']:.1%}, "
          f"AUROC {report['auroc']:.3f}" if report["auroc"] is not None
          else f"{len(cases)} validation examples - AUROC undefined (no errors or no successes)")
    print(f"{'Threshold':>10}{'Fallback':>10}{'Caught':>10}{'Needless':>10}{'Ceiling':>10}")
    for row in report["thresholds"]:
        caught = (f"{row['errors_caught']}" if row["errors_caught_rate"] is None
                  else f"{row['errors_caught']} ({row['errors_caught_rate']:.0%})")
        print(f"{row['threshold']:>10}{row['fallback_rate']:>10.1%}{caught:>10}"
              f"{row['correct_sent_to_llm']:>10}{row['accuracy_ceiling']:>10.1%}")
    print("\nT5 errors (confidence, input, actual -> expected):")
    for c in report["cases"]:
        if not c["correct"]:
            print(f"  {c['confidence']:.3f}  {c['input']!r}: {c['actual']} -> {c['expected']}")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
