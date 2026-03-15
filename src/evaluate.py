#!/usr/bin/env python3
"""
CLI for evaluating constraint generation models.

Usage:
    python -m src.evaluate --dataset training_data/financial_domain.json
    python -m src.evaluate --dataset training_data/healthcare_domain.json --output report.json
"""
import argparse
import json
import sys
from typing import Optional

from .model_evaluator import (
    load_evaluation_dataset,
    EvaluationSuite,
    generate_report,
    save_report,
)


class MockInterpreter:
    """Mock interpreter for testing evaluation without a real model."""

    def interpret(self, business_rule: str) -> dict:
        """Return empty dict - for testing evaluation pipeline only."""
        return {}


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for evaluation CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Evaluate constraint generation model"
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to evaluation dataset JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save evaluation report (default: stdout)"
    )
    parser.add_argument(
        "--model",
        choices=["t5", "llm", "mock"],
        default="mock",
        help="Model to evaluate (default: mock for testing)"
    )
    parser.add_argument(
        "--domain",
        help="Domain adapter to use (for T5 model)"
    )
    parser.add_argument(
        "--adapter-path",
        help="Path to LoRA adapter (for T5 model)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args(argv)

    try:
        # Load dataset
        if args.verbose:
            print(f"Loading dataset: {args.dataset}")

        test_cases = load_evaluation_dataset(args.dataset)

        if args.verbose:
            print(f"Loaded {len(test_cases)} test cases")

        # Create interpreter based on model choice
        interpreter: MockInterpreter  # Type hint for static analysis
        if args.model == "mock":
            interpreter = MockInterpreter()
            if args.verbose:
                print("Using mock interpreter (for testing pipeline)")
        elif args.model == "t5":
            # TODO: Implement T5 interpreter loading
            print("T5 interpreter not yet implemented", file=sys.stderr)
            print("Use --model mock to test evaluation pipeline", file=sys.stderr)
            return 1
        else:  # llm
            # TODO: Implement LLM interpreter loading
            print("LLM interpreter not yet implemented", file=sys.stderr)
            print("Use --model mock to test evaluation pipeline", file=sys.stderr)
            return 1

        # Run evaluation
        if args.verbose:
            print("Running evaluation...")

        suite = EvaluationSuite(interpreter=interpreter)
        results = suite.run(test_cases)

        # Generate report
        report = generate_report(results)

        # Output results
        if args.output:
            save_report(results, args.output)
            if args.verbose:
                print(f"Report saved to: {args.output}")
        else:
            # Print summary to stdout
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Total Tests:         {report['total_tests']}")
            print(f"JSON Correctness:    {report['json_correctness_rate']:.1%}")
            print(f"Schema Valid:        {report['schema_valid_rate']:.1%}")
            print(f"Exact Match:         {report['exact_match_rate']:.1%}")
            print(f"Semantic Match:      {report['semantic_match_rate']:.1%}")
            print(f"Avg Latency:         {report['average_latency_ms']:.2f}ms")
            print("=" * 60)

            if report['failures'] and args.verbose:
                print(f"\nFailures ({len(report['failures'])}):")
                for i, f in enumerate(report['failures'][:5]):  # Show first 5
                    print(f"\n  [{i+1}] Input: {f['input']}")
                    print(f"      Expected: {f['expected']}")
                    print(f"      Actual:   {f['actual']}")
                if len(report['failures']) > 5:
                    print(f"\n  ... and {len(report['failures']) - 5} more failures")

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
