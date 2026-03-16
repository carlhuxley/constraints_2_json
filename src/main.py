#!/usr/bin/env python3
"""
CLI for JSON Schema Constraint Enrichment.

Usage:
    python -m src.main --schema input.json --dict metadata.csv --output enriched.json
"""
import argparse
import json
import os
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .enricher import enrich_schema
from .llm_interpreter import OpenRouterClient
from .schema_validator import SchemaValidationError
from .hybrid_interpreter import create_hybrid_interpreter


def create_llm_client(provider: str, model: str):
    """
    Create an LLM client based on provider.

    Args:
        provider: LLM provider name (anthropic, openai, openrouter, etc.)
        model: Model identifier

    Returns:
        LLM client instance or None
    """
    if provider == "openrouter":
        return OpenRouterClient(model)

    # Placeholder - actual implementation would import provider SDK
    # For now, return None to skip LLM interpretation
    print(f"Note: LLM integration ({provider}/{model}) not configured.")
    print("Running without business rule interpretation.")
    return None


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Enrich JSON Schema with constraints from data dictionary"
    )
    parser.add_argument(
        "--schema", "-s",
        required=True,
        help="Path to input JSON Schema file"
    )
    parser.add_argument(
        "--dict", "-d",
        required=True,
        help="Path to data dictionary CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to write enriched schema (default: stdout)"
    )
    parser.add_argument(
        "--llm",
        choices=["anthropic", "openai", "openrouter", "none"],
        default="openrouter" if os.environ.get("OPENROUTER_API_KEY") else "none",
        help="LLM provider for business rule interpretation (defaults to openrouter if OPENROUTER_API_KEY is set)"
    )
    parser.add_argument(
        "--model", "-m",
        default="deepseek/deepseek-chat",
        help="LLM model to use (e.g. deepseek/deepseek-chat for DeepSeek V3 via OpenRouter)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip schema validation after enrichment"
    )
    parser.add_argument(
        "--adapter",
        help="Path to T5 LoRA adapter directory for local inference"
    )
    parser.add_argument(
        "--adapter-name",
        choices=["financial", "healthcare"],
        help="Named T5 adapter to use (looks in adapters/ directory)"
    )
    parser.add_argument(
        "--collect-training-data",
        action="store_true",
        help="Collect successful interpretations for future training"
    )
    parser.add_argument(
        "--training-log",
        default="training_data_collected.json",
        help="Path to save collected training data (default: training_data_collected.json)"
    )

    args = parser.parse_args(argv)

    # Create LLM client if requested
    llm_client = None
    if args.llm != "none":
        llm_client = create_llm_client(args.llm, args.model)

    # Create hybrid interpreter if adapter is specified
    hybrid_interpreter = None
    if args.adapter or args.adapter_name:
        if args.verbose:
            adapter_info = args.adapter or f"adapters/adapter_{args.adapter_name}"
            print(f"Loading T5 adapter: {adapter_info}")

        hybrid_interpreter = create_hybrid_interpreter(
            adapter_name=args.adapter_name,
            adapter_path=args.adapter,
            llm_client=llm_client,
            log_failures=True,
            collect_training_data=args.collect_training_data,
            training_log_path=args.training_log if args.collect_training_data else None,
        )

    try:
        if args.verbose:
            print(f"Loading schema: {args.schema}")
            print(f"Loading dictionary: {args.dict}")
            if args.no_validate:
                print("Schema validation: disabled")
            if hybrid_interpreter:
                print("Using: T5 with LLM fallback")
            elif llm_client:
                print("Using: LLM only")
            else:
                print("Using: Dictionary lookup only (no business rule interpretation)")

        result = enrich_schema(
            schema_path=args.schema,
            dict_path=args.dict,
            llm_client=llm_client,
            hybrid_interpreter=hybrid_interpreter,
            output_path=args.output,
            validate=not args.no_validate
        )

        if args.output:
            if args.verbose:
                print(f"Enriched schema written to: {args.output}")
        else:
            # Output to stdout
            print(json.dumps(result, indent=2))

        # Print hybrid interpreter stats if used
        if hybrid_interpreter and args.verbose:
            stats = hybrid_interpreter.get_stats()
            print(f"\nInterpreter stats:")
            print(f"  Total calls: {stats['total_calls']}")
            print(f"  T5 successes: {stats['t5_successes']} ({stats['t5_success_rate']})")
            print(f"  LLM fallbacks: {stats['llm_fallbacks']} ({stats['fallback_rate']})")
            if args.collect_training_data:
                print(f"  Training examples collected: {stats['training_examples_collected']}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in schema - {e}", file=sys.stderr)
        return 1
    except SchemaValidationError as e:
        print("Schema validation failed:", file=sys.stderr)
        for error in e.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
