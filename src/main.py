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

    args = parser.parse_args(argv)

    # Create LLM client if requested
    llm_client = None
    if args.llm != "none":
        llm_client = create_llm_client(args.llm, args.model)

    try:
        if args.verbose:
            print(f"Loading schema: {args.schema}")
            print(f"Loading dictionary: {args.dict}")

        result = enrich_schema(
            schema_path=args.schema,
            dict_path=args.dict,
            llm_client=llm_client,
            output_path=args.output
        )

        if args.output:
            if args.verbose:
                print(f"Enriched schema written to: {args.output}")
        else:
            # Output to stdout
            print(json.dumps(result, indent=2))

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in schema - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
