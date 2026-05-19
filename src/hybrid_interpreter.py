"""
Hybrid Interpreter - T5 with LLM fallback for constraint generation.

This module provides a hybrid approach that:
1. Attempts fast T5 inference first
2. Validates the output against the field type
3. Falls back to LLM if T5 fails or produces invalid output
4. Logs all failures for analysis and model improvement
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Protocol

from .schema_validator import validate_node_constraints


# Configure logging
logger = logging.getLogger(__name__)


class Interpreter(Protocol):
    """Protocol for constraint interpreters."""

    def interpret(self, business_rule: str) -> dict:
        """Convert business rule to JSON Schema constraints."""
        ...


@dataclass
class TrainingRecord:
    """Record of a successful interpretation for training data collection."""

    timestamp: str
    input: str  # business_rule
    output: dict  # JSON constraints
    field_name: str
    field_type: str
    source: str  # "t5" or "llm"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "input": self.input,
            "output": self.output,
            "field_name": self.field_name,
            "field_type": self.field_type,
            "source": self.source,
        }

    def to_training_example(self) -> dict:
        """Convert to training data format (input/output only)."""
        return {
            "input": self.input,
            "output": self.output,
            "field_type": self.field_type,
        }


@dataclass
class FailureRecord:
    """Record of a T5 interpretation failure."""

    timestamp: str
    field_name: str
    field_type: str
    business_rule: str
    t5_output: dict
    validation_errors: List[str]
    fallback_output: Optional[dict] = None
    fallback_source: str = "llm"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "field_name": self.field_name,
            "field_type": self.field_type,
            "business_rule": self.business_rule,
            "t5_output": self.t5_output,
            "validation_errors": self.validation_errors,
            "fallback_output": self.fallback_output,
            "fallback_source": self.fallback_source,
        }


@dataclass
class HybridStats:
    """Statistics for hybrid interpreter performance."""

    total_calls: int = 0
    t5_successes: int = 0
    t5_failures: int = 0
    llm_fallbacks: int = 0
    llm_failures: int = 0
    variations_generated: int = 0
    failures: List[FailureRecord] = field(default_factory=list)
    training_log: List[TrainingRecord] = field(default_factory=list)

    @property
    def t5_success_rate(self) -> float:
        """Percentage of successful T5 interpretations."""
        if self.total_calls == 0:
            return 0.0
        return self.t5_successes / self.total_calls

    @property
    def fallback_rate(self) -> float:
        """Percentage of calls that required LLM fallback."""
        if self.total_calls == 0:
            return 0.0
        return self.llm_fallbacks / self.total_calls

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "total_calls": self.total_calls,
            "t5_successes": self.t5_successes,
            "t5_failures": self.t5_failures,
            "llm_fallbacks": self.llm_fallbacks,
            "llm_failures": self.llm_failures,
            "variations_generated": self.variations_generated,
            "t5_success_rate": f"{self.t5_success_rate:.1%}",
            "fallback_rate": f"{self.fallback_rate:.1%}",
            "training_examples_collected": len(self.training_log),
        }


class HybridInterpreter:
    """
    Hybrid interpreter that uses T5 with LLM fallback.

    Provides fast local inference when possible, with automatic
    fallback to LLM for edge cases or failures.
    """

    def __init__(
        self,
        t5_interpreter: Optional[Interpreter] = None,
        llm_client: Optional[object] = None,
        log_failures: bool = True,
        failure_log_path: Optional[str] = None,
        collect_training_data: bool = False,
        training_log_path: Optional[str] = None,
    ):
        """
        Initialize hybrid interpreter.

        Args:
            t5_interpreter: T5 interpreter instance (optional)
            llm_client: LLM client for fallback (optional)
            log_failures: Whether to log T5 failures
            failure_log_path: Path to write failure log JSON
            collect_training_data: Whether to log successful interpretations
            training_log_path: Path to write training data JSON
        """
        self.t5 = t5_interpreter
        self.llm_client = llm_client
        self.log_failures = log_failures
        self.failure_log_path = failure_log_path or "t5_failures.json"
        self.collect_training_data = collect_training_data
        self.training_log_path = training_log_path or "training_data_collected.json"
        self.stats = HybridStats()

    def interpret(
        self,
        field_name: str,
        field_type: str,
        business_rule: str,
    ) -> dict:
        """
        Interpret business rule to JSON Schema constraints.

        Tries T5 first, validates output, falls back to LLM if needed.

        Args:
            field_name: Name of the field being constrained
            field_type: JSON Schema type (string, integer, etc.)
            business_rule: Free-text business rule

        Returns:
            Dict of JSON Schema constraints
        """
        self.stats.total_calls += 1

        # Try T5 first if available
        if self.t5 is not None:
            t5_result = self._try_t5(field_name, field_type, business_rule)
            if t5_result is not None:
                self.stats.t5_successes += 1
                self._log_training_data(
                    field_name, field_type, business_rule, t5_result, "t5"
                )
                return t5_result

            # T5 failed, try LLM fallback
            self.stats.t5_failures += 1

        # Fall back to LLM
        if self.llm_client is not None:
            llm_result = self._try_llm(field_name, field_type, business_rule)
            if llm_result:
                self.stats.llm_fallbacks += 1
                self._log_training_data(
                    field_name, field_type, business_rule, llm_result, "llm"
                )
                if self.collect_training_data:
                    self._log_llm_variations(field_name, field_type, business_rule, llm_result)
                return llm_result
            self.stats.llm_failures += 1

        # Both failed
        logger.warning(
            f"Both T5 and LLM failed for: {business_rule}"
        )
        return {}

    def _try_t5(
        self,
        field_name: str,
        field_type: str,
        business_rule: str,
    ) -> Optional[dict]:
        """
        Attempt T5 interpretation with validation.

        Args:
            field_name: Field name
            field_type: Field type
            business_rule: Business rule text

        Returns:
            Constraints dict if successful, None if failed
        """
        try:
            result = self.t5.interpret(business_rule)

            if not result:
                self._log_failure(
                    field_name, field_type, business_rule,
                    {}, ["Empty output from T5"]
                )
                return None

            # Validate constraints against field type
            test_node = {"type": field_type, **result}
            errors = validate_node_constraints(test_node, field_name)

            if errors:
                self._log_failure(
                    field_name, field_type, business_rule,
                    result, errors
                )
                return None

            return result

        except Exception as e:
            logger.exception(f"T5 interpretation error: {e}")
            self._log_failure(
                field_name, field_type, business_rule,
                {}, [f"Exception: {str(e)}"]
            )
            return None

    def _try_llm(
        self,
        field_name: str,
        field_type: str,
        business_rule: str,
    ) -> Optional[dict]:
        """
        Attempt LLM interpretation.

        Args:
            field_name: Field name
            field_type: Field type
            business_rule: Business rule text

        Returns:
            Constraints dict if successful, None if failed
        """
        try:
            from .llm_interpreter import interpret_business_rule

            result = interpret_business_rule(
                field_name=field_name,
                field_type=field_type,
                business_rule=business_rule,
                llm_client=self.llm_client,
            )

            # Update the last failure record with LLM result
            if self.stats.failures and self.log_failures:
                self.stats.failures[-1].fallback_output = result
                self._save_failures()

            return result if result else None

        except Exception as e:
            logger.exception(f"LLM interpretation error: {e}")
            return None

    def _log_failure(
        self,
        field_name: str,
        field_type: str,
        business_rule: str,
        t5_output: dict,
        errors: List[str],
    ) -> None:
        """Log a T5 failure for later analysis."""
        if not self.log_failures:
            return

        record = FailureRecord(
            timestamp=datetime.now().isoformat(),
            field_name=field_name,
            field_type=field_type,
            business_rule=business_rule,
            t5_output=t5_output,
            validation_errors=errors,
        )

        self.stats.failures.append(record)
        logger.info(
            f"T5 failure logged: {business_rule} -> {t5_output} ({errors})"
        )

        self._save_failures()

    def _save_failures(self) -> None:
        """Save failures to JSON file."""
        if not self.failure_log_path:
            return

        try:
            data = {
                "stats": self.stats.to_dict(),
                "failures": [f.to_dict() for f in self.stats.failures],
            }
            with open(self.failure_log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save failure log: {e}")

    def _log_llm_variations(
        self,
        field_name: str,
        field_type: str,
        business_rule: str,
        output: dict,
    ) -> None:
        """Generate and log input variations for a successful LLM result."""
        from .llm_interpreter import generate_rule_variations

        variations = generate_rule_variations(business_rule, self.llm_client)
        count = 0
        for variation in variations:
            if variation != business_rule:
                self._log_training_data(field_name, field_type, variation, output, "llm")
                count += 1
        if count:
            self.stats.variations_generated += count
            logger.info(f"Generated {count} input variations for: {business_rule}")

    def _log_training_data(
        self,
        field_name: str,
        field_type: str,
        business_rule: str,
        output: dict,
        source: str,
    ) -> None:
        """Log a successful interpretation for training data collection."""
        if not self.collect_training_data:
            return

        record = TrainingRecord(
            timestamp=datetime.now().isoformat(),
            input=business_rule,
            output=output,
            field_name=field_name,
            field_type=field_type,
            source=source,
        )

        self.stats.training_log.append(record)
        logger.debug(
            f"Training data logged: {business_rule} -> {output} (source: {source})"
        )

        self._save_training_data()

    def _save_training_data(self) -> None:
        """Save training data to JSON file."""
        if not self.training_log_path:
            return

        try:
            data = {
                "collected_at": datetime.now().isoformat(),
                "total_examples": len(self.stats.training_log),
                "sources": {
                    "t5": sum(1 for r in self.stats.training_log if r.source == "t5"),
                    "llm": sum(1 for r in self.stats.training_log if r.source == "llm"),
                },
                "examples": [r.to_training_example() for r in self.stats.training_log],
            }
            with open(self.training_log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

    def get_training_data(self) -> List[dict]:
        """
        Get collected training data in training format.

        Returns:
            List of {"input": str, "output": dict} training examples
        """
        return [r.to_training_example() for r in self.stats.training_log]

    def export_training_data(self, path: str, domain: str = "production") -> None:
        """
        Export collected training data in dataset format.

        Args:
            path: Output file path
            domain: Domain name for the dataset
        """
        data = {
            "domain": domain,
            "description": f"Training data collected from production ({len(self.stats.training_log)} examples)",
            "collected_at": datetime.now().isoformat(),
            "examples": [r.to_training_example() for r in self.stats.training_log],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.stats.training_log)} training examples to {path}")

    def get_stats(self) -> dict:
        """Get current statistics."""
        return self.stats.to_dict()

    def get_failures(self) -> List[dict]:
        """Get list of failure records."""
        return [f.to_dict() for f in self.stats.failures]


def create_hybrid_interpreter(
    adapter_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    llm_client: Optional[object] = None,
    log_failures: bool = True,
    failure_log_path: Optional[str] = None,
    collect_training_data: bool = False,
    training_log_path: Optional[str] = None,
) -> HybridInterpreter:
    """
    Create a hybrid interpreter with optional T5 and LLM.

    Args:
        adapter_name: Name of T5 adapter to load (e.g., "financial")
        adapter_path: Direct path to adapter (alternative to adapter_name)
        llm_client: LLM client for fallback
        log_failures: Whether to log T5 failures
        failure_log_path: Path for failure log
        collect_training_data: Whether to log successful interpretations
        training_log_path: Path to write collected training data

    Returns:
        Configured HybridInterpreter
    """
    t5_interpreter = None

    # Try to load T5 interpreter
    if adapter_name or adapter_path:
        try:
            from .t5_interpreter import load_adapter, T5Interpreter

            if adapter_path:
                t5_interpreter = T5Interpreter(adapter_path=adapter_path)
            else:
                t5_interpreter = load_adapter(adapter_name)

            logger.info(f"T5 interpreter loaded: {adapter_name or adapter_path}")

        except ImportError as e:
            logger.warning(
                f"T5 dependencies not available: {e}. "
                "Install with: pip install transformers peft torch"
            )
        except FileNotFoundError as e:
            logger.warning(f"T5 adapter not found: {e}")
        except Exception as e:
            logger.warning(f"Failed to load T5 interpreter: {e}")

    return HybridInterpreter(
        t5_interpreter=t5_interpreter,
        llm_client=llm_client,
        log_failures=log_failures,
        failure_log_path=failure_log_path,
        collect_training_data=collect_training_data,
        training_log_path=training_log_path,
    )
