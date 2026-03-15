"""
Model Evaluator - evaluate constraint generation models using DeepEval-style metrics.

This module provides metrics and evaluation tools for comparing LLM and T5
model performance on business rule to JSON Schema constraint conversion.
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Protocol
from statistics import mean

from .schema_validator import validate_node_constraints


class Interpreter(Protocol):
    """Protocol for constraint interpreters."""

    def interpret(self, business_rule: str) -> dict:
        """Convert business rule to JSON Schema constraint."""
        ...


@dataclass
class EvalTestCase:
    """A single evaluation test case."""

    input: str
    expected_output: str
    field_type: str = "string"
    constraint_type: Optional[str] = None

    def __post_init__(self):
        # Normalize expected_output to compact JSON string
        if isinstance(self.expected_output, dict):
            self.expected_output = json.dumps(self.expected_output, separators=(',', ':'))


@dataclass
class TestResult:
    """Result of evaluating a single test case."""

    input: str
    expected_output: str
    actual_output: str
    field_type: str
    constraint_type: Optional[str]
    json_correct: bool
    schema_valid: bool
    exact_match: bool
    semantic_match: bool
    latency_ms: float


@dataclass
class EvaluationResults:
    """Aggregate results from running evaluation."""

    test_results: List[TestResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return len(self.test_results)

    @property
    def json_correctness_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(1 for r in self.test_results if r.json_correct) / len(self.test_results)

    @property
    def schema_valid_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(1 for r in self.test_results if r.schema_valid) / len(self.test_results)

    @property
    def exact_match_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(1 for r in self.test_results if r.exact_match) / len(self.test_results)

    @property
    def semantic_match_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(1 for r in self.test_results if r.semantic_match) / len(self.test_results)

    @property
    def average_latency_ms(self) -> float:
        if not self.test_results:
            return 0.0
        return mean(r.latency_ms for r in self.test_results)

    def get_failures(self) -> List[TestResult]:
        """Get all test cases that failed exact match."""
        return [r for r in self.test_results if not r.exact_match]

    def group_by_constraint_type(self) -> Dict[str, 'EvaluationResults']:
        """Group results by constraint type."""
        groups: Dict[str, List[TestResult]] = {}
        for result in self.test_results:
            ctype = result.constraint_type or "unknown"
            if ctype not in groups:
                groups[ctype] = []
            groups[ctype].append(result)

        return {
            ctype: EvaluationResults(test_results=results)
            for ctype, results in groups.items()
        }


def create_evaluation_dataset(training_data: List[Dict]) -> List[EvalTestCase]:
    """
    Create evaluation dataset from training examples.

    Args:
        training_data: List of {"input": str, "output": dict} pairs

    Returns:
        List of EvalTestCase objects
    """
    dataset = []
    for example in training_data:
        output = example["output"]
        if isinstance(output, dict):
            output = json.dumps(output, separators=(',', ':'))

        dataset.append(EvalTestCase(
            input=example["input"],
            expected_output=output,
            field_type=example.get("field_type", "string"),
            constraint_type=example.get("constraint_type"),
        ))
    return dataset


def load_evaluation_dataset(path: str) -> List[EvalTestCase]:
    """
    Load evaluation dataset from JSON file.

    Args:
        path: Path to training data JSON file

    Returns:
        List of EvalTestCase objects
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = data.get("examples", data)
    return create_evaluation_dataset(examples)


class JsonCorrectnessMetric:
    """Metric for checking if output is valid JSON object."""

    def measure(self, output: str) -> float:
        """
        Measure if output is valid JSON object.

        Args:
            output: Model output string

        Returns:
            1.0 if valid JSON object, 0.0 otherwise
        """
        try:
            parsed = json.loads(output)
            # Must be a dict/object, not a primitive or array
            if isinstance(parsed, dict):
                return 1.0
            return 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0


class SchemaValidMetric:
    """Metric for checking if constraints are valid for field type."""

    def measure(self, actual_output: str, field_type: str) -> float:
        """
        Measure if constraints are valid for the field type.

        Args:
            actual_output: Model output JSON string
            field_type: The JSON Schema type (string, integer, etc.)

        Returns:
            1.0 if valid, 0.0 otherwise
        """
        try:
            parsed = json.loads(actual_output)
            if not isinstance(parsed, dict):
                return 0.0

            # Create a node with the type and constraints
            node = {"type": field_type, **parsed}
            errors = validate_node_constraints(node, "test")

            return 1.0 if len(errors) == 0 else 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0


class ExactMatchMetric:
    """Metric for checking exact match between expected and actual output."""

    def measure(self, actual_output: str, expected_output: str) -> float:
        """
        Measure if outputs match exactly.

        Args:
            actual_output: Model output JSON string
            expected_output: Expected JSON string

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        try:
            actual = json.loads(actual_output)
            expected = json.loads(expected_output)
            return 1.0 if actual == expected else 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0


class SemanticMatchMetric:
    """Metric for checking semantic equivalence (ignoring key order)."""

    def measure(self, actual_output: str, expected_output: str) -> float:
        """
        Measure if outputs are semantically equivalent.

        For JSON objects, this means same keys and values regardless of order.

        Args:
            actual_output: Model output JSON string
            expected_output: Expected JSON string

        Returns:
            1.0 if semantically equivalent, 0.0 otherwise
        """
        try:
            actual = json.loads(actual_output)
            expected = json.loads(expected_output)

            # For dicts, compare as sets of items
            if isinstance(actual, dict) and isinstance(expected, dict):
                return 1.0 if actual == expected else 0.0

            return 1.0 if actual == expected else 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0


class LatencyMetric:
    """Metric for tracking inference latency."""

    def __init__(self):
        self._latencies: List[float] = []
        self.last_latency_ms: float = 0.0

    def record(self, start: float, end: float) -> None:
        """Record latency from start/end timestamps."""
        latency_ms = (end - start) * 1000
        self._latencies.append(latency_ms)
        self.last_latency_ms = latency_ms

    def record_ms(self, latency_ms: float) -> None:
        """Record latency directly in milliseconds."""
        self._latencies.append(latency_ms)
        self.last_latency_ms = latency_ms

    @property
    def average_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return mean(self._latencies)

    @property
    def p99_ms(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class EvaluationSuite:
    """Run evaluation suite on a model interpreter."""

    def __init__(self, interpreter: Interpreter):
        """
        Initialize evaluation suite.

        Args:
            interpreter: Model interpreter with interpret() method
        """
        self.interpreter = interpreter
        self.json_metric = JsonCorrectnessMetric()
        self.schema_metric = SchemaValidMetric()
        self.exact_metric = ExactMatchMetric()
        self.semantic_metric = SemanticMatchMetric()
        self.latency_metric = LatencyMetric()

    def run(self, test_cases: List[EvalTestCase]) -> EvaluationResults:
        """
        Run evaluation on test cases.

        Args:
            test_cases: List of test cases to evaluate

        Returns:
            EvaluationResults with per-test and aggregate results
        """
        results = EvaluationResults()

        for test_case in test_cases:
            # Run inference with timing
            start = time.perf_counter()
            output_dict = self.interpreter.interpret(test_case.input)
            end = time.perf_counter()

            # Convert to JSON string
            actual_output = json.dumps(output_dict, separators=(',', ':'))

            # Calculate metrics
            json_correct = self.json_metric.measure(actual_output) == 1.0
            schema_valid = self.schema_metric.measure(actual_output, test_case.field_type) == 1.0
            exact_match = self.exact_metric.measure(actual_output, test_case.expected_output) == 1.0
            semantic_match = self.semantic_metric.measure(actual_output, test_case.expected_output) == 1.0
            latency_ms = (end - start) * 1000

            results.test_results.append(TestResult(
                input=test_case.input,
                expected_output=test_case.expected_output,
                actual_output=actual_output,
                field_type=test_case.field_type,
                constraint_type=test_case.constraint_type,
                json_correct=json_correct,
                schema_valid=schema_valid,
                exact_match=exact_match,
                semantic_match=semantic_match,
                latency_ms=latency_ms,
            ))

        return results


def compare_models(
    model_a: Interpreter,
    model_b: Interpreter,
    test_cases: List[EvalTestCase],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B"
) -> Dict[str, EvaluationResults]:
    """
    Compare two models on the same test cases.

    Args:
        model_a: First model interpreter
        model_b: Second model interpreter
        test_cases: Test cases to evaluate
        model_a_name: Display name for first model
        model_b_name: Display name for second model

    Returns:
        Dict mapping model names to their EvaluationResults
    """
    suite_a = EvaluationSuite(interpreter=model_a)
    suite_b = EvaluationSuite(interpreter=model_b)

    return {
        model_a_name: suite_a.run(test_cases),
        model_b_name: suite_b.run(test_cases),
    }


def generate_report(results: EvaluationResults) -> Dict[str, Any]:
    """
    Generate evaluation report from results.

    Args:
        results: EvaluationResults from running evaluation

    Returns:
        Report dict with summary statistics
    """
    return {
        "total_tests": results.total_tests,
        "json_correctness_rate": results.json_correctness_rate,
        "schema_valid_rate": results.schema_valid_rate,
        "exact_match_rate": results.exact_match_rate,
        "semantic_match_rate": results.semantic_match_rate,
        "average_latency_ms": results.average_latency_ms,
        "failures": [
            {
                "input": f.input,
                "expected": f.expected_output,
                "actual": f.actual_output,
            }
            for f in results.get_failures()
        ],
    }


def save_report(results: EvaluationResults, path: str) -> None:
    """
    Save evaluation report to file.

    Args:
        results: EvaluationResults from running evaluation
        path: Path to save report JSON
    """
    report = generate_report(results)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
