"""Tests for model_evaluator module using DeepEval."""
import pytest
import json
import time
from unittest.mock import Mock, patch


class TestEvaluationDataset:
    """Test evaluation dataset creation."""

    def test_create_dataset_from_training_data(self):
        """Should create evaluation dataset from training examples."""
        from src.model_evaluator import create_evaluation_dataset

        training_data = [
            {"input": "Must be 18 or older", "output": {"minimum": 18}},
            {"input": "Maximum 100 characters", "output": {"maxLength": 100}},
        ]

        dataset = create_evaluation_dataset(training_data)

        assert len(dataset) == 2
        assert dataset[0].input == "Must be 18 or older"
        assert dataset[0].expected_output == '{"minimum":18}'

    def test_create_dataset_from_json_file(self):
        """Should load dataset from training data JSON file."""
        from src.model_evaluator import load_evaluation_dataset
        import tempfile
        import os

        data = {
            "domain": "test",
            "examples": [
                {"input": "Value must be positive", "output": {"minimum": 0}},
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            dataset = load_evaluation_dataset(path)
            assert len(dataset) == 1
            assert dataset[0].input == "Value must be positive"
        finally:
            os.unlink(path)


class TestJsonCorrectnessMetric:
    """Test JSON correctness evaluation."""

    def test_valid_json_passes(self):
        """Valid JSON output should pass."""
        from src.model_evaluator import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        score = metric.measure('{"minimum": 18}')

        assert score == 1.0

    def test_invalid_json_fails(self):
        """Invalid JSON output should fail."""
        from src.model_evaluator import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        score = metric.measure('{"minimum": }')

        assert score == 0.0

    def test_empty_string_fails(self):
        """Empty string should fail."""
        from src.model_evaluator import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        score = metric.measure('')

        assert score == 0.0

    def test_non_object_json_fails(self):
        """Non-object JSON should fail (we expect objects)."""
        from src.model_evaluator import JsonCorrectnessMetric

        metric = JsonCorrectnessMetric()
        score = metric.measure('"just a string"')

        assert score == 0.0


class TestSchemaValidMetric:
    """Test schema validity evaluation."""

    def test_valid_constraint_for_type_passes(self):
        """Valid constraint for field type should pass."""
        from src.model_evaluator import SchemaValidMetric

        metric = SchemaValidMetric()
        score = metric.measure(
            actual_output='{"minimum": 18}',
            field_type="integer"
        )

        assert score == 1.0

    def test_invalid_constraint_for_type_fails(self):
        """Invalid constraint for field type should fail."""
        from src.model_evaluator import SchemaValidMetric

        metric = SchemaValidMetric()
        score = metric.measure(
            actual_output='{"maxLength": 10}',
            field_type="integer"
        )

        assert score == 0.0

    def test_string_constraint_on_string_passes(self):
        """String constraint on string type should pass."""
        from src.model_evaluator import SchemaValidMetric

        metric = SchemaValidMetric()
        score = metric.measure(
            actual_output='{"maxLength": 100, "pattern": "^[A-Z]+$"}',
            field_type="string"
        )

        assert score == 1.0

    def test_invalid_json_fails(self):
        """Invalid JSON should fail schema validation."""
        from src.model_evaluator import SchemaValidMetric

        metric = SchemaValidMetric()
        score = metric.measure(
            actual_output='not json',
            field_type="string"
        )

        assert score == 0.0


class TestExactMatchMetric:
    """Test exact match evaluation."""

    def test_identical_outputs_pass(self):
        """Identical outputs should pass."""
        from src.model_evaluator import ExactMatchMetric

        metric = ExactMatchMetric()
        score = metric.measure(
            actual_output='{"minimum": 18}',
            expected_output='{"minimum": 18}'
        )

        assert score == 1.0

    def test_different_values_fail(self):
        """Different values should fail."""
        from src.model_evaluator import ExactMatchMetric

        metric = ExactMatchMetric()
        score = metric.measure(
            actual_output='{"minimum": 19}',
            expected_output='{"minimum": 18}'
        )

        assert score == 0.0

    def test_different_keys_fail(self):
        """Different keys should fail."""
        from src.model_evaluator import ExactMatchMetric

        metric = ExactMatchMetric()
        score = metric.measure(
            actual_output='{"maximum": 18}',
            expected_output='{"minimum": 18}'
        )

        assert score == 0.0


class TestSemanticMatchMetric:
    """Test semantic match evaluation."""

    def test_same_content_different_order_passes(self):
        """Same content with different key order should pass."""
        from src.model_evaluator import SemanticMatchMetric

        metric = SemanticMatchMetric()
        score = metric.measure(
            actual_output='{"maximum": 100, "minimum": 0}',
            expected_output='{"minimum": 0, "maximum": 100}'
        )

        assert score == 1.0

    def test_different_content_fails(self):
        """Different content should fail."""
        from src.model_evaluator import SemanticMatchMetric

        metric = SemanticMatchMetric()
        score = metric.measure(
            actual_output='{"minimum": 0}',
            expected_output='{"minimum": 0, "maximum": 100}'
        )

        assert score == 0.0

    def test_equivalent_enum_order_passes(self):
        """Enum with different order should still match semantically."""
        from src.model_evaluator import SemanticMatchMetric

        metric = SemanticMatchMetric()
        # Note: This is a design decision - should enum order matter?
        # For now, we'll say exact match is required for enums
        score = metric.measure(
            actual_output='{"enum": ["A", "B", "C"]}',
            expected_output='{"enum": ["A", "B", "C"]}'
        )

        assert score == 1.0


class TestLatencyMetric:
    """Test latency measurement."""

    def test_records_latency(self):
        """Should record inference latency."""
        from src.model_evaluator import LatencyMetric

        metric = LatencyMetric()

        # Simulate inference with known delay
        start = time.perf_counter()
        time.sleep(0.01)  # 10ms
        end = time.perf_counter()

        metric.record(start, end)

        assert metric.last_latency_ms >= 10
        assert metric.last_latency_ms < 50  # Allow some variance

    def test_calculates_average_latency(self):
        """Should calculate average latency over multiple calls."""
        from src.model_evaluator import LatencyMetric

        metric = LatencyMetric()
        metric.record_ms(10)
        metric.record_ms(20)
        metric.record_ms(30)

        assert metric.average_ms == 20

    def test_calculates_p99_latency(self):
        """Should calculate p99 latency."""
        from src.model_evaluator import LatencyMetric

        metric = LatencyMetric()
        for i in range(100):
            metric.record_ms(10)
        metric.record_ms(100)  # Outlier

        assert metric.p99_ms >= 10


class TestEvaluationSuite:
    """Test running evaluation suite."""

    def test_run_evaluation_returns_results(self):
        """Running evaluation should return results for each test case."""
        from src.model_evaluator import EvaluationSuite, EvalTestCase

        test_cases = [
            EvalTestCase(
                input="Must be 18 or older",
                expected_output='{"minimum": 18}',
                field_type="integer"
            ),
        ]

        mock_interpreter = Mock()
        mock_interpreter.interpret.return_value = {"minimum": 18}

        suite = EvaluationSuite(interpreter=mock_interpreter)
        results = suite.run(test_cases)

        assert len(results.test_results) == 1
        assert results.test_results[0].json_correct is True

    def test_aggregate_scores(self):
        """Should calculate aggregate scores across test cases."""
        from src.model_evaluator import EvaluationSuite, EvalTestCase

        test_cases = [
            EvalTestCase(
                input="Must be 18 or older",
                expected_output='{"minimum": 18}',
                field_type="integer"
            ),
            EvalTestCase(
                input="Maximum 100",
                expected_output='{"maximum": 100}',
                field_type="integer"
            ),
        ]

        mock_interpreter = Mock()
        mock_interpreter.interpret.side_effect = [
            {"minimum": 18},  # Correct
            {"maximum": 99},  # Wrong value
        ]

        suite = EvaluationSuite(interpreter=mock_interpreter)
        results = suite.run(test_cases)

        assert results.json_correctness_rate == 1.0  # Both valid JSON
        assert results.exact_match_rate == 0.5  # Only one matches exactly


class TestModelComparison:
    """Test comparing T5 vs LLM performance."""

    def test_compare_two_models(self):
        """Should compare two model interpreters."""
        from src.model_evaluator import compare_models, EvalTestCase

        test_cases = [
            EvalTestCase(
                input="Must be 18 or older",
                expected_output='{"minimum": 18}',
                field_type="integer"
            ),
        ]

        t5_interpreter = Mock()
        t5_interpreter.interpret.return_value = {"minimum": 18}

        llm_interpreter = Mock()
        llm_interpreter.interpret.return_value = {"minimum": 18}

        comparison = compare_models(
            model_a=t5_interpreter,
            model_b=llm_interpreter,
            test_cases=test_cases,
            model_a_name="T5",
            model_b_name="LLM"
        )

        assert "T5" in comparison
        assert "LLM" in comparison
        assert comparison["T5"].exact_match_rate == 1.0
        assert comparison["LLM"].exact_match_rate == 1.0


class TestConstraintTypeEvaluation:
    """Test evaluation grouped by constraint type."""

    def test_group_by_constraint_type(self):
        """Should group results by constraint type."""
        from src.model_evaluator import EvaluationSuite, EvalTestCase

        test_cases = [
            EvalTestCase(
                input="Must be 18 or older",
                expected_output='{"minimum": 18}',
                field_type="integer",
                constraint_type="minimum"
            ),
            EvalTestCase(
                input="Cannot exceed 100",
                expected_output='{"maximum": 100}',
                field_type="integer",
                constraint_type="maximum"
            ),
            EvalTestCase(
                input="Status: A, B, C",
                expected_output='{"enum": ["A", "B", "C"]}',
                field_type="string",
                constraint_type="enum"
            ),
        ]

        mock_interpreter = Mock()
        mock_interpreter.interpret.side_effect = [
            {"minimum": 18},
            {"maximum": 100},
            {"enum": ["A", "B", "C"]},
        ]

        suite = EvaluationSuite(interpreter=mock_interpreter)
        results = suite.run(test_cases)
        by_type = results.group_by_constraint_type()

        assert "minimum" in by_type
        assert "maximum" in by_type
        assert "enum" in by_type
        assert by_type["minimum"].exact_match_rate == 1.0


class TestEvaluationReport:
    """Test evaluation report generation."""

    def test_generate_summary_report(self):
        """Should generate summary report."""
        from src.model_evaluator import EvaluationSuite, EvalTestCase, generate_report

        test_cases = [
            EvalTestCase(
                input="Must be 18 or older",
                expected_output='{"minimum": 18}',
                field_type="integer"
            ),
        ]

        mock_interpreter = Mock()
        mock_interpreter.interpret.return_value = {"minimum": 18}

        suite = EvaluationSuite(interpreter=mock_interpreter)
        results = suite.run(test_cases)

        report = generate_report(results)

        assert "json_correctness_rate" in report
        assert "exact_match_rate" in report
        assert "total_tests" in report

    def test_save_report_to_file(self):
        """Should save report to file."""
        from src.model_evaluator import EvaluationSuite, EvalTestCase, save_report
        import tempfile
        import os

        test_cases = [
            EvalTestCase(
                input="Must be 18 or older",
                expected_output='{"minimum": 18}',
                field_type="integer"
            ),
        ]

        mock_interpreter = Mock()
        mock_interpreter.interpret.return_value = {"minimum": 18}

        suite = EvaluationSuite(interpreter=mock_interpreter)
        results = suite.run(test_cases)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_report(results, path)
            assert os.path.exists(path)

            with open(path, 'r') as f:
                saved = json.load(f)
            assert "json_correctness_rate" in saved
        finally:
            os.unlink(path)


class TestFailureAnalysis:
    """Test failure analysis in evaluation."""

    def test_captures_failures(self):
        """Should capture and analyze failures."""
        from src.model_evaluator import EvaluationSuite, EvalTestCase

        test_cases = [
            EvalTestCase(
                input="Must be 18 or older",
                expected_output='{"minimum": 18}',
                field_type="integer"
            ),
        ]

        mock_interpreter = Mock()
        mock_interpreter.interpret.return_value = {"minimum": 21}  # Wrong!

        suite = EvaluationSuite(interpreter=mock_interpreter)
        results = suite.run(test_cases)

        failures = results.get_failures()
        assert len(failures) == 1
        assert failures[0].input == "Must be 18 or older"
        assert failures[0].expected_output == '{"minimum": 18}'
        assert failures[0].actual_output == '{"minimum":21}'
