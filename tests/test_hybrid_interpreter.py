"""Tests for hybrid interpreter module."""
import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock


class TestHybridInterpreter:
    """Test hybrid interpreter behavior."""

    def test_t5_success_no_fallback(self):
        """Should use T5 result when valid."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"minimum": 18}

        mock_llm = Mock()

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=mock_llm,
            log_failures=False,
        )

        result = interpreter.interpret("age", "integer", "Must be 18 or older")

        assert result == {"minimum": 18}
        assert mock_t5.interpret.called
        assert not mock_llm.complete.called
        assert interpreter.stats.t5_successes == 1
        assert interpreter.stats.llm_fallbacks == 0

    def test_t5_invalid_triggers_llm_fallback(self):
        """Should fall back to LLM when T5 produces invalid constraints."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        # maxLength is invalid for integer type
        mock_t5.interpret.return_value = {"maxLength": 100}

        mock_llm = Mock()

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {"maximum": 100}

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=mock_llm,
                log_failures=False,
            )

            result = interpreter.interpret("count", "integer", "Maximum 100")

            assert result == {"maximum": 100}
            assert interpreter.stats.t5_failures == 1
            assert interpreter.stats.llm_fallbacks == 1

    def test_t5_empty_triggers_llm_fallback(self):
        """Should fall back to LLM when T5 returns empty."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}

        mock_llm = Mock()

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {"minimum": 0}

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=mock_llm,
                log_failures=False,
            )

            result = interpreter.interpret("value", "number", "Must be positive")

            assert result == {"minimum": 0}
            assert interpreter.stats.t5_failures == 1

    def test_t5_exception_triggers_fallback(self):
        """Should fall back to LLM when T5 raises exception."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.side_effect = RuntimeError("Model error")

        mock_llm = Mock()

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {"enum": ["A", "B"]}

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=mock_llm,
                log_failures=False,
            )

            result = interpreter.interpret("status", "string", "Must be A or B")

            assert result == {"enum": ["A", "B"]}
            assert interpreter.stats.t5_failures == 1
            assert interpreter.stats.llm_fallbacks == 1

    def test_no_t5_uses_llm_only(self):
        """Should use LLM directly when no T5 available."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_llm = Mock()

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {"minimum": 0}

            interpreter = HybridInterpreter(
                t5_interpreter=None,
                llm_client=mock_llm,
                log_failures=False,
            )

            result = interpreter.interpret("value", "number", "Must be positive")

            assert result == {"minimum": 0}
            assert interpreter.stats.t5_successes == 0
            assert interpreter.stats.llm_fallbacks == 1

    def test_both_fail_returns_empty(self):
        """Should return empty dict when both T5 and LLM fail."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}

        mock_llm = Mock()

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {}

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=mock_llm,
                log_failures=False,
            )

            result = interpreter.interpret("field", "string", "Complex rule")

            assert result == {}
            assert interpreter.stats.t5_failures == 1
            assert interpreter.stats.llm_failures == 1


class TestFailureLogging:
    """Test failure logging functionality."""

    def test_logs_t5_failures(self, tmp_path):
        """Should log T5 failures to file."""
        from src.hybrid_interpreter import HybridInterpreter

        log_path = tmp_path / "failures.json"

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"maxLength": 100}  # Invalid for integer

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {"maximum": 100}

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                log_failures=True,
                failure_log_path=str(log_path),
            )

            interpreter.interpret("count", "integer", "Max 100")

        # Check log file was created
        assert log_path.exists()

        with open(log_path) as f:
            log_data = json.load(f)

        assert len(log_data["failures"]) == 1
        assert log_data["failures"][0]["business_rule"] == "Max 100"
        assert log_data["failures"][0]["t5_output"] == {"maxLength": 100}
        assert "maxLength" in str(log_data["failures"][0]["validation_errors"])

    def test_failure_record_includes_fallback(self, tmp_path):
        """Should include LLM fallback result in failure record."""
        from src.hybrid_interpreter import HybridInterpreter

        log_path = tmp_path / "failures.json"

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}  # Empty = failure

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {"minimum": 18}

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                log_failures=True,
                failure_log_path=str(log_path),
            )

            interpreter.interpret("age", "integer", "Must be 18+")

        with open(log_path) as f:
            log_data = json.load(f)

        assert log_data["failures"][0]["fallback_output"] == {"minimum": 18}


class TestHybridStats:
    """Test statistics tracking."""

    def test_calculates_success_rate(self):
        """Should calculate T5 success rate correctly."""
        from src.hybrid_interpreter import HybridStats

        stats = HybridStats(
            total_calls=10,
            t5_successes=8,
            t5_failures=2,
        )

        assert stats.t5_success_rate == 0.8

    def test_calculates_fallback_rate(self):
        """Should calculate fallback rate correctly."""
        from src.hybrid_interpreter import HybridStats

        stats = HybridStats(
            total_calls=10,
            llm_fallbacks=3,
        )

        assert stats.fallback_rate == 0.3

    def test_zero_calls_returns_zero_rates(self):
        """Should return 0 rates when no calls made."""
        from src.hybrid_interpreter import HybridStats

        stats = HybridStats()

        assert stats.t5_success_rate == 0.0
        assert stats.fallback_rate == 0.0


class TestCreateHybridInterpreter:
    """Test factory function."""

    def test_creates_without_t5(self):
        """Should create interpreter without T5 when not available."""
        from src.hybrid_interpreter import create_hybrid_interpreter

        interpreter = create_hybrid_interpreter(
            adapter_name=None,
            llm_client=Mock(),
        )

        assert interpreter.t5 is None
        assert interpreter.llm_client is not None

    def test_creates_with_missing_adapter(self):
        """Should handle missing adapter gracefully."""
        from src.hybrid_interpreter import create_hybrid_interpreter

        interpreter = create_hybrid_interpreter(
            adapter_name="nonexistent",
            llm_client=Mock(),
        )

        # Should create but without T5
        assert interpreter.t5 is None


class TestProductionLogging:
    """Test production training data collection."""

    def test_collects_t5_successes(self):
        """Should collect successful T5 interpretations."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"minimum": 18}

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            collect_training_data=True,
            log_failures=False,
        )

        interpreter.interpret("age", "integer", "Must be 18 or older")

        training_data = interpreter.get_training_data()
        assert len(training_data) == 1
        assert training_data[0]["input"] == "Must be 18 or older"
        assert training_data[0]["output"] == {"minimum": 18}
        assert training_data[0]["field_type"] == "integer"

    def test_collects_llm_successes(self):
        """Should collect successful LLM interpretations."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_llm = Mock()

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret:
            mock_interpret.return_value = {"enum": ["A", "B"]}

            interpreter = HybridInterpreter(
                t5_interpreter=None,
                llm_client=mock_llm,
                collect_training_data=True,
                log_failures=False,
            )

            interpreter.interpret("status", "string", "Must be A or B")

            training_data = interpreter.get_training_data()
            assert len(training_data) == 1
            assert training_data[0]["input"] == "Must be A or B"
            assert training_data[0]["output"] == {"enum": ["A", "B"]}

    def test_tracks_source_in_stats(self):
        """Should track source (t5 vs llm) in stats."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"minimum": 0}

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            collect_training_data=True,
            log_failures=False,
        )

        interpreter.interpret("value", "number", "Must be positive")

        assert len(interpreter.stats.training_log) == 1
        assert interpreter.stats.training_log[0].source == "t5"

    def test_does_not_collect_when_disabled(self):
        """Should not collect data when collect_training_data=False."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"minimum": 0}

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            collect_training_data=False,
            log_failures=False,
        )

        interpreter.interpret("value", "number", "Must be positive")

        assert len(interpreter.stats.training_log) == 0

    def test_saves_training_data_to_file(self, tmp_path):
        """Should save training data to JSON file."""
        from src.hybrid_interpreter import HybridInterpreter

        log_path = tmp_path / "training.json"

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"maxLength": 100}

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            collect_training_data=True,
            training_log_path=str(log_path),
            log_failures=False,
        )

        interpreter.interpret("name", "string", "Max 100 characters")

        assert log_path.exists()

        with open(log_path) as f:
            data = json.load(f)

        assert data["total_examples"] == 1
        assert data["sources"]["t5"] == 1
        assert data["examples"][0]["input"] == "Max 100 characters"

    def test_export_training_data(self, tmp_path):
        """Should export training data in dataset format."""
        from src.hybrid_interpreter import HybridInterpreter

        export_path = tmp_path / "exported.json"

        mock_t5 = Mock()
        mock_t5.interpret.side_effect = [
            {"minimum": 0},
            {"maximum": 100},
        ]

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            collect_training_data=True,
            log_failures=False,
        )

        interpreter.interpret("min", "integer", "Must be positive")
        interpreter.interpret("max", "integer", "Cannot exceed 100")

        interpreter.export_training_data(str(export_path), domain="test")

        with open(export_path) as f:
            data = json.load(f)

        assert data["domain"] == "test"
        assert len(data["examples"]) == 2


class TestLLMVariations:
    """Test LLM input variation logging on fallback."""

    def test_variations_logged_alongside_original(self):
        """Should log original rule plus each variation when LLM falls back."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}  # T5 fails

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret, \
             patch('src.llm_interpreter.generate_rule_variations') as mock_variations:

            mock_interpret.return_value = {"pattern": "^[a-zA-Z0-9]+$"}
            mock_variations.return_value = ["Only letters and numbers", "Alphanumeric only"]

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                collect_training_data=True,
                log_failures=False,
            )

            interpreter.interpret("customer_id", "string", "Must be alphanumeric")

            training_data = interpreter.get_training_data()
            inputs = [ex["input"] for ex in training_data]

            assert "Must be alphanumeric" in inputs
            assert "Only letters and numbers" in inputs
            assert "Alphanumeric only" in inputs

    def test_all_variations_share_same_output(self):
        """Variations should be paired with the same validated output as the original."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret, \
             patch('src.llm_interpreter.generate_rule_variations') as mock_variations:

            mock_interpret.return_value = {"pattern": "^[a-zA-Z0-9]+$"}
            mock_variations.return_value = ["Only letters and numbers", "Alphanumeric only"]

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                collect_training_data=True,
                log_failures=False,
            )

            interpreter.interpret("customer_id", "string", "Must be alphanumeric")

            training_data = interpreter.get_training_data()
            for ex in training_data:
                assert ex["output"] == {"pattern": "^[a-zA-Z0-9]+$"}

    def test_variations_not_generated_when_collection_disabled(self):
        """Should not call generate_rule_variations when collect_training_data=False."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret, \
             patch('src.llm_interpreter.generate_rule_variations') as mock_variations:

            mock_interpret.return_value = {"pattern": "^[a-zA-Z0-9]+$"}

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                collect_training_data=False,
                log_failures=False,
            )

            interpreter.interpret("customer_id", "string", "Must be alphanumeric")

            mock_variations.assert_not_called()

    def test_original_logged_even_if_variations_empty(self):
        """Should still log the original rule when variation generation returns nothing."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret, \
             patch('src.llm_interpreter.generate_rule_variations') as mock_variations:

            mock_interpret.return_value = {"pattern": "^[a-zA-Z0-9]+$"}
            mock_variations.return_value = []

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                collect_training_data=True,
                log_failures=False,
            )

            interpreter.interpret("customer_id", "string", "Must be alphanumeric")

            training_data = interpreter.get_training_data()
            assert len(training_data) == 1
            assert training_data[0]["input"] == "Must be alphanumeric"

    def test_variations_not_generated_on_t5_success(self):
        """Should not generate variations when T5 succeeds — no LLM call made."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"minimum": 18}

        with patch('src.llm_interpreter.generate_rule_variations') as mock_variations:

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                collect_training_data=True,
                log_failures=False,
            )

            interpreter.interpret("age", "integer", "Must be 18 or older")

            mock_variations.assert_not_called()

    def test_variations_generated_counter_incremented(self):
        """Should increment variations_generated stat by number of unique variations."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {}

        with patch('src.llm_interpreter.interpret_business_rule') as mock_interpret, \
             patch('src.llm_interpreter.generate_rule_variations') as mock_variations:

            mock_interpret.return_value = {"pattern": "^[a-zA-Z0-9]+$"}
            mock_variations.return_value = ["Only letters and numbers", "Alphanumeric only"]

            interpreter = HybridInterpreter(
                t5_interpreter=mock_t5,
                llm_client=Mock(),
                collect_training_data=True,
                log_failures=False,
            )

            interpreter.interpret("customer_id", "string", "Must be alphanumeric")

            assert interpreter.stats.variations_generated == 2
            assert interpreter.get_stats()["variations_generated"] == 2


class TestValidConstraintTypes:
    """Test that validation correctly identifies valid/invalid constraints."""

    def test_string_constraints_valid(self):
        """String constraints should pass for string type."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"maxLength": 100, "pattern": "^[A-Z]+$"}

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            log_failures=False,
        )

        result = interpreter.interpret("code", "string", "Max 100 chars, uppercase")

        assert result == {"maxLength": 100, "pattern": "^[A-Z]+$"}
        assert interpreter.stats.t5_successes == 1

    def test_number_constraints_valid(self):
        """Number constraints should pass for number type."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"minimum": 0, "maximum": 100}

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            log_failures=False,
        )

        result = interpreter.interpret("value", "number", "Between 0 and 100")

        assert result == {"minimum": 0, "maximum": 100}
        assert interpreter.stats.t5_successes == 1

    def test_enum_valid_for_any_type(self):
        """Enum should be valid for any type."""
        from src.hybrid_interpreter import HybridInterpreter

        mock_t5 = Mock()
        mock_t5.interpret.return_value = {"enum": ["A", "B", "C"]}

        interpreter = HybridInterpreter(
            t5_interpreter=mock_t5,
            llm_client=None,
            log_failures=False,
        )

        result = interpreter.interpret("status", "string", "Must be A, B, or C")

        assert result == {"enum": ["A", "B", "C"]}
        assert interpreter.stats.t5_successes == 1
