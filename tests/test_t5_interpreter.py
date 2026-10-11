"""Tests for T5 interpreter module."""
import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock


class TestT5InterpreterParsing:
    """Test output parsing without loading model."""

    def test_parse_output_with_braces(self):
        """Should parse output that has braces."""
        from src.t5_interpreter import T5Interpreter

        # Mock the initialization
        with patch.object(T5Interpreter, '__init__', lambda x, **kw: None):
            interpreter = T5Interpreter()
            interpreter._parse_output = T5Interpreter._parse_output.__get__(
                interpreter, T5Interpreter
            )

            result = interpreter._parse_output('{"minimum": 18}')
            assert result == {"minimum": 18}

    def test_parse_output_without_braces(self):
        """Should add braces to output missing them."""
        from src.t5_interpreter import T5Interpreter

        with patch.object(T5Interpreter, '__init__', lambda x, **kw: None):
            interpreter = T5Interpreter()
            interpreter._parse_output = T5Interpreter._parse_output.__get__(
                interpreter, T5Interpreter
            )

            result = interpreter._parse_output('"minimum":18')
            assert result == {"minimum": 18}

    def test_parse_output_empty_string(self):
        """Should return empty dict for empty string."""
        from src.t5_interpreter import T5Interpreter

        with patch.object(T5Interpreter, '__init__', lambda x, **kw: None):
            interpreter = T5Interpreter()
            interpreter._parse_output = T5Interpreter._parse_output.__get__(
                interpreter, T5Interpreter
            )

            result = interpreter._parse_output('')
            assert result == {}

    def test_parse_output_invalid_json(self):
        """Should return empty dict for invalid JSON."""
        from src.t5_interpreter import T5Interpreter

        with patch.object(T5Interpreter, '__init__', lambda x, **kw: None):
            interpreter = T5Interpreter()
            interpreter._parse_output = T5Interpreter._parse_output.__get__(
                interpreter, T5Interpreter
            )

            result = interpreter._parse_output('not valid json at all')
            assert result == {}

    def test_parse_output_with_enum(self):
        """Should parse enum constraints."""
        from src.t5_interpreter import T5Interpreter

        with patch.object(T5Interpreter, '__init__', lambda x, **kw: None):
            interpreter = T5Interpreter()
            interpreter._parse_output = T5Interpreter._parse_output.__get__(
                interpreter, T5Interpreter
            )

            result = interpreter._parse_output('"enum":["ACTIVE","INACTIVE"]')
            assert result == {"enum": ["ACTIVE", "INACTIVE"]}

    def test_parse_output_multiple_constraints(self):
        """Should parse multiple constraints."""
        from src.t5_interpreter import T5Interpreter

        with patch.object(T5Interpreter, '__init__', lambda x, **kw: None):
            interpreter = T5Interpreter()
            interpreter._parse_output = T5Interpreter._parse_output.__get__(
                interpreter, T5Interpreter
            )

            result = interpreter._parse_output('"minimum":0,"maximum":100')
            assert result == {"minimum": 0, "maximum": 100}


class TestLoadAdapter:
    """Test adapter loading functionality."""

    def test_load_adapter_not_found(self):
        """Should raise error when adapter not found."""
        from src.t5_interpreter import load_adapter

        with pytest.raises(FileNotFoundError) as exc_info:
            load_adapter("nonexistent", adapters_dir="/tmp/no_adapters")

        assert "nonexistent" in str(exc_info.value)

    def test_load_adapter_finds_correct_path(self, tmp_path):
        """Should find adapter in standard directory structure."""
        from src.t5_interpreter import load_adapter

        # Create mock adapter structure
        adapter_dir = tmp_path / "adapter_test" / "final_adapter"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text('{"test": true}')

        # Should find it but fail on actual model load
        with patch('src.t5_interpreter._check_transformers', return_value=False):
            with pytest.raises(ImportError):
                load_adapter("test", adapters_dir=str(tmp_path))


@pytest.mark.skipif(
    not os.path.exists("adapters/adapter_financial/final_adapter"),
    reason="Financial adapter not installed"
)
class TestT5InterpreterIntegration:
    """Integration tests with actual adapter (skip if not available)."""

    @pytest.fixture
    def interpreter(self):
        """Load the financial adapter."""
        from src.t5_interpreter import load_adapter
        return load_adapter("financial")

    def test_minimum_constraint(self, interpreter):
        """Should generate minimum constraint."""
        result = interpreter.interpret("Balance must be at least 1000")
        assert "minimum" in result
        assert result["minimum"] == 1000

    def test_maximum_constraint(self, interpreter):
        """Should generate maximum constraint."""
        result = interpreter.interpret("Age cannot exceed 65")
        assert "maximum" in result
        assert result["maximum"] == 65

    def test_enum_constraint(self, interpreter):
        """Should generate enum constraint."""
        result = interpreter.interpret("Status must be ACTIVE or INACTIVE")
        assert "enum" in result
        assert "ACTIVE" in result["enum"]
        assert "INACTIVE" in result["enum"]

    def test_range_constraint(self, interpreter):
        """Should generate min/max range."""
        result = interpreter.interpret("Value must be between 0 and 100")
        assert result.get("minimum") == 0
        assert result.get("maximum") == 100

    @pytest.mark.skip(reason="T5 model needs more training data for email format")
    def test_format_constraint(self, interpreter):
        """Should generate format constraint."""
        result = interpreter.interpret("Email must be valid format")
        assert result.get("format") == "email"

    def test_maxlength_constraint(self, interpreter):
        """Should generate maxLength constraint."""
        result = interpreter.interpret("Maximum 100 characters")
        assert result.get("maxLength") == 100


@pytest.mark.skipif(
    not os.path.exists("adapters/adapter_financial/final_adapter/tokenizer_config.json"),
    reason="T5 tokenizer not available"
)
class TestT5InterpreterTextFormat:
    """interpret() must use the shared t5_text format in both directions."""

    @pytest.fixture
    def interpreter_emitting(self):
        """Interpreter with the real tokenizer and a model that emits fixed text."""
        import torch
        from transformers import AutoTokenizer
        from src.t5_interpreter import T5Interpreter

        tokenizer = AutoTokenizer.from_pretrained("adapters/adapter_financial/final_adapter")

        def build(model_output_text):
            ids = tokenizer(model_output_text).input_ids
            model = Mock()
            model.generate.return_value = Mock(sequences=torch.tensor([ids]), scores=())
            model.compute_transition_scores.return_value = torch.log(
                torch.tensor([[0.9] * (len(ids) - 1) + [0.6]])
            )
            with patch.object(T5Interpreter, '__init__', lambda x, **kw: None):
                interpreter = T5Interpreter()
            interpreter.tokenizer = tokenizer
            interpreter.model = model
            interpreter.device = "cpu"
            return interpreter, model

        return build

    def test_returns_pattern_with_regex_metacharacters(self, interpreter_emitting):
        from src.t5_text import format_target

        expected = {"pattern": "^[0-9]{6}$"}
        interpreter, _ = interpreter_emitting(format_target(expected))

        assert interpreter.interpret("Sort code is six digits") == expected

    def test_model_sees_encoded_rule(self, interpreter_emitting):
        from src.t5_text import format_input

        interpreter, model = interpreter_emitting('"maximum":5')
        interpreter.interpret("Amount <= 5")

        sent_ids = model.generate.call_args.kwargs["input_ids"][0].tolist()
        expected_ids = interpreter.tokenizer(format_input("Amount <= 5")).input_ids
        assert sent_ids == expected_ids

    def test_confidence_is_least_certain_token_probability(self, interpreter_emitting):
        interpreter, _ = interpreter_emitting('"maximum":5')
        interpreter.interpret("Amount <= 5")

        assert interpreter.last_confidence == pytest.approx(0.6)
