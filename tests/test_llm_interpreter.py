"""Tests for llm_interpreter module following TDD."""
import pytest
from unittest.mock import Mock, MagicMock
from src.llm_interpreter import interpret_business_rule, LLMClient


class TestInterpretMinimumRules:
    """Test interpreting minimum value rules."""

    def test_interprets_age_minimum(self):
        """Should extract minimum from 'must be 18 or older' rule."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"minimum": 18}'

        result = interpret_business_rule(
            field_name="customer_age",
            field_type="integer",
            business_rule="Must be 18 or older",
            llm_client=mock_client
        )

        assert result["minimum"] == 18

    def test_interprets_positive_value_rule(self):
        """Should extract minimum from 'must be positive' rule."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"minimum": 0, "exclusiveMinimum": true}'

        result = interpret_business_rule(
            field_name="quantity",
            field_type="integer",
            business_rule="Must be a positive value",
            llm_client=mock_client
        )

        assert "minimum" in result or "exclusiveMinimum" in result


class TestInterpretRangeRules:
    """Test interpreting range rules."""

    def test_interprets_between_rule(self):
        """Should extract min and max from 'between X and Y' rule."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"minimum": 0, "maximum": 100}'

        result = interpret_business_rule(
            field_name="percentage",
            field_type="number",
            business_rule="Value must be between 0 and 100",
            llm_client=mock_client
        )

        assert result["minimum"] == 0
        assert result["maximum"] == 100

    def test_interprets_max_age_rule(self):
        """Should extract maximum from age limit rule."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"maximum": 120}'

        result = interpret_business_rule(
            field_name="age",
            field_type="integer",
            business_rule="Maximum age is 120 years",
            llm_client=mock_client
        )

        assert result["maximum"] == 120


class TestInterpretPatternRules:
    """Test interpreting pattern rules."""

    def test_interprets_state_code_rule(self):
        """Should extract pattern for state code."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"pattern": "^[A-Z]{2}$"}'

        result = interpret_business_rule(
            field_name="state_code",
            field_type="string",
            business_rule="Must be a valid US state code (2 uppercase letters)",
            llm_client=mock_client
        )

        assert result["pattern"] == "^[A-Z]{2}$"

    def test_interprets_email_pattern_rule(self):
        """Should interpret email format rule."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"format": "email"}'

        result = interpret_business_rule(
            field_name="email",
            field_type="string",
            business_rule="Must be a valid email address",
            llm_client=mock_client
        )

        assert result.get("format") == "email" or "pattern" in result


class TestInterpretEnumRules:
    """Test interpreting enumeration rules."""

    def test_interprets_status_enum(self):
        """Should extract enum from status list."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"enum": ["ACTIVE", "INACTIVE", "PENDING"]}'

        result = interpret_business_rule(
            field_name="status",
            field_type="string",
            business_rule="Status must be ACTIVE, INACTIVE, or PENDING",
            llm_client=mock_client
        )

        assert result["enum"] == ["ACTIVE", "INACTIVE", "PENDING"]

    def test_interprets_yes_no_enum(self):
        """Should extract enum from yes/no options."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"enum": ["Y", "N"]}'

        result = interpret_business_rule(
            field_name="active_flag",
            field_type="string",
            business_rule="Must be Y or N",
            llm_client=mock_client
        )

        assert result["enum"] == ["Y", "N"]


class TestInterpretLengthRules:
    """Test interpreting length rules."""

    def test_interprets_max_length(self):
        """Should extract maxLength from length rule."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"maxLength": 255}'

        result = interpret_business_rule(
            field_name="description",
            field_type="string",
            business_rule="Maximum 255 characters allowed",
            llm_client=mock_client
        )

        assert result["maxLength"] == 255

    def test_interprets_exact_length(self):
        """Should extract minLength and maxLength for exact length."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{"minLength": 5, "maxLength": 5}'

        result = interpret_business_rule(
            field_name="zip_code",
            field_type="string",
            business_rule="Must be exactly 5 digits",
            llm_client=mock_client
        )

        assert result["minLength"] == 5
        assert result["maxLength"] == 5


class TestLLMClientInterface:
    """Test LLM client interface requirements."""

    def test_calls_client_complete(self):
        """Should call llm_client.complete with prompt."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = '{}'

        interpret_business_rule(
            field_name="test",
            field_type="string",
            business_rule="Test rule",
            llm_client=mock_client
        )

        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args[0][0]
        assert "test" in call_args.lower()
        assert "string" in call_args.lower()
        assert "test rule" in call_args.lower()


class TestErrorHandling:
    """Test error handling in interpretation."""

    def test_handles_invalid_json_response(self):
        """Should handle invalid JSON from LLM gracefully."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = "not valid json"

        result = interpret_business_rule(
            field_name="test",
            field_type="string",
            business_rule="Test rule",
            llm_client=mock_client
        )

        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)

    def test_handles_empty_response(self):
        """Should handle empty response from LLM."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = ""

        result = interpret_business_rule(
            field_name="test",
            field_type="string",
            business_rule="Test rule",
            llm_client=mock_client
        )

        assert isinstance(result, dict)

    def test_handles_json_with_extra_text(self):
        """Should extract JSON even with surrounding text."""
        mock_client = Mock(spec=LLMClient)
        mock_client.complete.return_value = 'Here is the result: {"minimum": 18}'

        result = interpret_business_rule(
            field_name="age",
            field_type="integer",
            business_rule="Must be 18+",
            llm_client=mock_client
        )

        assert result.get("minimum") == 18
