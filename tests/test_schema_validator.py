"""Tests for schema_validator module."""
import pytest
from src.schema_validator import (
    validate_schema,
    validate_node_constraints,
    validate_schema_constraints,
    SchemaValidationError,
)


class TestValidateSchema:
    """Test JSON Schema Draft 7 meta-schema validation."""

    def test_valid_simple_schema(self):
        """Valid schema should pass validation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        is_valid, errors = validate_schema(schema)
        assert is_valid is True
        assert errors == []

    def test_valid_schema_with_constraints(self):
        """Schema with valid constraints should pass."""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "maxLength": 255
                },
                "count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100
                }
            }
        }
        is_valid, errors = validate_schema(schema)
        assert is_valid is True
        assert errors == []

    def test_invalid_type_value(self):
        """Schema with invalid type value should fail."""
        schema = {
            "type": "invalid_type"
        }
        is_valid, errors = validate_schema(schema)
        assert is_valid is False
        assert len(errors) > 0

    def test_invalid_constraint_type(self):
        """Schema with wrong constraint value type should fail."""
        schema = {
            "type": "string",
            "maxLength": "not_a_number"
        }
        is_valid, errors = validate_schema(schema)
        assert is_valid is False
        assert len(errors) > 0

    def test_empty_schema_is_valid(self):
        """Empty schema is technically valid."""
        schema = {}
        is_valid, errors = validate_schema(schema)
        assert is_valid is True
        assert errors == []


class TestValidateNodeConstraints:
    """Test type-constraint compatibility validation."""

    def test_string_constraints_on_string(self):
        """String constraints on string type should pass."""
        node = {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "pattern": "^[a-z]+$"
        }
        errors = validate_node_constraints(node, "field")
        assert errors == []

    def test_number_constraints_on_number(self):
        """Number constraints on number type should pass."""
        node = {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "multipleOf": 0.5
        }
        errors = validate_node_constraints(node, "field")
        assert errors == []

    def test_integer_constraints_on_integer(self):
        """Integer constraints on integer type should pass."""
        node = {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        }
        errors = validate_node_constraints(node, "field")
        assert errors == []

    def test_array_constraints_on_array(self):
        """Array constraints on array type should pass."""
        node = {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True
        }
        errors = validate_node_constraints(node, "field")
        assert errors == []

    def test_string_constraint_on_number_fails(self):
        """maxLength on number type should fail."""
        node = {
            "type": "number",
            "maxLength": 10
        }
        errors = validate_node_constraints(node, "field")
        assert len(errors) == 1
        assert "maxLength" in errors[0]
        assert "number" in errors[0]

    def test_number_constraint_on_string_fails(self):
        """minimum on string type should fail."""
        node = {
            "type": "string",
            "minimum": 0
        }
        errors = validate_node_constraints(node, "field")
        assert len(errors) == 1
        assert "minimum" in errors[0]
        assert "string" in errors[0]

    def test_multiple_incompatible_constraints(self):
        """Multiple incompatible constraints should all be reported."""
        node = {
            "type": "string",
            "minimum": 0,
            "maximum": 100,
            "multipleOf": 5
        }
        errors = validate_node_constraints(node, "field")
        assert len(errors) == 3

    def test_union_type_with_compatible_constraint(self):
        """Constraint valid for any type in union should pass."""
        node = {
            "type": ["string", "null"],
            "maxLength": 100
        }
        errors = validate_node_constraints(node, "field")
        assert errors == []

    def test_union_type_with_incompatible_constraint(self):
        """Constraint not valid for any type in union should fail."""
        node = {
            "type": ["integer", "null"],
            "maxLength": 100
        }
        errors = validate_node_constraints(node, "field")
        assert len(errors) == 1

    def test_no_type_skips_validation(self):
        """Node without type should skip constraint validation."""
        node = {
            "maxLength": 100
        }
        errors = validate_node_constraints(node, "field")
        assert errors == []

    def test_error_includes_path(self):
        """Error message should include the field path."""
        node = {
            "type": "string",
            "minimum": 0
        }
        errors = validate_node_constraints(node, "user.profile.age")
        assert "user.profile.age" in errors[0]


class TestValidateSchemaConstraints:
    """Test full schema validation with constraint checking."""

    def test_valid_nested_schema(self):
        """Valid nested schema should pass all checks."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "maxLength": 100},
                        "age": {"type": "integer", "minimum": 0}
                    }
                }
            }
        }
        is_valid, errors = validate_schema_constraints(schema)
        assert is_valid is True
        assert errors == []

    def test_invalid_nested_constraint(self):
        """Invalid constraint in nested property should be caught."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minimum": 0}
                    }
                }
            }
        }
        is_valid, errors = validate_schema_constraints(schema)
        assert is_valid is False
        assert any("user.name" in e for e in errors)

    def test_validates_array_items(self):
        """Should validate constraints in array item schemas."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minimum": 0  # Invalid for string
                    }
                }
            }
        }
        is_valid, errors = validate_schema_constraints(schema)
        assert is_valid is False
        assert any("tags[]" in e for e in errors)

    def test_validates_definitions(self):
        """Should validate constraints in definitions."""
        schema = {
            "definitions": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "zipcode": {
                            "type": "string",
                            "minimum": 0  # Invalid for string
                        }
                    }
                }
            }
        }
        is_valid, errors = validate_schema_constraints(schema)
        assert is_valid is False
        assert any("definitions.Address" in e for e in errors)


class TestSchemaValidationError:
    """Test SchemaValidationError exception."""

    def test_error_contains_errors_list(self):
        """Exception should contain list of errors."""
        errors = ["Error 1", "Error 2"]
        exc = SchemaValidationError(errors)
        assert exc.errors == errors
        assert "2 error" in str(exc)

    def test_error_message_format(self):
        """Exception message should indicate error count."""
        exc = SchemaValidationError(["Single error"])
        assert "1 error" in str(exc)
