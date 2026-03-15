"""
Schema Validator - validates enriched JSON schemas against JSON Schema Draft 7.

This module provides validation functions to ensure enriched schemas are valid
according to the JSON Schema specification and that constraints are compatible
with their target node types.
"""
from typing import Tuple, List
from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Schema validation failed with {len(errors)} error(s)")


# Constraints that are only valid for specific types
TYPE_SPECIFIC_CONSTRAINTS = {
    "string": {"minLength", "maxLength", "pattern", "format"},
    "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
    "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
    "array": {"minItems", "maxItems", "uniqueItems", "items", "contains"},
    "object": {"minProperties", "maxProperties", "required", "properties", "additionalProperties", "patternProperties"},
}

# Reverse mapping: constraint -> valid types
CONSTRAINT_VALID_TYPES = {}
for type_name, constraints in TYPE_SPECIFIC_CONSTRAINTS.items():
    for constraint in constraints:
        if constraint not in CONSTRAINT_VALID_TYPES:
            CONSTRAINT_VALID_TYPES[constraint] = set()
        CONSTRAINT_VALID_TYPES[constraint].add(type_name)


def validate_schema(schema: dict) -> Tuple[bool, List[str]]:
    """
    Validate a schema against JSON Schema Draft 7 meta-schema.

    Args:
        schema: The JSON Schema dict to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as e:
        errors.append(f"Schema error: {e.message}")
        # Walk the error context for nested errors
        for error in e.context:
            errors.append(f"  - {error.message}")

    return (len(errors) == 0, errors)


def validate_node_constraints(node: dict, path: str = "") -> List[str]:
    """
    Validate that constraints are compatible with the node's declared type.

    Args:
        node: A schema node dict
        path: The path to this node (for error messages)

    Returns:
        List of validation error messages
    """
    errors = []

    node_type = node.get("type")
    if not node_type:
        return errors

    # Handle type as a list (e.g., ["string", "null"])
    if isinstance(node_type, list):
        valid_types = set(node_type)
    else:
        valid_types = {node_type}

    for constraint, allowed_types in CONSTRAINT_VALID_TYPES.items():
        if constraint in node:
            # Check if any of the node's types are valid for this constraint
            if not valid_types & allowed_types:
                type_str = ", ".join(sorted(valid_types))
                allowed_str = ", ".join(sorted(allowed_types))
                path_prefix = f"At '{path}': " if path else ""
                errors.append(
                    f"{path_prefix}Constraint '{constraint}' is not valid for type '{type_str}' "
                    f"(valid for: {allowed_str})"
                )

    return errors


def validate_schema_constraints(schema: dict) -> Tuple[bool, List[str]]:
    """
    Validate both the schema structure and type-constraint compatibility.

    This performs a full validation including:
    1. JSON Schema Draft 7 meta-schema validation
    2. Type-constraint compatibility checks for all nodes

    Args:
        schema: The JSON Schema dict to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # First validate against meta-schema
    is_valid, meta_errors = validate_schema(schema)
    errors.extend(meta_errors)

    # Then check type-constraint compatibility
    constraint_errors = _walk_and_validate_constraints(schema)
    errors.extend(constraint_errors)

    return (len(errors) == 0, errors)


def _walk_and_validate_constraints(schema: dict, path: str = "") -> List[str]:
    """
    Recursively walk schema and validate constraints at each node.

    Args:
        schema: Schema or subschema to validate
        path: Current path for error messages

    Returns:
        List of validation errors
    """
    errors = []

    # Validate current node
    errors.extend(validate_node_constraints(schema, path))

    # Walk properties
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            prop_path = f"{path}.{prop_name}" if path else prop_name
            errors.extend(_walk_and_validate_constraints(prop_schema, prop_path))

    # Walk array items
    if "items" in schema and isinstance(schema["items"], dict):
        items_path = f"{path}[]" if path else "[]"
        errors.extend(_walk_and_validate_constraints(schema["items"], items_path))

    # Walk definitions
    if "definitions" in schema:
        for def_name, def_schema in schema["definitions"].items():
            def_path = f"definitions.{def_name}"
            errors.extend(_walk_and_validate_constraints(def_schema, def_path))

    # Walk $defs (Draft 2019-09+ style)
    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            def_path = f"$defs.{def_name}"
            errors.extend(_walk_and_validate_constraints(def_schema, def_path))

    return errors
