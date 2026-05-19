"""
Schema Enricher - orchestrates constraint enrichment from data dictionary.

This module provides the main orchestration logic for enriching JSON Schemas
with constraints from an Informatica data dictionary.
"""
import json
from typing import Optional, Union

from .schema_navigator import walk_schema
from .dict_lookup import DataDictionary
from .node_updater import update_node
from .llm_interpreter import interpret_business_rule, LLMClient
from .schema_validator import validate_schema_constraints, SchemaValidationError
from .hybrid_interpreter import HybridInterpreter


class SchemaEnricher:
    """
    Orchestrates JSON Schema enrichment from data dictionary.

    Walks the schema tree, looks up each field in the dictionary,
    and applies constraints to schema nodes.
    """

    def __init__(
        self,
        schema: dict,
        dict_path: str,
        llm_client: Optional[LLMClient] = None,
        hybrid_interpreter: Optional[HybridInterpreter] = None,
        validate: bool = True
    ):
        """
        Initialize the enricher.

        Args:
            schema: JSON Schema dict to enrich
            dict_path: Path to data dictionary CSV
            llm_client: Optional LLM client for business rule interpretation
            hybrid_interpreter: Optional hybrid interpreter (T5 + LLM fallback)
            validate: Whether to validate the enriched schema (default True)
        """
        self.schema = schema
        self.dictionary = DataDictionary(dict_path)
        self.llm_client = llm_client
        self.hybrid_interpreter = hybrid_interpreter
        self.validate = validate
        self._audit: list = []

    def enrich(self, validate: Optional[bool] = None) -> dict:
        """
        Enrich the schema with constraints from the dictionary.

        Walks all schema properties and applies matching constraints.

        Args:
            validate: Override the instance validate setting for this call

        Returns:
            The enriched schema (modified in-place)

        Raises:
            SchemaValidationError: If validation is enabled and schema is invalid
        """
        for path, node in walk_schema(self.schema):
            self._enrich_node(path, node)

        # Determine whether to validate
        should_validate = validate if validate is not None else self.validate

        if should_validate:
            is_valid, errors = validate_schema_constraints(self.schema)
            if not is_valid:
                raise SchemaValidationError(errors)

        return self.schema

    def get_audit_log(self) -> list:
        """Return the per-field audit log for this enrichment run."""
        return self._audit

    def _enrich_node(self, path: str, node: dict) -> None:
        """
        Enrich a single schema node.

        Args:
            path: Field path (e.g., "user.address.zipcode")
            node: The schema node dict to enrich
        """
        field_type = node.get("type", "unknown")

        # Get constraints from dictionary
        constraints = self.dictionary.get_constraints(path)

        if not constraints:
            self._audit.append({
                "field_path": path,
                "field_type": field_type,
                "business_rule": None,
                "dictionary_keys": [],
                "interpreter_source": "no_match",
                "interpreter_keys": [],
                "constraints_applied": [],
            })
            return

        dictionary_keys = list(constraints.keys())

        # Handle business rule with interpreter if available
        business_rule = constraints.pop("x-business-rule", None)
        interpreter_source = "no_rule"
        interpreter_keys = []

        if business_rule:
            constraints["x-business-rule"] = business_rule
            field_name = path.split(".")[-1]

            # Prefer hybrid interpreter (T5 + LLM fallback)
            if self.hybrid_interpreter:
                interpreted = self.hybrid_interpreter.interpret(
                    field_name=field_name,
                    field_type=field_type,
                    business_rule=business_rule,
                )
                interpreter_source = self.hybrid_interpreter.last_source or "none"
                interpreter_keys = list(interpreted.keys())
                for key, value in interpreted.items():
                    if key not in constraints:
                        constraints[key] = value

            # Fall back to LLM-only if no hybrid interpreter
            elif self.llm_client:
                llm_constraints = interpret_business_rule(
                    field_name=field_name,
                    field_type=field_type,
                    business_rule=business_rule,
                    llm_client=self.llm_client
                )
                interpreter_source = "llm"
                interpreter_keys = list(llm_constraints.keys())
                for key, value in llm_constraints.items():
                    if key not in constraints:
                        constraints[key] = value

        self._audit.append({
            "field_path": path,
            "field_type": field_type,
            "business_rule": business_rule,
            "dictionary_keys": dictionary_keys,
            "interpreter_source": interpreter_source,
            "interpreter_keys": interpreter_keys,
            "constraints_applied": list(constraints.keys()),
        })

        # Update the node with constraints
        update_node(node, constraints)


def enrich_schema(
    schema_path: str,
    dict_path: str,
    llm_client: Optional[LLMClient] = None,
    hybrid_interpreter: Optional[HybridInterpreter] = None,
    output_path: Optional[str] = None,
    audit_log_path: Optional[str] = None,
    validate: bool = True
) -> dict:
    """
    Convenience function to enrich a schema from file paths.

    Args:
        schema_path: Path to input JSON Schema file
        dict_path: Path to data dictionary CSV
        llm_client: Optional LLM client for business rule interpretation
        hybrid_interpreter: Optional hybrid interpreter (T5 + LLM fallback)
        output_path: Optional path to write enriched schema
        validate: Whether to validate the enriched schema (default True)

    Returns:
        The enriched schema dict

    Raises:
        SchemaValidationError: If validation is enabled and schema is invalid
    """
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    enricher = SchemaEnricher(
        schema, dict_path, llm_client,
        hybrid_interpreter=hybrid_interpreter,
        validate=validate
    )
    result = enricher.enrich()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

    if audit_log_path:
        audit_data = {
            "schema": schema_path,
            "dictionary": dict_path,
            "fields": enricher.get_audit_log(),
        }
        with open(audit_log_path, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2)

    return result
