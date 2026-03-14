"""
Schema Enricher - orchestrates constraint enrichment from data dictionary.

This module provides the main orchestration logic for enriching JSON Schemas
with constraints from an Informatica data dictionary.
"""
import json
from typing import Optional

from .schema_navigator import walk_schema
from .dict_lookup import DataDictionary
from .node_updater import update_node
from .llm_interpreter import interpret_business_rule, LLMClient


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
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize the enricher.

        Args:
            schema: JSON Schema dict to enrich
            dict_path: Path to data dictionary CSV
            llm_client: Optional LLM client for business rule interpretation
        """
        self.schema = schema
        self.dictionary = DataDictionary(dict_path)
        self.llm_client = llm_client

    def enrich(self) -> dict:
        """
        Enrich the schema with constraints from the dictionary.

        Walks all schema properties and applies matching constraints.

        Returns:
            The enriched schema (modified in-place)
        """
        for path, node in walk_schema(self.schema):
            self._enrich_node(path, node)

        return self.schema

    def _enrich_node(self, path: str, node: dict) -> None:
        """
        Enrich a single schema node.

        Args:
            path: Field path (e.g., "user.address.zipcode")
            node: The schema node dict to enrich
        """
        # Get constraints from dictionary
        constraints = self.dictionary.get_constraints(path)

        if not constraints:
            return

        # Handle business rule with LLM if available
        business_rule = constraints.pop("x-business-rule", None)
        if business_rule:
            # Always add the business rule as extension
            constraints["x-business-rule"] = business_rule

            # If LLM is available, interpret the rule
            if self.llm_client:
                field_type = node.get("type", "string")
                llm_constraints = interpret_business_rule(
                    field_name=path.split(".")[-1],
                    field_type=field_type,
                    business_rule=business_rule,
                    llm_client=self.llm_client
                )
                # Merge LLM constraints (they take precedence for new keys)
                for key, value in llm_constraints.items():
                    if key not in constraints:
                        constraints[key] = value

        # Update the node with constraints
        update_node(node, constraints)


def enrich_schema(
    schema_path: str,
    dict_path: str,
    llm_client: Optional[LLMClient] = None,
    output_path: Optional[str] = None
) -> dict:
    """
    Convenience function to enrich a schema from file paths.

    Args:
        schema_path: Path to input JSON Schema file
        dict_path: Path to data dictionary CSV
        llm_client: Optional LLM client for business rule interpretation
        output_path: Optional path to write enriched schema

    Returns:
        The enriched schema dict
    """
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    enricher = SchemaEnricher(schema, dict_path, llm_client)
    result = enricher.enrich()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

    return result
