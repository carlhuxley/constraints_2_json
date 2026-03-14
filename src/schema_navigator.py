"""
Schema Navigator - walks JSON Schema tree yielding (path, node) tuples.

This module provides functionality to traverse a JSON Schema and yield
each property node along with its path, handling nested objects and arrays.
"""
from typing import Iterator


def walk_schema(schema: dict, path: str = "") -> Iterator[tuple[str, dict]]:
    """
    Yield (field_path, property_node) for each property in schema.

    Recursively walks the JSON Schema tree, yielding tuples for each property.
    Handles nested objects and array items.

    Args:
        schema: The JSON Schema dict to walk
        path: Current path prefix (used in recursion)

    Yields:
        Tuples of (field_path, property_node) for each property found
    """
    if "properties" not in schema:
        return

    for name, prop in schema["properties"].items():
        current_path = f"{path}.{name}" if path else name
        yield (current_path, prop)

        # Recurse into nested objects
        if prop.get("type") == "object" and "properties" in prop:
            yield from walk_schema(prop, current_path)

        # Handle array items
        if prop.get("type") == "array" and "items" in prop:
            items_schema = prop["items"]
            items_path = f"{current_path}[]"
            yield (items_path, items_schema)

            # Recurse into array item objects
            if items_schema.get("type") == "object" and "properties" in items_schema:
                yield from walk_schema(items_schema, items_path)
