"""
Node Updater - merges constraints into JSON Schema nodes in-place.

This module provides the update_node function for adding constraints
to schema nodes without overwriting existing values.
"""


def update_node(node: dict, constraints: dict) -> None:
    """
    Merge constraints into JSON Schema node in-place.

    Adds new constraints without overwriting existing values.
    Description fields are appended rather than replaced.

    Args:
        node: The JSON Schema node dict to update (modified in-place)
        constraints: Dict of constraints to add
    """
    for key, value in constraints.items():
        if key == "description":
            _handle_description(node, value)
        elif key not in node:
            # Add new constraint only if not already present
            node[key] = value
        # Skip if key already exists (preserve original)


def _handle_description(node: dict, new_description: str) -> None:
    """
    Handle description field by appending to existing.

    Args:
        node: The node to update
        new_description: Description text to add
    """
    if not new_description or not new_description.strip():
        return

    existing = node.get("description", "")
    if existing:
        # Append to existing description
        node["description"] = f"{existing.strip()}\n{new_description.strip()}"
    else:
        # Set new description
        node["description"] = new_description.strip()
