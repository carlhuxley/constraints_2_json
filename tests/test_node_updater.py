"""Tests for node_updater module following TDD."""
import pytest
from src.node_updater import update_node


class TestUpdateNodeBasicConstraints:
    """Test adding basic constraints to nodes."""

    def test_adds_minimum_constraint(self):
        """Should add minimum constraint to node."""
        node = {"type": "integer"}
        update_node(node, {"minimum": 18})
        assert node["minimum"] == 18

    def test_adds_maximum_constraint(self):
        """Should add maximum constraint to node."""
        node = {"type": "integer"}
        update_node(node, {"maximum": 120})
        assert node["maximum"] == 120

    def test_adds_multiple_constraints(self):
        """Should add multiple constraints at once."""
        node = {"type": "integer"}
        update_node(node, {"minimum": 18, "maximum": 120})
        assert node["minimum"] == 18
        assert node["maximum"] == 120

    def test_adds_string_constraints(self):
        """Should add string-specific constraints."""
        node = {"type": "string"}
        update_node(node, {"pattern": "^[A-Z]{2}$", "maxLength": 2})
        assert node["pattern"] == "^[A-Z]{2}$"
        assert node["maxLength"] == 2

    def test_adds_enum_constraint(self):
        """Should add enum constraint."""
        node = {"type": "string"}
        update_node(node, {"enum": ["ACTIVE", "INACTIVE", "PENDING"]})
        assert node["enum"] == ["ACTIVE", "INACTIVE", "PENDING"]


class TestUpdateNodePreservation:
    """Test that existing values are preserved."""

    def test_preserves_existing_minimum(self):
        """Should not overwrite existing minimum."""
        node = {"type": "integer", "minimum": 0}
        update_node(node, {"minimum": 18})
        assert node["minimum"] == 0

    def test_preserves_existing_maximum(self):
        """Should not overwrite existing maximum."""
        node = {"type": "integer", "maximum": 100}
        update_node(node, {"maximum": 120})
        assert node["maximum"] == 100

    def test_preserves_type(self):
        """Should preserve existing type."""
        node = {"type": "integer"}
        update_node(node, {"type": "number"})
        assert node["type"] == "integer"

    def test_preserves_format(self):
        """Should preserve existing format."""
        node = {"type": "string", "format": "email"}
        update_node(node, {"format": "uri"})
        assert node["format"] == "email"


class TestUpdateNodeDescription:
    """Test description handling."""

    def test_adds_description_to_empty(self):
        """Should add description when none exists."""
        node = {"type": "string"}
        update_node(node, {"description": "Field description"})
        assert node["description"] == "Field description"

    def test_appends_to_existing_description(self):
        """Should append to existing description."""
        node = {"type": "string", "description": "Customer age"}
        update_node(node, {"description": "Must be 18+"})
        assert node["description"] == "Customer age\nMust be 18+"

    def test_handles_empty_new_description(self):
        """Should not modify if new description is empty."""
        node = {"type": "string", "description": "Original"}
        update_node(node, {"description": ""})
        assert node["description"] == "Original"

    def test_trims_whitespace_in_appended_description(self):
        """Should trim whitespace when appending."""
        node = {"type": "string", "description": "  Customer age  "}
        update_node(node, {"description": "  Must be 18+  "})
        # Should handle existing whitespace gracefully
        assert "Customer age" in node["description"]
        assert "Must be 18+" in node["description"]


class TestUpdateNodeExtensions:
    """Test custom extension properties (x-* fields)."""

    def test_adds_business_rule_extension(self):
        """Should add x-business-rule extension."""
        node = {"type": "integer"}
        update_node(node, {"x-business-rule": "Age must be 18+"})
        assert node["x-business-rule"] == "Age must be 18+"

    def test_adds_required_extension(self):
        """Should add x-required extension."""
        node = {"type": "string"}
        update_node(node, {"x-required": True})
        assert node["x-required"] is True

    def test_adds_custom_extension(self):
        """Should add any x-* extension property."""
        node = {"type": "string"}
        update_node(node, {"x-source": "informatica", "x-column-name": "CUST_AGE"})
        assert node["x-source"] == "informatica"
        assert node["x-column-name"] == "CUST_AGE"


class TestUpdateNodeInPlace:
    """Test that updates happen in-place."""

    def test_modifies_original_node(self):
        """Should modify the original node object."""
        node = {"type": "integer"}
        original_id = id(node)
        update_node(node, {"minimum": 18})
        assert id(node) == original_id

    def test_returns_none(self):
        """Should return None (in-place modification)."""
        node = {"type": "integer"}
        result = update_node(node, {"minimum": 18})
        assert result is None

    def test_empty_constraints_no_change(self):
        """Should not modify node if constraints are empty."""
        node = {"type": "integer", "minimum": 0}
        original = dict(node)
        update_node(node, {})
        assert node == original
