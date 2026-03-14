"""Tests for schema_navigator module following TDD."""
import pytest
from src.schema_navigator import walk_schema


class TestWalkSchemaFlatProperties:
    """Test walking flat schema properties."""

    def test_walks_simple_properties(self):
        """Should yield tuples for each property in flat schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        result = list(walk_schema(schema))
        paths = [path for path, _ in result]

        assert "name" in paths
        assert "age" in paths

    def test_yields_property_node(self):
        """Each tuple should contain the property node dict."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 100}
            }
        }
        result = list(walk_schema(schema))

        assert len(result) == 1
        path, node = result[0]
        assert path == "name"
        assert node == {"type": "string", "maxLength": 100}


class TestWalkSchemaNestedObjects:
    """Test walking nested object properties."""

    def test_walks_nested_object(self):
        """Should yield paths for nested object properties."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }
        result = list(walk_schema(schema))
        paths = [path for path, _ in result]

        assert "address" in paths
        assert "address.street" in paths
        assert "address.city" in paths

    def test_deeply_nested_structure(self):
        """Should handle deeply nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "settings": {
                                    "type": "object",
                                    "properties": {
                                        "theme": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result = list(walk_schema(schema))
        paths = [path for path, _ in result]

        assert "user" in paths
        assert "user.profile" in paths
        assert "user.profile.settings" in paths
        assert "user.profile.settings.theme" in paths


class TestWalkSchemaArrays:
    """Test walking array schemas."""

    def test_walks_array_items(self):
        """Should yield path for array and its items schema."""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        result = list(walk_schema(schema))
        paths = [path for path, _ in result]

        assert "tags" in paths
        assert "tags[]" in paths

    def test_walks_array_with_object_items(self):
        """Should recurse into array item objects."""
        schema = {
            "type": "object",
            "properties": {
                "orders": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "total": {"type": "number"}
                        }
                    }
                }
            }
        }
        result = list(walk_schema(schema))
        paths = [path for path, _ in result]

        assert "orders" in paths
        assert "orders[]" in paths
        assert "orders[].id" in paths
        assert "orders[].total" in paths


class TestWalkSchemaEdgeCases:
    """Test edge cases for schema walking."""

    def test_empty_schema(self):
        """Should handle empty schema."""
        schema = {}
        result = list(walk_schema(schema))
        assert result == []

    def test_schema_without_properties(self):
        """Should handle schema without properties key."""
        schema = {"type": "object"}
        result = list(walk_schema(schema))
        assert result == []

    def test_preserves_all_property_attributes(self):
        """Should preserve all attributes in yielded node."""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "maxLength": 255,
                    "description": "User email address"
                }
            }
        }
        result = list(walk_schema(schema))
        _, node = result[0]

        assert node["type"] == "string"
        assert node["format"] == "email"
        assert node["maxLength"] == 255
        assert node["description"] == "User email address"
