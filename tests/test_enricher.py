"""Tests for schema enricher module following TDD."""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from src.enricher import SchemaEnricher, enrich_schema


class TestSchemaEnricherInitialization:
    """Test SchemaEnricher initialization."""

    def test_creates_enricher_with_schema_and_dict(self):
        """Should create enricher with schema and dictionary."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        csv_content = """Column Name,Data Type,Length,Nullable
name,VARCHAR,100,Y
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            enricher = SchemaEnricher(schema, csv_path)
            assert enricher is not None
            assert enricher.schema == schema
        finally:
            os.unlink(csv_path)


class TestSchemaEnrichmentWithoutLLM:
    """Test schema enrichment without LLM interpretation."""

    def test_enriches_simple_field(self):
        """Should add constraints from dictionary to schema."""
        schema = {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string"}
            }
        }

        csv_content = """Column Name,Data Type,Length,Nullable,Description
customer_name,VARCHAR,100,N,Customer full name
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            enricher = SchemaEnricher(schema, csv_path)
            result = enricher.enrich()

            prop = result["properties"]["customer_name"]
            assert prop["maxLength"] == 100
            assert prop["description"] == "Customer full name"
        finally:
            os.unlink(csv_path)

    def test_enriches_with_enum(self):
        """Should add enum constraint from valid values."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"}
            }
        }

        csv_content = """Column Name,Data Type,Length,Nullable,Valid_Values
status,VARCHAR,20,N,ACTIVE|INACTIVE|PENDING
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            enricher = SchemaEnricher(schema, csv_path)
            result = enricher.enrich()

            prop = result["properties"]["status"]
            assert prop["enum"] == ["ACTIVE", "INACTIVE", "PENDING"]
        finally:
            os.unlink(csv_path)

    def test_preserves_existing_constraints(self):
        """Should not overwrite existing schema constraints."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0}
            }
        }

        csv_content = """Column Name,Data Type,Length,Nullable
age,NUMBER,3,N
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            # Disable validation - this test checks enrichment preserves existing values,
            # not type-constraint compatibility (which is tested in test_schema_validator.py)
            enricher = SchemaEnricher(schema, csv_path, validate=False)
            result = enricher.enrich()

            prop = result["properties"]["age"]
            assert prop["minimum"] == 0  # Original preserved
        finally:
            os.unlink(csv_path)


class TestSchemaEnrichmentWithLLM:
    """Test schema enrichment with LLM interpretation."""

    def test_enriches_with_business_rule(self):
        """Should use LLM to interpret business rules."""
        schema = {
            "type": "object",
            "properties": {
                "customer_age": {"type": "integer"}
            }
        }

        csv_content = """Column Name,Data Type,Length,Nullable,Business_Rule
customer_age,NUMBER,3,N,Must be 18 or older
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            mock_llm = Mock()
            mock_llm.complete.return_value = '{"minimum": 18}'

            # Disable validation - this test checks LLM integration,
            # not type-constraint compatibility (which is tested in test_schema_validator.py)
            enricher = SchemaEnricher(schema, csv_path, llm_client=mock_llm, validate=False)
            result = enricher.enrich()

            prop = result["properties"]["customer_age"]
            assert prop["minimum"] == 18
            assert prop["x-business-rule"] == "Must be 18 or older"
        finally:
            os.unlink(csv_path)


class TestNestedSchemaEnrichment:
    """Test enrichment of nested schemas."""

    def test_enriches_nested_properties(self):
        """Should enrich nested object properties."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "zipcode": {"type": "string"}
                    }
                }
            }
        }

        csv_content = """Column Name,Data Type,Length,Nullable
zipcode,VARCHAR,10,N
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            enricher = SchemaEnricher(schema, csv_path)
            result = enricher.enrich()

            prop = result["properties"]["address"]["properties"]["zipcode"]
            assert prop["maxLength"] == 10
        finally:
            os.unlink(csv_path)


class TestEnrichSchemaFunction:
    """Test the convenience function."""

    def test_enrich_schema_from_files(self):
        """Should enrich schema from file paths."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        csv_content = """Column Name,Data Type,Length,Nullable
name,VARCHAR,50,Y
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as sf:
            json.dump(schema, sf)
            schema_path = sf.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as df:
            df.write(csv_content)
            dict_path = df.name

        try:
            result = enrich_schema(schema_path, dict_path)
            assert result["properties"]["name"]["maxLength"] == 50
        finally:
            os.unlink(schema_path)
            os.unlink(dict_path)
