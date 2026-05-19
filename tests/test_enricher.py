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


class TestAuditLog:
    """Test per-field audit log generation."""

    def _make_enricher(self, schema, csv_content, **kwargs):
        """Helper: write temp files and return a SchemaEnricher."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        enricher = SchemaEnricher(schema, csv_path, **kwargs)
        os.unlink(csv_path)
        return enricher

    def test_audit_entry_for_dictionary_match(self):
        """Should record an entry for every field found in the dictionary."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        csv = "Column Name,Data Type,Length,Nullable\nage,INT,0,N\n"

        enricher = self._make_enricher(schema, csv)
        enricher.enrich(validate=False)

        entries = enricher.get_audit_log()
        assert len(entries) == 1
        assert entries[0]["field_path"] == "age"
        assert entries[0]["field_type"] == "integer"
        assert entries[0]["interpreter_source"] == "no_rule"

    def test_audit_entry_for_no_dictionary_match(self):
        """Should record a no_match entry for fields absent from the dictionary."""
        schema = {"type": "object", "properties": {"unknown_field": {"type": "string"}}}
        csv = "Column Name,Data Type,Length,Nullable\nother,VARCHAR,10,Y\n"

        enricher = self._make_enricher(schema, csv)
        enricher.enrich(validate=False)

        entries = enricher.get_audit_log()
        assert len(entries) == 1
        assert entries[0]["interpreter_source"] == "no_match"
        assert entries[0]["constraints_applied"] == []

    def test_audit_records_dictionary_keys(self):
        """Should record which keys came from the dictionary."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        csv = "Column Name,Data Type,Length,Nullable,Description\nname,VARCHAR,50,N,Full name\n"

        enricher = self._make_enricher(schema, csv)
        enricher.enrich(validate=False)

        entry = enricher.get_audit_log()[0]
        assert "maxLength" in entry["dictionary_keys"]
        assert "x-required" in entry["dictionary_keys"]
        assert "description" in entry["dictionary_keys"]

    def test_audit_records_interpreter_source_and_keys(self):
        """Should record T5/LLM source and which keys the interpreter contributed."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        csv = "Column Name,Data Type,Length,Nullable,Business Rule\nage,INT,0,N,Must be at least 18\n"

        mock_hi = Mock()
        mock_hi.interpret.return_value = {"minimum": 18}
        mock_hi.last_source = "t5"

        enricher = self._make_enricher(schema, csv, hybrid_interpreter=mock_hi)
        enricher.enrich(validate=False)

        entry = enricher.get_audit_log()[0]
        assert entry["interpreter_source"] == "t5"
        assert "minimum" in entry["interpreter_keys"]
        assert entry["business_rule"] == "Must be at least 18"

    def test_audit_log_written_to_file(self, tmp_path):
        """Should write audit log JSON to the specified path."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        csv_content = "Column Name,Data Type,Length,Nullable\nname,VARCHAR,100,Y\n"
        audit_path = tmp_path / "audit.json"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as sf:
            json.dump(schema, sf)
            schema_path = sf.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as df:
            df.write(csv_content)
            dict_path = df.name

        try:
            enrich_schema(schema_path, dict_path, audit_log_path=str(audit_path), validate=False)

            assert audit_path.exists()
            with open(audit_path) as f:
                data = json.load(f)

            assert data["schema"] == schema_path
            assert data["dictionary"] == dict_path
            assert isinstance(data["fields"], list)
            assert data["fields"][0]["field_path"] == "name"
        finally:
            os.unlink(schema_path)
            os.unlink(dict_path)
