"""Tests for dict_lookup module following TDD."""
import pytest
import tempfile
import os
from src.dict_lookup import DataDictionary


class TestDataDictionaryLoading:
    """Test loading data dictionary from CSV."""

    def test_loads_csv_file(self):
        """Should load and parse CSV file."""
        csv_content = """Column Name,Data Type,Length,Nullable
customer_age,NUMBER,3,N
customer_name,VARCHAR,100,Y
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            assert dd is not None
            assert "customer_age" in dd.fields
            assert "customer_name" in dd.fields
        finally:
            os.unlink(csv_path)

    def test_parses_field_info(self):
        """Should parse field information from CSV."""
        csv_content = """Column Name,Data Type,Length,Nullable,Valid_Values,Business_Rule,Description
customer_age,NUMBER,3,N,,Must be 18 or older,Customer age in years
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            field = dd.fields["customer_age"]
            assert field["data_type"] == "NUMBER"
            assert field["length"] == "3"
            assert field["nullable"] == "N"
            assert field["business_rule"] == "Must be 18 or older"
        finally:
            os.unlink(csv_path)


class TestDataDictionaryLookup:
    """Test field lookup functionality."""

    def test_lookup_exact_name(self):
        """Should find field by exact name."""
        csv_content = """Column Name,Data Type,Length,Nullable
order_total,NUMBER,10,N
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            result = dd.lookup("order_total")
            assert result is not None
            assert result["data_type"] == "NUMBER"
        finally:
            os.unlink(csv_path)

    def test_lookup_leaf_name(self):
        """Should find field by leaf name from path."""
        csv_content = """Column Name,Data Type,Length,Nullable
zipcode,VARCHAR,10,Y
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            result = dd.lookup("customer.address.zipcode")
            assert result is not None
            assert result["data_type"] == "VARCHAR"
        finally:
            os.unlink(csv_path)

    def test_lookup_returns_none_for_unknown(self):
        """Should return None for unknown field."""
        csv_content = """Column Name,Data Type,Length,Nullable
customer_age,NUMBER,3,N
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            result = dd.lookup("unknown_field")
            assert result is None
        finally:
            os.unlink(csv_path)

    def test_lookup_case_insensitive(self):
        """Should handle case-insensitive lookup."""
        csv_content = """Column Name,Data Type,Length,Nullable
CUSTOMER_AGE,NUMBER,3,N
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            result = dd.lookup("customer_age")
            assert result is not None
        finally:
            os.unlink(csv_path)


class TestDataDictionaryConstraintParsing:
    """Test constraint parsing from CSV columns."""

    def test_parses_length_constraint(self):
        """Should parse Length column for maxLength."""
        csv_content = """Column Name,Data Type,Length,Nullable
customer_name,VARCHAR,50,Y
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            constraints = dd.get_constraints("customer_name")
            assert constraints.get("maxLength") == 50
        finally:
            os.unlink(csv_path)

    def test_parses_enum_from_valid_values(self):
        """Should parse Valid_Values column into enum."""
        csv_content = """Column Name,Data Type,Length,Nullable,Valid_Values
status,VARCHAR,20,N,ACTIVE|INACTIVE|PENDING
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            constraints = dd.get_constraints("status")
            assert constraints.get("enum") == ["ACTIVE", "INACTIVE", "PENDING"]
        finally:
            os.unlink(csv_path)

    def test_parses_nullable_to_required(self):
        """Should map Nullable 'N' to required hint."""
        csv_content = """Column Name,Data Type,Length,Nullable
customer_id,NUMBER,10,N
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            constraints = dd.get_constraints("customer_id")
            assert constraints.get("x-required") is True
        finally:
            os.unlink(csv_path)

    def test_includes_business_rule(self):
        """Should include business rule in constraints."""
        csv_content = """Column Name,Data Type,Length,Nullable,Business_Rule
customer_age,NUMBER,3,N,Must be 18 or older
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            constraints = dd.get_constraints("customer_age")
            assert constraints.get("x-business-rule") == "Must be 18 or older"
        finally:
            os.unlink(csv_path)

    def test_includes_description(self):
        """Should include description in constraints."""
        csv_content = """Column Name,Data Type,Length,Nullable,Description
customer_age,NUMBER,3,N,Customer age in years
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            dd = DataDictionary(csv_path)
            constraints = dd.get_constraints("customer_age")
            assert constraints.get("description") == "Customer age in years"
        finally:
            os.unlink(csv_path)
