"""
Data Dictionary Lookup - loads and queries field constraints from CSV.

This module provides the DataDictionary class for loading Informatica
data dictionary metadata and looking up field constraints.
"""
import csv
from typing import Optional

# Data types that should have maxLength applied
STRING_DATA_TYPES = frozenset([
    "varchar", "varchar2", "nvarchar", "nvarchar2",
    "char", "nchar",
    "string", "text", "ntext",
    "clob", "nclob",
])

# Data types that are numeric (maxLength should NOT apply)
NUMERIC_DATA_TYPES = frozenset([
    "number", "numeric",
    "integer", "int", "smallint", "bigint", "tinyint",
    "decimal", "dec", "float", "real", "double",
    "money", "smallmoney",
])


class DataDictionary:
    """
    Loads and queries data dictionary from CSV file.

    Supports lookup by exact name or leaf name, with case-insensitive matching.
    """

    def __init__(self, csv_path: str):
        """
        Load data dictionary from CSV file.

        Args:
            csv_path: Path to the CSV file containing field metadata
        """
        self.fields: dict[str, dict] = {}
        self._load(csv_path)

    def _load(self, csv_path: str) -> None:
        """Load and index CSV by field name."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize column names (handle variations)
                normalized = self._normalize_row(row)
                field_name = normalized.get("column_name", "").strip()
                if field_name:
                    # Store both original and lowercase for lookup
                    self.fields[field_name.lower()] = normalized

    def _normalize_row(self, row: dict) -> dict:
        """Normalize column names to consistent format."""
        normalized = {}
        for key, value in row.items():
            # Convert to lowercase and replace spaces/special chars
            norm_key = key.lower().strip().replace(" ", "_")
            normalized[norm_key] = value.strip() if value else ""
        return normalized

    def lookup(self, field_path: str) -> Optional[dict]:
        """
        Find field info by path, trying various name formats.

        Args:
            field_path: Field path like "user.address.zipcode"

        Returns:
            Field info dict if found, None otherwise
        """
        field_path_lower = field_path.lower()

        # Try exact match
        if field_path_lower in self.fields:
            return self.fields[field_path_lower]

        # Try leaf name (last component of path)
        leaf_name = field_path.split(".")[-1].lower()
        # Remove array brackets if present
        leaf_name = leaf_name.rstrip("[]")

        if leaf_name in self.fields:
            return self.fields[leaf_name]

        return None

    def get_constraints(self, field_path: str) -> dict:
        """
        Get JSON Schema constraints for a field.

        Args:
            field_path: Field path to look up

        Returns:
            Dict of JSON Schema constraints derived from field metadata
        """
        field_info = self.lookup(field_path)
        if not field_info:
            return {}

        constraints = {}

        # Get the data type to determine which constraints apply
        data_type = field_info.get("data_type", "").lower()

        # Parse length -> maxLength (only for string types)
        length = field_info.get("length", "")
        if length and length.isdigit():
            # Only apply maxLength to string data types
            if data_type in STRING_DATA_TYPES:
                constraints["maxLength"] = int(length)

        # Parse valid_values -> enum
        valid_values = field_info.get("valid_values", "")
        if valid_values:
            # Support both pipe and comma separators
            if "|" in valid_values:
                constraints["enum"] = [v.strip() for v in valid_values.split("|")]
            elif "," in valid_values:
                constraints["enum"] = [v.strip() for v in valid_values.split(",")]

        # Parse nullable -> x-required
        nullable = field_info.get("nullable", "").upper()
        if nullable == "N":
            constraints["x-required"] = True

        # Include business rule
        business_rule = field_info.get("business_rule", "")
        if business_rule:
            constraints["x-business-rule"] = business_rule

        # Include description
        description = field_info.get("description", "")
        if description:
            constraints["description"] = description

        return constraints
