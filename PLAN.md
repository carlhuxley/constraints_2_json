# Plan: JSON Schema Node Navigation + Constraint Enrichment

## Overview
Build a Python tool that:
1. Loads an existing JSON Schema
2. Walks the schema tree to find each property node
3. Looks up each field's constraints from the Informatica data dictionary
4. Uses LLM to interpret business rules into JSON Schema constraints
5. Adds constraints in-place to each node
6. Handles nested schemas (objects within objects)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  JSON Schema    │────▶│  Schema          │
│  (Input)        │     │  Navigator       │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Node A   │ │ Node B   │ │ Node C   │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   ▼            ▼            ▼
              ┌──────────────────────────────────┐
              │     Data Dictionary Lookup       │
              │     (CSV/Informatica metadata)   │
              └────────────────┬─────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │     LLM Constraint Interpreter   │
              │     (for business rules)         │
              └────────────────┬─────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │     In-Place Node Update         │
              └──────────────────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │     Enriched JSON Schema         │
              └──────────────────────────────────┘
```

## Processing Flow (Per Node)

```
for each property node in JSON Schema:
    1. Extract field path (e.g., "user.address.zipcode")
    2. Lookup field in data dictionary by name
    3. If found:
       a. Extract simple constraints (min, max, pattern, enum)
       b. If business_rule text exists:
          - Call LLM to interpret into JSON Schema keywords
       c. Merge constraints into the node
    4. If not found:
       - Log warning (field exists in schema but not in dict)
    5. Move to next node
```

## Files to Create

### 1. `constraints_2_json/schema_navigator.py`
Navigate JSON Schema tree:
- Recursive traversal of `properties`
- Handle nested objects (`type: object` with nested `properties`)
- Handle arrays (`items` schema)
- Handle `$ref` references (optional, can skip initially)
- Yield (path, node) tuples for each property

```python
def walk_schema(schema: dict, path: str = "") -> Iterator[tuple[str, dict]]:
    """Yield (field_path, property_node) for each property in schema."""
    if "properties" in schema:
        for name, prop in schema["properties"].items():
            current_path = f"{path}.{name}" if path else name
            yield (current_path, prop)
            # Recurse into nested objects
            if prop.get("type") == "object":
                yield from walk_schema(prop, current_path)
            # Handle array items
            if prop.get("type") == "array" and "items" in prop:
                yield from walk_schema(prop["items"], f"{current_path}[]")
```

### 2. `constraints_2_json/dict_lookup.py`
Data dictionary lookup:
- Load and index CSV by field name
- Support fuzzy matching (optional: for column name variations)
- Handle hierarchical field names (nested paths)

```python
class DataDictionary:
    def __init__(self, csv_path: str):
        self.fields = {}  # field_name -> constraint dict
        self._load(csv_path)

    def lookup(self, field_path: str) -> Optional[dict]:
        """Find constraints for a field, trying various name formats."""
        # Try exact match: "user.address.zipcode"
        # Try leaf name: "zipcode"
        # Try normalized: "ZIP_CODE" -> "zipcode"
```

### 3. `constraints_2_json/llm_interpreter.py`
Interpret business rules via LLM:
- Single-field context (stays within limits)
- Structured output parsing
- Returns JSON Schema constraint fragment

```python
def interpret_business_rule(
    field_name: str,
    field_type: str,
    business_rule: str,
    llm_client: LLMClient
) -> dict:
    """Convert business rule text to JSON Schema constraints."""
    prompt = f"""
    Convert this business rule to JSON Schema constraints:

    Field: {field_name}
    Type: {field_type}
    Rule: {business_rule}

    Return only valid JSON with applicable constraints:
    {{"minimum": ..., "maximum": ..., "pattern": ..., "enum": [...], etc.}}
    """
    response = llm_client.complete(prompt)
    return json.loads(response)
```

### 4. `constraints_2_json/node_updater.py`
Update JSON Schema nodes in-place:
- Merge constraints without overwriting existing
- Handle type-specific keywords
- Preserve existing descriptions (append, don't replace)

```python
def update_node(node: dict, constraints: dict) -> None:
    """Merge constraints into JSON Schema node in-place."""
    for key, value in constraints.items():
        if key == "description":
            # Append to existing description
            existing = node.get("description", "")
            node["description"] = f"{existing}\n{value}".strip()
        elif key not in node:
            # Add new constraint
            node[key] = value
        # Skip if already present (preserve original)
```

### 5. `constraints_2_json/main.py`
CLI orchestration:
- Load schema, load dictionary
- Walk schema and process each node
- Save enriched schema

## Informatica Data Dictionary Columns

| Column              | Maps To                                    |
|---------------------|--------------------------------------------|
| Column Name         | Lookup key                                 |
| Data Type           | Validates against schema type              |
| Length              | maxLength (strings)                        |
| Precision/Scale     | multipleOf (numbers)                       |
| Nullable            | Inverse of required                        |
| Valid Values        | enum array                                 |
| Business Rule       | LLM interpretation → various constraints   |
| Description         | description (append)                       |

## Example Transformation

**Input JSON Schema Node:**
```json
{
  "customer_age": {
    "type": "integer",
    "description": "Customer's age"
  }
}
```

**Data Dictionary Row:**
```csv
Column Name,Data Type,Nullable,Business Rule
customer_age,NUMBER(3),N,Must be 18 or older for account opening. Maximum age 120.
```

**After Enrichment:**
```json
{
  "customer_age": {
    "type": "integer",
    "description": "Customer's age\nMust be 18 or older for account opening. Maximum age 120.",
    "minimum": 18,
    "maximum": 120,
    "x-business-rule": "Must be 18 or older for account opening. Maximum age 120."
  }
}
```

## CLI Usage

```bash
python -m src.main \
  --schema input_schema.json \
  --dict informatica_metadata.csv \
  --output enriched_schema.json \
  --llm openrouter \
  --model deepseek/deepseek-chat
```

## Implementation Steps

1. **Create package structure**
   - `__init__.py`, basic setup

2. **Implement schema navigator** (`schema_navigator.py`)
   - Recursive tree walker
   - Handle nested objects and arrays

3. **Implement dictionary lookup** (`dict_lookup.py`)
   - CSV loading and indexing
   - Name matching logic

4. **Implement LLM interpreter** (`llm_interpreter.py`)
   - LLM client wrapper
   - Prompt template
   - Response parsing

5. **Implement node updater** (`node_updater.py`)
   - Merge logic
   - Type-specific handling

6. **Implement CLI** (`main.py`)
   - Wire everything together
   - Progress logging

## Verification

1. Create sample JSON Schema with nested objects
2. Create matching data dictionary with business rules
3. Run tool and verify:
   - All nodes visited
   - Constraints correctly merged
   - LLM interpretations are valid JSON Schema
   - Original schema structure preserved
