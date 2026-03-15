Feature: JSON Schema Constraint Enrichment
  As a data engineer
  I want to enrich JSON Schema with constraints from a data dictionary
  So that my schemas have accurate validation rules

  # ============================================================
  # Schema Navigator Scenarios
  # ============================================================

  Scenario: Walk flat schema properties
    Given a JSON Schema with properties "name" type string and "age" type integer
    When I call walk_schema
    Then I should get tuples with paths "name" and "age"
    And each tuple should contain the property node

  Scenario: Walk nested object properties
    Given a JSON Schema with nested object "address" containing "street" and "city"
    When I call walk_schema
    Then I should get path "address" for the object
    And I should get path "address.street" for the nested property
    And I should get path "address.city" for the nested property

  Scenario: Walk array with item schema
    Given a JSON Schema with array "tags" containing string items
    When I call walk_schema
    Then I should get path "tags" for the array
    And I should get path "tags[]" for the array items schema

  Scenario: Walk deeply nested structure
    Given a JSON Schema with "user.profile.settings.theme" nested path
    When I call walk_schema
    Then I should yield all intermediate and leaf paths

  # ============================================================
  # Data Dictionary Lookup Scenarios
  # ============================================================

  Scenario: Load data dictionary from CSV
    Given a CSV with field "customer_age" and data_type "NUMBER"
    When I create a DataDictionary from the CSV
    Then lookup "customer_age" should return the field info

  Scenario: Lookup field by exact name
    Given a data dictionary with field "order_total"
    When I lookup "order_total"
    Then I should get the field constraints

  Scenario: Lookup field by leaf name
    Given a data dictionary with field "zipcode"
    When I lookup "customer.address.zipcode"
    Then I should find "zipcode" by leaf name matching

  Scenario: Lookup returns None for unknown field
    Given a data dictionary without field "unknown_field"
    When I lookup "unknown_field"
    Then I should get None

  Scenario: Parse constraint columns from CSV
    Given a CSV row with Length "50" and Nullable "N" and Valid_Values "A,B,C"
    When I lookup the field
    Then I should get maxLength 50
    And I should get enum ["A", "B", "C"]

  # ============================================================
  # Node Updater Scenarios
  # ============================================================

  Scenario: Update node with constraints
    Given a JSON Schema node with type integer
    When I call update_node with minimum 18 and maximum 120
    Then the node should have minimum 18 and maximum 120

  Scenario: Update node preserves existing values
    Given a JSON Schema node with existing minimum 0
    When I call update_node with minimum 18
    Then the node should retain minimum 0

  Scenario: Update node appends to description
    Given a JSON Schema node with description "Customer age"
    When I call update_node with description "Must be 18+"
    Then the description should be "Customer age\nMust be 18+"

  Scenario: Update node adds new constraints
    Given a JSON Schema node with type string
    When I call update_node with pattern "^[A-Z]{2}$" and maxLength 2
    Then the node should have pattern "^[A-Z]{2}$"
    And the node should have maxLength 2

  Scenario: Update node with business rule extension
    Given a JSON Schema node with type integer
    When I call update_node with x-business-rule "Age must be 18+"
    Then the node should have x-business-rule "Age must be 18+"

  # ============================================================
  # LLM Interpreter Scenarios
  # ============================================================

  Scenario: Interpret age minimum rule
    Given business rule text "Must be 18 or older"
    When I interpret the rule for field type integer
    Then I should get a constraint dict with minimum 18

  Scenario: Interpret range rule
    Given business rule text "Value must be between 0 and 100"
    When I interpret the rule for field type number
    Then I should get minimum 0 and maximum 100

  Scenario: Interpret pattern rule
    Given business rule text "Must be a valid US state code (2 uppercase letters)"
    When I interpret the rule for field type string
    Then I should get pattern "^[A-Z]{2}$"

  Scenario: Interpret enum rule
    Given business rule text "Status must be ACTIVE, INACTIVE, or PENDING"
    When I interpret the rule for field type string
    Then I should get enum ["ACTIVE", "INACTIVE", "PENDING"]

  Scenario: Interpret length rule
    Given business rule text "Maximum 255 characters allowed"
    When I interpret the rule for field type string
    Then I should get maxLength 255

  # ============================================================
  # End-to-End Integration Scenarios
  # ============================================================

  Scenario: Enrich simple schema from dictionary
    Given a JSON Schema with property "customer_age" type integer
    And a data dictionary with "customer_age" having minimum 18 and maximum 120
    When I run the enrichment process
    Then the enriched schema should have minimum 18 on "customer_age"
    And the enriched schema should have maximum 120 on "customer_age"

  Scenario: Enrich schema with business rule interpretation
    Given a JSON Schema with property "status" type string
    And a data dictionary with "status" having business_rule "Must be ACTIVE or INACTIVE"
    When I run the enrichment process with LLM interpretation
    Then the enriched schema should have enum ["ACTIVE", "INACTIVE"] on "status"

  # ============================================================
  # Schema Validation Scenarios - Meta-Schema Validation
  # ============================================================

  Scenario: Valid simple schema passes validation
    Given a JSON Schema with properties "name" type string and "age" type integer
    When I call validate_schema
    Then validation should pass with no errors

  Scenario: Valid schema with constraints passes validation
    Given a JSON Schema with property "email" type string with format "email" and maxLength 255
    And a property "count" type integer with minimum 0 and maximum 100
    When I call validate_schema
    Then validation should pass with no errors

  Scenario: Detect invalid type value
    Given a JSON Schema with type "invalid_type"
    When I call validate_schema
    Then validation should fail with errors

  Scenario: Detect invalid constraint value type
    Given a JSON Schema with type string and maxLength "not_a_number"
    When I call validate_schema
    Then validation should fail with errors

  Scenario: Empty schema is valid
    Given an empty JSON Schema
    When I call validate_schema
    Then validation should pass with no errors

  # ============================================================
  # Schema Validation Scenarios - Type-Constraint Compatibility
  # ============================================================

  Scenario: String constraints on string type pass
    Given a JSON Schema node type string with minLength 1, maxLength 100, and pattern "^[a-z]+$"
    When I call validate_node_constraints
    Then validation should return no errors

  Scenario: Number constraints on number type pass
    Given a JSON Schema node type number with minimum 0, maximum 100, and multipleOf 0.5
    When I call validate_node_constraints
    Then validation should return no errors

  Scenario: Integer constraints on integer type pass
    Given a JSON Schema node type integer with minimum 0 and maximum 100
    When I call validate_node_constraints
    Then validation should return no errors

  Scenario: Array constraints on array type pass
    Given a JSON Schema node type array with minItems 1, maxItems 10, and uniqueItems true
    When I call validate_node_constraints
    Then validation should return no errors

  Scenario: String constraint on number type fails
    Given a JSON Schema node type number with maxLength 10
    When I call validate_node_constraints
    Then validation should return error mentioning "maxLength" and "number"

  Scenario: Number constraint on string type fails
    Given a JSON Schema node type string with minimum 0
    When I call validate_node_constraints
    Then validation should return error mentioning "minimum" and "string"

  Scenario: Multiple incompatible constraints all reported
    Given a JSON Schema node type string with minimum 0, maximum 100, and multipleOf 5
    When I call validate_node_constraints
    Then validation should return 3 errors

  Scenario: Union type with compatible constraint passes
    Given a JSON Schema node type ["string", "null"] with maxLength 100
    When I call validate_node_constraints
    Then validation should return no errors

  Scenario: Union type with incompatible constraint fails
    Given a JSON Schema node type ["integer", "null"] with maxLength 100
    When I call validate_node_constraints
    Then validation should return 1 error

  Scenario: Node without type skips constraint validation
    Given a JSON Schema node without type but with maxLength 100
    When I call validate_node_constraints
    Then validation should return no errors

  Scenario: Error message includes field path
    Given a JSON Schema node type string with minimum 0 at path "user.profile.age"
    When I call validate_node_constraints
    Then the error should include "user.profile.age"

  # ============================================================
  # Schema Validation Scenarios - Full Schema Validation
  # ============================================================

  Scenario: Valid nested schema passes all checks
    Given a JSON Schema with nested "user.name" type string maxLength 100
    And nested "user.age" type integer minimum 0
    When I call validate_schema_constraints
    Then validation should pass with no errors

  Scenario: Invalid constraint in nested property caught
    Given a JSON Schema with nested "user.name" type string with minimum 0
    When I call validate_schema_constraints
    Then validation should fail
    And error should include "user.name"

  Scenario: Validates array item schemas
    Given a JSON Schema with array "tags" containing items type string with minimum 0
    When I call validate_schema_constraints
    Then validation should fail
    And error should include "tags[]"

  Scenario: Validates definitions
    Given a JSON Schema with definition "Address" containing "zipcode" type string with minimum 0
    When I call validate_schema_constraints
    Then validation should fail
    And error should include "definitions.Address"

  # ============================================================
  # Schema Validation Scenarios - Exception Handling
  # ============================================================

  Scenario: SchemaValidationError contains errors list
    Given a SchemaValidationError with errors ["Error 1", "Error 2"]
    Then the exception errors property should contain both errors
    And the exception message should mention "2 error"

  Scenario: SchemaValidationError message format
    Given a SchemaValidationError with errors ["Single error"]
    Then the exception message should mention "1 error"

  # ============================================================
  # Schema Validation Scenarios - Integration
  # ============================================================

  Scenario: CLI validates by default
    Given an input schema file and data dictionary
    When I run the CLI without --no-validate flag
    Then validation should be performed
    And validation errors should be printed to stderr

  Scenario: CLI skips validation with --no-validate
    Given an input schema file and data dictionary
    When I run the CLI with --no-validate flag
    Then validation should be skipped
    And the enriched schema should be output even if it has constraint issues

  Scenario: Enricher raises SchemaValidationError on invalid schema
    Given a JSON Schema that will have incompatible constraints after enrichment
    When I call enricher.enrich() with validation enabled
    Then a SchemaValidationError should be raised
    And the error should contain a list of validation errors
