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
