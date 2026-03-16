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

  # ============================================================
  # Model Evaluation Scenarios - DeepEval Integration
  # ============================================================

  Scenario: Create evaluation dataset from training data
    Given training data with business rules and expected JSON constraints
    When I create an evaluation dataset
    Then the dataset should contain test cases with input and expected output

  Scenario: Evaluate JSON correctness metric
    Given a model output '{"minimum": 18}'
    When I evaluate with JsonCorrectnessMetric
    Then the metric should pass for valid JSON
    And the metric should fail for invalid JSON like '{"minimum": }'

  Scenario: Evaluate schema validity metric
    Given a model output '{"minimum": 18}' for field type integer
    When I evaluate with SchemaValidMetric
    Then the metric should pass for valid schema constraints
    And the metric should fail for '{"maxLength": 10}' on integer type

  Scenario: Evaluate exact match metric
    Given expected output '{"minimum": 18}'
    And actual output '{"minimum": 18}'
    When I evaluate with ExactMatchMetric
    Then the metric should pass for identical outputs
    And the metric should fail for '{"minimum": 19}'

  Scenario: Evaluate semantic match metric
    Given expected output '{"minimum": 0, "maximum": 100}'
    And actual output '{"maximum": 100, "minimum": 0}'
    When I evaluate with SemanticMatchMetric
    Then the metric should pass for equivalent JSON with different key order

  Scenario: Track latency metric
    Given a model interpreter
    When I run inference and measure latency
    Then LatencyMetric should record the response time in milliseconds

  Scenario: Run evaluation suite on model
    Given an evaluation dataset with 10 test cases
    When I run the evaluation suite
    Then I should get aggregate scores for each metric
    And I should get per-test-case results

  Scenario: Compare T5 vs LLM performance
    Given a T5 interpreter and an LLM interpreter
    And an evaluation dataset
    When I run comparison evaluation
    Then I should get side-by-side metric scores
    And I should get latency comparison

  Scenario: Evaluate by constraint type
    Given an evaluation dataset with various constraint types
    When I run evaluation grouped by constraint type
    Then I should get accuracy for minimum/maximum constraints
    And I should get accuracy for pattern constraints
    And I should get accuracy for enum constraints

  Scenario: Evaluate by domain
    Given evaluation datasets for financial and healthcare domains
    When I run evaluation for each domain
    Then I should get per-domain accuracy scores

  Scenario: Generate evaluation report
    Given completed evaluation results
    When I generate an evaluation report
    Then the report should include summary statistics
    And the report should include failure analysis
    And the report should be saved to a file

  Scenario: CLI runs evaluation
    Given a trained model adapter
    When I run 'python -m src.evaluate --model t5 --domain financial'
    Then evaluation should run on the domain dataset
    And results should be printed to stdout

  # ============================================================
  # Data Dictionary Type-Constraint Scenarios
  # ============================================================

  Scenario: maxLength not applied to NUMBER data type
    Given a data dictionary with "customer_id" having data_type NUMBER and Length 10
    When I get constraints for "customer_id"
    Then maxLength should NOT be in the constraints

  Scenario: maxLength not applied to INTEGER data type
    Given a data dictionary with "order_count" having data_type INTEGER and Length 5
    When I get constraints for "order_count"
    Then maxLength should NOT be in the constraints

  Scenario: maxLength not applied to DECIMAL data type
    Given a data dictionary with "price" having data_type DECIMAL and Length 10
    When I get constraints for "price"
    Then maxLength should NOT be in the constraints

  Scenario: maxLength applied to VARCHAR data type
    Given a data dictionary with "customer_name" having data_type VARCHAR and Length 100
    When I get constraints for "customer_name"
    Then maxLength should be 100

  Scenario: maxLength applied to CHAR data type
    Given a data dictionary with "state_code" having data_type CHAR and Length 2
    When I get constraints for "state_code"
    Then maxLength should be 2

  Scenario: maxLength applied to STRING data type
    Given a data dictionary with "notes" having data_type STRING and Length 500
    When I get constraints for "notes"
    Then maxLength should be 500

  Scenario: maxLength applied to TEXT data type
    Given a data dictionary with "description" having data_type TEXT and Length 2000
    When I get constraints for "description"
    Then maxLength should be 2000

  # ============================================================
  # T5 Interpreter Scenarios
  # ============================================================

  Scenario: T5 interprets minimum constraint
    Given a trained T5 adapter for financial domain
    When I interpret "Balance must be at least 1000"
    Then I should get {"minimum": 1000}

  Scenario: T5 interprets maximum constraint
    Given a trained T5 adapter for financial domain
    When I interpret "Age cannot exceed 65"
    Then I should get {"maximum": 65}

  Scenario: T5 interprets enum constraint
    Given a trained T5 adapter for financial domain
    When I interpret "Status must be ACTIVE or INACTIVE"
    Then I should get {"enum": ["ACTIVE", "INACTIVE"]}

  Scenario: T5 interprets range constraint
    Given a trained T5 adapter for financial domain
    When I interpret "Value must be between 0 and 100"
    Then I should get {"minimum": 0, "maximum": 100}

  Scenario: T5 interprets format constraint
    Given a trained T5 adapter for financial domain
    When I interpret "Email must be valid format"
    Then I should get {"format": "email"}

  Scenario: T5 interprets maxLength constraint
    Given a trained T5 adapter for financial domain
    When I interpret "Maximum 100 characters"
    Then I should get {"maxLength": 100}

  Scenario: T5 output parsed without braces
    Given T5 model output '"minimum":18'
    When I parse the output
    Then I should get {"minimum": 18}

  Scenario: T5 output parsed with braces
    Given T5 model output '{"minimum":18}'
    When I parse the output
    Then I should get {"minimum": 18}

  Scenario: Load T5 adapter by name
    Given adapters directory with "adapter_financial/final_adapter"
    When I call load_adapter("financial")
    Then a T5Interpreter should be returned

  # ============================================================
  # Hybrid Interpreter Scenarios
  # ============================================================

  Scenario: Hybrid uses T5 when valid
    Given a hybrid interpreter with T5 and LLM
    When T5 returns valid constraints
    Then T5 result should be used
    And LLM should not be called

  Scenario: Hybrid falls back to LLM on invalid T5 output
    Given a hybrid interpreter with T5 and LLM
    When T5 returns maxLength for integer field (invalid)
    Then LLM should be called as fallback
    And LLM result should be used

  Scenario: Hybrid falls back to LLM on empty T5 output
    Given a hybrid interpreter with T5 and LLM
    When T5 returns empty result
    Then LLM should be called as fallback

  Scenario: Hybrid falls back to LLM on T5 exception
    Given a hybrid interpreter with T5 and LLM
    When T5 raises an exception
    Then LLM should be called as fallback

  Scenario: Hybrid uses LLM only when no T5
    Given a hybrid interpreter with only LLM
    When I interpret a business rule
    Then LLM should be used directly

  Scenario: Hybrid returns empty when both fail
    Given a hybrid interpreter with T5 and LLM
    When both T5 and LLM fail
    Then empty dict should be returned

  Scenario: Hybrid logs T5 failures
    Given a hybrid interpreter with failure logging enabled
    When T5 fails with validation errors
    Then failure should be logged with timestamp and details
    And failure log should be saved to file

  Scenario: Hybrid tracks statistics
    Given a hybrid interpreter
    When I make multiple interpretation calls
    Then stats should include total_calls
    And stats should include t5_successes
    And stats should include t5_failures
    And stats should include llm_fallbacks
    And stats should include t5_success_rate

  # ============================================================
  # Production Data Collection Scenarios
  # ============================================================

  Scenario: Collect T5 successful interpretations
    Given a hybrid interpreter with training data collection enabled
    When T5 successfully interprets a rule
    Then the training example should be collected
    And the example should have source "t5"

  Scenario: Collect LLM successful interpretations
    Given a hybrid interpreter with training data collection enabled
    When LLM successfully interprets a rule
    Then the training example should be collected
    And the example should have source "llm"

  Scenario: Training data includes input and output
    Given collected training data
    Then each example should have "input" (business rule)
    And each example should have "output" (JSON constraints)
    And each example should have "field_type"

  Scenario: Do not collect when disabled
    Given a hybrid interpreter with training data collection disabled
    When I interpret business rules
    Then no training data should be collected

  Scenario: Save training data to file
    Given a hybrid interpreter with training_log_path configured
    When successful interpretations occur
    Then training data should be saved to the file
    And the file should include total_examples count
    And the file should include source breakdown

  Scenario: Export training data in dataset format
    Given collected training data from production
    When I call export_training_data with domain "production"
    Then a JSON file should be created
    And the file should have "domain" field
    And the file should have "examples" array
    And the format should match training data files
