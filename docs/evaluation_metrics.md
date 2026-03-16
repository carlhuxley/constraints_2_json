# Evaluation Metrics

This document describes the evaluation metrics used to assess the performance of the constraint generation models (T5 and LLM).

## Overview

The project uses **custom evaluation metrics** implemented in `src/model_evaluator.py`. These metrics follow DeepEval-style patterns but are purpose-built for evaluating business rule to JSON Schema constraint conversion.

## Metrics Summary

| Metric | Type | Description | Use Case |
|--------|------|-------------|----------|
| JSON Correctness | Accuracy | Is output valid JSON object? | Basic validity check |
| Schema Valid | Accuracy | Are constraints valid for field type? | Type compatibility |
| Exact Match | Accuracy | Does output exactly match expected? | Strict accuracy |
| Semantic Match | Accuracy | Equivalent ignoring key order? | Flexible accuracy |
| Latency | Performance | Inference time (avg, p99) | Speed comparison |

## Metric Implementations

### 1. JSON Correctness Metric

**Location:** `src/model_evaluator.py:156`

Measures whether the model output is a valid JSON object.

```python
class JsonCorrectnessMetric:
    def measure(self, output: str) -> float:
        # Returns 1.0 if valid JSON object, 0.0 otherwise
```

**Examples:**
- `{"minimum": 18}` → 1.0 (valid)
- `{"minimum": }` → 0.0 (invalid JSON)
- `"just a string"` → 0.0 (not an object)

---

### 2. Schema Valid Metric

**Location:** `src/model_evaluator.py:179`

Validates that generated constraints are compatible with the field type.

```python
class SchemaValidMetric:
    def measure(self, actual_output: str, field_type: str) -> float:
        # Returns 1.0 if constraints valid for type, 0.0 otherwise
```

**Examples:**
- `{"minimum": 18}` on `integer` → 1.0 (valid)
- `{"maxLength": 10}` on `integer` → 0.0 (invalid - maxLength is for strings)
- `{"pattern": "^[A-Z]+$"}` on `string` → 1.0 (valid)

---

### 3. Exact Match Metric

**Location:** `src/model_evaluator.py:207`

Checks if the model output exactly matches the expected output.

```python
class ExactMatchMetric:
    def measure(self, actual_output: str, expected_output: str) -> float:
        # Returns 1.0 if exact match, 0.0 otherwise
```

**Examples:**
- Actual: `{"minimum": 18}`, Expected: `{"minimum": 18}` → 1.0
- Actual: `{"minimum": 19}`, Expected: `{"minimum": 18}` → 0.0

---

### 4. Semantic Match Metric

**Location:** `src/model_evaluator.py:229`

Checks semantic equivalence, ignoring key ordering in JSON objects.

```python
class SemanticMatchMetric:
    def measure(self, actual_output: str, expected_output: str) -> float:
        # Returns 1.0 if semantically equivalent, 0.0 otherwise
```

**Examples:**
- Actual: `{"maximum": 100, "minimum": 0}`, Expected: `{"minimum": 0, "maximum": 100}` → 1.0
- Actual: `{"minimum": 0}`, Expected: `{"minimum": 0, "maximum": 100}` → 0.0

---

### 5. Latency Metric

**Location:** `src/model_evaluator.py:258`

Tracks inference latency for performance comparison.

```python
class LatencyMetric:
    def record(self, start: float, end: float) -> None

    @property
    def average_ms(self) -> float

    @property
    def p99_ms(self) -> float
```

**Typical Values:**
- T5 (local): ~50-100ms
- LLM (API): ~500-2000ms

---

## Operational Metrics

In addition to evaluation metrics, the `HybridInterpreter` tracks operational metrics:

**Location:** `src/hybrid_interpreter.py:90`

| Metric | Description |
|--------|-------------|
| `total_calls` | Total interpretation requests |
| `t5_successes` | Successful T5 interpretations |
| `t5_failures` | Failed T5 interpretations |
| `llm_fallbacks` | Times LLM was used as fallback |
| `t5_success_rate` | Percentage handled by T5 |
| `fallback_rate` | Percentage requiring LLM |

**Example Output:**
```
Interpreter stats:
  Total calls: 18
  T5 successes: 15 (83.3%)
  LLM fallbacks: 3 (16.7%)
```

---

## Evaluation Suite

The `EvaluationSuite` class runs all metrics on a set of test cases:

**Location:** `src/model_evaluator.py:291`

```python
from src.model_evaluator import EvaluationSuite, load_evaluation_dataset

# Load test cases from training data
test_cases = load_evaluation_dataset("training_data/financial_domain.json")

# Run evaluation
suite = EvaluationSuite(interpreter=my_model)
results = suite.run(test_cases)

# Access results
print(f"Exact Match Rate: {results.exact_match_rate:.1%}")
print(f"Average Latency: {results.average_latency_ms:.2f}ms")
```

---

## Aggregate Results

The `EvaluationResults` class provides aggregate statistics:

| Property | Description |
|----------|-------------|
| `total_tests` | Number of test cases |
| `json_correctness_rate` | % valid JSON outputs |
| `schema_valid_rate` | % type-compatible constraints |
| `exact_match_rate` | % exact matches |
| `semantic_match_rate` | % semantic matches |
| `average_latency_ms` | Mean inference time |

---

## Model Comparison

Compare T5 vs LLM performance:

```python
from src.model_evaluator import compare_models

results = compare_models(
    model_a=t5_interpreter,
    model_b=llm_interpreter,
    test_cases=test_cases,
    model_a_name="T5",
    model_b_name="LLM"
)

print(f"T5 Exact Match: {results['T5'].exact_match_rate:.1%}")
print(f"LLM Exact Match: {results['LLM'].exact_match_rate:.1%}")
```

---

## CLI Usage

Run evaluation from command line:

```bash
# Evaluate with mock interpreter (test pipeline)
python -m src.evaluate --dataset training_data/financial_domain.json -v

# Save report to file
python -m src.evaluate --dataset training_data/financial_domain.json --output report.json
```

---

## Test Coverage

All metrics are covered by unit tests in `tests/test_model_evaluator.py`:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestJsonCorrectnessMetric` | 4 | Valid, invalid, empty, non-object |
| `TestSchemaValidMetric` | 4 | Valid type, invalid type, string constraints |
| `TestExactMatchMetric` | 3 | Identical, different values, different keys |
| `TestSemanticMatchMetric` | 3 | Different order, different content, enums |
| `TestLatencyMetric` | 3 | Recording, average, p99 |
| `TestEvaluationSuite` | 2 | Run evaluation, aggregate scores |
| `TestModelComparison` | 1 | Compare two models |
| `TestConstraintTypeEvaluation` | 1 | Group by constraint type |
| `TestEvaluationReport` | 2 | Generate report, save to file |
| `TestFailureAnalysis` | 1 | Capture failures |

**Run tests:**
```bash
PYTHONPATH=. pytest tests/test_model_evaluator.py -v
```

---

## Feature Scenarios (BDD)

Evaluation scenarios in `features/schema_enrichment.feature`:

- Scenario: Create evaluation dataset from training data (line 296)
- Scenario: Evaluate JSON correctness metric (line 301)
- Scenario: Evaluate schema validity metric (line 307)
- Scenario: Evaluate exact match metric (line 313)
- Scenario: Evaluate semantic match metric (line 320)
- Scenario: Track latency metric (line 326)
- Scenario: Run evaluation suite on model (line 331)
- Scenario: Evaluate by constraint type (line 344)
- Scenario: Evaluate by domain (line 351)
- Scenario: Generate evaluation report (line 356)
- Scenario: CLI runs evaluation (line 363)

---

## Why These Metrics?

| Requirement | Our Implementation | Rationale |
|-------------|-------------------|-----------|
| Accuracy | Exact Match, Semantic Match | Measures correctness of generated constraints |
| Validity | JSON Correctness, Schema Valid | Ensures outputs are usable |
| Performance | Latency (avg, p99) | Critical for production (T5 ~50ms vs LLM ~500ms) |
| Reliability | Fallback Rate | Measures model self-sufficiency |
| Debugging | Failure Analysis | Identifies patterns for retraining |

**Note:** Traditional classification metrics (F1, ROC-AUC) are not directly applicable as this is a generation task, not classification. However, Exact Match Rate serves a similar purpose to accuracy in classification contexts.
