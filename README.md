# Constraints to JSON Schema

Convert natural language business rules into JSON Schema constraints using a hybrid T5 + LLM approach.

## What It Does

Takes a JSON Schema and a data dictionary with business rules:

```
Schema field: "age" (integer)
Business rule: "Must be at least 18 years old"
         ↓
JSON Schema constraint: {"minimum": 18}
```

The tool enriches your schema with proper JSON Schema constraints (`minimum`, `maximum`, `pattern`, `enum`, `format`, etc.) based on human-readable business rules.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key (for LLM fallback)
export OPENROUTER_API_KEY="your-key"

# Enrich a schema
python -m src.main \
    --schema examples/financial_schema.json \
    --dict examples/financial_dictionary.csv \
    --output enriched.json \
    --adapter-name financial \
    --verbose
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  JSON Schema    │     │ Data Dictionary │     │   Enriched      │
│  (input.json)   │ ──▶ │  (rules.csv)    │ ──▶ │   Schema        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Hybrid Interpreter │
                    │  ┌───────┐ ┌──────┐ │
                    │  │  T5   │→│ LLM  │ │
                    │  │ 50ms  │ │500ms │ │
                    │  └───────┘ └──────┘ │
                    └─────────────────────┘
```

1. **T5 Model** (local, fast) - Fine-tuned model handles common patterns
2. **LLM Fallback** (API, slower) - Handles complex/novel rules when T5 fails
3. **Validation** - Ensures generated constraints are valid for field types

## Supported Constraints

| Type | Example Rule | Generated Constraint |
|------|--------------|---------------------|
| Minimum | "Must be at least 18" | `{"minimum": 18}` |
| Maximum | "Cannot exceed 100" | `{"maximum": 100}` |
| Range | "Between 0 and 100" | `{"minimum": 0, "maximum": 100}` |
| Pattern | "Must be 10 digits" | `{"pattern": "^[0-9]{10}$"}` |
| Enum | "Status: ACTIVE, INACTIVE" | `{"enum": ["ACTIVE", "INACTIVE"]}` |
| Format | "Valid email required" | `{"format": "email"}` |
| Length | "Max 100 characters" | `{"maxLength": 100}` |

## CLI Options

```bash
python -m src.main --help

Options:
  --schema, -s      Input JSON Schema file (required)
  --dict, -d        Data dictionary CSV file (required)
  --output, -o      Output file (default: stdout)
  --adapter-name    Use named adapter: financial, healthcare
  --adapter         Path to custom LoRA adapter
  --llm             LLM provider: openrouter, anthropic, openai, none
  --model           LLM model (default: deepseek/deepseek-chat)
  --verbose, -v     Show detailed output
  --collect-training-data   Log successful interpretations for retraining
```

## Project Structure

```
constraints_2_json/
├── src/
│   ├── main.py              # CLI entry point
│   ├── enricher.py          # Schema enrichment logic
│   ├── hybrid_interpreter.py # T5 + LLM hybrid
│   ├── t5_interpreter.py    # T5 LoRA inference
│   ├── llm_interpreter.py   # LLM API client
│   ├── schema_validator.py  # Constraint validation
│   ├── evaluate.py          # Model evaluation CLI
│   └── train_adapter.py     # Adapter training scripts
├── adapters/
│   └── adapter_financial/   # Pre-trained financial domain adapter
├── training_data/
│   ├── financial_domain.json
│   └── healthcare_domain.json
├── examples/
│   ├── financial_schema.json
│   └── financial_dictionary.csv
├── docs/                    # Documentation
└── tests/                   # Test suite
```

## Documentation

| Document | Description |
|----------|-------------|
| [Deployment Guide](docs/deployment.md) | Installation, configuration, Docker setup |
| [Training Workflow](docs/training_workflow.md) | How to train/retrain T5 adapters |
| [Evaluation Metrics](docs/evaluation_metrics.md) | Model evaluation and metrics |
| [Fine-Tuning Strategy](docs/fine_tuning_strategy.md) | Why T5 + LoRA, architecture decisions |

## Examples

### Financial Domain

```bash
python -m src.main \
    -s examples/financial_schema.json \
    -d examples/financial_dictionary.csv \
    -o enriched_financial.json \
    --adapter-name financial \
    -v
```

### Healthcare Domain

```bash
python -m src.main \
    -s examples/healthcare_schema.json \
    -d examples/healthcare_dictionary.csv \
    -o enriched_healthcare.json \
    --adapter-name healthcare \
    -v
```

### LLM Only (No Local Model)

```bash
python -m src.main \
    -s examples/financial_schema.json \
    -d examples/financial_dictionary.csv \
    --llm openrouter \
    --model deepseek/deepseek-chat
```

## Evaluation

Run model evaluation on test data:

```bash
# Quick evaluation with mock interpreter
python -m src.evaluate --dataset training_data/financial_domain.json -v

# Evaluate T5 adapter
python -c "
from src.t5_interpreter import T5Interpreter
from src.model_evaluator import load_evaluation_dataset, EvaluationSuite, generate_report

t5 = T5Interpreter('adapters/adapter_financial/final_adapter')
test_cases = load_evaluation_dataset('training_data/financial_domain.json')
suite = EvaluationSuite(interpreter=t5)
results = suite.run(test_cases)
report = generate_report(results)

print(f'Exact Match: {report[\"exact_match_rate\"]:.1%}')
print(f'Schema Valid: {report[\"schema_valid_rate\"]:.1%}')
"
```

## Training

See [Training Workflow](docs/training_workflow.md) for complete instructions.

```bash
# Merge training data
python -c "
import json
with open('training_data/financial_domain.json') as f: orig = json.load(f)
with open('training_data/financial_domain_supplemental.json') as f: supp = json.load(f)
orig['examples'].extend([ex for ex in supp['examples'] if 'input' in ex])
with open('training_data/financial_domain_combined.json', 'w') as f:
    json.dump(orig, f, indent=2)
"

# Train adapter (requires GPU - use Google Colab if needed)
python -m src.train_adapter_simple \
    --dataset training_data/financial_domain_combined.json \
    --output adapters/adapter_financial_v2 \
    --epochs 3
```

## Configuration

### Environment Variables

```bash
# Required for LLM fallback
OPENROUTER_API_KEY=your-api-key

# Optional
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
```

### Data Dictionary Format

CSV with columns:
```csv
field_name,field_type,description,business_rule
age,integer,Customer age,Must be at least 18 years old
email,string,Contact email,Must be a valid email address
status,string,Account status,"Status: ACTIVE, INACTIVE, CLOSED"
```

## Requirements

- Python 3.10+
- PyTorch (for T5 inference)
- transformers, peft (for LoRA)
- See `requirements.txt` for full list

## License

MIT
