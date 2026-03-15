# Fine-Tuning Strategy: T5 with LoRA Adapters

## Overview

Replace LLM API calls with a fine-tuned T5 model for converting business rule text to JSON Schema constraints. Use LoRA adapters to support multiple domains.

## Why T5 Over LLM?

The task is well-suited for a smaller fine-tuned model because:

1. **Constrained output space** - Output is a small JSON object with known keys (`minimum`, `maximum`, `pattern`, `enum`, `maxLength`, etc.)
2. **Short inputs** - Business rules are typically 1-2 sentences
3. **Seq2seq is T5's strength** - Text-to-structured-output is exactly what T5 was designed for
4. **Patterns are repetitive** - "Must be X or older" → `{"minimum": X}` is learnable

## Comparison

| Aspect | LLM (GPT/Claude) | Fine-tuned T5 |
|--------|------------------|---------------|
| Latency | 500ms-2s | 10-50ms |
| Cost | $0.001-0.01/call | Free (local) |
| Accuracy | High (zero-shot) | High (if trained well) |
| Generalization | Excellent | Good for seen patterns |
| Setup | API key | Training pipeline |

## Training Data Sources

### 1. Production Data Collection

Log (business_rule, llm_output) pairs from the current system:

```python
# In llm_interpreter.py
logger.info(f"TRAINING_PAIR|{business_rule}|{json.dumps(result)}")
```

Harvest validated pairs over time until 500-1000 examples collected.

### 2. Synthetic Data Generation

Generate pairs programmatically. Patterns are predictable:

```
"Must be 18 or older" → {"minimum": 18}
"At least 5 characters" → {"minLength": 5}
"Cannot exceed 100" → {"maximum": 100}
"Valid values: A, B, C" → {"enum": ["A", "B", "C"]}
```

## Free Fine-Tuning Platforms

| Platform | GPU | Time Limit | Enough for T5? |
|----------|-----|------------|----------------|
| Google Colab Free | T4 (16GB) | ~12hr/day | Yes |
| Kaggle | T4/P100 | 30hr/week | Yes |
| Lightning.ai | T4 | 22hr/month | Yes |
| Local CPU | - | Unlimited | Slow but works |

T5-small (60M params) fine-tunes in ~30 mins on free Colab with 1000 examples.

## Why LoRA?

For T5-small/base, full fine-tuning is feasible. However, LoRA enables **domain-specific adapters**:

### Architecture

```
t5-base (850MB, loaded once)
├── adapter_financial.bin    (~10MB) - Banking/finance constraints
├── adapter_healthcare.bin   (~10MB) - HIPAA/PHI compliance
├── adapter_ecommerce.bin    (~10MB) - Product/inventory rules
└── adapter_hr.bin           (~10MB) - Employee/payroll rules
```

### Benefits

| Benefit | Description |
|---------|-------------|
| Memory efficient | Load base model once, swap 10MB adapters |
| Domain-specific accuracy | Each adapter learns domain vocabulary |
| Easy updates | Retrain one adapter without touching others |
| A/B testing | Try new adapter alongside production one |
| Client customization | Different adapter per client/tenant |

### Runtime Switching

```python
from peft import PeftModel

base_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Switch based on schema domain
if schema.get("x-domain") == "healthcare":
    model = PeftModel.from_pretrained(base_model, "adapters/healthcare")
elif schema.get("x-domain") == "financial":
    model = PeftModel.from_pretrained(base_model, "adapters/financial")
```

## Training Cost

Each adapter takes ~30 mins on free Google Colab with ~500 domain-specific examples.

## Implementation Plan

1. Generate synthetic datasets for each domain (~500 examples each)
2. Create Colab notebook for LoRA fine-tuning
3. Train domain-specific adapters
4. Integrate adapter loading into `llm_interpreter.py`
5. Add `--domain` CLI flag to select adapter

## Domain Examples

### Financial Domain
- "Balance must not be negative" → `{"minimum": 0}`
- "Interest rate between 0 and 30 percent" → `{"minimum": 0, "maximum": 30}`
- "Account number must be 10 digits" → `{"pattern": "^[0-9]{10}$"}`
- "Transaction type: DEBIT, CREDIT, TRANSFER" → `{"enum": ["DEBIT", "CREDIT", "TRANSFER"]}`

### Healthcare Domain
- "Patient age must be between 0 and 150" → `{"minimum": 0, "maximum": 150}`
- "MRN must be 8 alphanumeric characters" → `{"pattern": "^[A-Z0-9]{8}$"}`
- "Blood type: A+, A-, B+, B-, O+, O-, AB+, AB-" → `{"enum": ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]}`
- "Dosage cannot exceed 1000mg" → `{"maximum": 1000}`
