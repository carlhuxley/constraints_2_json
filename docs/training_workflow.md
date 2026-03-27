# T5 Adapter Training Workflow

This document describes the complete workflow for training and retraining T5 LoRA adapters for constraint generation.

## Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Collect Data   │ ──▶ │  Prepare Data    │ ──▶ │  Train Adapter  │
│  (Production)   │     │  (Merge/Augment) │     │  (GPU Required) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Deploy         │ ◀── │  Evaluate        │ ◀── │  Adapter Ready  │
│  (Replace Old)  │     │  (Run Evals)     │     │  (.bin files)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Step 1: Data Collection

### Automatic Collection (Production)

The `HybridInterpreter` automatically logs training examples when T5 fails and LLM succeeds:

```python
# In src/hybrid_interpreter.py
# When T5 fails but LLM provides a valid result, the pair is logged
```

Collected data is saved to `training_data_collected.json`:

```json
{
  "collected_at": "2026-03-16T21:34:38",
  "total_examples": 2,
  "examples": [
    {"input": "Must be positive", "output": {"minimum": 0}, "field_type": "integer"},
    {"input": "Cannot exceed 100", "output": {"maximum": 100}, "field_type": "integer"}
  ]
}
```

### Manual Collection (Gap Analysis)

Run evaluation to identify failures:

```bash
# Run evaluation
source venv/bin/activate
python -c "
from src.t5_interpreter import T5Interpreter
from src.model_evaluator import load_evaluation_dataset, EvaluationSuite, generate_report

interpreter = T5Interpreter('adapters/adapter_financial/final_adapter')
test_cases = load_evaluation_dataset('training_data/financial_domain.json')
suite = EvaluationSuite(interpreter=interpreter)
results = suite.run(test_cases)
report = generate_report(results)

# Print failures for analysis
for f in report['failures'][:20]:
    print(f\"Input: {f['input']}\")
    print(f\"Expected: {f['expected']}\")
    print(f\"Actual: {f['actual']}\")
    print()
"
```

Common failure patterns to look for:
- Confusing `>=` with `<=` (minimum vs maximum)
- Missing constraint types
- Over-generating constraints
- Natural language variations not understood

---

## Step 2: Prepare Training Data

### Data File Structure

Training data is stored in `training_data/` as JSON:

```json
{
  "domain": "financial",
  "description": "Business rule to JSON Schema constraint pairs",
  "examples": [
    {"input": "Age >= 18", "output": {"minimum": 18}},
    {"input": "Age <= 65", "output": {"maximum": 65}}
  ]
}
```

### Merging Data Files

When you have supplemental data, merge it with the original:

```bash
python -c "
import json

# Load original
with open('training_data/financial_domain.json') as f:
    orig = json.load(f)

# Load supplemental
with open('training_data/financial_domain_supplemental.json') as f:
    supp = json.load(f)

# Filter out comment entries
supp_examples = [ex for ex in supp['examples'] if 'input' in ex]

# Merge
orig['examples'].extend(supp_examples)

# Save combined
with open('training_data/financial_domain_combined.json', 'w') as f:
    json.dump(orig, f, indent=2)

print(f'Combined: {len(orig[\"examples\"])} examples')
"
```

### Adding Collected Production Data

```bash
python -c "
import json

# Load combined training data
with open('training_data/financial_domain_combined.json') as f:
    data = json.load(f)

# Load collected production data
with open('training_data_collected.json') as f:
    collected = json.load(f)

# Add collected examples
for ex in collected['examples']:
    data['examples'].append({
        'input': ex['input'],
        'output': ex['output']
    })

# Save updated file
with open('training_data/financial_domain_combined.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f'Added {len(collected[\"examples\"])} production examples')
print(f'Total: {len(data[\"examples\"])} examples')
"
```

### Data Quality Checklist

Before training, verify:

- [ ] All examples have `input` (string) and `output` (dict) fields
- [ ] Outputs are valid JSON Schema constraints
- [ ] Contrastive pairs exist (e.g., both `>=` and `<=` examples)
- [ ] Diverse values used (not just 0, 1, 100)
- [ ] Multiple phrasings per pattern

---

## Step 3: Train the Adapter

### Option A: Google Colab (Free GPU)

1. Upload training data to Colab or Google Drive

2. Use this notebook code:

```python
# Install dependencies
!pip install transformers peft torch accelerate

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Configuration
BASE_MODEL = "t5-base"
TRAINING_DATA = "financial_domain_combined.json"  # Upload this file
OUTPUT_DIR = "adapter_financial_v2"
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Load data
with open(TRAINING_DATA) as f:
    data = json.load(f)
examples = [ex for ex in data["examples"] if "input" in ex]
print(f"Loaded {len(examples)} examples")

# Dataset class
class ConstraintDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_text = f"convert: {ex['input']}"
        target_text = json.dumps(ex['output'], separators=(',', ':'))

        inputs = self.tokenizer(input_text, max_length=self.max_length,
                                truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length,
                                 truncation=True, padding="max_length", return_tensors="pt")

        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Create dataloader
dataset = ConstraintDataset(examples, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1} avg loss: {total_loss / len(dataloader):.4f}")

# Save adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# Test inference
model.eval()
test_input = "Age >= 18"
inputs = tokenizer(f"convert: {test_input}", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Test: '{test_input}' -> {result}")
```

3. Download the adapter folder and place in `adapters/adapter_financial_v2/final_adapter/`

### Option B: Local GPU (CUDA 7.0+)

```bash
# Ensure compatible GPU
nvidia-smi

# Run training
source venv/bin/activate
python -m src.train_adapter_simple \
    --dataset training_data/financial_domain_combined.json \
    --output adapters/adapter_financial_v2 \
    --epochs 3 \
    --batch-size 8
```

### Option C: Cloud GPU (Kaggle, Lightning.ai)

Same as Colab, just upload the data and notebook.

### Training Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `base_model` | t5-base | Use t5-small for faster training |
| `epochs` | 3 | 3-5 usually sufficient |
| `batch_size` | 8 | Reduce if OOM errors |
| `learning_rate` | 1e-4 | Standard for LoRA |
| `lora_r` | 16 | Rank of LoRA matrices |
| `lora_alpha` | 32 | Usually 2x lora_r |
| `lora_dropout` | 0.1 | Regularization |

---

## Step 4: Evaluate the Adapter

### Quick Sanity Check

```bash
source venv/bin/activate
python -c "
from src.t5_interpreter import T5Interpreter

# Load new adapter
t5 = T5Interpreter('adapters/adapter_financial_v2/final_adapter')

# Test the patterns that were failing
tests = [
    ('Age >= 18', {'minimum': 18}),
    ('Age <= 65', {'maximum': 65}),
    ('Score >= 10', {'minimum': 10}),
    ('Score <= 10', {'maximum': 10}),
    ('Value at most 100', {'maximum': 100}),
    ('Minimum deposit is \$100', {'minimum': 100}),
]

print('Sanity Check Results:')
print('=' * 50)
for input_text, expected in tests:
    actual = t5.interpret(input_text)
    status = '✓' if actual == expected else '✗'
    print(f'{status} \"{input_text}\"')
    print(f'    Expected: {expected}')
    print(f'    Actual:   {actual}')
    print()
"
```

### Full Evaluation

```bash
source venv/bin/activate
python -c "
from src.t5_interpreter import T5Interpreter
from src.model_evaluator import load_evaluation_dataset, EvaluationSuite, generate_report

# Load new adapter
interpreter = T5Interpreter('adapters/adapter_financial_v2/final_adapter')

# Run on combined dataset
test_cases = load_evaluation_dataset('training_data/financial_domain_combined.json')
suite = EvaluationSuite(interpreter=interpreter)
results = suite.run(test_cases)
report = generate_report(results)

print('=' * 60)
print('EVALUATION RESULTS - New Adapter')
print('=' * 60)
print(f'Total Tests:         {report[\"total_tests\"]}')
print(f'JSON Correctness:    {report[\"json_correctness_rate\"]:.1%}')
print(f'Schema Valid:        {report[\"schema_valid_rate\"]:.1%}')
print(f'Exact Match:         {report[\"exact_match_rate\"]:.1%}')
print(f'Semantic Match:      {report[\"semantic_match_rate\"]:.1%}')
print(f'Avg Latency:         {report[\"average_latency_ms\"]:.2f}ms')
print('=' * 60)
"
```

### Compare Old vs New

```bash
source venv/bin/activate
python -c "
from src.t5_interpreter import T5Interpreter
from src.model_evaluator import load_evaluation_dataset, EvaluationSuite, generate_report

test_cases = load_evaluation_dataset('training_data/financial_domain_combined.json')

# Old adapter
old_t5 = T5Interpreter('adapters/adapter_financial/final_adapter')
old_suite = EvaluationSuite(interpreter=old_t5)
old_results = old_suite.run(test_cases)
old_report = generate_report(old_results)

# New adapter
new_t5 = T5Interpreter('adapters/adapter_financial_v2/final_adapter')
new_suite = EvaluationSuite(interpreter=new_t5)
new_results = new_suite.run(test_cases)
new_report = generate_report(new_results)

print('Comparison: Old vs New Adapter')
print('=' * 60)
print(f'Metric              Old         New         Delta')
print('-' * 60)
print(f'Exact Match:        {old_report[\"exact_match_rate\"]:6.1%}      {new_report[\"exact_match_rate\"]:6.1%}      {new_report[\"exact_match_rate\"] - old_report[\"exact_match_rate\"]:+6.1%}')
print(f'Semantic Match:     {old_report[\"semantic_match_rate\"]:6.1%}      {new_report[\"semantic_match_rate\"]:6.1%}      {new_report[\"semantic_match_rate\"] - old_report[\"semantic_match_rate\"]:+6.1%}')
print(f'Schema Valid:       {old_report[\"schema_valid_rate\"]:6.1%}      {new_report[\"schema_valid_rate\"]:6.1%}      {new_report[\"schema_valid_rate\"] - old_report[\"schema_valid_rate\"]:+6.1%}')
print('=' * 60)
"
```

---

## Step 5: Deploy the New Adapter

### Replace Old Adapter

```bash
# Backup old adapter
mv adapters/adapter_financial/final_adapter adapters/adapter_financial/final_adapter_backup

# Deploy new adapter
cp -r adapters/adapter_financial_v2/final_adapter adapters/adapter_financial/final_adapter
```

### Update Adapter Path (if using different location)

In your code, update the adapter path:

```python
# Before
interpreter = T5Interpreter('adapters/adapter_financial/final_adapter')

# After (if using versioned adapters)
interpreter = T5Interpreter('adapters/adapter_financial_v2/final_adapter')
```

### Verify Deployment

```bash
# Run the enrichment on a test schema
source venv/bin/activate
python -m src.main enrich examples/financial_schema.json --output test_output.json

# Check for any T5 failures
cat t5_failures.json
```

---

## Appendix: Current Training Data Files

| File | Examples | Purpose |
|------|----------|---------|
| `financial_domain.json` | 509 | Original training data |
| `financial_domain_supplemental.json` | 278 | Gap-filling data (>= vs <= pairs, "at most" patterns) |
| `financial_domain_combined.json` | 787 | Merged file for retraining |
| `healthcare_domain.json` | - | Healthcare domain (separate adapter) |
| `training_data_collected.json` | varies | Auto-collected from production |

---

## Appendix: Troubleshooting

### "CUDA capability X.X < 7.0"

Your GPU is too old for the installed PyTorch. Options:
1. Use Google Colab (free T4 GPU)
2. Install older PyTorch version compatible with your GPU
3. Train on CPU (very slow, not recommended)

### Out of Memory (OOM)

Reduce batch size:
```bash
python -m src.train_adapter_simple --batch-size 2
```

Or use gradient accumulation (modify training script).

### Poor Accuracy After Training

1. Check training data quality
2. Increase epochs (try 5-10)
3. Add more contrastive examples
4. Check for data leakage between train/validation splits

### Model Outputs Empty `{}`

The model hasn't learned the pattern. Add more examples of that specific pattern type.
