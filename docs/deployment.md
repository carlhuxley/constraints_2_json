# Deployment Notes

This document provides instructions for deploying and running the JSON Schema Constraint Enrichment tool.

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running Locally](#running-locally)
7. [Running in a Sandbox](#running-in-a-sandbox)
8. [Docker Deployment](#docker-deployment)
9. [Runbook for Stakeholders](#runbook-for-stakeholders)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Constraint Enrichment Tool converts business rules from data dictionaries into JSON Schema constraints. It supports two inference modes:

| Mode | Description | Latency | Cost |
|------|-------------|---------|------|
| **T5 (Local)** | Fine-tuned T5 model with LoRA adapters | ~50-100ms | Free |
| **LLM (API)** | OpenRouter API (DeepSeek, GPT-4, etc.) | ~500-2000ms | Per-token |
| **Hybrid** | T5 first, LLM fallback on failure | Variable | Reduced |

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| RAM | 4GB (8GB recommended for T5) |
| Disk | 2GB (including model weights) |
| OS | Linux, macOS, Windows (WSL2) |

### For T5 Local Inference

| Component | Requirement |
|-----------|-------------|
| RAM | 8GB minimum |
| GPU | Optional (CUDA 7.0+ capability) |
| Disk | 500MB per adapter |

---

## Dependencies

### Core Dependencies

```
jsonschema>=4.0.0      # JSON Schema validation
requests>=2.28.0       # HTTP client for LLM API
python-dotenv>=1.0.0   # Environment variable loading
```

### T5 Local Inference (Optional)

```
transformers>=4.36.0   # Hugging Face Transformers
peft>=0.7.0            # Parameter-Efficient Fine-Tuning
torch>=2.0.0           # PyTorch
```

### Development/Testing

```
pytest>=8.0.0          # Test framework
deepeval>=1.0.0        # Evaluation metrics (optional)
```

### Full requirements.txt

```
jsonschema>=4.0.0
requests>=2.28.0
python-dotenv>=1.0.0
pytest>=8.0.0
deepeval>=1.0.0
transformers>=4.36.0
peft>=0.7.0
torch>=2.0.0
```

---

## Installation

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/carlhuxley/constraints_2_json.git
cd constraints_2_json

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
PYTHONPATH=. pytest tests/ -v
```

### Option 2: Minimal Installation (LLM-only)

```bash
# Install only core dependencies (no T5)
pip install jsonschema requests python-dotenv

# Set API key
export OPENROUTER_API_KEY="your-api-key"
```

### Option 3: Full Installation with T5

```bash
# Install all dependencies
pip install -r requirements.txt

# Download/extract T5 adapter
# Place adapter files in: adapters/adapter_financial/final_adapter/
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for LLM mode
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx

# Optional: Hugging Face token for faster model downloads
HF_TOKEN=hf_xxxxxxxxxxxx

# Optional: Force CPU mode for T5
CUDA_VISIBLE_DEVICES=""
```

### T5 Adapter Setup

Place trained LoRA adapters in the `adapters/` directory:

```
adapters/
├── adapter_financial/
│   └── final_adapter/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── adapter_healthcare/
    └── final_adapter/
        └── ...
```

---

## Running Locally

### Basic Usage (LLM Mode)

```bash
# Enrich schema using LLM
PYTHONPATH=. python -m src.main \
  --schema examples/financial_schema.json \
  --dict examples/financial_dictionary.csv \
  --output enriched_schema.json \
  -v
```

### T5 Mode with LLM Fallback

```bash
# Use T5 adapter with LLM fallback
PYTHONPATH=. python -m src.main \
  --schema examples/financial_schema.json \
  --dict examples/financial_dictionary.csv \
  --adapter-name financial \
  --output enriched_schema.json \
  -v
```

### Direct Adapter Path

```bash
# Specify adapter path directly
PYTHONPATH=. python -m src.main \
  --schema input.json \
  --dict dictionary.csv \
  --adapter /path/to/adapter \
  --output output.json
```

### Collect Training Data

```bash
# Collect successful interpretations for retraining
PYTHONPATH=. python -m src.main \
  --schema input.json \
  --dict dictionary.csv \
  --adapter-name financial \
  --collect-training-data \
  --training-log collected_data.json \
  --output output.json
```

### CLI Options Reference

| Flag | Description |
|------|-------------|
| `--schema, -s` | Path to input JSON Schema file (required) |
| `--dict, -d` | Path to data dictionary CSV (required) |
| `--output, -o` | Output file path (default: stdout) |
| `--adapter` | Path to T5 LoRA adapter directory |
| `--adapter-name` | Named adapter: `financial` or `healthcare` |
| `--llm` | LLM provider: `openrouter`, `none` |
| `--model, -m` | LLM model (default: `deepseek/deepseek-chat`) |
| `--no-validate` | Skip schema validation |
| `--collect-training-data` | Log successful interpretations |
| `--training-log` | Path for collected training data |
| `--verbose, -v` | Enable verbose output |

---

## Running in a Sandbox

### Google Colab

```python
# Install dependencies
!pip install jsonschema requests python-dotenv transformers peft torch

# Clone repository
!git clone https://github.com/carlhuxley/constraints_2_json.git
%cd constraints_2_json

# Set API key
import os
os.environ["OPENROUTER_API_KEY"] = "your-key"

# Run enrichment
!PYTHONPATH=. python -m src.main \
  --schema examples/financial_schema.json \
  --dict examples/financial_dictionary.csv \
  -v
```

### Jupyter Notebook

```python
import sys
sys.path.insert(0, '/path/to/constraints_2_json')

from src.enricher import enrich_schema
from src.llm_interpreter import OpenRouterClient

client = OpenRouterClient("deepseek/deepseek-chat")
result = enrich_schema(
    schema_path="examples/financial_schema.json",
    dict_path="examples/financial_dictionary.csv",
    llm_client=client
)
print(result)
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY adapters/ adapters/

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.main"]
```

### Build and Run

```bash
# Build image
docker build -t constraint-enricher .

# Run with LLM
docker run -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  -v $(pwd)/data:/data \
  constraint-enricher \
  --schema /data/schema.json \
  --dict /data/dictionary.csv \
  --output /data/enriched.json

# Run with T5 adapter
docker run -v $(pwd)/data:/data \
  -v $(pwd)/adapters:/app/adapters \
  constraint-enricher \
  --schema /data/schema.json \
  --dict /data/dictionary.csv \
  --adapter-name financial \
  --output /data/enriched.json
```

---

## Runbook for Stakeholders

### Quick Start (5 minutes)

1. **Install Python 3.10+** from [python.org](https://www.python.org/downloads/)

2. **Clone and setup:**
   ```bash
   git clone https://github.com/carlhuxley/constraints_2_json.git
   cd constraints_2_json
   pip install jsonschema requests python-dotenv
   ```

3. **Get API key** from [OpenRouter](https://openrouter.ai/keys)

4. **Run:**
   ```bash
   export OPENROUTER_API_KEY="your-key"
   PYTHONPATH=. python -m src.main \
     --schema examples/financial_schema.json \
     --dict examples/financial_dictionary.csv \
     -v
   ```

### Common Operations

| Task | Command |
|------|---------|
| Enrich a schema | `python -m src.main -s schema.json -d dict.csv -o output.json` |
| Run tests | `pytest tests/ -v` |
| Evaluate model | `python -m src.evaluate --dataset training_data/financial_domain.json` |
| View help | `python -m src.main --help` |

### Input File Formats

**JSON Schema** (input):
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "age": { "type": "integer" },
    "email": { "type": "string" }
  }
}
```

**Data Dictionary CSV** (input):
```csv
Column Name,Data Type,Length,Nullable,Valid_Values,Business_Rule,Description
age,NUMBER,3,N,,Must be at least 18 years old,Customer age
email,VARCHAR,100,Y,,Must be a valid email address,Contact email
```

**Enriched Schema** (output):
```json
{
  "properties": {
    "age": {
      "type": "integer",
      "minimum": 18,
      "x-business-rule": "Must be at least 18 years old"
    },
    "email": {
      "type": "string",
      "format": "email",
      "maxLength": 100
    }
  }
}
```

### Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| T5 Success Rate | >80% | Check interpreter stats in verbose output |
| Schema Valid | 100% | No validation errors on output |
| Latency (T5) | <100ms | Check interpreter stats |
| Test Coverage | 100% pass | Run `pytest tests/ -v` |

### Escalation Path

| Issue | Action |
|-------|--------|
| API key invalid | Regenerate at [OpenRouter](https://openrouter.ai/keys) |
| T5 adapter not loading | Check `adapters/` directory structure |
| Schema validation fails | Run with `--no-validate` and inspect output |
| Tests failing | Check `t5_failures.json` for model issues |

---

## Troubleshooting

### Common Issues

**1. "OPENROUTER_API_KEY not set"**
```bash
export OPENROUTER_API_KEY="your-key"
# or add to .env file
```

**2. "transformers and peft are required"**
```bash
pip install transformers peft torch
```

**3. "CUDA error: no kernel image available"**
```bash
# GPU not compatible - force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

**4. "Adapter not found"**
```bash
# Check adapter directory structure
ls -la adapters/adapter_financial/final_adapter/
# Should contain: adapter_config.json, adapter_model.safetensors
```

**5. "Size mismatch in loading state_dict"**
```bash
# Adapter trained on different base model
# Check adapter_config.json for base_model_name_or_path
cat adapters/adapter_financial/final_adapter/adapter_config.json | grep base_model
```

### Getting Help

- **Issues:** https://github.com/carlhuxley/constraints_2_json/issues
- **Documentation:** `docs/` directory
- **Tests:** `pytest tests/ -v` to verify installation

---

## Project Structure

```
constraints_2_json/
├── src/
│   ├── main.py              # CLI entry point
│   ├── enricher.py          # Schema enrichment orchestration
│   ├── dict_lookup.py       # Data dictionary parsing
│   ├── llm_interpreter.py   # LLM-based rule interpretation
│   ├── t5_interpreter.py    # T5-based rule interpretation
│   ├── hybrid_interpreter.py # T5 + LLM fallback
│   ├── schema_navigator.py  # Schema tree walking
│   ├── node_updater.py      # Schema node updates
│   ├── schema_validator.py  # JSON Schema validation
│   ├── model_evaluator.py   # Evaluation metrics
│   └── evaluate.py          # Evaluation CLI
├── adapters/                # T5 LoRA adapters
├── training_data/           # Training datasets
├── examples/                # Sample schemas and dictionaries
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── features/                # BDD feature specifications
└── requirements.txt         # Python dependencies
```
