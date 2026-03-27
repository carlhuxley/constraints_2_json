#!/usr/bin/env python3
"""
Train a LoRA adapter for T5 constraint generation.

Usage:
    python -m src.train_adapter --dataset training_data/financial_domain_combined.json
    python -m src.train_adapter --dataset training_data/financial_domain_combined.json --epochs 5
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


class ConstraintDataset(Dataset):
    """Dataset for constraint generation training."""

    def __init__(self, examples: list, tokenizer, max_input_length: int = 128, max_target_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format input as during inference
        input_text = f"convert: {example['input']}"
        target_text = json.dumps(example['output'], separators=(',', ':'))

        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }


def load_dataset(path: str) -> list:
    """Load training examples from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Filter out comment entries
    examples = [ex for ex in data["examples"] if "input" in ex and "output" in ex]
    return examples


def train(
    dataset_path: str,
    output_dir: str = "adapters/adapter_retrained",
    base_model: str = "t5-base",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    validation_split: float = 0.1,
    device: Optional[str] = None,
):
    """
    Train a LoRA adapter for constraint generation.

    Args:
        dataset_path: Path to training data JSON
        output_dir: Where to save the trained adapter
        base_model: Base T5 model to use
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        validation_split: Fraction of data to use for validation
        device: Device to train on (auto-detected if not specified)
    """
    print(f"=" * 60)
    print("T5 LoRA Adapter Training")
    print(f"=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {base_model}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print()

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            try:
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 7:
                    device = "cuda"
                else:
                    print(f"CUDA capability {capability} < 7.0, using CPU")
                    device = "cpu"
            except Exception:
                device = "cpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Device: {device}")
    print()

    # Load data
    print("Loading dataset...")
    examples = load_dataset(dataset_path)
    print(f"Loaded {len(examples)} examples")

    # Split into train/val
    split_idx = int(len(examples) * (1 - validation_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    print(f"Train: {len(train_examples)}, Validation: {len(val_examples)}")
    print()

    # Load tokenizer and model
    print(f"Loading {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    # Configure LoRA
    print("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    # Create datasets
    train_dataset = ConstraintDataset(train_examples, tokenizer)
    val_dataset = ConstraintDataset(val_examples, tokenizer)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=False,  # Disable for CPU compatibility
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        use_cpu=True if device == "cpu" else False,  # Force CPU if needed
        no_cuda=True if device == "cpu" else False,  # Disable CUDA
    )

    # Create trainer
    # Note: newer transformers versions use processing_class instead of tokenizer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    print(f"This may take a while on CPU...")
    print()

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    duration = end_time - start_time
    print()
    print(f"Training completed in {duration}")

    # Save the adapter
    final_output = os.path.join(output_dir, "final_adapter")
    print(f"Saving adapter to {final_output}...")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    print()
    print(f"=" * 60)
    print("Training complete!")
    print(f"=" * 60)
    print(f"Adapter saved to: {final_output}")
    print()
    print("To use this adapter:")
    print(f"  python -m src.evaluate --dataset {dataset_path} --model t5 --adapter-path {final_output}")

    return final_output


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for T5 constraint generation"
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="adapters/adapter_retrained",
        help="Output directory for trained adapter"
    )
    parser.add_argument(
        "--base-model",
        default="t5-base",
        help="Base T5 model to use (default: t5-base)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to train on (default: auto-detect)"
    )

    args = parser.parse_args(argv)

    try:
        train(
            dataset_path=args.dataset,
            output_dir=args.output,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            device=args.device,
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
