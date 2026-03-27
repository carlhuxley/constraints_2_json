#!/usr/bin/env python3
"""
Simple training script for T5 LoRA adapter (no HuggingFace Trainer).

Usage:
    CUDA_VISIBLE_DEVICES="" python -m src.train_adapter_simple --dataset training_data/financial_domain_combined.json
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

# Force CPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


class ConstraintDataset(Dataset):
    """Dataset for constraint generation training."""

    def __init__(self, examples: list, tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = f"convert: {example['input']}"
        target_text = json.dumps(example['output'], separators=(',', ':'))

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Replace padding token id with -100 for loss calculation
        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_dataset(path: str) -> list:
    """Load training examples from JSON file."""
    with open(path) as f:
        data = json.load(f)
    examples = [ex for ex in data["examples"] if "input" in ex and "output" in ex]
    return examples


def train(
    dataset_path: str,
    output_dir: str = "adapters/adapter_retrained",
    base_model: str = "t5-base",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    validation_split: float = 0.1,
    log_every: int = 25,
):
    """Train a LoRA adapter for constraint generation."""

    print("=" * 60)
    print("T5 LoRA Adapter Training (Simple)")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {base_model}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Device: cpu (forced)")
    print()

    # Load data
    print("Loading dataset...")
    examples = load_dataset(dataset_path)
    print(f"Loaded {len(examples)} examples")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(examples)

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

    # Create datasets and dataloaders
    train_dataset = ConstraintDataset(train_examples, tokenizer)
    val_dataset = ConstraintDataset(val_examples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Training loop
    print("Starting training...")
    print(f"Total steps: {total_steps}")
    print()

    model.train()
    start_time = datetime.now()
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                print(f"  Step {global_step}: avg_loss={avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        avg_train_loss = epoch_loss / epoch_steps

        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  New best model saved to {checkpoint_dir}")

        print()
        model.train()

    end_time = datetime.now()
    duration = end_time - start_time

    # Save final adapter
    final_output = os.path.join(output_dir, "final_adapter")
    os.makedirs(final_output, exist_ok=True)
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Duration: {duration}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final adapter saved to: {final_output}")
    print()
    print("To evaluate:")
    print(f"  python -c \"")
    print(f"from src.t5_interpreter import T5Interpreter")
    print(f"t5 = T5Interpreter('{final_output}')")
    print(f"print(t5.interpret('Age >= 18'))  # Should output: {{'minimum': 18}}")
    print(f"print(t5.interpret('Age <= 65'))  # Should output: {{'maximum': 65}}")
    print(f"\"")

    return final_output


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Train T5 LoRA adapter (simple)")
    parser.add_argument("--dataset", "-d", required=True, help="Training data JSON")
    parser.add_argument("--output", "-o", default="adapters/adapter_financial_v2", help="Output dir")
    parser.add_argument("--base-model", default="t5-base", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")

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
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
