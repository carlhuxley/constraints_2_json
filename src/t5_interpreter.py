"""
T5 Interpreter - converts business rules to JSON Schema constraints using fine-tuned T5.

This module provides a fast, local alternative to LLM-based interpretation
using a T5 model fine-tuned with LoRA adapters for specific domains.
"""
import json
import os
from typing import Optional

# Lazy imports to avoid loading torch when not needed
_transformers_available = None


def _check_transformers():
    """Check if transformers is available."""
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers
            import peft
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


class T5Interpreter:
    """
    Interpreter that uses a fine-tuned T5 model with LoRA adapters
    to convert business rules to JSON Schema constraints.
    """

    @staticmethod
    def _detect_base_model(adapter_path: str) -> str:
        """
        Detect the base model from adapter_config.json.

        Args:
            adapter_path: Path to the adapter directory

        Returns:
            Base model name/path
        """
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path", "")
            if base_model:
                return base_model

        # Default fallback
        return "google/flan-t5-small"

    def __init__(
        self,
        adapter_path: str,
        base_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize T5 interpreter with a LoRA adapter.

        Args:
            adapter_path: Path to the LoRA adapter directory
            base_model: Base T5 model to use (auto-detected from adapter if not specified)
            device: Device to run inference on (default: auto-detect)
        """
        if not _check_transformers():
            raise ImportError(
                "transformers and peft are required for T5Interpreter. "
                "Install with: pip install transformers peft torch"
            )

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel

        self.adapter_path = adapter_path

        # Auto-detect base model from adapter config if not specified
        if base_model is None:
            base_model = self._detect_base_model(adapter_path)
        self.base_model_name = base_model

        # Auto-detect device
        if device is None:
            import torch
            device = "cpu"  # Default to CPU
            if torch.cuda.is_available():
                # Check if CUDA is actually usable (capability >= 7.0 for recent PyTorch)
                try:
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 7:
                        device = "cuda"
                except Exception:
                    pass
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        self.device = device

        # Load tokenizer from adapter (it may have custom config)
        tokenizer_path = adapter_path if os.path.exists(
            os.path.join(adapter_path, "tokenizer_config.json")
        ) else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load base model
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()

    def interpret(self, business_rule: str) -> dict:
        """
        Convert a business rule to JSON Schema constraints.

        Args:
            business_rule: Free-text business rule (e.g., "Must be at least 18")

        Returns:
            Dict of JSON Schema constraints (e.g., {"minimum": 18})
        """
        # Format input as during training
        input_text = f"convert: {business_rule}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).to(self.device)

        # Generate
        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=1,
                do_sample=False,
            )

        # Decode
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse to JSON
        return self._parse_output(raw_output)

    def _parse_output(self, raw_output: str) -> dict:
        """
        Parse model output to JSON dict.

        Handles outputs with or without outer braces.

        Args:
            raw_output: Raw string from model

        Returns:
            Parsed JSON dict, empty dict if parsing fails
        """
        raw_output = raw_output.strip()

        if not raw_output:
            return {}

        # Add braces if missing
        if not raw_output.startswith("{"):
            raw_output = "{" + raw_output + "}"

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to fix common issues
            # Sometimes model outputs unquoted keys
            try:
                import re
                # Add quotes around unquoted keys
                fixed = re.sub(r'(\w+):', r'"\1":', raw_output)
                return json.loads(fixed)
            except (json.JSONDecodeError, Exception):
                return {}

    def interpret_with_context(
        self,
        field_name: str,
        field_type: str,
        business_rule: str,
    ) -> dict:
        """
        Convert business rule with field context.

        This method matches the signature used by the LLM interpreter
        for compatibility.

        Args:
            field_name: Name of the field being constrained
            field_type: JSON Schema type (string, integer, etc.)
            business_rule: Free-text business rule

        Returns:
            Dict of JSON Schema constraints
        """
        # For now, we just use the business rule
        # Future: Could include field context in prompt
        return self.interpret(business_rule)


def load_adapter(
    adapter_name: str,
    adapters_dir: str = "adapters",
    base_model: Optional[str] = None,
) -> T5Interpreter:
    """
    Load a T5 interpreter with a named adapter.

    Args:
        adapter_name: Name of the adapter (e.g., "financial", "healthcare")
        adapters_dir: Directory containing adapter folders
        base_model: Base T5 model to use (auto-detected from adapter if not specified)

    Returns:
        Configured T5Interpreter instance
    """
    # Look for adapter in standard locations
    possible_paths = [
        os.path.join(adapters_dir, f"adapter_{adapter_name}", "final_adapter"),
        os.path.join(adapters_dir, f"adapter_{adapter_name}"),
        os.path.join(adapters_dir, adapter_name, "final_adapter"),
        os.path.join(adapters_dir, adapter_name),
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(
            os.path.join(path, "adapter_config.json")
        ):
            return T5Interpreter(adapter_path=path, base_model=base_model)

    raise FileNotFoundError(
        f"Adapter '{adapter_name}' not found. Searched: {possible_paths}"
    )
