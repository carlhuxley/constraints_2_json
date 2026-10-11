"""
T5 text format - the single definition of what T5 sees in training and inference.

The T5 SentencePiece vocabulary has no token for several characters this task needs
({ } ^ \\ < ~). The tokenizer maps them to <unk>, so the model can never read
"Amount <= 5" (it sees "Amount = 5") or write a pattern like "^[0-9]{3}$" (it can
only produce "[0-9]3$"). Each such character is therefore swapped for a placeholder
word the vocabulary does contain, and swapped back on the model's output.

Training scripts and T5Interpreter must both go through this module so the prompt
and encoding can never drift apart again.
"""
import json


PROMPT_PREFIX = "convert: "

# Placeholders are upper-case nonsense words: representable by T5, and not
# something a real business rule or JSON Schema constraint would contain.
_PLACEHOLDERS = {
    "{": "QLBRQ",
    "}": "QRBRQ",
    "^": "QCARQ",
    "\\": "QBSLQ",
    "<": "QLTQ",
    "~": "QTILQ",
}


def _encode(text: str) -> str:
    for char, placeholder in _PLACEHOLDERS.items():
        text = text.replace(char, placeholder)
    return text


def format_input(business_rule: str) -> str:
    """Model input text for a business rule."""
    return PROMPT_PREFIX + _encode(business_rule)


def format_target(constraints: dict) -> str:
    """Training target text for a constraint dict."""
    return _encode(json.dumps(constraints, separators=(",", ":")))


def decode_output(raw_output: str) -> str:
    """Restore the real characters in decoded model output."""
    for char, placeholder in _PLACEHOLDERS.items():
        raw_output = raw_output.replace(placeholder, char)
    return raw_output
