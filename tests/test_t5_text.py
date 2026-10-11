"""Tests for t5_text: the text format shared by T5 training and inference.

These run against the real T5 tokenizer, because the behaviour under test is
"survives the T5 vocabulary", which a fake tokenizer cannot show.
"""
import glob
import json
import os

import pytest


TOKENIZER_PATH = "adapters/adapter_financial/final_adapter"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(TOKENIZER_PATH, "tokenizer_config.json")),
    reason="T5 tokenizer not available",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def through_t5(tokenizer, text: str) -> str:
    """What the model can actually see/produce for this text."""
    return tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens=True)


def all_examples():
    examples = []
    for path in sorted(glob.glob("training_data/*.json")):
        for e in json.load(open(path, encoding="utf-8"))["examples"]:
            if "input" in e:
                examples.append(e)
    return examples


class TestTargetsSurviveTokenizer:

    def test_every_dataset_answer_round_trips_exactly(self, tokenizer):
        from src.t5_text import format_target, decode_output

        lost = []
        for e in all_examples():
            expected = json.dumps(e["output"], separators=(",", ":"))
            seen = decode_output(through_t5(tokenizer, format_target(e["output"])))
            if seen != expected:
                lost.append((expected, seen))

        assert lost == []

    def test_regex_metacharacters_round_trip(self, tokenizer):
        from src.t5_text import format_target, decode_output

        constraints = {"pattern": "^\\d{3}-[A-Z]{2,4}<x>~$"}
        seen = decode_output(through_t5(tokenizer, format_target(constraints)))

        assert json.loads(seen) == constraints

    def test_no_encoded_answer_exceeds_training_max_length(self, tokenizer):
        from src.t5_text import format_target

        max_length = 128  # train_adapter*.py and T5Interpreter truncate here
        too_long = [
            e["output"] for e in all_examples()
            if len(tokenizer(format_target(e["output"])).input_ids) > max_length
        ]

        assert too_long == []


class TestInputsSurviveTokenizer:

    def test_at_most_rule_is_not_seen_as_equality(self, tokenizer):
        # Without encoding, "<" becomes <unk> and "Amount <= 5" reads as "Amount = 5".
        from src.t5_text import format_input

        at_most = through_t5(tokenizer, format_input("Amount <= 5"))
        equals = through_t5(tokenizer, format_input("Amount = 5"))

        assert at_most != equals

    def test_no_dataset_rule_loses_characters(self, tokenizer):
        from src.t5_text import format_input

        lossy = [
            e["input"] for e in all_examples()
            if tokenizer.unk_token_id in tokenizer(format_input(e["input"])).input_ids
        ]

        assert lossy == []

    def test_input_starts_with_shared_prompt_prefix(self):
        from src.t5_text import PROMPT_PREFIX, format_input

        assert format_input("Must be at least 18") == PROMPT_PREFIX + "Must be at least 18"


class TestTrainingExamplesUseSharedFormat:
    """Both training scripts must build examples exactly as inference reads them."""

    @pytest.fixture(params=["src.train_adapter", "src.train_adapter_simple"])
    def dataset_class(self, request, monkeypatch):
        import importlib
        # train_adapter_simple hides the GPU at import time; keep that out of the session.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        return importlib.import_module(request.param).ConstraintDataset

    def test_example_is_encoded_like_inference(self, tokenizer, dataset_class):
        from src.t5_text import format_input, format_target

        example = {"input": "Code is 3 digits, at most <= 999", "output": {"pattern": "^[0-9]{3}$"}}
        item = dataset_class([example], tokenizer)[0]

        def text(ids):
            ids = [i for i in ids.tolist() if i >= 0]  # drop -100 loss padding
            return tokenizer.decode(ids, skip_special_tokens=True)

        assert text(item["input_ids"]) == through_t5(tokenizer, format_input(example["input"]))
        assert text(item["labels"]) == through_t5(tokenizer, format_target(example["output"]))
