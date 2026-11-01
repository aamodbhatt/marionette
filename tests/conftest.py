"""Shared fixtures: a tiny random Qwen3 and a byte-level tokenizer with a chat template.

Nothing here touches the network or downloads weights; the whole suite runs on CPU.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from marionette.lm import LanguageModel

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{% endif %}{% endif %}"
)


def make_tokenizer() -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=SPECIALS[1:2],
    )
    fast.add_tokens(["<think>", "</think>"])
    fast.chat_template = CHAT_TEMPLATE
    return fast


def make_model(vocab_size: int, n_layers: int = 4, seed: int = 0) -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    return make_tokenizer()


@pytest.fixture(scope="session")
def tiny_lm(tokenizer: Any) -> LanguageModel:
    return LanguageModel(model=make_model(len(tokenizer)), tokenizer=tokenizer, name="tiny-qwen3")
