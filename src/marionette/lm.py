"""A thin, hookable view of a Hugging Face decoder-only chat model.

Everything marionette does to the model goes through forward hooks on decoder blocks:
reading the residual stream (probes, SAE), adding vectors to it (steering), or replacing it
(SAE splices for evaluation). One invariant matters for honesty of the demo: within a block's
hook, *reads happen before edits*, so the polygraph never sees a residual the truth serum has
already modified at that layer.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from marionette.data.schema import Message
from marionette.runtime import torch_dtype

ReadFn = Callable[[torch.Tensor], None]
EditFn = Callable[[torch.Tensor], torch.Tensor]


class _StopForward(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised inside a hook to skip the blocks after the last layer that is needed."""


def _first(output: Any) -> torch.Tensor:
    """Decoder blocks return a tensor in recent transformers and a tuple in older ones."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unexpected decoder block output type: {type(output)!r}")


def _replace_first(output: Any, new: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return new
    return (new, *output[1:])


@dataclass
class LanguageModel:
    """A causal chat LM plus the hook points marionette reads from and writes to."""

    model: Any
    tokenizer: Any
    name: str

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device  # type: ignore[no-any-return]

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.model.layers  # type: ignore[no-any-return]

    @property
    def end_of_turn_id(self) -> int:
        return int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))

    def check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} out of range for {self.n_layers} decoder blocks")

    # ---- chat formatting -------------------------------------------------------------------

    def render_prompt(self, messages: Sequence[Message]) -> str:
        """The chat-template text up to and including the assistant header (non-thinking mode)."""
        rendered = self.tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise TypeError("chat template did not return a string")
        return rendered

    def encode_dialog(self, messages: Sequence[Message], response: str) -> tuple[list[int], int]:
        """Token ids of prompt + response + end-of-turn, and the index where the response starts.

        Prompt and response are tokenised separately so the response span is exact; this is how
        the response was produced at generation time too (the prompt ends on a newline token).
        """
        prompt_ids = self.tokenizer(self.render_prompt(messages), add_special_tokens=False)[
            "input_ids"
        ]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        return [*prompt_ids, *response_ids, self.end_of_turn_id], len(prompt_ids)

    # ---- hooks ---------------------------------------------------------------------------

    @contextlib.contextmanager
    def hooks(
        self,
        reads: Mapping[int, ReadFn] | None = None,
        edits: Mapping[int, EditFn] | None = None,
        stop_after: int | None = None,
    ) -> Iterator[None]:
        """Install read and edit hooks on decoder blocks for the duration of the context.

        ``reads[l]`` sees block ``l``'s output *before* ``edits[l]`` transforms it. With
        ``stop_after``, the forward pass aborts after that block (callers catch nothing: use
        `run_until`).
        """
        reads = reads or {}
        edits = edits or {}
        layers = set(reads) | set(edits) | ({stop_after} if stop_after is not None else set())
        for layer in layers:
            self.check_layer(layer)
        handles = []
        for layer in sorted(layers):

            def hook(_m: nn.Module, _i: Any, output: Any, layer: int = layer) -> Any:
                hidden = _first(output)
                if layer in reads:
                    reads[layer](hidden)
                if layer in edits:
                    hidden = edits[layer](hidden)
                    output = _replace_first(output, hidden)
                if layer == stop_after:
                    raise _StopForward
                return output

            handles.append(self.blocks[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def run_until(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, layers: Sequence[int]
    ) -> dict[int, torch.Tensor]:
        """Residual stream after each requested block, running only the blocks up to the last."""
        captured: dict[int, torch.Tensor] = {}
        reads: dict[int, ReadFn] = {layer: _store(captured, layer) for layer in layers}
        with self.hooks(reads=reads, stop_after=max(layers)), contextlib.suppress(_StopForward):
            self.model.model(
                input_ids=input_ids.to(self.device),
                attention_mask=None if attention_mask is None else attention_mask.to(self.device),
                use_cache=False,
            )
        missing = set(layers) - set(captured)
        if missing:
            raise RuntimeError(f"hooked blocks never ran: {sorted(missing)}")
        return captured


def _store(sink: dict[int, torch.Tensor], layer: int) -> ReadFn:
    def read(hidden: torch.Tensor) -> None:
        sink[layer] = hidden

    return read


def additive_edit(vector: torch.Tensor) -> EditFn:
    """An edit that adds a fixed residual-space vector at every position."""

    def edit(hidden: torch.Tensor) -> torch.Tensor:
        return hidden + vector.to(device=hidden.device, dtype=hidden.dtype)

    return edit


def load_language_model(
    name: str,
    dtype: str,
    device: torch.device,
    revision: str | None = None,
    n_layers: int | None = None,
) -> LanguageModel:
    """Load weights straight onto ``device`` (never staged through host RAM).

    With ``n_layers``, only the first ``n_layers`` decoder blocks are instantiated, which is all
    that SAE training and activation capture at an early layer need.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    config = AutoConfig.from_pretrained(name, revision=revision)
    if n_layers is not None:
        if not 0 < n_layers <= config.num_hidden_layers:
            raise ValueError(f"n_layers={n_layers} out of range 1..{config.num_hidden_layers}")
        config.num_hidden_layers = n_layers
        if getattr(config, "layer_types", None):
            config.layer_types = config.layer_types[:n_layers]
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        config=config,
        dtype=torch_dtype(dtype),  # type: ignore[arg-type]
        device_map={"": str(device)},
    )
    model.eval()  # type: ignore[no-untyped-call]
    return LanguageModel(model=model, tokenizer=tokenizer, name=name)
