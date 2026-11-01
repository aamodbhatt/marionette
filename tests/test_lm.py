"""Hooks, pooling capture and steering on a tiny randomly initialised Qwen3."""

from __future__ import annotations

import numpy as np
import torch

from marionette.capture import capture, load_activations, pooling_reads, response_mask
from marionette.data.schema import Example, Message
from marionette.lm import LanguageModel, additive_edit
from marionette.steering import ProbeBundle, truth_serum


def ids_for(lm: LanguageModel, text: str = "hello world, tiny model") -> torch.Tensor:
    return torch.tensor([lm.tokenizer(text, add_special_tokens=False)["input_ids"]])


def test_run_until_matches_hidden_states(tiny_lm: LanguageModel) -> None:
    ids = ids_for(tiny_lm)
    got = tiny_lm.run_until(ids, None, [0, 2])
    with torch.inference_mode():
        ref = tiny_lm.model.model(input_ids=ids, output_hidden_states=True).hidden_states
    # hidden_states[i + 1] is the output of block i (the last entry is post-norm, so skip it).
    assert torch.allclose(got[0], ref[1], atol=1e-5)
    assert torch.allclose(got[2], ref[3], atol=1e-5)


def test_reads_see_the_residual_before_edits_at_the_same_layer(tiny_lm: LanguageModel) -> None:
    ids = ids_for(tiny_lm)
    clean = tiny_lm.run_until(ids, None, [1])[1]
    seen: list[torch.Tensor] = []
    vec = torch.ones(tiny_lm.d_model) * 5
    with (
        torch.inference_mode(),
        tiny_lm.hooks(reads={1: lambda h: seen.append(h.clone())}, edits={1: additive_edit(vec)}),
    ):
        tiny_lm.model(input_ids=ids)
    assert torch.allclose(seen[0], clean, atol=1e-6)


def test_edits_change_downstream_logits_and_zero_edit_is_identity(tiny_lm: LanguageModel) -> None:
    ids = ids_for(tiny_lm)
    with torch.inference_mode():
        base = tiny_lm.model(input_ids=ids).logits
        with tiny_lm.hooks(edits={1: additive_edit(torch.zeros(tiny_lm.d_model))}):
            same = tiny_lm.model(input_ids=ids).logits
        with tiny_lm.hooks(edits={1: additive_edit(torch.ones(tiny_lm.d_model) * 3)}):
            moved = tiny_lm.model(input_ids=ids).logits
    assert torch.allclose(base, same)
    assert not torch.allclose(base, moved)
    # Hooks are removed on exit.
    with torch.inference_mode():
        assert torch.allclose(tiny_lm.model(input_ids=ids).logits, base)


def test_truth_serum_subtracts_alpha_times_the_direction(tiny_lm: LanguageModel) -> None:
    d = tiny_lm.d_model
    vec = torch.arange(d, dtype=torch.float32)
    bundle = ProbeBundle(
        model="t",
        layer=1,
        w=torch.zeros(d),
        b=0.0,
        platt_a=1.0,
        platt_c=0.0,
        threshold_logit=0.0,
        steer={1: vec, 2: vec},
    )
    assert truth_serum(bundle, 0.0) == {}
    edits = truth_serum(bundle, 2.0)
    assert sorted(edits) == [1, 2]
    h = torch.zeros(1, 3, d)
    assert torch.allclose(edits[1](h), -2.0 * vec.expand(1, 3, d))


def test_encode_dialog_marks_the_response_span(tiny_lm: LanguageModel) -> None:
    msgs = [Message(role="user", content="Is it true?")]
    ids, start = tiny_lm.encode_dialog(msgs, "Yes it is.")
    decoded = tiny_lm.tokenizer.decode(ids[start:-1])
    assert decoded == "Yes it is."
    assert ids[-1] == tiny_lm.end_of_turn_id
    assert tiny_lm.tokenizer.decode(ids[:start]).endswith("</think>\n\n")


def test_pooling_reads_compute_masked_mean_and_max() -> None:
    h = torch.randn(2, 5, 3)
    mask = response_mask([5, 4], [1, 2], 5)
    sums, maxes = torch.empty(2, 1, 3), torch.empty(2, 1, 3)
    pooling_reads(mask, 1, sums, maxes)[0](h)
    assert torch.allclose(sums[0, 0], h[0, 1:4].mean(0))
    assert torch.allclose(maxes[1, 0], h[1, 2:3].amax(0))


def test_capture_writes_pooled_activations_per_layer(tiny_lm: LanguageModel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    exs = [
        Example(
            id=f"e{i}",
            dataset="roleplay",
            family="used_car",
            split="train",
            condition="deceptive" if i % 2 else "honest",
            label=i % 2,
            messages=[Message(role="user", content=f"question {i}")],
            response=f"answer {i}!",
        )
        for i in range(5)
    ]
    capture(tiny_lm, exs, tmp_path, batch_size=2, max_len=256)
    acts = load_activations(tmp_path)
    assert acts.mean.shape == (5, tiny_lm.n_layers, tiny_lm.d_model)
    assert acts.labels.tolist() == [0, 1, 0, 1, 0]
    ids, start = tiny_lm.encode_dialog(exs[3].messages, exs[3].response or "")
    h = tiny_lm.run_until(torch.tensor([ids]), None, [2])[2][0, start:-1]
    np.testing.assert_allclose(acts.layer(2, "mean")[3], h.mean(0).numpy(), atol=2e-2)
