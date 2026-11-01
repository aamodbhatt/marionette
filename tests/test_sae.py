"""SAE math: BatchTopK sparsity, AuxK, decoder norms, threshold inference, persistence."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from marionette.sae import BatchTopKSAE, SAESpec


def make(n: int = 64, d: int = 16, k: int = 4, aux_k: int = 8) -> BatchTopKSAE:
    return BatchTopKSAE(SAESpec(d_in=d, n_latents=n, k=k, aux_k=aux_k, layer=0, model="t"), seed=0)


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 12), st.integers(1, 6), st.integers(0, 10_000))
def test_batch_topk_keeps_exactly_k_times_batch(batch: int, k: int, seed: int) -> None:
    pre = torch.rand(batch, 32, generator=torch.Generator().manual_seed(seed)) + 0.01
    out = BatchTopKSAE.batch_topk(pre, k)
    assert int((out > 0).sum()) == k * batch
    kept = out[out > 0]
    dropped = pre[out == 0]
    if len(dropped):
        assert float(kept.min()) >= float(dropped.max())


def test_batch_topk_lets_tokens_use_different_numbers_of_latents() -> None:
    pre = torch.tensor([[5.0, 4.0, 3.0, 0.1], [0.2, 0.1, 0.0, 0.0]])
    out = BatchTopKSAE.batch_topk(pre, 2)
    assert (out[0] > 0).sum() == 3 and (out[1] > 0).sum() == 1


def test_decoder_rows_unit_norm_after_init_and_step() -> None:
    sae = make()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)
    with torch.no_grad():
        sae.W_dec.mul_(3.0)
    sae.normalise_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


def test_parallel_gradient_is_removed() -> None:
    sae = make()
    sae.W_dec.grad = torch.randn_like(sae.W_dec)
    sae.remove_parallel_decoder_grad()
    dots = (sae.W_dec.grad * sae.W_dec).sum(1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)


def test_aux_loss_is_zero_without_dead_latents_and_only_trains_dead_ones() -> None:
    sae = make()
    x = torch.randn(32, 16)
    out = sae.training_loss(x, aux_coef=1.0, dead_tokens=10**9)
    assert float(out["aux"]) == 0.0
    dead = torch.zeros(64, dtype=torch.bool)
    dead[:8] = True
    pre = sae.pre_acts(x)
    err = torch.randn(32, 16)
    aux = sae.aux_loss(pre, dead, err)
    sae.zero_grad()
    aux.backward()
    grad = sae.W_dec.grad
    assert grad is not None
    assert float(grad[8:].abs().sum()) == 0.0
    assert float(grad[:8].abs().sum()) > 0.0


def test_dead_counters_reset_when_a_latent_fires() -> None:
    sae = make()
    x = torch.randn(16, 16)
    sae.training_loss(x, 0.0, dead_tokens=5)
    fired = sae.batch_topk(sae.pre_acts(x), 4).gt(0).any(0)
    assert torch.all(sae.tokens_since_fired[fired] == 0)
    assert torch.all(sae.tokens_since_fired[~fired] == 16)


def test_training_reduces_reconstruction_error() -> None:
    torch.manual_seed(0)
    basis = torch.randn(8, 16)
    data = torch.relu(torch.randn(512, 8)) @ basis
    sae = make(n=32, k=3)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        out = sae.training_loss(data, 1 / 32, dead_tokens=10_000)
        first = first if first is not None else float(out["fvu"])
        opt.zero_grad()
        out["loss"].backward()
        sae.remove_parallel_decoder_grad()
        opt.step()
        sae.normalise_decoder()
        sae.update_threshold(out["min_kept"], 0.9)
    assert float(out["fvu"]) < 0.5 * first
    # The learned global threshold gives a sparse, batch-independent encoding.
    single = sae.encode(data[:1])
    assert 0 < int((single > 0).sum()) < 32


def test_save_load_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    sae = make()
    with torch.no_grad():
        sae.threshold.fill_(0.3)
        sae.input_scale.fill_(2.0)
    sae.save(tmp_path, {"steps": 1})
    again = BatchTopKSAE.load(tmp_path)
    x = torch.randn(4, 16)
    assert torch.allclose(sae(x), again(x))
    assert again.spec == sae.spec


def test_encode_decode_work_in_raw_units() -> None:
    sae = make()
    with torch.no_grad():
        sae.input_scale.fill_(4.0)
    f = torch.zeros(1, 64)
    f[0, 3] = 1.0
    expected = (sae.W_dec[3] + sae.b_dec) / 4.0
    assert torch.allclose(sae.decode(f)[0], expected)


def test_per_token_fvu_diagnostics_expose_outlier_tokens() -> None:
    from marionette.sae_train import _per_token_fvu

    x = torch.randn(1000, 16, generator=torch.Generator().manual_seed(0))
    x[:5] *= 50
    out = _per_token_fvu(x, (0.1 * x).pow(2).sum(-1))
    assert out["top1pct_norm_tokens_share_of_variance"] > 0.8
    assert 0.005 < out["median_token_fvu"] < 0.02
    assert 0.005 < out["fvu_without_top1pct_norm_tokens"] < 0.02
