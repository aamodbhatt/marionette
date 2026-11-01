"""BatchTopK sparse autoencoder with an AuxK loss for dead latents.

* **BatchTopK** (Bussmann et al., 2024): during training, keep the ``k * batch`` largest
  pre-activations across the whole batch rather than ``k`` per token, so the average L0 is
  exactly ``k`` while individual tokens may use more or fewer latents.
* **Inference threshold**: after training, a single global threshold (an EMA of the smallest
  activation BatchTopK kept) replaces the batch dependence, so one token can be encoded alone.
* **AuxK** (Gao et al., 2024): latents that have not fired for ``dead_tokens`` tokens are asked
  to reconstruct the main model's residual error, which revives them.
* Decoder rows are kept at unit norm, and the gradient component parallel to each row is
  removed before every step, so latent activations carry the scale.

Callers work in raw residual units: inputs are scaled internally by ``input_scale`` (set from
data so that the mean squared norm is ``d_in``), and reconstructions are scaled back.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn


@dataclass(frozen=True)
class SAESpec:
    """Shape and identity of a trained SAE."""

    d_in: int
    n_latents: int
    k: int
    aux_k: int
    layer: int
    model: str


class BatchTopKSAE(nn.Module):
    """Sparse autoencoder ``x -> f -> x_hat`` with ``f`` non-negative and sparse."""

    def __init__(self, spec: SAESpec, seed: int = 0) -> None:
        super().__init__()
        self.spec = spec
        gen = torch.Generator().manual_seed(seed)
        w_dec = torch.randn(spec.n_latents, spec.d_in, generator=gen)
        w_dec /= w_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)
        self.W_enc = nn.Parameter(w_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(spec.n_latents))
        self.b_dec = nn.Parameter(torch.zeros(spec.d_in))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("tokens_since_fired", torch.zeros(spec.n_latents, dtype=torch.long))
        self.threshold: torch.Tensor
        self.input_scale: torch.Tensor
        self.tokens_since_fired: torch.Tensor

    # ---- core maps (normalised units) -------------------------------------------------------

    def pre_acts(self, xn: torch.Tensor) -> torch.Tensor:
        return torch.relu((xn - self.b_dec) @ self.W_enc + self.b_enc)

    @staticmethod
    def batch_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
        """Keep the ``k * batch`` largest entries of ``pre`` across the batch; zero the rest."""
        flat = pre.flatten()
        n_keep = min(k * pre.shape[0], flat.numel())
        values, idx = flat.topk(n_keep, sorted=False)
        out = torch.zeros_like(flat)
        out[idx] = values
        return out.view_as(pre)

    def decode_normalised(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    # ---- public API (raw residual units) ----------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference encoding with the global threshold (batch-independent)."""
        pre = self.pre_acts(x.float() * self.input_scale)
        return pre * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decode_normalised(f) / self.input_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def feature_directions(self) -> torch.Tensor:
        """Residual-space direction of each latent's decoder row, ``[n_latents, d_in]`` (unit)."""
        return torch.nn.functional.normalize(self.W_dec, dim=1)

    # ---- training ---------------------------------------------------------------------------

    def training_loss(
        self, xn: torch.Tensor, aux_coef: float, dead_tokens: int
    ) -> dict[str, torch.Tensor]:
        """Loss on a batch of *normalised* inputs; also updates the dead-latent counters."""
        pre = self.pre_acts(xn)
        f = self.batch_topk(pre, self.spec.k)
        recon = self.decode_normalised(f)
        err = xn - recon
        mse = err.pow(2).sum(-1).mean()
        total_var = (xn - xn.mean(0)).pow(2).sum(-1).mean()
        with torch.no_grad():
            fired = (f > 0).any(0)
            self.tokens_since_fired += xn.shape[0]
            self.tokens_since_fired[fired] = 0
            dead = self.tokens_since_fired > dead_tokens
        aux = self.aux_loss(pre, dead, err)
        return {
            "loss": mse + aux_coef * aux,
            "mse": mse.detach(),
            "fvu": (mse / total_var).detach(),
            "aux": aux.detach(),
            "l0": (f > 0).float().sum(-1).mean().detach(),
            "dead_fraction": dead.float().mean(),
            "min_kept": f[f > 0].min().detach() if (f > 0).any() else torch.tensor(0.0),
        }

    def aux_loss(self, pre: torch.Tensor, dead: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        """Normalised MSE of rebuilding the residual error from the top-``aux_k`` dead latents."""
        n_dead = int(dead.sum())
        if n_dead == 0:
            return torch.zeros((), device=pre.device)
        k_aux = min(self.spec.aux_k, n_dead)
        masked = pre.masked_fill(~dead, 0.0)
        values, idx = masked.topk(k_aux, dim=-1, sorted=False)
        f_aux = torch.zeros_like(pre).scatter_(-1, idx, values)
        err_hat = f_aux @ self.W_dec
        target = err.detach()
        denom = (target - target.mean(0)).pow(2).sum(-1).mean().clamp(min=1e-8)
        return (err_hat - target).pow(2).sum(-1).mean() / denom

    @torch.no_grad()
    def remove_parallel_decoder_grad(self) -> None:
        """Project out the gradient component that would change decoder row norms."""
        if self.W_dec.grad is None:
            return
        w = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.grad -= (self.W_dec.grad * w).sum(1, keepdim=True) * w

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))

    @torch.no_grad()
    def update_threshold(self, min_kept: torch.Tensor, beta: float) -> None:
        if float(min_kept) <= 0:
            return
        if float(self.threshold) == 0:
            self.threshold.copy_(min_kept)
        else:
            self.threshold.mul_(beta).add_((1 - beta) * min_kept)

    # ---- persistence ------------------------------------------------------------------------

    def save(self, out_dir: Path, extra: dict[str, object] | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            str(out_dir / "sae.safetensors"),
        )
        (out_dir / "sae.json").write_text(
            json.dumps({"spec": asdict(self.spec), **(extra or {})}, indent=2) + "\n"
        )

    @classmethod
    def load(cls, out_dir: Path, device: torch.device | str = "cpu") -> BatchTopKSAE:
        meta_path = out_dir / "sae.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} not found; run `marionette sae train` first")
        spec = SAESpec(**json.loads(meta_path.read_text())["spec"])
        sae = cls(spec)
        sae.load_state_dict(load_file(str(out_dir / "sae.safetensors")))
        return sae.to(device).eval()
