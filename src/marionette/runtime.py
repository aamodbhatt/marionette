"""Device selection, mixed precision and seeding shared by all stages."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator

import numpy as np
import torch

from marionette.config import DType


def resolve_device(spec: str) -> torch.device:
    """Map a config device string ("auto", "cpu", "cuda", "cuda:1") to a torch device."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {spec!r} requested but CUDA is not available")
    return device


def torch_dtype(name: DType) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


@contextlib.contextmanager
def autocast(device: torch.device, enabled: bool = True) -> Iterator[None]:
    """bf16 autocast on CUDA; a no-op on CPU, where bf16 matmuls are slower than fp32."""
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code may still read the legacy global RNG
    torch.manual_seed(seed)


def configure_matmul() -> None:
    """Allow TF32 for the few fp32 matmuls left outside autocast (geometric median, stats)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
