"""Small training helpers shared by the baseline ports (no method logic here)."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    """Exponential moving average of a module's weights (ema_pytorch.EMA, update_every=1)."""

    def __init__(self, model: nn.Module, beta: float):
        self.beta = beta
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - self.beta)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
