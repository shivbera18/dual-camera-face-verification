from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA adapter for a frozen Linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base(x)
            + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        )


class LoRAConv2d(nn.Module):
    """Low-rank 1x1 adapter added in parallel to a frozen Conv2d layer."""

    def __init__(self, base: nn.Conv2d, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        for param in self.base.parameters():
            param.requires_grad = False
        self.down = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scaling


def _replace_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_keywords: tuple[str, ...] = ("classifier.1",),
    include_conv: bool = False,
) -> nn.Module:
    """Return a copy of model with LoRA injected into matching Linear/optional Conv2d layers."""
    adapted = deepcopy(model)
    for param in adapted.parameters():
        param.requires_grad = False

    named_modules = dict(adapted.named_modules())
    for full_name, module in list(named_modules.items()):
        if not full_name or not any(key in full_name for key in target_keywords):
            continue
        parent_name, child_name = (
            full_name.rsplit(".", 1) if "." in full_name else ("", full_name)
        )
        parent = adapted.get_submodule(parent_name) if parent_name else adapted
        if isinstance(module, nn.Linear):
            _replace_child(
                parent, child_name, LoRALinear(module, rank=rank, alpha=alpha)
            )
        elif include_conv and isinstance(module, nn.Conv2d):
            _replace_child(
                parent, child_name, LoRAConv2d(module, rank=rank, alpha=alpha)
            )
    return adapted


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
