"""LoRA adapters for parameter-efficient fine-tuning."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.efficientnet import DeepfakeClassifier, _load_state_into
from src.utils.config import resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: Optional[float] = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        if alpha is None:
            alpha = float(rank)
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)
        self.scaling = alpha / max(rank, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = (self.B @ self.A) * self.scaling
        return F.linear(x, self.weight + delta_w, self.bias)


def _is_target_module(name: str, target_modules: Iterable[str]) -> bool:
    return any(name == t or name.endswith("." + t) or t in name for t in target_modules)


def inject_lora(
    model: DeepfakeClassifier,
    rank: int = 4,
    alpha: Optional[float] = None,
    target_modules: Iterable[str] = ("classifier.1",),
) -> DeepfakeClassifier:
    for p in model.parameters():
        p.requires_grad = False
    replaced = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _is_target_module(full_name, target_modules):
                lora = LoRALinear(child.in_features, child.out_features, rank=rank, alpha=alpha)
                lora.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora.bias.data.copy_(child.bias.data)
                setattr(parent, child_name, lora)
                replaced += 1
    if replaced == 0:
        logger.warning("inject_lora: no target modules matched %s", tuple(target_modules))
    else:
        logger.info("inject_lora: replaced %d Linear layer(s) rank=%d", replaced, rank)
    return model


def build_lora_classifier(
    checkpoint: Optional[Union[str, Path]] = None,
    rank: int = 4,
    alpha: Optional[float] = None,
    target_modules: Iterable[str] = ("classifier.1",),
    device: Union[str, torch.device] = "cpu",
) -> DeepfakeClassifier:
    """Build a DeepfakeClassifier with LoRA injected, THEN load checkpoint.

    Critical ordering: injecting LoRA first means the saved state_dict's
    `classifier.1.A` / `classifier.1.B` keys have a target to load into.
    """
    model = DeepfakeClassifier(pretrained=False, freeze_backbone=True)
    model = inject_lora(model, rank=rank, alpha=alpha, target_modules=target_modules)
    if checkpoint:
        _load_state_into(model, checkpoint, device)
    return model.to(device)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
