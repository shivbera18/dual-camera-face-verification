"""EfficientNet-B0 deepfake classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as tvm

from src.utils.config import get_model_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeepfakeClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tvm.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 1),
        )
        if freeze_backbone:
            self.freeze_backbone()
        self._input_size = tuple(get_model_config()["efficientnet"]["input_size"][:2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
        logger.info("backbone frozen")

    def unfreeze_last_blocks(self, n: int = 3) -> None:
        blocks = list(self.backbone.features.children())
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        logger.info("unfroze last %d blocks", n)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)


def _load_state_into(model: nn.Module, checkpoint: Union[str, Path], device) -> bool:
    ckpt_path = resolve(checkpoint) if not Path(str(checkpoint)).is_absolute() else Path(str(checkpoint))
    if not ckpt_path.exists():
        return False
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    logger.info("loaded checkpoint: %s", ckpt_path)
    return True


def build_classifier(
    checkpoint: Optional[Union[str, Path]] = None,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> DeepfakeClassifier:
    model = DeepfakeClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if checkpoint:
        _load_state_into(model, checkpoint, device)
    return model.to(device)
