"""EfficientNet-B0 deepfake classifier."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torchvision.models as tvm

from src.utils.config import get_model_config
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
        self.sigmoid = nn.Sigmoid()
        if freeze_backbone:
            self.freeze_backbone()
        self._input_size = tuple(get_model_config()["efficientnet"]["input_size"][:2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return self.sigmoid(logits).squeeze(-1)

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


def build_classifier(
    checkpoint: str | None = None,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> DeepfakeClassifier:
    model = DeepfakeClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if checkpoint:
        from src.utils.config import resolve
        ckpt_path = resolve(checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            model.load_state_dict(state, strict=False)
            logger.info("loaded checkpoint: %s", ckpt_path)
    return model.to(device)
