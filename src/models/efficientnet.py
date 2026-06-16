from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class DeepfakeClassifier(nn.Module):
    """EfficientNet-B0 with a binary real/fake head. Forward returns logits."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, 1)
        )
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("classifier")

    def unfreeze_last_blocks(self, n: int = 3) -> None:
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        blocks = list(self.backbone.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def load_deepfake_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    pretrained: bool = False,
) -> tuple[DeepfakeClassifier, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DeepfakeClassifier(pretrained=pretrained)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, metadata
