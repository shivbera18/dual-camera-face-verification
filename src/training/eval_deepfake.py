from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.models.efficientnet import load_deepfake_checkpoint
from src.training.augmentation import get_val_transform
from src.training.dataset import DeepfakeDataset
from src.training.evaluate import evaluate_classifier
from src.training.train_baseline import get_device
from src.utils.config import get_dataset_config, get_model_config, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientNet deepfake checkpoint on a manifest split."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--artifact-prefix", default=None)
    args = parser.parse_args()

    dataset_cfg = get_dataset_config()
    model_cfg = get_model_config()["efficientnet"]
    checkpoint = resolve_project_path(args.checkpoint or model_cfg["checkpoint"])
    manifest = resolve_project_path(args.manifest or dataset_cfg["splits"]["manifest"])
    device = get_device(args.device)
    model, metadata = load_deepfake_checkpoint(
        checkpoint, device=device, pretrained=False
    )
    ds = DeepfakeDataset(
        manifest,
        args.split,
        transform=get_val_transform(),
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size or model_cfg["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    prefix = args.artifact_prefix or f"artifacts/metrics/deepfake_{args.split}"
    report = evaluate_classifier(
        model, loader, device, threshold=args.threshold, artifact_prefix=prefix
    )
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_metadata"] = {
        k: v
        for k, v in metadata.items()
        if k not in {"model_state", "optimizer_state", "scheduler_state"}
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
