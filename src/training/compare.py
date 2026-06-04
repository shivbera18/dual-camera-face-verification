"""Compare baseline vs LoRA on the test split."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch

from src.models.efficientnet import build_classifier
from src.models.lora import build_lora_classifier, count_trainable_params, count_total_params
from src.training.augmentation import get_val_transform
from src.training.dataset import DeepfakeDataset
from src.training.evaluate import evaluate_classifier, save_report
from src.utils.config import resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_test_loader(manifest: str, batch_size: int, num_workers: int):
    ds = DeepfakeDataset(manifest, "test", transform=get_val_transform())
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def _load_model(ckpt: str, rank: Optional[int], alpha: Optional[float], device):
    if rank is not None:
        return build_lora_classifier(
            checkpoint=ckpt, rank=rank, alpha=alpha, device=device
        )
    return build_classifier(
        checkpoint=ckpt, pretrained=False, freeze_backbone=True, device=device
    )


@torch.no_grad()
def _measure_latency(model, loader, device, n_batches: int = 10) -> float:
    model.eval()
    it = iter(loader)
    t0 = time.time()
    n = 0
    for _ in range(n_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device)
        _ = model.predict_proba(x)
        n += x.size(0)
    elapsed = (time.time() - t0) / max(n, 1)
    return float(elapsed * 1000.0)


def compare(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = _make_test_loader(args.manifest, args.batch_size, args.num_workers)
    baseline = _load_model(args.baseline_ckpt, rank=None, alpha=None, device=device)
    lora = _load_model(args.lora_ckpt, rank=args.rank, alpha=args.alpha, device=device)
    base_metrics = evaluate_classifier(baseline, test_loader, threshold=0.5, device=device)
    lora_metrics = evaluate_classifier(lora, test_loader, threshold=0.5, device=device)
    base_latency = _measure_latency(baseline, test_loader, device)
    lora_latency = _measure_latency(lora, test_loader, device)
    table = {
        "metric": ["trainable_params", "total_params", "roc_auc", "f1", "inference_ms_per_image"],
        "baseline": [
            count_trainable_params(baseline),
            count_total_params(baseline),
            base_metrics["roc_auc"],
            base_metrics["f1"],
            base_latency,
        ],
        "lora": [
            count_trainable_params(lora),
            count_total_params(lora),
            lora_metrics["roc_auc"],
            lora_metrics["f1"],
            lora_latency,
        ],
    }
    md_lines = ["| metric | baseline | lora |", "|---|---:|---:|"]
    for i, m in enumerate(table["metric"]):
        md_lines.append(f"| {m} | {table['baseline'][i]} | {table['lora'][i]} |")
    out = "\n".join(md_lines) + "\n"
    out_path = resolve("artifacts/metrics/baseline_vs_lora.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out)
    save_report({"table": table}, "artifacts/metrics/baseline_vs_lora.json")
    print(out)
    logger.info("comparison written: %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="data/splits/deepfake_manifest.csv")
    p.add_argument("--baseline_ckpt", type=str, default="artifacts/models/efficientnet_b0_baseline_best.pth")
    p.add_argument("--lora_ckpt", type=str, default="artifacts/models/efficientnet_b0_lora_best.pth")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    compare(parse_args())
