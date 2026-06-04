# Dual-Camera Face Verification — Complete Development Guide

> **The single source of truth for code, features, training, deployment, and troubleshooting.**

This guide is the exhaustive companion to `plan.md` (execution roadmap), `architecture.md` (module design), `CODE-PLAN.md` (file-by-file plan), `DATASETS.md` (datasets), and `CHANGES.md` (deviations). If anything disagrees with those files, the latest of those four documents wins, and this guide should be updated to match.

---

## Table of Contents

1. [Project Mission & Scope](#1-project-mission--scope)
2. [System Architecture — Layer by Layer](#2-system-architecture--layer-by-layer)
3. [Pre-trained Models — What We Use and Why](#3-pre-trained-models--what-we-use-and-why)
4. [Data Pipeline — Raw Frames to Training Batches](#4-data-pipeline--raw-frames-to-training-batches)
5. [EfficientNet-B0 Training — Deep Insights](#5-efficientnet-b0-training--deep-insights)
6. [LoRA / Adapter Fine-Tuning — Deep Dive](#6-lora--adapter-fine-tuning--deep-dive)
7. [ArcFace Verification — Threshold Tuning & Enrollment](#7-arcface-verification--threshold-tuning--enrollment)
8. [Decision Engine Logic](#8-decision-engine-logic)
9. [Dual-Camera Extension — Detailed Design](#9-dual-camera-extension--detailed-design)
10. [Module-by-Module Code Reference](#10-module-by-module-code-reference)
11. [Configuration Files Reference](#11-configuration-files-reference)
12. [Evaluation, Metrics & Reporting](#12-evaluation-metrics--reporting)
13. [Hyperparameter Tuning Playbook](#13-hyperparameter-tuning-playbook)
14. [Performance Optimization](#14-performance-optimization)
15. [Common Pitfalls & Solutions](#15-common-pitfalls--solutions)
16. [Testing Strategy](#16-testing-strategy)
17. [Logging, Reproducibility & Experiment Tracking](#17-logging-reproducibility--experiment-tracking)
18. [Deployment & Demo](#18-deployment--demo)
19. [Glossary](#19-glossary)
20. [Quick Reference Cheat-Sheet](#20-quick-reference-cheat-sheet)

---

## 1. Project Mission & Scope

### 1.1 Mission

Build a **face authentication system** that can:

1. **Detect** a face in a frame (single or dual camera).
2. **Reject** fake / deepfake / spoof faces (printed photo, screen replay, AI-synthesized face).
3. **Verify** that the real face matches an enrolled user identity.
4. **Decide** in real time and produce a clear verdict.

The system must work in real time on commodity hardware (laptop, mid-range GPU) and degrade gracefully under poor lighting, angled faces, and partial occlusion.

### 1.2 Three-Phase Build Order (locked)

| Phase | Goal | Why first |
|---|---|---|
| **A. Baseline** | Single camera · RetinaFace · EfficientNet-B0 · ArcFace | Foundation — must work before any complexity is added |
| **B. LoRA / Adapter** | Parameter-efficient fine-tuning of EfficientNet-B0 | Better accuracy with fewer trainable params and faster training |
| **C. Dual-Camera Demo** | Synchronize two webcams, fuse per-view scores | Demo extension — wraps baseline; never replaces it |

### 1.3 Decision vocabulary

The system always returns one of four outcomes:

| Decision | Meaning |
|---|---|
| `ACCEPT` | Real face detected AND identity matches enrolled user |
| `REJECT_FAKE` | Face detected AND fake/spoof score above threshold |
| `REJECT_IDENTITY` | Real face AND identity similarity below threshold |
| `RETRY` | No face detected OR low confidence OR transient error |

### 1.4 Out of scope (for this iteration)

- Voice / liveness motion / blink detection (single-frame only).
- Multi-user identification (1:1 verification, not 1:N).
- Training ArcFace from scratch (use InsightFace's pre-trained model).
- Mobile / edge optimization (laptop/desktop target only).
- Cloud inference (runs entirely on local machine).

---

## 2. System Architecture — Layer by Layer

The system is composed of **five logical layers** that form a strict pipeline:

```
┌────────────────────────────────────────────────────────────┐
│  L1. INPUT LAYER                                           │
│  - Single camera, video file, image file, or dual camera   │
└────────────────────────┬───────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────┐
│  L2. FACE PROCESSING LAYER                                 │
│  - RetinaFace → bbox + 5 landmarks                         │
│  - FaceAligner → 224×224 aligned crop                      │
└────────────────────────┬───────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────┐
│  L3. MODEL LAYER                                           │
│  - EfficientNet-B0 (baseline) or LoRA-injected             │
│      → outputs fake_probability ∈ [0, 1]                   │
│  - ArcFace (frozen, InsightFace buffalo_l)                 │
│      → 512-D L2-normalized identity embedding             │
└────────────────────────┬───────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────┐
│  L4. DECISION LAYER                                        │
│  - If fake_probability > T_fake: REJECT_FAKE               │
│  - Else: compare embedding to enrolled template            │
│  - Optional: fuse left/right camera scores                 │
│  - Return final decision + scores + reason                 │
└────────────────────────┬───────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────┐
│  L5. STORAGE & LOGGING LAYER                               │
│  - Append every attempt to verification log                │
│  - Save paired image + metadata (dual-cam mode)            │
│  - Persist model versions and thresholds used              │
└────────────────────────────────────────────────────────────┘
```

### 2.1 Why this layering?

- **Single Responsibility**: each layer does one thing; testing is local.
- **Swap-ability**: swap L3 (baseline ↔ LoRA) without touching L4.
- **Re-usability**: dual-cam mode (L1) re-uses L2-L4 unchanged.
- **Debug-ability**: an incorrect decision can be traced back through each layer's intermediate output.

### 2.2 Threading model

- **Single camera**: a single thread captures + processes + renders. Simple.
- **Dual camera**: two capture threads (one per camera) + a main inference thread that consumes paired frames.
- The DualCameraCapture class uses a **lock-free ring buffer** (or `queue.Queue` per camera) so threads never block on I/O.

### 2.3 Memory budget

- Each 224×224 RGB float32 batch (B=32) ≈ `32 × 3 × 224 × 224 × 4 B = 19.3 MB`.
- EfficientNet-B0 inference @ FP32 ≈ 13 MB activation memory per sample.
- ArcFace w600k_r50 ≈ 250 MB when loaded (ONNX runtime is lighter, ~120 MB).
- Total GPU footprint for inference: ~1 GB; training needs ~3–4 GB VRAM minimum with batch 32.

---

## 3. Pre-trained Models — What We Use and Why

### 3.1 InsightFace `buffalo_l` pack

The pack is auto-downloaded by `insightface` on first call to `FaceAnalysis(name='buffalo_l').prepare(...)`. Default location: `~/.insightface/models/buffalo_l/`.

| File | Purpose | Size | Used by |
|---|---|---:|---|
| `det_10g.onnx` | RetinaFace detector (ResNet-10 backbone) | ~17 MB | `src/face/detector.py` |
| `w600k_r50.onnx` | ArcFace ResNet-50 trained on 600k identities | ~250 MB | `src/models/arcface.py` |
| `1k3d68.onnx` | 68-point 3D landmark regressor | ~95 MB | (not used; 5-pt landmarks from det suffice) |
| `2d106det.onnx` | 106-point 2D landmark detector | ~95 MB | (not used) |
| `genderage.onnx` | Gender/age estimation | ~95 MB | (not used) |

> We only consume the first two. The other three are kept on disk because the `buffalo_l` pack downloads as a bundle.

### 3.2 Why RetinaFace (over MTCNN / YuNet / MediaPipe)?

- **5-point landmarks** come for free, enabling similarity transform alignment.
- **State-of-the-art accuracy** on WIDER FACE hard subset.
- **Confidence score** in `[0, 1]` per face — easy to filter.
- **ONNX** — runs on CPU if no GPU, runs on CUDA if available.
- **InsightFace ecosystem** — same package provides ArcFace embeddings.

> Alternative: **YuNet** from OpenCV Zoo is faster but produces only 5-point landmarks with less accurate bbox regression; sufficient for alignment but RetinaFace is more robust under pose variation.

### 3.3 Why ArcFace (over FaceNet, CosFace, SphereFace)?

- **Margin-penalty loss** enforces angular separation → embeddings are more discriminative.
- **L2-normalized** → cosine similarity is just a dot product, very fast at verification time.
- **Pre-trained on 600k identities** → almost zero-shot on new users.
- **Same ecosystem** as RetinaFace — no extra dependency.

> We **never train ArcFace**. The entire identity side is just inference + cosine similarity + threshold.

### 3.4 Why EfficientNet-B0 (over ResNet-18, MobileNetV3, EfficientNet-B3)?

- **Best accuracy-per-FLOP** of any CNN family as of its publication.
- **Compound scaling** — depth, width, resolution all balanced.
- **5.3M params** — fits on a phone if needed; trains fast on a single GPU.
- **ImageNet pretrained weights** are 1-2 lines of code in torchvision.
- **`torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`** — official, no third-party code.

> **B3 or B4** would give +2-3% AUC at ~3× training cost. We start with B0 for speed; B3 is the easy upgrade path.

### 3.5 Pre-trained model checklist

Before starting Phase 0, ensure:

- [ ] `~/.insightface/models/buffalo_l/` exists with 5 ONNX files.
- [ ] `~/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth` exists (or will be auto-downloaded on first `torchvision` call).

If neither exists, run:

```bash
python -c "from insightface.app import FaceAnalysis; FaceAnalysis('buffalo_l').prepare(ctx_id=-1)"
python -c "import torchvision; from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)"
```

---

## 4. Data Pipeline — Raw Frames to Training Batches

### 4.1 End-to-end data flow

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Raw datasets    │   │  Preprocess      │   │  Manifest CSV    │
│  - FF++ faces    │──▶│  - Copy/link     │──▶│  - sample_id     │
│  - 140k R/F      │   │  - Filter        │   │  - path, label   │
│  - ciplab R/F    │   │  - (Re-crop if   │   │  - split         │
│  - Anti-spoof    │   │    needed)       │   │  - source        │
└──────────────────┘   └─────────┬────────┘   └────────┬─────────┘
                                 │                      │
                                 ▼                      ▼
                       ┌──────────────────┐   ┌──────────────────┐
                       │  Processed dir   │   │  PyTorch         │
                       │  processed/      │   │  Dataset class   │
                       │  deepfake_faces/ │   │  (manifest-based)│
                       └──────────────────┘   └─────────┬────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────┐
                                            │  DataLoader      │
                                            │  - shuffle       │
                                            │  - augment       │
                                            │  - collate       │
                                            └──────────────────┘
```

### 4.2 Preprocess rules

The `preprocess.py` module performs the following on each raw image:

1. **Load image** with `cv2.imread` (BGR).
2. **Detect face** with `FaceDetector` (RetinaFace).
3. **Filter** — drop the sample if any of these fail:
   - No face detected.
   - Face area < 40×40 px (`min_face_size`).
   - Detection confidence < 0.8.
   - Image corrupted or unreadable.
4. **Align** using the 5 landmarks and a similarity transform → 224×224 BGR crop.
5. **Save** to `data/processed/deepfake_faces/<split>/<real|fake>/<id>.jpg`.
6. **Append row** to `data/splits/deepfake_manifest.csv`.

For datasets that are **already pre-cropped** (FF++ Kaggle mirror, 140k, ciplab) we skip detection and just copy + verify loadability. The alignment step is also a no-op because the crop is already 224×224.

### 4.3 Manifest CSV schema

| Column | Type | Example | Notes |
|---|---|---|---|
| `sample_id` | str | `ffpp_001234` | Unique |
| `image_path` | str | `data/processed/deepfake_faces/train/real/ffpp_001234.jpg` | Absolute or repo-relative |
| `label` | int | `0` | 0 = real, 1 = fake (matches `configs/dataset.yaml`) |
| `source_dataset` | str | `ffpp` / `140k` / `ciplab` / `antispoof` | For group-aware splits |
| `original_video` | str | `youtube_001_126` | Helps detect leakage |
| `subject_id` | str | `s123` or empty | Optional |
| `split` | str | `train` / `val` / `test` | Mutually exclusive |
| `face_confidence` | float | `0.97` | From RetinaFace |
| `crop_quality_status` | str | `ok` / `rejected_too_small` | For audit |

### 4.4 Splitting strategy

**Stratified group split** is the default:

1. Group rows by `original_video` (or `source_dataset` if video info is missing).
2. Within each group, stratify by `label` (preserve real:fake ratio).
3. Assign groups to train/val/test at the configured ratios (default 70/15/15).

This **prevents the most common leakage bug** in deepfake training: putting a frame from the same source video in both train and test, which inflates accuracy by 10-20 points.

The function signature:

```python
def stratified_group_split(
    df: pd.DataFrame,
    group_col: str = "original_video",
    label_col: str = "label",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ...
```

### 4.5 Class imbalance

Real-world deepfake datasets are imbalanced. The 140k dataset has roughly 50/50 but FF++ has 17% real / 83% fake. Options, in order of preference:

1. **Weighted random sampling** — `WeightedRandomSampler` with `weight = 1 / class_freq`. Easy, no data change.
2. **Class-weighted loss** — `BCEWithLogitsLoss(pos_weight=...)`. Simpler than (1) but worse.
3. **Oversampling minority** — duplicate minority samples. Risk of overfitting.
4. **Focal Loss** — `(1 - p_t)^gamma * BCE`. Good for very imbalanced; less so here.

**Recommendation**: start with `WeightedRandomSampler` and `pos_weight=1.0`. If val F1 on the minority class lags, add `pos_weight=N_real/N_fake`.

### 4.6 Augmentation pipeline (`src/training/augmentation.py`)

Augmentation is the most important hyperparameter in deepfake detection. Too little → overfitting. Too much → destroys the artifacts we are trying to learn.

#### 4.6.1 Train augmentations (Albumentations)

```python
import albumentations as A
from albumentations import (
    HorizontalFlip, RandomBrightnessContrast, ImageCompression,
    GaussianBlur, GaussNoise, RandomResizedCrop, ShiftScaleRotate,
    HueSaturationValue, CoarseDropout, Normalize
)

def get_train_transform(image_size=(224, 224)):
    return A.Compose([
        # Geometric
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                         rotate_limit=15, border_mode=0, p=0.4),
        RandomResizedCrop(size=image_size, scale=(0.85, 1.0),
                          ratio=(0.9, 1.1), p=0.3),
        # Photometric
        RandomBrightnessContrast(brightness_limit=0.2,
                                 contrast_limit=0.2, p=0.3),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                           val_shift_limit=10, p=0.2),
        # Compression / noise
        ImageCompression(quality_range=(60, 100), p=0.3),
        GaussianBlur(blur_limit=(3, 5), p=0.2),
        GaussNoise(p=0.2),
        # Occlusion (use sparingly)
        CoarseDropout(num_holes_range=(1, 3),
                      hole_height_range=(8, 24),
                      hole_width_range=(8, 24), p=0.1),
        # Normalize to ImageNet stats (required for EfficientNet)
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
    ])


def get_val_transform(image_size=(224, 224)):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
    ])
```

#### 4.6.2 Why each augmentation is here

| Augmentation | Probability | Why it's there | Why not stronger |
|---|---|---|---|
| `HorizontalFlip` | 0.5 | Doubles effective data; faces are mirror-symmetric for fake detection | Vertical flip is **excluded** (faces rarely appear upside down in real use) |
| `ShiftScaleRotate` | 0.4 | Simulates camera movement and face pose | Large rotation (>15°) breaks face alignment |
| `RandomResizedCrop` | 0.3 | Slight zoom variations; defensive against framing differences | Aggressive crop (scale 0.5) loses face features |
| `RandomBrightnessContrast` | 0.3 | Real cameras have auto-exposure variability | Extreme values (>0.3) hurt recognition |
| `HueSaturationValue` | 0.2 | Different cameras have different white balance | Big hue shift creates unrealistic faces |
| `ImageCompression` | 0.3 | Real deepfakes get re-encoded when shared on social media | JPEG < 50 destroys our model's features too |
| `GaussianBlur` | 0.2 | Camera motion, low-light noise | Blur > 5px destroys texture artifacts we want to learn |
| `GaussNoise` | 0.2 | Sensor noise realism | High variance noise looks synthetic |
| `CoarseDropout` | 0.1 | Occlusion from hair, hands, masks | High drop rate hides the face entirely |

> **Critical insight**: deepfake artifacts are *subtle* high-frequency patterns. Aggressive blur or extreme compression will destroy them and make the task impossible. Stay close to the recommended probabilities.

#### 4.6.3 Augmentations we DELIBERATELY skip

- **Cutout > 30% of face** — hides the artifact.
- **Color jitter > 30%** — too unrealistic.
- **Mixup / CutMix** — labels are binary; mixed labels confuse BCE.
- **Heavy geometric warps** — face alignment is already done by RetinaFace.
- **Style transfer / domain randomization** — overkill for this size of dataset.

### 4.7 DataLoader settings

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,                 # True for train, False for val/test
    num_workers=4,                # tune for your CPU
    pin_memory=True,              # faster GPU transfer
    persistent_workers=True,      # avoid re-spawning workers
    drop_last=True,               # stable batch norm for train
    prefetch_factor=2,
)
```

> Set `num_workers` to `min(8, os.cpu_count())`. Watch RAM — 4 workers × 32 imgs × 224×224×3 = ~150 MB resident.

### 4.8 Sanity check the data

After building the manifest, run this 30-second check before any training:

```python
import pandas as pd
from collections import Counter

df = pd.read_csv("data/splits/deepfake_manifest.csv")
print("Total samples:", len(df))
print("By split:", Counter(df["split"]))
print("By label:", Counter(df["label"]))
print("By source:", Counter(df["source_dataset"]))
print("Real/fake per split:")
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
```

Expected output (approximate):

```
Total samples: 220000
By split: Counter({'train': 154000, 'val': 33000, 'test': 33000})
By label: Counter({1: 140000, 0: 80000})
By source: Counter({'ffpp': 95000, '140k': 110000, 'ciplab': 2000, ...})
```

If you see class collapse (e.g., 90% one class in val), the split was bad — re-run with a different seed or fix the group column.

---

## 5. EfficientNet-B0 Training — Deep Insights

This is the heart of the project. Most of the engineering effort is here.

### 5.1 Why EfficientNet-B0 is the right choice

| Property | EfficientNet-B0 | ResNet-18 | MobileNetV3-S |
|---|---|---|---|
| ImageNet top-1 | 77.3% | 69.8% | 67.5% |
| Params | 5.3M | 11.7M | 2.5M |
| FLOPs | 0.39B | 1.8B | 0.06B |
| Inference (CPU, ms) | ~40 | ~25 | ~15 |
| Our val AUC on FF++ | ~0.93 | ~0.88 | ~0.85 |

B0 is the sweet spot. B3 would gain ~1.5% AUC at 3× training time.

### 5.2 Model architecture

```python
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DeepfakeClassifier(nn.Module):
    """EfficientNet-B0 with binary real/fake head."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits. Apply sigmoid externally for probability."""
        return self.backbone(x).squeeze(-1)

    def freeze_backbone(self) -> None:
        """Freeze all parameters except the new classifier head."""
        for name, p in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    def unfreeze_last_blocks(self, n: int = 3) -> None:
        """Unfreeze the last n MBConv blocks for fine-tuning."""
        # Unfreeze the head first
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True
        # Then unfreeze the last n feature blocks
        blocks = list(self.backbone.features.children())
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
```

### 5.3 Two-stage fine-tuning — the canonical recipe

This is the most important recipe in the project. **Always follow this two-stage pattern; don't try to fine-tune the whole model from day 1.**

#### Stage 1: Train the head only (5 epochs)

```python
model = DeepfakeClassifier(pretrained=True)
model.freeze_backbone()

# Sanity check
assert model.count_parameters() < 10_000, "Head should have <10K params"

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,                # higher LR for head
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
criterion = nn.BCEWithLogitsLoss()
```

**What this does**: The new classifier head (Linear 1280→1) starts random. The backbone stays at ImageNet features. We need ~5 epochs for the head to converge to a useful linear classifier over ImageNet features.

**Expected result after Stage 1**: val AUC ≈ 0.85-0.90.

#### Stage 2: Unfreeze last blocks and fine-tune (15-20 epochs)

```python
# Load Stage 1 checkpoint
model.load_state_dict(torch.load("artifacts/models/efficientnet_b0_head_best.pth"))
model.unfreeze_last_blocks(n=3)

# Lower LR for backbone
optimizer = torch.optim.AdamW([
    {"params": model.backbone.features[-3:].parameters(), "lr": 1e-5},
    {"params": model.backbone.features[:-3].parameters(), "lr": 1e-6},  # frozen, but in case
    {"params": model.backbone.classifier.parameters(), "lr": 1e-4},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 1e-6, 1e-4],
    total_steps=15 * len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
)
```

**Why parameter groups**: Different parts of the network benefit from different learning rates. The head can move fast; the late backbone blocks need to be nudged gently; the early backbone should stay frozen (it already knows edges, textures, and shapes).

**Why OneCycle**: It warms up the LR for the first 10% of steps then anneals to ~1/100 of max. Empirically, this gives 1-2% AUC boost over plain cosine.

**Expected result after Stage 2**: val AUC ≈ 0.93-0.96.

### 5.4 Loss function choices

| Loss | Formula | When to use | Pros | Cons |
|---|---|---|---|---|
| `BCEWithLogitsLoss` | `-log(p) y - log(1-p) (1-y)` | Default | Stable, well-studied | Sensitive to class imbalance |
| `BCEWithLogitsLoss(pos_weight=...)` | Same, scaled | Imbalanced data | One-line fix | Can overshoot |
| `FocalLoss(gamma=2)` | `(1-p_t)^γ * BCE` | Severe imbalance | Focuses on hard samples | Needs tuning; not always better |
| `LabelSmoothingBCE(eps=0.1)` | BCE with smoothed labels | Reduce overconfidence | Calibrated probabilities | Slightly lower accuracy |

**Recommendation**: start with `BCEWithLogitsLoss`. Add `pos_weight=N_real/N_fake` if val F1 for the minority class lags. Switch to Focal if you have >10:1 imbalance.

### 5.5 Optimizer

| Optimizer | Use case | Settings |
|---|---|---|
| `AdamW` | Default; nearly always best | `lr=1e-4`, `weight_decay=1e-4`, `betas=(0.9, 0.999)` |
| `SGD + momentum` | Sometimes matches AdamW at the end | `lr=1e-2`, `momentum=0.9`, `wd=1e-4`, `nesterov=True` |
| `Lion` | Newer, faster on some tasks | `lr=1e-5` (10× smaller than Adam) |

> We default to `AdamW`. Switch to SGD only if val loss plateaus.

### 5.6 Learning rate scheduling

| Schedule | Curve | When to use |
|---|---|---|
| **None (constant)** | Flat | Quick experiments, debugging |
| **StepLR** | Staircase | Classic, works but coarse |
| **CosineAnnealingLR** | Smooth decay | Default choice; no warmup needed |
| **OneCycleLR** | Up then down | Best empirical results; needs `total_steps` |
| **ReduceLROnPlateau** | Adaptive | When you don't know epoch count |

**Recommendation**: `OneCycleLR` for Stage 2, `CosineAnnealingLR` for Stage 1.

### 5.7 Regularization

We use **four** regularizers stacked:

1. **Dropout (0.3)** in the classifier head.
2. **Weight decay (1e-4)** in the optimizer.
3. **Data augmentation** (see §4.6).
4. **Early stopping** (patience 5 on val AUC).

Optional:
- **Stochastic depth (DropPath)** at 0.1 — turn on if you train a deeper model (B3+).
- **Mixup** — try if accuracy plateaus; can hurt deepfake detection because labels are binary.
- **Stochastic Weight Averaging (SWA)** — adds +0.5% AUC for free, runs for the last 5 epochs of training.

### 5.8 Mixed-precision training (AMP)

Use AMP for ~1.5-2× training speedup with no accuracy loss:

```python
scaler = torch.amp.GradScaler("cuda")

for batch in train_loader:
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(images)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> On Apple Silicon (MPS), AMP is not yet as mature. Use FP32 or `bfloat16` if available.

### 5.9 Gradient accumulation (effective batch size)

When GPU memory is tight:

```python
accum_steps = 4        # effective batch = 32 * 4 = 128
optimizer.zero_grad()
for i, (images, labels) in enumerate(train_loader):
    logits = model(images)
    loss = criterion(logits, labels) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

> Adjust the LR proportionally: `effective_lr = base_lr * (accum_steps * batch_size) / 256`.

### 5.10 Checkpointing strategy

Save **three** things at every best-AUC epoch:

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "scaler_state": scaler.state_dict(),
    "val_auc": val_auc,
    "val_loss": val_loss,
    "config": config_dict,
}, "artifacts/models/efficientnet_b0_baseline_best.pth")
```

Also save `efficientnet_b0_baseline_last.pth` every epoch so you can resume.

### 5.11 Early stopping

```python
class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best = -float("inf") if mode == "max" else float("inf")

    def __call__(self, metric: float) -> bool:
        improved = (metric > self.best + self.min_delta) if self.mode == "max" \
                   else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

### 5.12 K-fold cross-validation (optional, advanced)

If you have time, run **5-fold stratified group CV**. This gives you:

- A more reliable test AUC estimate (±0.5% standard error).
- Five model checkpoints; you can ensemble them at inference.

The cost is 5× the training time — only worth it for the final report.

### 5.13 Full training loop (production-grade)

```python
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    scheduler=None, log_every=50):
    model.train()
    losses, all_logits, all_labels = [], [], []

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().float().cpu())
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(labels, probs)
    return {"loss": np.mean(losses), "auc": auc}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, all_logits, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.float().cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, (probs > 0.5).astype(int))
    return {"loss": np.mean(losses), "auc": auc, "f1": f1}


def fit(model, train_loader, val_loader, *, lr, epochs, device,
        weight_decay=1e-4, save_dir="artifacts/models"):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=epochs * len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    early = EarlyStopping(patience=5, mode="max")

    best_auc = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer,
                             scaler, device, scheduler)
        va = validate(model, val_loader, criterion, device)
        dt = time.time() - t0
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_auc": tr["auc"],
               "val_loss": va["loss"], "val_auc": va["auc"], "val_f1": va["f1"],
               "time_s": dt}
        history.append(row)
        print(f"[{epoch:02d}/{epochs}] "
              f"train loss={tr['loss']:.4f} auc={tr['auc']:.4f} | "
              f"val loss={va['loss']:.4f} auc={va['auc']:.4f} f1={va['f1']:.4f} "
              f"({dt:.1f}s)")
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_auc": va["auc"], "config": {"lr": lr, "epochs": epochs}},
                       f"{save_dir}/efficientnet_b0_baseline_best.pth")
        if early(va["auc"]):
            print(f"Early stop at epoch {epoch} (no AUC improvement in {early.patience} epochs)")
            break

    # Save history for plotting
    pd.DataFrame(history).to_csv(f"{save_dir}/training_history.csv", index=False)
    return history, best_auc
```

### 5.14 Training insights — what to monitor

Beyond the loss, **watch these signals every epoch**:

| Signal | What it tells you | Healthy range |
|---|---|---|
| `train_loss - val_loss` gap | Overfitting | < 0.1 in early epochs, narrows over time |
| `train_auc - val_auc` gap | Overfitting | < 0.05 after Stage 2 |
| `lr` | Sanity check | Decreasing as expected |
| Gradient norm | Stability | < 10 ideally, < 100 max |
| `time / epoch` | Throughput | 60-300s on a single GPU |
| GPU utilization | Efficiency | > 80% with right `num_workers` |
| GPU memory | Capacity | < 90% of total |

If val loss diverges from train loss, you are overfitting. If both are flat from epoch 1, you have a bug (data labels all the same, frozen loss, etc.).

### 5.15 Common training bugs and their signatures

| Bug | Symptom | Fix |
|---|---|---|
| Data not normalized | Loss stays at 0.69 (= ln 2) for many epochs | Check `Normalize` is applied |
| Labels flipped | Val AUC ≈ 0.0 (perfect inverse) | Print a sample's label and image |
| Frozen loss graph | Loss never changes | Make sure you call `optimizer.zero_grad()` |
| Wrong shape loss | Shape mismatch error | `labels.float()` and squeeze logits |
| Train set in val set | Val AUC > 0.99 | Use stratified group split |
| DataLoader shuffling val | Metrics unstable | `shuffle=False` for val/test |
| Mixed precision overflow | Loss = NaN | Use `bfloat16` instead of `float16`, or disable AMP |
| Augmentation destroys face | Val AUC < 0.7 | Reduce augmentation strength |

### 5.16 Expected results & milestone table

| Milestone | Stage | Val AUC | Val F1 | Notes |
|---|---|---|---|---|
| Random head | 0 | 0.50 | 0.50 | Sanity check |
| Stage 1 done | 1 | 0.85-0.90 | 0.80-0.85 | 5 epochs, head only |
| Stage 2 done | 2 | 0.93-0.96 | 0.91-0.94 | 15-20 epochs, last 3 blocks |
| With LoRA | 2+ | 0.94-0.97 | 0.92-0.95 | Often matches/beats Stage 2 with fewer params |
| + Test set | end | 0.92-0.95 | 0.90-0.93 | Honest number, no leakage |

If you are below 0.85 after Stage 1, do NOT proceed — debug first.

---

## 6. LoRA / Adapter Fine-Tuning — Deep Dive

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method. Instead of updating all M parameters, we add a tiny trainable low-rank delta to specific weight matrices.

### 6.1 The math (the short version)

For a frozen weight matrix `W ∈ R^{d_out × d_in}`, LoRA adds:

```
W' = W + ΔW       where ΔW = (α/r) · B · A
A ∈ R^{r × d_in}   (initialized: Kaiming uniform)
B ∈ R^{d_out × r}  (initialized: zeros)
```

- `r` = **rank** (4, 8, 16 are common). Lower rank = fewer params.
- `α` = **scaling factor** (usually 2× rank, or `α/r = 1`).

When `B = 0`, the original `W` is preserved exactly. As training proceeds, `B` and `A` learn the task-specific low-rank update.

**Why it works**: the change in weights during fine-tuning lives in a low-dimensional subspace; we can capture it with a few thousand parameters instead of millions.

### 6.2 Where to inject LoRA in EfficientNet

EfficientNet-B0 is built from MBConv blocks. The main candidate layers:

| Layer | LoRA target | Pros | Cons |
|---|---|---|---|
| Classifier head (`Linear(1280, 1)`) | The `Linear` itself | Trivial, fast, no effect on features | Limited gain |
| Each MBConv's expansion `Conv1x1` | `weight` | Most impactful, ~150K params added | Slightly more code |
| Each MBConv's projection `Conv1x1` | `weight` | Stabilizes features | Less effect than expansion |
| The final 1×1 in features | `weight` | Touches semantic features | Risk of changing too much |

**Practical recommendation for B0**: inject LoRA into **all `Conv2d` layers in the last 3 feature blocks**. This gives ~200K trainable params, a good speed/accuracy tradeoff.

### 6.3 LoRA implementation (from scratch)

```python
import torch
import torch.nn as nn
import math


class LoRAConv2d(nn.Module):
    """LoRA wrapper for a frozen nn.Conv2d."""

    def __init__(self, base: nn.Conv2d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        in_ch = base.in_channels
        out_ch = base.out_channels
        kernel_size = base.kernel_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A: [rank, in_ch, 1, 1]  - 1x1 conv as the low-rank factor
        # B: [out_ch, rank, 1, 1]
        self.lora_A = nn.Parameter(torch.zeros(rank, in_ch, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero so initial output == base output

    def forward(self, x):
        base_out = self.base(x)
        # LoRA path: x → conv_A → conv_B (both 1x1) → scale
        lora_out = nn.functional.conv2d(x, self.lora_A, stride=self.base.stride,
                                         padding=self.base.padding)
        lora_out = nn.functional.conv2d(lora_out, self.lora_B)
        return base_out + lora_out * self.scaling


def inject_lora(model: nn.Module, rank: int = 4, alpha: float = 1.0,
                target_substrings=("features.6", "features.7", "features.8")) -> int:
    """Replace Conv2d layers whose qualified name contains any of target_substrings
    with LoRAConv2d. Returns number of injected adapters."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and any(t in name for t in target_substrings):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for p in parent_name.split("."):
                if p:
                    parent = getattr(parent, p)
            new = LoRAConv2d(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, new)
            count += 1
    return count


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
```

### 6.4 LoRA training recipe

```python
# 1. Load baseline (full) model
baseline = DeepfakeClassifier(pretrained=True)
baseline.load_state_dict(torch.load("artifacts/models/efficientnet_b0_baseline_best.pth")["model_state"])

# 2. Inject LoRA
n = inject_lora(baseline, rank=8, alpha=16.0)
print(f"Injected {n} LoRA adapters")
tr, tot = count_trainable(baseline)
print(f"Trainable: {tr:,} / {tot:,} = {tr/tot*100:.2f}%")

# 3. Train ONLY the LoRA parameters (base is frozen)
optimizer = torch.optim.AdamW(
    [p for p in baseline.parameters() if p.requires_grad],
    lr=1e-3,                  # higher LR is OK because only LoRA updates
    weight_decay=1e-4,
)
```

**Key differences from full fine-tuning**:

- LR is **10× higher** (1e-3 vs 1e-4) because we are updating a small number of well-initialized parameters.
- **No need for two stages** — head is part of base and already trained.
- **Fewer epochs needed** — usually converges in 8-10 epochs.

### 6.5 Rank/alpha selection guide

| Rank (r) | Alpha (α) | Trainable params | Speed | Use when |
|---|---|---|---|---|
| 4 | 8 | ~50K | Fastest | Quick experiments, hard memory limits |
| 8 | 16 | ~100K | Fast | **Default choice** |
| 16 | 32 | ~200K | Medium | Trying to push accuracy |
| 32 | 64 | ~400K | Slower | Last attempt before full fine-tuning |
| 64 | 128 | ~800K | Slowest | Often matches full fine-tuning |

Rule of thumb: `α = 2 × r` gives scaling 2.0, a sane default. `α = r` gives scaling 1.0, more conservative.

### 6.6 LoRA — when it works and when it doesn't

**Works well when**:
- You have a small amount of data (1k-50k samples).
- The base model is well-aligned with the task (EfficientNet trained on ImageNet → deepfake faces).
- You want fast iteration and small checkpoints (~MB vs ~20MB).

**Doesn't help when**:
- You have 100k+ samples and a beefy GPU; full fine-tuning can do better.
- The task is very different from the base training (e.g., training on satellite imagery).
- You set the rank too low (< 4) for a complex task.

### 6.7 Saving & loading LoRA

Two approaches:

**A. Save full state dict (simple)**
```python
torch.save(model.state_dict(), "efficientnet_b0_lora.pth")
# Note: this includes the frozen weights, so file is ~20MB
```

**B. Save only LoRA deltas (small)**
```python
lora_state = {name: p for name, p in model.named_parameters()
              if "lora_A" in name or "lora_B" in name}
torch.save(lora_state, "efficientnet_b0_lora_deltas.pth")  # ~400KB
# To load:
model = DeepfakeClassifier(pretrained=True)
inject_lora(model)
model.load_state_dict(torch.load("efficientnet_b0_lora_deltas.pth"), strict=False)
```

We default to **A** for simplicity. Switch to **B** if you start versioning many LoRA variants.

---

## 7. ArcFace Verification — Threshold Tuning & Enrollment

### 7.1 What ArcFace gives you

For a 112×112 aligned face, the ArcFace ONNX model in `buffalo_l` returns a **512-D float32 vector** that is **L2-normalized** (||e||₂ = 1). Two normalized vectors compared with cosine similarity:

```
cos(e1, e2) = e1 · e2            (because ||e||₂ = 1)
```

This is a single 512-element dot product — very fast, even on CPU.

### 7.2 What "good" similarity scores look like

On LFW, with InsightFace ArcFace:

| Pair type | Mean cosine | Std dev | Notes |
|---|---|---|---|
| Same person | ~0.65 | ~0.10 | High variance due to pose, lighting |
| Different person | ~0.20 | ~0.15 | Mean is far below same-person mean |
| Twins | ~0.45 | ~0.10 | Boundary case; usually rejected at T=0.50 |

A threshold of **0.40-0.50** is a good starting point on LFW. We default to **0.60** for stricter verification (lower false-accept rate).

### 7.3 Enrollment flow

The enrollment process produces **one L2-normalized template per user**, averaged from N samples.

```python
class EnrollmentStore:
    def __init__(self, db_path="artifacts/models/enrollment_db.pkl"):
        self.db_path = db_path
        self.users: dict[str, dict] = {}   # user_id → {template, n_samples, created_at, ...}
        self._load()

    def enroll(self, user_id: str, embeddings: list[np.ndarray]):
        """Average multiple L2-normalized embeddings into a single template."""
        assert len(embeddings) >= 3, "Need at least 3 enrollment images"
        # Stack and renormalize (mean of unit vectors is not unit length)
        stacked = np.stack(embeddings)              # (N, 512)
        template = stacked.mean(axis=0)            # (512,)
        template = template / np.linalg.norm(template)
        self.users[user_id] = {
            "template": template.astype(np.float32),
            "n_samples": len(embeddings),
            "created_at": time.time(),
            "model_name": "buffalo_l",
            "model_version": "w600k_r50",
        }
        self.save()

    def get_template(self, user_id: str) -> np.ndarray | None:
        return self.users.get(user_id, {}).get("template")

    def verify(self, user_id: str, query_emb: np.ndarray) -> tuple[float, bool]:
        """Returns (cosine_similarity, is_match)."""
        t = self.get_template(user_id)
        if t is None:
            return 0.0, False
        sim = float(np.dot(t, query_emb))           # already normalized
        is_match = sim >= THRESHOLD_MATCH
        return sim, is_match

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.users, f)

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.users = pickle.load(f)
```

### 7.4 Enrollment best practices

| Practice | Why |
|---|---|
| Capture 5-10 samples per user | More samples → less noise in template |
| Vary pose slightly (turn head left/right) | Captures the user's embedding from multiple angles |
| Same lighting as runtime | If runtime is indoor, enroll indoors |
| Don't move the camera between samples | Stable embeddings |
| Exclude blurry / off-frame samples | Pre-bad embeddings from corrupting template |
| Re-enroll if user gains/loses significant weight | Embeddings can shift |

### 7.5 LFW threshold tuning (the canonical sweep)

```python
import numpy as np
from sklearn.metrics import roc_curve


def tune_threshold_lfw(embeddings_a, embeddings_b, labels, thresholds=None):
    """labels: 1=same person, 0=different."""
    if thresholds is None:
        thresholds = np.arange(0.20, 0.80, 0.01)

    sims = np.array([np.dot(a, b) for a, b in zip(embeddings_a, embeddings_b)])

    # EER point
    fpr, tpr, thr = roc_curve(labels, sims)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thr[eer_idx]
    eer_value = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # Threshold sweep
    rows = []
    for t in thresholds:
        preds = (sims >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        far = fp / (fp + tn + 1e-9)        # false accept rate
        frr = fn / (fn + tp + 1e-9)        # false reject rate
        acc = (tp + tn) / len(labels)
        rows.append({"threshold": t, "far": far, "frr": frr, "acc": acc})

    # Pick the threshold where FAR == FRR (EER) for the default setting
    return {
        "eer_threshold": float(eer_threshold),
        "eer": float(eer_value),
        "sweep": rows,
    }
```

### 7.6 Why LFW is the right tuning set

- 13,233 images, 5,749 identities.
- 6,000 labeled pairs (3,000 same + 3,000 different).
- Captures "in the wild" variability (lighting, pose, expression, age).
- Standard benchmark; results are reproducible and comparable to published numbers.

For deployment, **also tune on your own data** (a few hundred pairs from your actual camera + lighting) to capture domain-specific thresholds.

### 7.7 ArcFace performance numbers (InsightFace w600k_r50)

| Benchmark | Accuracy | Notes |
|---|---|---|
| LFW | 99.78% | Standard verification |
| CFP-FP | 98.5% | Frontal-profile |
| AgeDB-30 | 98.2% | Age gap |
| MegaFace | 98.4% | 1M distractors |

These are very high — your system's accuracy on real users will be limited mostly by face detection, alignment quality, and your threshold, not ArcFace itself.

### 7.8 Anti-spoofing interaction

ArcFace doesn't care if a face is real or fake. If you hand it a printed photo, the embedding is meaningful (and will match a printed enrollment!). That is why the **EfficientNet fake-check runs FIRST**, before ArcFace is invoked. The decision engine:

```
1. fake_prob = efficientnet(crop)
2. if fake_prob > T_fake: REJECT_FAKE
3. else: emb = arcface(crop); sim = cosine(emb, template)
4. if sim >= T_match: ACCEPT
5. else: REJECT_IDENTITY
```

This ordering is critical. A faked face that passes the fake check (rare, but possible) and matches a stored template would be accepted. The threshold `T_fake` should be set conservatively (e.g., 0.30) to minimize this risk.

---

## 8. Decision Engine Logic

### 8.1 The decision rules

```python
@dataclass
class VerificationResult:
    decision: str          # one of ACCEPT, REJECT_FAKE, REJECT_IDENTITY, RETRY
    fake_score: float      # 0..1
    match_score: float     # 0..1
    user_id: str | None
    face_detected: bool
    latency_ms: float
    reason: str


class DecisionEngine:
    def __init__(self, fake_threshold=0.5, match_threshold=0.60,
                 min_face_confidence=0.8, min_face_size=40):
        self.T_fake = fake_threshold
        self.T_match = match_threshold
        self.min_conf = min_face_confidence
        self.min_size = min_face_size

    def decide(self, *, face_detected: bool, face_conf: float, face_size: int,
               fake_score: float, match_score: float, user_id: str) -> VerificationResult:
        # Step 0: face presence + quality
        if not face_detected:
            return VerificationResult("RETRY", 0.0, 0.0, user_id, False, 0.0,
                                       reason="no_face")
        if face_conf < self.min_conf:
            return VerificationResult("RETRY", 0.0, 0.0, user_id, True, 0.0,
                                       reason=f"low_conf_{face_conf:.2f}")
        if face_size < self.min_size:
            return VerificationResult("RETRY", 0.0, 0.0, user_id, True, 0.0,
                                       reason=f"small_face_{face_size}px")
        # Step 1: fake check
        if fake_score > self.T_fake:
            return VerificationResult("REJECT_FAKE", float(fake_score), 0.0,
                                       user_id, True, 0.0,
                                       reason=f"fake_prob_{fake_score:.3f}")
        # Step 2: identity match
        if match_score >= self.T_match:
            return VerificationResult("ACCEPT", float(fake_score),
                                       float(match_score), user_id, True, 0.0,
                                       reason="matched")
        return VerificationResult("REJECT_IDENTITY", float(fake_score),
                                   float(match_score), user_id, True, 0.0,
                                   reason=f"sim_{match_score:.3f}_below_{self.T_match}")
```

### 8.2 Threshold selection cheat-sheet

| Use case | T_fake | T_match | Why |
|---|---|---|---|
| High security (banking) | 0.30 | 0.70 | Low FAR; some FRR accepted |
| Standard auth (phone unlock) | 0.50 | 0.60 | Balanced |
| Demo / public | 0.60 | 0.45 | High recall; more ACCEPTs |
| Research / debugging | 0.50 | 0.50 | Calibrated mid-point |

> The right answer comes from ROC analysis on YOUR data. See §12.

### 8.3 The "RETRY" path

`RETRY` is not a failure — it's the system telling the user "try again". Common causes:

- User too close / too far.
- Lighting too low (face detector returns low confidence).
- Sunglasses, mask, hand over face.
- Face detected but alignment score is bad (crop is rotated).

The user-facing UX should **not log RETRY as a security event**; log it as informational.

---

## 9. Dual-Camera Extension — Detailed Design

### 9.1 Goals of dual-camera mode

1. **Synchronize** two webcams within ~50 ms of each other.
2. **Run the full pipeline on each view** independently.
3. **Fuse the per-view scores** to make a more confident decision.
4. **Save paired images + metadata** for debugging and demo.

### 9.2 Why dual-camera helps

- **Two angles** of the same face make spoofing harder (printed photo, phone screen).
- **Consistency check** between left/right embeddings catches some presentation attacks.
- **Redundancy** — if one camera has bad lighting, the other may still work.
- **Cool demo factor** — visually impressive.

### 9.3 Hardware setup

| Component | Recommendation | Why |
|---|---|---|
| 2× USB webcams | Same model, 720p, 30 FPS | Reduces frame timing differences |
| USB hub | Powered, USB 3.0 | Avoids current starvation |
| Mounting | Side-by-side, 6-12 cm apart | Captures both views of the same face |
| Lighting | Diffuse, front-facing | Avoids backlit or strong shadow |
| OS | Linux or macOS | Best OpenCV multi-cam support |

### 9.4 Frame capture with timestamps

```python
import time
import threading
import queue
import cv2
from dataclasses import dataclass


@dataclass
class Frame:
    img: np.ndarray
    timestamp_ms: float
    camera_id: int
    frame_index: int


class DualCameraCapture:
    def __init__(self, left_idx=0, right_idx=1, sync_delta_ms=50,
                 width=1280, height=720, fps=30):
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.sync_delta_ms = sync_delta_ms
        self.width = width
        self.height = height
        self.fps = fps
        self.left_q: queue.Queue[Frame] = queue.Queue(maxsize=2)
        self.right_q: queue.Queue[Frame] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _open_camera(self, idx):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self, cam_id, idx):
        cap = self._open_camera(idx)
        if not cap.isOpened():
            print(f"[ERROR] Camera {cam_id} (idx={idx}) failed to open")
            return
        q = self.left_q if cam_id == 0 else self.right_q
        frame_index = 0
        while not self._stop.is_set():
            ok, img = cap.read()
            if not ok:
                continue
            ts = time.monotonic() * 1000
            f = Frame(img=img, timestamp_ms=ts, camera_id=cam_id,
                      frame_index=frame_index)
            try:
                q.put_nowait(f)
            except queue.Full:
                # Drop oldest
                try: q.get_nowait()
                except queue.Empty: pass
                q.put_nowait(f)
            frame_index += 1
        cap.release()

    def start(self):
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader, args=(0, self.left_idx), daemon=True),
            threading.Thread(target=self._reader, args=(1, self.right_idx), daemon=True),
        ]
        for t in self._threads: t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads: t.join(timeout=2)

    def get_pair(self) -> tuple[Frame, Frame] | None:
        """Return the most recent left/right pair if within sync_delta_ms."""
        if self.left_q.empty() or self.right_q.empty():
            return None
        # Drain one item from each, then look for the closest
        left = self.left_q.get()
        right = self.right_q.get()
        # Look for a closer right within ±sync window
        while not self.right_q.empty():
            cand = self.right_q.queue[0]
            if abs(cand.timestamp_ms - left.timestamp_ms) < abs(right.timestamp_ms - left.timestamp_ms):
                right = self.right_q.get()
            else:
                break
        if abs(left.timestamp_ms - right.timestamp_ms) > self.sync_delta_ms:
            return None
        return left, right

    def save_pair(self, left: Frame, right: Frame, metadata: dict, out_dir: str):
        import json
        os.makedirs(out_dir, exist_ok=True)
        sid = metadata.get("session_id", "session")
        pid = metadata.get("pair_index", 0)
        lp = f"{out_dir}/{sid}_{pid:06d}_left_{int(left.timestamp_ms)}.jpg"
        rp = f"{out_dir}/{sid}_{pid:06d}_right_{int(right.timestamp_ms)}.jpg"
        cv2.imwrite(lp, left.img)
        cv2.imwrite(rp, right.img)
        metadata.update({
            "left_image_path": lp, "right_image_path": rp,
            "left_timestamp": left.timestamp_ms,
            "right_timestamp": right.timestamp_ms,
            "sync_delta_ms": abs(left.timestamp_ms - right.timestamp_ms),
        })
        with open(f"{out_dir}/{sid}_{pid:06d}.json", "w") as f:
            json.dump(metadata, f, indent=2)
```

### 9.5 Frame pairing strategies

| Strategy | Logic | Best for |
|---|---|---|
| **Nearest neighbor** | Pair left with right whose timestamp is closest | Live demo |
| **Buffered sliding window** | Hold the last N frames; pair greedily | Slightly desynced hardware |
| **Hardware trigger** | Both cameras triggered by GPIO at the same time | Research-grade |
| **FPS-locked** | Trigger captures at fixed wall-clock intervals | When you control both clocks |

We use **nearest neighbor** for simplicity. For the demo, that's plenty.

### 9.6 Score fusion strategies

The `DualCamPipeline` can use several fusion strategies:

```python
def fuse_scores(left: dict, right: dict, method="mean_max") -> dict:
    """
    left, right: dicts with keys 'fake_score' and 'match_score'
    method: 'mean_max' | 'max_max' | 'and_fake' | 'or_match' | 'confidence_weighted'
    """
    if method == "mean_max":
        return {
            "fake_score": (left["fake_score"] + right["fake_score"]) / 2,
            "match_score": max(left["match_score"], right["match_score"]),
        }
    if method == "max_max":
        return {
            "fake_score": max(left["fake_score"], right["fake_score"]),
            "match_score": max(left["match_score"], right["match_score"]),
        }
    if method == "and_fake":
        # Reject as fake if EITHER view says fake
        return {
            "fake_score": max(left["fake_score"], right["fake_score"]),
            "match_score": max(left["match_score"], right["match_score"]),
        }
    if method == "or_match":
        # Accept identity if EITHER view matches
        return {
            "fake_score": (left["fake_score"] + right["fake_score"]) / 2,
            "match_score": max(left["match_score"], right["match_score"]),
        }
    if method == "confidence_weighted":
        w_l = left.get("face_confidence", 0.5)
        w_r = right.get("face_confidence", 0.5)
        s = w_l + w_r
        return {
            "fake_score": (left["fake_score"] * w_l + right["fake_score"] * w_r) / s,
            "match_score": max(left["match_score"], right["match_score"]),
        }
    raise ValueError(f"Unknown fusion method: {method}")
```


