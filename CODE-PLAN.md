# Code Writing Plan

Detailed implementation order, file-by-file. Follow phases in sequence — each phase depends on previous.

Reference: `plan.md` (execution roadmap) · `architecture.md` (module design) · `configs/` (hyperparams)

---

## Phase 0 — Project Skeleton

**Goal:** Importable package structure, config loader, logger. Nothing model-specific yet.

### Files to create

```
src/
├── __init__.py                  (already exists)
├── utils/
│   ├── __init__.py
│   ├── config.py                # load yaml configs into dataclass/dict
│   └── logger.py                # colorlog wrapper, file + console handlers
```

### `src/utils/config.py`
- `load_config(path: str) -> dict` — loads any yaml, returns dict
- `get_dataset_config()` — loads `configs/dataset.yaml`
- `get_model_config()` — loads `configs/model.yaml`
- `get_pipeline_config()` — loads `configs/pipeline.yaml`

### `src/utils/logger.py`
- `get_logger(name, log_file=None)` — returns logger with color console + rotating file handler
- Log format: `[timestamp] [level] [module] message`
- Save logs to `artifacts/logs/`

**Done when:** `from src.utils.config import get_model_config` works without error.

---

## Phase 1 — Face Detection & Alignment

**Goal:** Given any image, detect face, return aligned 224×224 crop. Used in ALL later phases.

### Files to create

```
src/face/
├── __init__.py
├── detector.py                  # RetinaFace wrapper via InsightFace
└── aligner.py                   # crop + align using landmarks
```

### `src/face/detector.py`
```python
class FaceDetector:
    def __init__(self, model_name='buffalo_l', det_size=(640,640), min_confidence=0.8)
    def detect(self, img: np.ndarray) -> list[FaceResult]
        # returns list sorted by face area desc
        # each FaceResult: bbox, landmarks, confidence
    def detect_best(self, img: np.ndarray) -> FaceResult | None
        # largest face near center; None if none found
```

### `src/face/aligner.py`
```python
def align_face(img: np.ndarray, landmarks: np.ndarray, output_size=(224,224)) -> np.ndarray
    # affine warp using 5-point landmarks
    # returns BGR uint8 crop

def preprocess_for_efficientnet(crop: np.ndarray) -> torch.Tensor
    # normalize to ImageNet mean/std, return [1,3,224,224]

def preprocess_for_arcface(crop: np.ndarray) -> np.ndarray
    # InsightFace expects 112×112 BGR; resize + normalize
```

**Test:** Load one image from `data/raw/lfw/lfw/`, detect + align, save crop to disk. Visual check.

---

## Phase 2 — Data Preprocessing Pipeline

**Goal:** Process raw datasets → labeled 224×224 face crops in `data/processed/deepfake_faces/` + manifest CSV.

### Files to create

```
src/training/
├── __init__.py
├── preprocess.py                # extract + align faces from image datasets
└── build_manifest.py            # create deepfake_manifest.csv
```

### `src/training/preprocess.py`
```python
def process_ffpp(metadata_csv, faces_dir, output_dir, split_ratios=(0.70,0.15,0.15))
    # reads metadata.csv (REAL/FAKE labels)
    # copies already-extracted crops to processed/deepfake_faces/
    # skips if face too small or unreadable

def process_140k(base_dir, output_dir)
    # source: data/raw/celebdf/real_vs_fake/real-vs-fake/
    # already split into train/test — map to our structure

def process_ciplab(base_dir, output_dir, split='val')
    # small dataset — use entirely as val set

def run_retinaface_on_dir(img_dir, output_dir, detector: FaceDetector)
    # for datasets that are NOT pre-cropped (anti-spoofing videos)
    # extract frames → detect face → save crop
```

### `src/training/build_manifest.py`
- Walks `data/processed/deepfake_faces/`
- Writes `data/splits/deepfake_manifest.csv`

**Manifest columns** (from `plan.md §5.3`):
```
sample_id, image_path, label, source_dataset, original_video, subject_id, split, face_confidence, crop_quality_status
```

**Output check:** `data/processed/deepfake_faces/train/real/` and `train/fake/` non-empty. Manifest row count matches file count.

---

## Phase 3 — EfficientNet-B0 Baseline

**Goal:** Train real/fake binary classifier. Save best checkpoint + metrics.

### Files to create

```
src/models/
├── __init__.py
├── efficientnet.py              # model definition
└── lora.py                      # LoRA adapters (Phase 6 only — stub for now)

src/training/
├── dataset.py                   # PyTorch Dataset class
├── augmentation.py              # albumentations pipeline
├── train_baseline.py            # training loop
└── evaluate.py                  # metrics
```

### `src/models/efficientnet.py`
```python
class DeepfakeClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True)
        # torchvision.models.efficientnet_b0(pretrained=True)
        # replace classifier head: Linear(1280 → 1) + Sigmoid
    def forward(self, x) -> torch.Tensor  # shape [B, 1], range [0,1]
    def freeze_backbone(self)
    def unfreeze_last_blocks(self, n=3)
```

### `src/training/dataset.py`
```python
class DeepfakeDataset(Dataset):
    def __init__(self, manifest_csv, split, transform=None)
    def __getitem__(self, idx) -> (tensor, label)
    # label: 0=real, 1=fake
```

### `src/training/augmentation.py`
```python
def get_train_transform() -> A.Compose
    # horizontal flip, brightness/contrast, jpeg compression,
    # gaussian blur, gaussian noise, random resize-crop

def get_val_transform() -> A.Compose
    # resize 224×224, normalize only
```

### `src/training/train_baseline.py`
- Stage 1: freeze backbone → train head only, 5 epochs, lr=1e-3
- Stage 2: unfreeze last 3 blocks → fine-tune, 15 epochs, lr=1e-4
- Early stopping on val AUC (patience=5)
- Save `artifacts/models/efficientnet_b0_baseline_best.pth`
- Save `artifacts/metrics/baseline_training_curves.png`

### `src/training/evaluate.py`
```python
def evaluate_classifier(model, dataloader, threshold=0.5) -> dict
    # returns: accuracy, precision, recall, f1, roc_auc
    # saves confusion matrix PNG to artifacts/metrics/

def plot_roc(fpr, tpr, auc, save_path)
def save_report(metrics: dict, save_path)  # saves baseline_report.json
```

**Run:** `python -m src.training.train_baseline`

**Done when:** `artifacts/models/efficientnet_b0_baseline_best.pth` exists, `baseline_report.json` shows AUC > 0.85.

---

## Phase 4 — ArcFace Enrollment & Verification

**Goal:** Enroll user from images/webcam. Verify identity using cosine similarity.

### Files to create

```
src/enrollment/
├── __init__.py
├── store.py                     # enrollment database (pickle / json)
└── enroll.py                    # enrollment flow

src/models/
└── arcface.py                   # ArcFace wrapper around InsightFace
```

### `src/models/arcface.py`
```python
class ArcFaceExtractor:
    def __init__(self, model_pack='buffalo_l')
        # wraps insightface FaceAnalysis
    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray
        # returns normalized 512-D vector
    def similarity(self, emb1, emb2) -> float
        # cosine similarity, range [-1, 1]
```

### `src/enrollment/store.py`
```python
class EnrollmentStore:
    def __init__(self, db_path='artifacts/models/enrollment_db.pkl')
    def enroll(self, user_id: str, embeddings: list[np.ndarray])
        # averages embeddings → one template per user
    def get_template(self, user_id: str) -> np.ndarray | None
    def list_users(self) -> list[str]
    def delete_user(self, user_id: str)
    def save(self) / def load(self)
```

### `src/enrollment/enroll.py`
- `enroll_from_images(user_id, image_paths, detector, aligner, extractor, store)` — batch enroll from files
- `enroll_from_webcam(user_id, n_samples, detector, aligner, extractor, store)` — interactive webcam capture

**LFW threshold tuning** (`src/training/tune_threshold.py`):
- Load LFW pairs from `data/raw/lfw/pairs.txt`
- Compute cosine similarity for all 6,000 pairs
- Sweep threshold 0.3–0.9, step 0.01
- Report FAR / FRR / EER
- Save `artifacts/metrics/arcface_threshold_report.json`

---

## Phase 5 — Single-Camera End-to-End Pipeline

**Goal:** Given webcam/image input → ACCEPT / REJECT_FAKE / REJECT_IDENTITY / RETRY.

### Files to create

```
src/inference/
├── __init__.py
├── decision.py                  # DecisionEngine
├── pipeline.py                  # single-camera pipeline
└── result.py                    # VerificationResult dataclass

src/input/
├── __init__.py
└── camera.py                    # OpenCV camera wrapper
```

### `src/inference/result.py`
```python
@dataclass
class VerificationResult:
    decision: str          # ACCEPT | REJECT_FAKE | REJECT_IDENTITY | RETRY
    fake_score: float
    match_score: float
    user_id: str | None
    face_detected: bool
    latency_ms: float
    reason: str
```

### `src/inference/decision.py`
```python
class DecisionEngine:
    def __init__(self, fake_threshold=0.5, match_threshold=0.60)
    def decide(self, fake_score, match_score, face_detected) -> VerificationResult
```

### `src/inference/pipeline.py`
```python
class SingleCamPipeline:
    def __init__(self, detector, aligner, deepfake_model, arcface, store, decision_engine)
    def run(self, img: np.ndarray, user_id: str) -> VerificationResult
    def run_live(self, camera_index=0, user_id=str)
        # OpenCV window with live overlay
        # show: fake score, match score, decision
```

**Run:** `python -m src.inference.pipeline --user_id alice --camera 0`

**Done when:** Live demo works. Decision displays in OpenCV window with correct scores.

---

## Phase 6 — LoRA / Adapter Fine-Tuning

**Start only after Phase 3 baseline metrics are saved.**

### Files to create / modify

```
src/models/
└── lora.py                      # LoRA adapter layers

src/training/
└── train_lora.py                # LoRA training loop
```

### `src/models/lora.py`
```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0)
    # W' = W + alpha * (A @ B)  where A=[in,r], B=[r,out]

def inject_lora(model: DeepfakeClassifier, rank=4, target_modules=['fc']) -> DeepfakeClassifier
    # replaces target Linear layers with LoRALinear
    # freezes original weights

def count_trainable_params(model) -> int
```

### `src/training/train_lora.py`
- Load baseline checkpoint
- Inject LoRA adapters
- Train on same manifest, same split
- Save `artifacts/models/efficientnet_b0_lora_best.pth`
- Save `artifacts/metrics/lora_report.json`

### Comparison (`src/training/compare.py`)
- Load baseline + LoRA checkpoints
- Evaluate both on test split
- Print + save `artifacts/metrics/baseline_vs_lora.md`

| Metric | Baseline | LoRA |
|---|---|---|
| Trainable params | | |
| ROC-AUC | | |
| F1 | | |
| Inference ms | | |

---

## Phase 7 — Dual-Camera Capture & Demo

**Start only after Phase 5 single-cam demo is stable.**

### Files to create

```
src/input/
├── dual_camera.py               # two-stream capture + frame pairing
└── frame.py                     # Frame dataclass with timestamp

src/inference/
└── dual_pipeline.py             # dual-camera pipeline + score fusion
```

### `src/input/frame.py`
```python
@dataclass
class Frame:
    img: np.ndarray
    timestamp_ms: float
    camera_id: int              # 0=left, 1=right
    frame_index: int
```

### `src/input/dual_camera.py`
```python
class DualCameraCapture:
    def __init__(self, left_idx=0, right_idx=1, sync_delta_ms=50)
    def start(self)              # starts two threads
    def stop(self)
    def get_pair(self) -> tuple[Frame, Frame] | None
        # returns (left, right) pair if sync_delta <= threshold
        # returns None if no valid pair available
    def save_pair(self, left, right, metadata: dict, output_dir: str)
```

### `src/inference/dual_pipeline.py`
```python
class DualCamPipeline:
    def __init__(self, single_pipeline: SingleCamPipeline, fusion='mean_fake_max_match')
    def run(self, left_frame: Frame, right_frame: Frame, user_id: str) -> VerificationResult
        # runs SingleCamPipeline on both frames
        # fuses scores per configs/pipeline.yaml
        # returns single VerificationResult with fused scores
    def run_live(self, user_id: str)
        # side-by-side OpenCV window
        # shows: left cam | right cam
        # overlays: face boxes, fake scores, match scores, fused decision
```

**Run:** `python -m src.inference.dual_pipeline --user_id alice`

---

## Phase 8 — Evaluation Scripts

### Files to create

```
src/training/
├── eval_deepfake.py             # full EfficientNet eval report
└── eval_verification.py        # ArcFace threshold sweep + FAR/FRR
```

### `src/training/eval_deepfake.py`
- Loads checkpoint (baseline or LoRA, flag `--model`)
- Runs on test split of manifest
- Outputs: accuracy, precision, recall, F1, ROC-AUC, confusion matrix PNG, report JSON
- Saves to `artifacts/metrics/`

### `src/training/eval_verification.py`
- Loads LFW pairs
- Extracts ArcFace embeddings for all 13,233 LFW images
- Computes 6,000 pair similarities
- Sweeps threshold → FAR, FRR, EER curve
- Saves: `arcface_threshold_sweep.png`, `arcface_threshold_report.json`

---

## File Map (final `src/` tree)

```
src/
├── __init__.py
├── utils/
│   ├── config.py
│   └── logger.py
├── face/
│   ├── detector.py              # FaceDetector (RetinaFace)
│   └── aligner.py               # align_face, preprocess_*
├── models/
│   ├── efficientnet.py          # DeepfakeClassifier
│   ├── arcface.py               # ArcFaceExtractor
│   └── lora.py                  # LoRALinear, inject_lora
├── training/
│   ├── dataset.py               # DeepfakeDataset
│   ├── augmentation.py          # train/val transforms
│   ├── preprocess.py            # raw → processed crops
│   ├── build_manifest.py        # write deepfake_manifest.csv
│   ├── train_baseline.py        # EfficientNet baseline training
│   ├── train_lora.py            # LoRA fine-tuning
│   ├── evaluate.py              # metrics helpers
│   ├── eval_deepfake.py         # full eval report
│   ├── eval_verification.py     # ArcFace threshold sweep
│   ├── tune_threshold.py        # LFW-based threshold tuning
│   └── compare.py               # baseline vs LoRA table
├── enrollment/
│   ├── store.py                 # EnrollmentStore
│   └── enroll.py                # enroll_from_images, enroll_from_webcam
├── input/
│   ├── camera.py                # OpenCV camera wrapper
│   ├── frame.py                 # Frame dataclass
│   └── dual_camera.py           # DualCameraCapture
└── inference/
    ├── result.py                # VerificationResult dataclass
    ├── decision.py              # DecisionEngine
    ├── pipeline.py              # SingleCamPipeline
    └── dual_pipeline.py         # DualCamPipeline
```

---

## Phase Order Summary

| Phase | What | Depends On |
|---|---|---|
| 0 | Config + logger | nothing |
| 1 | FaceDetector + FaceAligner | Phase 0 |
| 2 | Preprocessing + manifest | Phase 1 |
| 3 | EfficientNet baseline training | Phase 2 |
| 4 | ArcFace enrollment + LFW tuning | Phase 1 |
| 5 | Single-cam end-to-end demo | Phase 3 + 4 |
| 6 | LoRA adapters | Phase 3 done + metrics saved |
| 7 | Dual-camera capture + demo | Phase 5 stable |
| 8 | Evaluation scripts | Phase 3 + 4 |

---

## Coding Rules

- No global state. Pass config dict or dataclass explicitly.
- Every class gets `__init__` + at most 3 public methods. No fat classes.
- No `print()` in library code. Use logger.
- No hardcoded paths. All paths come from `configs/dataset.yaml`.
- Type hints on all function signatures.
- Each phase gets one smoke test: run script on 5 images, check output shape/file exists.
- Checkpoints saved with metadata dict: `{'epoch', 'val_auc', 'model_state', 'config'}`.
- No notebooks in `src/`. Notebooks go in `notebooks/` if needed for exploration only.
