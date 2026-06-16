# How to Run, Prepare Data, Train, and Evaluate

This project implements a dual-camera face ID verification system with:

- **RetinaFace / InsightFace `buffalo_l`** for face detection and landmarks
- **EfficientNet-B0 ImageNet-pretrained** for real/fake deepfake classification
- **ArcFace / InsightFace `buffalo_l`** for identity embedding verification
- **Optional LoRA adapters** for parameter-efficient fine-tuning
- **Single-camera and dual-camera runtime demos**

---

## 1. Dataset status

The raw datasets were checked against `DATASETS.md` and `configs/dataset.yaml`.

Current inventory found:

| Dataset | Expected path | Current status |
|---|---|---|
| FF++ faces | `data/raw/faceforensicspp/faces_224/` | Present: 95,634 images |
| FF++ labels | `data/raw/faceforensicspp/metadata.csv` | Present: 95,634 rows; 16,293 real / 79,341 fake |
| 140k real/fake | `data/raw/celebdf/real_vs_fake/real-vs-fake/` | Present: 140,000 images with train/valid/test splits |
| ciplab real/fake | `data/raw/celebdf/real_and_fake_face/` | Present: 2,041 images |
| Anti-spoofing | `data/raw/custom/single_cam/` | Present: 36 videos + 9 images |
| LFW | `data/raw/lfw/lfw/` | Present: 13,233 images |
| LFW pairs | `data/raw/lfw/pairs.txt` | Present: 6,001 non-empty lines |

The raw dataset is complete. I also ran symlink-based preprocessing in this workspace, so the generated training artifacts are now present:

- `data/processed/deepfake_faces/` with 237,675 processed/symlinked images
- `data/splits/deepfake_manifest.csv` with 237,675 rows

If these generated artifacts are deleted later, recreate them with the preprocessing command below.

---

## 2. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If InsightFace or EfficientNet weights are not already cached, they will download on first use. You can pre-download them:

```bash
python -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1)"
python -c "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)"
```

Use `ctx_id=-1` for CPU and `ctx_id=0` for GPU where supported by ONNX Runtime.

---

## 3. Check dataset completeness

```bash
python -m src.training.check_dataset
```

Output is also saved to:

```text
artifacts/metrics/dataset_check_report.json
```

Important fields:

- `status.raw_dataset_complete`: should be `true`
- `status.processed_dataset_ready`: becomes `true` after preprocessing

---

## 4. Prepare the processed training dataset

The downloaded FF++, 140k, and ciplab datasets are already face crops, so preprocessing creates a reproducible manifest and symlinks into the final split layout instead of duplicating hundreds of thousands of files.

Recommended full command:

```bash
python -m src.training.preprocess --link-mode symlink
```

Optional slower but stricter check that opens every image:

```bash
python -m src.training.preprocess --link-mode symlink --verify-images
```

Optional anti-spoofing crop extraction using RetinaFace:

```bash
python -m src.training.preprocess --link-mode symlink --include-antispoof
```

Debug/smoke-test preprocessing:

```bash
python -m src.training.preprocess --link-mode symlink --max-samples 100
```

Generated layout:

```text
data/processed/deepfake_faces/train/real/
data/processed/deepfake_faces/train/fake/
data/processed/deepfake_faces/val/real/
data/processed/deepfake_faces/val/fake/
data/processed/deepfake_faces/test/real/
data/processed/deepfake_faces/test/fake/
data/splits/deepfake_manifest.csv
```

Manifest columns:

```text
sample_id,image_path,label,source_dataset,original_video,subject_id,split,face_confidence,crop_quality_status
```

Labels:

- `0` = real
- `1` = fake

---

## 5. Train EfficientNet-B0 baseline

Full training follows the pre-decided pipeline:

1. Stage 1: freeze EfficientNet backbone and train only the classifier head.
2. Stage 2: unfreeze the last EfficientNet blocks and fine-tune.
3. Save best checkpoint by validation ROC-AUC.
4. Evaluate on test split and save all metrics.

Run:

```bash
python -m src.training.train_baseline
```

Useful GPU command (NVIDIA):

```bash
python -m src.training.train_baseline \
    --device cuda \
    --batch-size 64 \
    --num-workers 8 \
    --epochs-stage1 5 \
    --epochs-stage2 15
```

Apple Silicon (M1/M2/M3 Pro) optimized command:

*Note: The codebase now natively detects your M3 Pro and activates the `mps` (Metal Performance Shaders) backend for PyTorch. It also automatically hardware-accelerates InsightFace via the `CoreMLExecutionProvider`. This guarantees maximum performance without passing complicated flags.*

```bash
python -m src.training.train_baseline \
    --device auto \
    --batch-size 32 \
    --num-workers 4 \
    --epochs-stage1 5 \
    --epochs-stage2 15
```
*(The learning rate scheduler bug has been fixed, so `--epochs-stage1 5` will now correctly decay the learning rate smoothly over exactly 5 epochs before unfreezing the backbone).*

Quick smoke training:

```bash
python -m src.training.train_baseline --run-name smoke --checkpoint artifacts/models/efficientnet_b0_smoke_best.pth --no-pretrained --max-train-samples 1000 --max-val-samples 300 --max-test-samples 300 --epochs-stage1 1 --epochs-stage2 1 --batch-size 16
```

Baseline outputs:

```text
artifacts/models/efficientnet_b0_baseline_best.pth
artifacts/models/efficientnet_b0_baseline_last.pth
artifacts/metrics/baseline_training_history.csv
artifacts/metrics/baseline_training_curves.png
artifacts/metrics/baseline_report.json
artifacts/metrics/baseline_test_report.json
artifacts/metrics/baseline_test_confusion_matrix.png
artifacts/metrics/baseline_test_roc.png
artifacts/metrics/baseline_test_pr.png
```

Metrics reported include:

- Accuracy
- Balanced accuracy
- Precision
- Recall / TPR
- Specificity / TNR
- F1-score
- ROC-AUC
- PR-AUC
- False positive rate
- False negative rate
- False positives / false negatives / true positives / true negatives
- Best-F1 threshold sweep
- Mean / p50 / p95 inference latency per image

---

## 6. Evaluate a trained checkpoint

```bash
python -m src.training.eval_deepfake --split test --threshold 0.5
```

With a custom checkpoint:

```bash
python -m src.training.eval_deepfake --checkpoint artifacts/models/efficientnet_b0_baseline_best.pth --split test
```

---

## 7. Tune ArcFace verification threshold on LFW

```bash
python -m src.training.eval_verification
```

Outputs:

```text
artifacts/metrics/arcface_threshold_report.json
artifacts/metrics/arcface_threshold_sweep.png
artifacts/metrics/lfw_arcface_embeddings.pkl
```

Metrics reported:

- LFW verification ROC-AUC
- EER
- EER threshold
- Best accuracy threshold
- FAR / false accept rate across threshold sweep
- FRR / false reject rate across threshold sweep
- Accuracy across threshold sweep

---

## 8. LoRA fine-tuning and comparison

After baseline training finishes:

```bash
python -m src.training.train_lora --base-checkpoint artifacts/models/efficientnet_b0_baseline_best.pth
```

Compare baseline vs LoRA:

```bash
python -m src.training.compare
```

Outputs:

```text
artifacts/models/efficientnet_b0_lora_best.pth
artifacts/metrics/lora_report.json
artifacts/metrics/baseline_vs_lora.md
```

---

## 9. Enroll a user

Enroll from images:

```bash
python -m src.enrollment.enroll --user-id alice --images path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg
```

Enroll interactively from webcam:

```bash
python -m src.enrollment.enroll --user-id alice --webcam --camera 0 --samples 10
```

Enrollment DB:

```text
artifacts/models/enrollment_db.pkl
```

---

## 10. Single-camera verification demo

Image input:

```bash
python -m src.inference.pipeline --user-id alice --image path/to/test.jpg
```

Webcam demo:

```bash
python -m src.inference.pipeline --user-id alice --camera 0
```

Decision values:

- `ACCEPT`: real face and identity matched
- `REJECT_FAKE`: fake/deepfake score above threshold
- `REJECT_IDENTITY`: real face but identity mismatch or no enrollment
- `RETRY`: no face, low confidence, or too-small face

---

## 11. Dual-camera verification demo

Connect two webcams and run:

```bash
python -m src.inference.dual_pipeline --user-id alice --left-camera 0 --right-camera 1
```

Configured in `configs/pipeline.yaml`:

```yaml
dual_camera:
  sync_delta_ms: 50
  fake_threshold: 0.5
  match_threshold: 0.60
  fusion:
    fake_score: mean
    match_score: max
  fallback_single_view: true
```

---

## 12. Recommended final project run order

```bash
python -m src.training.check_dataset
python -m src.training.preprocess --link-mode symlink
python -m src.training.train_baseline --device auto
python -m src.training.eval_deepfake --split test
python -m src.training.eval_verification
python -m src.training.train_lora
python -m src.training.compare
python -m src.enrollment.enroll --user-id alice --webcam --samples 10
python -m src.inference.pipeline --user-id alice --camera 0
python -m src.inference.dual_pipeline --user-id alice --left-camera 0 --right-camera 1
```

Full training and ArcFace threshold tuning can take a long time depending on CPU/GPU. Use the smoke commands first to verify the pipeline, then run the full commands for final metrics.
