# Dual-Camera Face Verification Project — Technical Deep Dive

This document is a complete, implementation-grounded explanation of how the project works end-to-end.

It covers:
- System architecture and module responsibilities
- End-to-end pipeline (single camera + dual camera)
- All math used in detection, depth, classification, and verification
- Why each model was selected
- How training and evaluation are done
- Exact meaning of generated files in `artifacts/metrics` and `artifacts/models`
- Meaning of LFW files and `.pkl` files

---

## 1) Project objective and security design

This project implements a **multi-stage identity verification system** that combines:
1. **Face detection + alignment** (RetinaFace via InsightFace)
2. **Deepfake / real-face classification** (EfficientNet-B0)
3. **Identity verification** (ArcFace embedding similarity)
4. **Stereo liveness consistency** (dual camera disparity-based geometry check)

Security philosophy: **gated zero-trust pipeline**.
A sample is accepted only if all required checks pass.

---

## 2) Codebase architecture (what each folder does)

## `src/face/`
- `detector.py`: RetinaFace wrapper (`FaceDetector`) and face selection policy
- `aligner.py`: 5-point landmark alignment, preprocessing for EfficientNet/ArcFace
- `depth.py`: stereo disparity-based depth variance estimation (anti-flat-spoof logic)

## `src/models/`
- `efficientnet.py`: EfficientNet-B0 binary real/fake classifier head
- `lora.py`: LoRA adapter modules (`LoRALinear`, optional `LoRAConv2d`, injection utilities)
- `arcface.py`: ArcFace embedding extraction + cosine similarity

## `src/inference/`
- `pipeline.py`: single-camera runtime pipeline
- `dual_pipeline.py`: dual-camera fusion and stereo override logic
- `decision.py`: final rule engine (`ACCEPT`, `REJECT_FAKE`, `REJECT_IDENTITY`, `RETRY`)
- `result.py`: output schema (`VerificationResult`)

## `src/training/`
- `preprocess.py`: dataset prep and manifest creation
- `dataset.py`: manifest-backed PyTorch dataset + weighted sampler
- `train_baseline.py`: baseline EfficientNet training
- `train_lora.py`: LoRA fine-tuning
- `evaluate.py`: metric computation + all plots + report JSONs
- `eval_deepfake.py`: evaluate deepfake classifier checkpoint
- `eval_verification.py`: LFW threshold tuning for ArcFace
- `compare.py`: baseline vs LoRA evaluation summary

## `src/enrollment/`
- `enroll.py`: user enrollment from images/webcam
- `store.py`: serialized enrollment DB (`enrollment_db.pkl`)

## `src/input/`
- `camera.py`: single camera wrapper
- `dual_camera.py`: dual stream threads, frame timestamp pairing
- `frame.py`: frame dataclass

---

## 3) Runtime pipeline (step by step)

## A) Single-camera pipeline (`src/inference/pipeline.py`)

For each frame:
1. Detect best face (`FaceDetector.detect_best`)
2. Align face using 5 landmarks (`align_face`) to `224x224`
3. Run deepfake model and compute:
   - `fake_score = sigmoid(logit)`
4. If `user_id` is provided:
   - fetch template from `EnrollmentStore`
   - compute ArcFace similarity (`match_score`)
5. Pass all scores to `DecisionEngine.decide`
6. Return structured result (`VerificationResult`)

If no face is detected, decision becomes `RETRY` with reason `no_face`.

## B) Dual-camera pipeline (`src/inference/dual_pipeline.py`)

For paired left/right frames:
1. Detect best face in both views
2. Estimate stereo depth variance (`estimate_stereo_depth`)
3. Run single-pipeline logic independently on left and right
4. Fuse scores:
   - fake score fusion = mean or max (configurable)
   - match score fusion = max or mean (configurable)
5. Depth override logic:
   - if both faces are detected and `0 < depth_variance < 1.0`, treat as likely flat spoof
   - force high fake score (`>=0.99`) before final decision
6. Decision engine outputs final label and reason

This creates a **hybrid anti-spoof gate**: geometry + learned deepfake signal.

---

## 4) Decision logic and states

Implemented in `src/inference/decision.py`.

Decision order:
1. No face detected -> `RETRY`
2. Face confidence below threshold -> `RETRY`
3. Face size below threshold -> `RETRY`
4. `fake_score > fake_threshold` -> `REJECT_FAKE`
5. No enrollment template -> `REJECT_IDENTITY`
6. `match_score >= match_threshold` -> `ACCEPT`
7. Otherwise -> `REJECT_IDENTITY`

Configured defaults (`configs/pipeline.yaml`):
- `fake_threshold = 0.5`
- `match_threshold = 0.60`
- `min_face_confidence = 0.8`
- `min_face_size = 40`
- dual sync tolerance: `50 ms`

---

## 5) Math used in this project

## 5.1 EfficientNet classification probability
Deepfake classifier outputs logits `z`. Probability of fake:

`p(fake) = sigmoid(z) = 1/(1 + exp(-z))`

Loss used for training:
- Binary cross-entropy with logits (`BCEWithLogitsLoss`)

Decision threshold:
- predict fake if `p(fake) >= threshold`.

## 5.2 ArcFace similarity
ArcFace embeddings are L2-normalized vectors (`512-D`).
Similarity:

`s = cos(theta) = (e1 · e2) / (||e1|| ||e2||)`

With normalized embeddings, this is equivalent to dot product in `[-1,1]`.
Higher means more likely same identity.

## 5.3 Face alignment (5-point affine)
Given detected 5 landmarks and canonical ArcFace template landmarks, affine transform `A` is estimated via `cv2.estimateAffinePartial2D`, then warp is applied.

Purpose:
- normalize pose/scale/rotation
- reduce nuisance variation before classification/verification

## 5.4 Stereo geometry (implemented practical variant)
In `src/face/depth.py`:
1. Use dense 106 landmarks from both cameras
2. Estimate fundamental matrix `F` with RANSAC
3. Perform uncalibrated rectification (`stereoRectifyUncalibrated`)
4. Compute disparity with SGBM
5. Extract disparity inside face ROI
6. Compute face disparity standard deviation

Interpretation used in code:
- very low disparity variance -> flatter surface -> likely replay/print spoof
- higher variance -> more 3D structure -> likely live face

Note: This implementation uses **depth variance as a geometric liveness heuristic** rather than direct metric depth reconstruction.

## 5.5 Evaluation metrics formulas
Given confusion counts (`TP`, `TN`, `FP`, `FN`):

- Accuracy = `(TP + TN) / (TP + TN + FP + FN)`
- Balanced Accuracy = `(TPR + TNR)/2`
- Precision = `TP / (TP + FP)`
- Recall / TPR = `TP / (TP + FN)`
- Specificity / TNR = `TN / (TN + FP)`
- F1 = `2 * (Precision * Recall)/(Precision + Recall)`
- FPR = `FP / (FP + TN)`
- FNR = `FN / (FN + TP)`

Curve metrics:
- ROC-AUC: ranking quality over thresholds
- PR-AUC (AP): precision-recall area
- DET: plots FPR vs FNR (log-log in this code)

Calibration:
- compares predicted probabilities with empirical positive frequency per bin

Latency:
- per-image latency from batched inference timing
- mean / p50 / p95 reported

---

## 6) Why these models were selected (rationale)

## RetinaFace (via InsightFace `buffalo_l`)
Why:
- robust face detection in unconstrained settings
- gives both 5 landmarks and dense landmarks (106 in this model pack)
- also exposes embeddings through the same package integration path

Tradeoff:
- heavier than Haar/HOG, but far more robust and practical for real security pipelines

## EfficientNet-B0 for deepfake classification
Why:
- strong accuracy/efficiency tradeoff
- suitable for edge/real-time constraints
- widely available, stable pretrained weights
- easy to fine-tune and compare with LoRA variants

Tradeoff:
- not the absolute largest SOTA, but much lighter and easier to deploy

## ArcFace embeddings (`buffalo_l`)
Why:
- standard, strong face verification embeddings
- cosine-threshold verification is simple and production-friendly
- strong benchmark behavior on LFW-style pair verification

Tradeoff:
- threshold must be tuned per deployment and operating condition

## LoRA for parameter-efficient adaptation
Why:
- reduce trainable parameters drastically while keeping pretrained backbone frozen
- enables lighter adaptation and potentially faster retraining cycles
- easier checkpoint portability for adapter-style updates

In this workspace, reported LoRA trainable params: **5,124**.

---

## 7) Training workflow (how the project is built)

## Baseline training (`train_baseline.py`)
Two-stage strategy:
1. Freeze backbone, train classification head
2. Unfreeze last N blocks (default 3), fine-tune

Core details:
- optimizer: AdamW
- schedulers:
  - stage 1: CosineAnnealingLR
  - stage 2: OneCycleLR
- early stopping on validation AUC
- weighted sampler to mitigate class imbalance
- best checkpoint by validation AUC

Main outputs:
- best and last checkpoints
- training history CSV and curves
- full test metrics/plots/report JSON

## LoRA fine-tuning (`train_lora.py`)
1. Load baseline checkpoint
2. Inject LoRA adapters into selected modules (default target `classifier.1`)
3. Train only LoRA params
4. Save best LoRA checkpoint by val AUC
5. Evaluate on test split and generate full metric artifacts

Checkpoint includes:
- `base_model_state`
- `model_state` (LoRA-injected model)
- `lora_config`
- `val_metrics`

---

## 8) Evaluation workflow

## Deepfake evaluation (`eval_deepfake.py` + `evaluate.py`)
- Loads manifest split (train/val/test)
- Predicts probabilities
- Computes scalar metrics at threshold (usually 0.5)
- Finds best-F1 threshold by sweep
- Saves plots and report JSON

## ArcFace threshold tuning (`eval_verification.py`)
- Loads LFW pairs (`pairs.txt`)
- Computes/caches embeddings for images
- Computes pair cosine similarities
- Sweeps thresholds (0.20 to 0.89 step 0.01)
- Reports FAR/FRR/accuracy curves
- Computes EER and EER threshold

---

## 9) Artifact catalog — what every generated file means

Below is a direct interpretation of files currently in `artifacts/metrics`.

## 9.1 Deepfake classifier artifacts (baseline)

- `deepfake_test_report.json`
  - Main test report at threshold 0.5
  - Contains `metrics_at_threshold`, `best_f1_threshold`, `latency`

- `deepfake_test_confusion_matrix.png`
  - Absolute confusion matrix at current threshold

- `deepfake_test_confusion_matrix_pct.png`
  - Row-normalized confusion matrix (%)

- `deepfake_test_roc.png`
  - ROC curve (`TPR` vs `FPR`), includes ROC-AUC

- `deepfake_test_pr.png`
  - Precision-Recall curve, includes AP/PR-AUC

- `deepfake_test_det.png`
  - DET curve (`FPR` vs `FNR`) on log scales

- `deepfake_test_score_dist.png`
  - Histogram of predicted fake probabilities for real vs fake classes
  - Better separation means easier thresholding

- `deepfake_test_threshold_sweep.png`
  - Accuracy and F1 as threshold changes
  - Helps choose operational cutoff

- `deepfake_test_calibration.png`
  - Probability calibration plot vs ideal diagonal

- `deepfake_test_latency_dist.png`
  - Histogram of per-image inference latency
  - mean and p95 lines shown

## 9.2 LoRA classifier artifacts

Equivalent files for LoRA model:
- `lora_test_report.json`
- `lora_test_confusion_matrix.png`
- `lora_test_confusion_matrix_pct.png`
- `lora_test_roc.png`
- `lora_test_pr.png`
- `lora_test_det.png`
- `lora_test_score_dist.png`
- `lora_test_threshold_sweep.png`
- `lora_test_calibration.png`
- `lora_test_latency_dist.png`

Additional LoRA summary files:
- `lora_report.json`
  - high-level LoRA run report
  - includes best val AUC, checkpoint path, trainable params, full test metrics, and epoch history

- `lora_training_history.csv`
  - epoch-wise training/validation metrics (loss, AUC, F1, time)

## 9.3 ArcFace verification artifacts

- `arcface_threshold_report.json`
  - LFW pair verification summary:
    - pairs total/evaluated/missing
    - ROC-AUC
    - EER + EER threshold
    - best-accuracy threshold row
    - default threshold row (0.60)
    - full threshold sweep entries

- `arcface_threshold_sweep.png`
  - FAR, FRR, and accuracy vs cosine threshold
  - includes EER threshold marker

- `lfw_arcface_embeddings.pkl`
  - cache map: `image_path -> 512-D embedding`
  - avoids recomputing embeddings for every threshold tuning run

## 9.4 Stereo depth artifact

- `latest_stereo_depth.png`
  - debug image produced by `estimate_stereo_depth`
  - normalized, color-mapped face disparity patch
  - overlays computed depth variance value

---

## 10) Model artifact catalog (`artifacts/models`)

- `efficientnet_b0_baseline_best.pth`
  - best baseline checkpoint selected by validation AUC

- `efficientnet_b0_baseline_last.pth`
  - last baseline epoch checkpoint

- `efficientnet_b0_lora_best.pth`
  - best LoRA checkpoint
  - contains base model state + lora config + lora-adapted weights

- `enrollment_db.pkl`
  - user enrollment database
  - stores per-user template embedding + metadata

- `smoke_*.pth`, `*_smoke_*`
  - smoke-test checkpoints from reduced or quick runs
  - useful for pipeline sanity checks, not final performance reporting

---

## 11) LFW files and `.pkl` files explained

## LFW raw files

- `data/raw/lfw/lfw/`
  - directory of identity subfolders and images

- `data/raw/lfw/pairs.txt`
  - pair protocol for verification benchmark
  - each row defines a positive or negative pair

## `lfw_arcface_embeddings.pkl`

Purpose:
- cache image embeddings from ArcFace extraction
- massively speeds up repeated threshold tuning

Structure:
- Python dict, key = absolute image path string
- value = `np.ndarray` shape `(512,)`, normalized embedding

Current snapshot summary (from local inspection):
- entries: `7690`
- sample vector length: `512`

## `enrollment_db.pkl`

Purpose:
- persistent enrollment templates for runtime verification

Structure:
- dict keyed by `user_id`
- each value contains:
  - `user_id`
  - `embedding_template` (normalized 512-D)
  - `num_samples`
  - `created_at`
  - `model_name`
  - `model_version`

Current snapshot includes 4 users (example IDs found locally).

---

## 12) Current metric snapshot (from generated reports)

## Baseline deepfake model (`deepfake_test_report.json`)
At threshold `0.5`:
- Accuracy: `0.96945`
- Precision: `0.97818`
- Recall: `0.97388`
- Specificity: `0.96165`
- F1: `0.97602`
- ROC-AUC: `0.99598`
- PR-AUC: `0.99767`
- FPR: `0.03835`
- FNR: `0.02612`
- Latency mean/p50/p95 ms: `0.4318 / 0.3669 / 0.5004`

Best-F1 threshold row:
- threshold: `0.32`
- F1: `0.97695`

## LoRA deepfake model (`lora_test_report.json`)
At threshold `0.5`:
- Accuracy: `0.96960`
- Precision: `0.97932`
- Recall: `0.97292`
- Specificity: `0.96373`
- F1: `0.97611`
- ROC-AUC: `0.99593`
- PR-AUC: `0.99764`
- FPR: `0.03627`
- FNR: `0.02708`
- Latency mean/p50/p95 ms: `0.3480 / 0.3145 / 0.3297`

Best-F1 threshold row:
- threshold: `0.31`
- F1: `0.97690`

## ArcFace LFW threshold report (`arcface_threshold_report.json`)
- pairs total: `6000`
- pairs evaluated: `5983`
- missing embeddings: `17`
- ROC-AUC: `0.97884`
- EER: `0.04897`
- EER threshold: `0.09888`
- best-accuracy threshold (in sweep): `~0.25`

Important note:
- The configured runtime default `match_threshold=0.60` is conservative.
- LFW sweep indicates that deployment threshold should be chosen by application risk policy, not fixed blindly.

---

## 13) End-to-end data flow summary

1. **Input acquisition**
   - Single camera or dual camera frames
   - In dual mode, pair frames by timestamp (<= 50 ms)

2. **Face detection + landmarks**
   - RetinaFace via InsightFace `buffalo_l`
   - choose best face by size/center scoring

3. **Alignment + preprocessing**
   - landmark-based affine alignment
   - preprocess for EfficientNet and ArcFace

4. **Deepfake classification**
   - EfficientNet-B0 (baseline or LoRA)
   - fake probability from sigmoid(logit)

5. **Identity verification**
   - ArcFace embedding cosine similarity vs enrolled template

6. **Stereo liveness geometry (dual only)**
   - depth variance via disparity map inside face region
   - optional spoof penalty if variance is too flat

7. **Decision engine**
   - gate checks in strict order
   - outputs structured reason code and scores

8. **Artifacts and logs**
   - plots, reports, checkpoints, enrollment DB, embedding caches

---

## 14) Practical interpretation of outputs

- If `fake_score` is high but `match_score` is high too:
  - system should still reject (`REJECT_FAKE`) due to stage ordering
- If deepfake passes but identity fails:
  - result is `REJECT_IDENTITY`
- If face detection quality is low:
  - `RETRY` (not immediate rejection)

This separation gives better diagnostics and safer runtime behavior.

---

## 15) Known implementation details worth understanding

1. **Dual depth gate is heuristic-based**
   - uses disparity variance threshold (`<1.0`) to flag flat spoof
   - threshold may need environment-specific tuning

2. **ArcFace threshold in config (0.60) is strict**
   - LFW sweep suggests alternate tradeoff points

3. **LoRA checkpoint is adapter-dependent**
   - requires loading with matching base model and LoRA config

4. **Evaluation artifacts depend on `artifact_prefix`**
   - filenames are generated systematically in `evaluate.py`

---

## 16) File-to-code mapping (quick lookup)

- Metric formulas + plotting: `src/training/evaluate.py`
- Deepfake eval entrypoint: `src/training/eval_deepfake.py`
- ArcFace LFW eval: `src/training/eval_verification.py`
- Baseline training: `src/training/train_baseline.py`
- LoRA training: `src/training/train_lora.py`
- Single runtime inference: `src/inference/pipeline.py`
- Dual runtime inference: `src/inference/dual_pipeline.py`
- Decision rules: `src/inference/decision.py`
- Stereo depth anti-spoof logic: `src/face/depth.py`
- Enrollment DB format: `src/enrollment/store.py`

---

## 17) If you want this even more thesis-ready

Recommended next additions:
1. Add **per-condition tables** (low light/backlight/motion/occlusion) from measured runs
2. Add **ablation numbers** (without stereo, without deepfake, without ArcFace)
3. Add **confidence intervals** from repeated runs
4. Add one page with **failure-case images** linked to reason codes
5. Add a **versioned experiment manifest** (config hash -> artifact bundle)

---

This document was generated from actual code and current artifact outputs in this workspace, not generic templates.
