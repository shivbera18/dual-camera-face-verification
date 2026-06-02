# Architecture Reference: RetinaFace + EfficientNet + ArcFace System

This document describes the basic architecture, runtime modules, storage layout, and two-camera demo design. The main execution plan is in `plan.md`.

---

## 1) High-Level Architecture

The system has five main layers:

1. **Input Layer**
   - Single camera input for baseline development
   - Video/image file input for training and testing
   - Two-webcam input for final demo extension

2. **Face Processing Layer**
   - RetinaFace face detection
   - Facial landmark extraction
   - Face alignment and cropping

3. **Model Layer**
   - EfficientNet-B0 for real/fake classification
   - ArcFace for identity embedding extraction
   - Optional LoRA/adapters for EfficientNet fine-tuning

4. **Decision Layer**
   - Deepfake threshold check
   - Identity similarity check
   - Optional two-view score fusion
   - Final decision: `ACCEPT`, `REJECT`, or `RETRY`

5. **Storage and Logging Layer**
   - Dataset manifests
   - Processed face crops
   - Model checkpoints
   - Enrollment embeddings
   - Verification attempt logs

---

## 2) Baseline Runtime Flow

Single-camera baseline flow:

1. Input frame is captured from camera/video/image.
2. RetinaFace detects face bounding box and landmarks.
3. Face is aligned and cropped.
4. EfficientNet-B0 predicts `fake_probability`.
5. If `fake_probability` is above threshold, reject immediately.
6. If face is real, ArcFace extracts a 512-D embedding.
7. Embedding is compared with enrolled templates using cosine similarity.
8. Decision module returns result with scores and reason.

Suggested decision values:
- `T_fake`: start with `0.50`, tune using validation ROC curve
- `T_match`: start with `0.60`, tune using LFW/custom verification data

---

## 3) Two-Webcam Demo Runtime Flow

The two-camera mode is a final demo extension. It does not replace the baseline; it wraps the same pipeline around two synchronized views.

1. Open two webcam streams:
   - `camera_left`: index `0`
   - `camera_right`: index `1`
2. Start one capture thread per camera.
3. Each captured frame receives:
   - timestamp
   - camera id
   - frame index
4. Pair the nearest left/right frames using timestamp difference.
5. Accept a pair only if sync difference is below the configured threshold, e.g. `50 ms`.
6. Run the baseline face pipeline independently on both frames.
7. Fuse scores:
   - `fake_probability_fused = mean(left_fake_probability, right_fake_probability)`
   - `match_similarity_fused = max(left_similarity, right_similarity)` initially
8. Final decision uses fused scores.

Recommended dual-camera decision rules:
- If face is missing in both views: `RETRY`
- If face is present in one view only: allow fallback in demo mode, but log warning
- If both views are valid and fused fake score is high: `REJECT`
- If fused fake score is low and match score is above threshold: `ACCEPT`

---

## 4) Two-Camera Capture and Storage Design

## 4.1 Capture modes

Use three capture modes:

1. **Enrollment capture**
   - Capture real user faces only
   - Save multiple image pairs per user
   - Use for ArcFace template creation

2. **Verification capture**
   - Capture live authentication attempts
   - Save only metadata by default
   - Save images only when debugging or collecting test data

3. **Dataset collection capture**
   - Capture real and fake samples for evaluation
   - Save both camera images and metadata
   - Used for final report and demo testing

## 4.2 Storage layout for dual-camera captures

Use this layout:

- `data/raw/custom/dual_cam/enrollment/<user_id>/<session_id>/left/`
- `data/raw/custom/dual_cam/enrollment/<user_id>/<session_id>/right/`
- `data/raw/custom/dual_cam/evaluation/real/<subject_id>/<session_id>/left/`
- `data/raw/custom/dual_cam/evaluation/real/<subject_id>/<session_id>/right/`
- `data/raw/custom/dual_cam/evaluation/fake/<attack_type>/<session_id>/left/`
- `data/raw/custom/dual_cam/evaluation/fake/<attack_type>/<session_id>/right/`
- `data/raw/custom/dual_cam/metadata/`

## 4.3 File naming convention

Use consistent names:

- Left image: `<session_id>_<pair_index>_left_<timestamp>.jpg`
- Right image: `<session_id>_<pair_index>_right_<timestamp>.jpg`
- Pair metadata: `<session_id>_<pair_index>.json`

Example fields in metadata:
- `session_id`
- `pair_index`
- `user_id` or `subject_id`
- `label`: `real` or `fake`
- `attack_type`: `none`, `printed_photo`, `screen_replay`, `deepfake_replay`
- `left_image_path`
- `right_image_path`
- `left_timestamp`
- `right_timestamp`
- `sync_delta_ms`
- `camera_left_index`
- `camera_right_index`
- `lighting_condition`
- `distance_cm`
- `notes`

---

## 5) Module Responsibilities

## 5.1 `InputManager`
Responsibilities:
- Load image/video/camera frames
- Support single-camera mode
- Support dual-camera mode
- Attach timestamps and frame IDs

## 5.2 `DualCameraCapture`
Responsibilities:
- Start/stop two camera streams
- Capture frames in parallel
- Pair frames by nearest timestamp
- Drop stale or badly synchronized frames

## 5.3 `FaceDetector`
Responsibilities:
- Run RetinaFace
- Select the best face when multiple faces are present
- Return bounding box, landmarks, confidence

Face selection rule:
- Prefer largest face near frame center
- Reject faces with confidence below threshold

## 5.4 `FaceAligner`
Responsibilities:
- Align face using RetinaFace landmarks
- Produce `224x224` crop for EfficientNet
- Produce ArcFace-compatible crop/embedding input

## 5.5 `DeepfakeDetector`
Responsibilities:
- Load EfficientNet baseline or LoRA model
- Normalize face crop
- Return fake probability and confidence

## 5.6 `EmbeddingExtractor`
Responsibilities:
- Load ArcFace from InsightFace
- Extract 512-D normalized embedding

## 5.7 `EnrollmentStore`
Responsibilities:
- Store user embeddings
- Average multiple embeddings into one template
- Save templates with user id and metadata

## 5.8 `DecisionEngine`
Responsibilities:
- Apply fake threshold
- Apply ArcFace similarity threshold
- Fuse two-camera scores when enabled
- Return final decision and reason code

## 5.9 `Logger`
Responsibilities:
- Save inference metadata
- Save errors and decision reasons
- Save model version and threshold values used during each attempt

---

## 6) Suggested Project Directory Architecture

- `src/input/`
  - camera and video loading
  - dual camera capture
- `src/face/`
  - RetinaFace wrapper
  - alignment utilities
- `src/models/`
  - EfficientNet model definition
  - LoRA/adapters implementation
  - ArcFace wrapper
- `src/training/`
  - preprocessing scripts
  - training scripts
  - evaluation scripts
- `src/enrollment/`
  - enrollment capture
  - embedding database management
- `src/inference/`
  - single-camera inference
  - dual-camera inference
  - decision engine
- `src/utils/`
  - config loading
  - logging
  - metrics
- `configs/`
  - training config
  - inference config
  - camera config
- `artifacts/`
  - checkpoints
  - logs
  - metrics

---

## 7) Core Data Artifacts

## 7.1 Dataset manifest
Tracks processed training data.

Required columns:
- `sample_id`
- `path`
- `label`
- `source`
- `subject_id`
- `split`
- `face_detected`
- `quality_score`

## 7.2 Enrollment database
Stores identity templates.

Required fields:
- `user_id`
- `embedding_template`
- `num_samples`
- `created_at`
- `model_name`
- `model_version`

## 7.3 Verification log
Stores every attempt.

Required fields:
- `attempt_id`
- `timestamp`
- `mode`: `single_camera` or `dual_camera`
- `fake_probability`
- `match_similarity`
- `decision`
- `reason`
- `thresholds`
- `model_versions`

---

## 8) Minimal API / CLI Design

Recommended commands/scripts:

- `prepare_deepfake_data`: extract RetinaFace crops from FaceForensics++
- `train_baseline`: train EfficientNet-B0 baseline
- `train_lora`: fine-tune adapters/LoRA version
- `evaluate_deepfake`: evaluate real/fake classifier
- `evaluate_verification`: evaluate ArcFace threshold
- `enroll_user`: capture or load images and create ArcFace template
- `verify_single`: run single-camera verification
- `verify_dual`: run two-webcam verification demo

---

## 9) Configuration Files

Recommended config files:

- `configs/training.yaml`
  - dataset paths
  - batch size
  - image size
  - learning rate
  - epochs
  - augmentation settings

- `configs/inference.yaml`
  - fake threshold
  - match threshold
  - selected checkpoint
  - RetinaFace confidence threshold

- `configs/camera.yaml`
  - camera indices
  - resolution
  - FPS
  - sync threshold
  - save images true/false

---

## 10) Architecture Principle

The baseline should work without two cameras. The dual-camera demo should reuse the same detection, deepfake, verification, and decision modules. This keeps the system clean and avoids maintaining two separate pipelines.
