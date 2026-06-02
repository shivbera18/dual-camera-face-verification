# Full Project Plan: RetinaFace + EfficientNet + ArcFace with LoRA and Dual-Webcam Demo

## 1) Project Objective

Build a face-authentication system that can detect a face, reject fake/deepfake faces, and verify the identity of a real user.

The project will be developed in this order:

1. **Baseline model first**
   - RetinaFace for face detection and landmark extraction
   - EfficientNet-B0 for real/fake face classification
   - ArcFace for face verification

2. **Advanced improvement second**
   - Add LoRA/adapters or parameter-efficient fine-tuning to improve EfficientNet-B0 without retraining the full model heavily

3. **Demo extension last**
   - Add two-webcam input mode
   - Capture face images from two cameras
   - Store synchronized image pairs and metadata
   - Run the same baseline pipeline on both views and fuse the results

This order is important. The single-camera baseline must work before adding LoRA or two-camera complexity.

---

## 2) Core Technology Stack

## 2.1 RetinaFace
Purpose:
- Detect face bounding box
- Extract facial landmarks
- Support face alignment before model input

Plan:
- Use pre-trained RetinaFace from InsightFace
- Do not train RetinaFace initially
- Use it consistently for training preprocessing and runtime inference

Why:
- Accurate face detection
- Landmark output is useful for aligned crops
- Reduces noise in EfficientNet and ArcFace inputs

## 2.2 EfficientNet-B0
Purpose:
- Classify aligned face crop as `real` or `fake`

Plan:
- Train baseline EfficientNet-B0 first
- Use ImageNet pre-trained weights
- Fine-tune for binary classification
- Later apply LoRA/adapters and compare against baseline

Why:
- Lightweight enough for demo
- Good accuracy-to-speed tradeoff
- Easy to train and deploy compared to larger models

## 2.3 ArcFace
Purpose:
- Extract identity embeddings
- Verify user identity using cosine similarity

Plan:
- Use pre-trained ArcFace from InsightFace
- Do not train ArcFace from scratch
- Build enrollment and verification workflow around ArcFace embeddings

Why:
- Strong face verification performance
- Saves training time
- Makes the project more practical and focused

---

## 3) System Pipeline

## 3.1 Baseline single-camera pipeline

1. Capture image/frame from camera, video, or image file
2. Run RetinaFace
3. Select best face if multiple faces are detected
4. Align/crop face using landmarks
5. Send aligned face to EfficientNet-B0
6. If fake probability is high, reject
7. If face is real, extract ArcFace embedding
8. Compare embedding with enrolled user template
9. Return decision:
   - `ACCEPT`: real face and identity matched
   - `REJECT_FAKE`: fake/deepfake detected
   - `REJECT_IDENTITY`: real face but identity not matched
   - `RETRY`: no face, bad face, or low confidence

## 3.2 Advanced LoRA pipeline

Same as baseline, but EfficientNet-B0 is replaced with:
- `EfficientNet-B0 baseline + LoRA/adapters`

The LoRA model must be evaluated against the baseline. If it does not improve or at least match baseline performance, keep the baseline for final demo.

## 3.3 Final two-webcam demo pipeline

1. Capture one frame from left webcam and one frame from right webcam
2. Attach timestamp to each frame
3. Pair frames if timestamp difference is below sync threshold
4. Run RetinaFace + EfficientNet + ArcFace on both camera frames
5. Fuse fake scores and identity scores
6. Return final decision
7. Store optional image pair and metadata for report/demo evidence

---

## 4) Dataset Plan

## 4.1 Mandatory datasets

| Dataset | Approx Size | Purpose | Required For |
|---|---:|---|---|
| FaceForensics++ c23 subset | ~3 GB practical subset | Real/fake face training | EfficientNet baseline |
| LFW | ~200 MB | Face verification evaluation | ArcFace threshold tuning |
| Custom single-camera captures | ~1 GB | Test in local camera conditions | Final baseline testing |
| Custom dual-camera captures | ~1–2 GB | Demo extension testing | Two-webcam demo |

## 4.2 Recommended datasets

| Dataset | Purpose |
|---|---|
| Celeb-DF | Harder deepfake generalization test |
| WIDER FACE | Optional face detector behavior evaluation |

## 4.3 Optional datasets

| Dataset | Purpose |
|---|---|
| CASIA-WebFace | Future face recognition fine-tuning only |
| VGGFace2 | Future face recognition fine-tuning only |

## 4.4 Download links

- FaceForensics++: `https://github.com/ondyari/FaceForensics`
- LFW: `http://vis-www.cs.umass.edu/lfw/`
- Celeb-DF: `https://github.com/yuezunli/celeb-deepfakeforensics`
- WIDER FACE: `http://shuoyang1213.me/WIDERFACE/`
- CASIA-WebFace: `https://www.kaggle.com/datasets/debarghamitraroy/casia-webface`
- VGGFace2: `https://github.com/ox-vgg/vgg_face2`

---

## 5) Data Storage Plan

## 5.1 Main folder layout

Use this project layout:

- `data/raw/faceforensicspp/`
- `data/raw/lfw/`
- `data/raw/celebdf/` optional
- `data/raw/custom/single_cam/`
- `data/raw/custom/dual_cam/`
- `data/processed/deepfake_faces/`
- `data/processed/verification_faces/`
- `data/splits/`
- `artifacts/models/`
- `artifacts/metrics/`
- `artifacts/logs/`
- `configs/`
- `src/`

## 5.2 Processed EfficientNet dataset layout

For real/fake training:

- `data/processed/deepfake_faces/train/real/`
- `data/processed/deepfake_faces/train/fake/`
- `data/processed/deepfake_faces/val/real/`
- `data/processed/deepfake_faces/val/fake/`
- `data/processed/deepfake_faces/test/real/`
- `data/processed/deepfake_faces/test/fake/`

## 5.3 Dataset manifest

Create a manifest file:

- `data/splits/deepfake_manifest.csv`

Required columns:
- `sample_id`
- `image_path`
- `label`
- `source_dataset`
- `original_video`
- `subject_id` if available
- `split`
- `face_confidence`
- `crop_quality_status`

This manifest prevents confusion when training and makes results reproducible.

---

## 6) Data Preparation Plan

## 6.1 FaceForensics++ preprocessing

Steps:
1. Download FaceForensics++ subset, preferably c23 compression first
2. Extract frames from original and manipulated videos
3. Use RetinaFace to detect faces
4. Reject frames where:
   - no face is detected
   - face is too small
   - detection confidence is low
   - face is heavily blurred or cut off
5. Align and crop face to `224x224`
6. Save processed crops into train/val/test folders
7. Write every crop into manifest CSV

Recommended starting labels:
- `real = 0`
- `fake = 1`

Recommended first split:
- train: 70%
- validation: 15%
- test: 15%

Important:
- Avoid leaking frames from the same original video across train and test when possible.
- Keep original and fake versions grouped carefully to avoid overly optimistic metrics.

## 6.2 LFW preparation

Use LFW only for verification threshold evaluation:
1. Load LFW image pairs
2. Extract ArcFace embeddings
3. Compute cosine similarity for same-person and different-person pairs
4. Sweep threshold values
5. Select threshold that gives acceptable FAR/FRR balance

## 6.3 Custom data preparation

Collect your own samples for final testing:

Single-camera:
- real user enrollment samples
- real verification attempts
- phone screen replay fake samples
- laptop screen replay fake samples
- deepfake video replay if available

Dual-camera:
- left/right real face pairs
- left/right screen replay pairs
- left/right printed photo pairs if possible

Custom data is not for replacing FaceForensics++ training initially. It is mainly for final validation and optional LoRA/domain adaptation.

---

## 7) Baseline EfficientNet-B0 Training Plan

## 7.1 Model setup

- Backbone: EfficientNet-B0
- Initial weights: ImageNet
- Input: aligned `224x224x3` face crop
- Output: one sigmoid value (`fake_probability`)
- Loss: binary cross-entropy
- Optimizer: Adam
- Initial learning rate: `1e-4`
- Batch size: 16 or 32 depending on GPU memory
- Epochs: 10–20
- Early stopping: monitor validation AUC or validation loss

## 7.2 Augmentation

Use realistic augmentations:
- horizontal flip
- brightness/contrast changes
- JPEG compression
- mild Gaussian blur
- mild Gaussian noise
- random resize/crop within safe range

Avoid augmentations that destroy deepfake artifacts too aggressively.

## 7.3 Training stages

Stage 1:
- Freeze most EfficientNet backbone
- Train only classifier head
- Purpose: stable starting point

Stage 2:
- Unfreeze last blocks
- Fine-tune with lower learning rate
- Purpose: adapt features to fake/real artifacts

Stage 3:
- Evaluate on held-out test split
- Save best checkpoint and metrics

## 7.4 Baseline outputs

Save:
- `artifacts/models/efficientnet_b0_baseline_best.h5` or equivalent
- `artifacts/metrics/baseline_training_curves.png`
- `artifacts/metrics/baseline_confusion_matrix.png`
- `artifacts/metrics/baseline_report.json`

---

## 8) ArcFace Verification Plan

## 8.1 Enrollment process

For each user:
1. Capture 5–10 good face images
2. Run RetinaFace detection and alignment
3. Extract ArcFace embedding for each image
4. Normalize embeddings
5. Average embeddings to create one user template
6. Store template in enrollment database

## 8.2 Verification process

1. Capture query face
2. Run RetinaFace detection/alignment
3. Extract ArcFace embedding
4. Compare with enrolled template using cosine similarity
5. Accept if similarity is above threshold

## 8.3 Threshold tuning

Start with:
- `T_match = 0.60`

Tune using:
- LFW pairs
- custom same-person/different-person attempts

Report:
- true accept rate
- false accept rate
- false reject rate
- selected threshold

---

## 9) LoRA / Adapter Fine-Tuning Plan

## 9.1 When to start

Start LoRA only after:
- EfficientNet baseline is trained
- baseline test metrics are recorded
- single-camera end-to-end demo works

## 9.2 Goal

Improve or maintain deepfake detection with fewer trainable parameters.

## 9.3 Practical implementation options

Option A: adapter head only
- Freeze EfficientNet backbone
- Add small trainable adapter layers before classifier
- Lowest risk and easiest to debug

Option B: last-block adapters
- Freeze early EfficientNet blocks
- Add LoRA/adapters to final MBConv blocks or projection layers
- More advanced and potentially better

Option C: mixed approach
- Train adapter head first
- Then unfreeze or adapt last few blocks

## 9.4 Training data for LoRA

Use:
- FaceForensics++ train split
- optional Celeb-DF samples
- optional custom fake captures

Keep validation protocol the same as baseline for fair comparison.

## 9.5 LoRA success criteria

LoRA/adapters are useful only if:
- ROC-AUC is equal or better than baseline
- F1 is equal or better than baseline
- trainable parameter count is lower
- inference speed is not significantly worse

## 9.6 LoRA outputs

Save:
- `artifacts/models/efficientnet_b0_lora_best.*`
- `artifacts/metrics/lora_report.json`
- `artifacts/metrics/baseline_vs_lora.md`

---

## 10) Two-Webcam Demo Plan

## 10.1 Objective

The two-webcam mode is a final extension for demo. It should show that the system can process two face views and make a stronger decision using both views.

Do not depend on two webcams for training the baseline.

## 10.2 Hardware setup

Recommended:
- 2 USB webcams, preferably same model
- 720p resolution is enough
- 30 FPS target
- Fixed mount so both cameras face the user
- Practical baseline distance: 6–10 cm

## 10.3 Capture design

The capture module will:
1. Open left camera and right camera
2. Start separate capture loops or threads
3. Save each frame with timestamp
4. Pair left and right frames if timestamps are close enough
5. Drop unpaired or stale frames

Recommended sync threshold:
- `sync_delta_ms <= 50`

## 10.4 How two face images will be captured

For each valid frame pair:
1. Capture left image from camera 0
2. Capture right image from camera 1
3. Timestamp both images immediately after capture
4. Check time difference
5. If valid, run RetinaFace on both images
6. If face is detected in both images:
   - crop left face
   - crop right face
   - run EfficientNet on both crops
   - run ArcFace on both crops if fake check passes
7. If face is detected in only one image:
   - use fallback mode for demo if allowed
   - log warning: `single_view_fallback`
8. Save image pair only if collection/debug mode is enabled

## 10.5 How dual-camera images will be stored

Store paired captures like this:

- `data/raw/custom/dual_cam/enrollment/<user_id>/<session_id>/left/`
- `data/raw/custom/dual_cam/enrollment/<user_id>/<session_id>/right/`
- `data/raw/custom/dual_cam/evaluation/real/<subject_id>/<session_id>/left/`
- `data/raw/custom/dual_cam/evaluation/real/<subject_id>/<session_id>/right/`
- `data/raw/custom/dual_cam/evaluation/fake/<attack_type>/<session_id>/left/`
- `data/raw/custom/dual_cam/evaluation/fake/<attack_type>/<session_id>/right/`
- `data/raw/custom/dual_cam/metadata/`

File naming:
- `<session_id>_<pair_index>_left_<timestamp>.jpg`
- `<session_id>_<pair_index>_right_<timestamp>.jpg`
- `<session_id>_<pair_index>.json`

Metadata JSON should include:
- `session_id`
- `pair_index`
- `label`
- `attack_type`
- `left_image_path`
- `right_image_path`
- `left_timestamp`
- `right_timestamp`
- `sync_delta_ms`
- `camera_indices`
- `face_detected_left`
- `face_detected_right`
- `fake_score_left`
- `fake_score_right`
- `match_score_left`
- `match_score_right`
- `fused_fake_score`
- `fused_match_score`
- `decision`

## 10.6 Score fusion for two webcams

Start simple:

- `fused_fake_score = (fake_left + fake_right) / 2`
- `fused_match_score = max(match_left, match_right)`

Decision:
- If `fused_fake_score > T_fake`: reject as fake
- Else if `fused_match_score > T_match`: accept
- Else reject identity

Advanced optional fusion:
- weighted average based on face detection confidence
- require both views to pass fake threshold
- compare left/right ArcFace embeddings for consistency

## 10.7 Demo UI / console output

Show:
- left camera preview
- right camera preview
- detected face boxes
- fake score left/right
- match score left/right
- fused fake score
- fused match score
- final decision

---

## 11) Evaluation Plan

## 11.1 EfficientNet metrics

Report:
- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- false positives and false negatives examples

## 11.2 ArcFace metrics

Report:
- threshold sweep
- same-person similarity distribution
- different-person similarity distribution
- final threshold
- false accept rate
- false reject rate

## 11.3 End-to-end metrics

Report:
- real user accept rate
- fake/deepfake reject rate
- average latency per frame
- FPS
- failure cases

## 11.4 Baseline vs LoRA comparison

Compare:
- trainable parameters
- training time
- ROC-AUC
- F1-score
- inference latency
- final recommendation

## 11.5 Single-camera vs dual-camera demo comparison

Compare:
- single-camera decision
- left-camera decision
- right-camera decision
- fused dual-camera decision
- latency impact
- stability under poor lighting or angled face

---

## 12) Detailed Timeline

## Week 1: Setup and basic inference
- Install dependencies
- Verify InsightFace RetinaFace and ArcFace
- Create folder structure
- Create configs
- Run face detection on sample images

## Week 2: Dataset preparation
- Download FaceForensics++ and LFW
- Build face extraction script with RetinaFace
- Generate processed face crops
- Create manifest and train/val/test splits

## Week 3: Baseline EfficientNet training
- Train classifier head first
- Track validation loss and AUC
- Save checkpoint and initial metrics

## Week 4: Baseline improvement
- Fine-tune last EfficientNet blocks
- Tune augmentations and learning rate
- Select best baseline model

## Week 5: ArcFace verification module
- Build enrollment flow
- Create local user templates
- Evaluate threshold on LFW/custom data
- Integrate identity verification

## Week 6: End-to-end baseline demo
- Combine RetinaFace + EfficientNet + ArcFace
- Implement decision engine
- Add logging
- Run single-camera live demo

## Week 7: LoRA/adapters implementation
- Add adapter/LoRA-like fine-tuning path
- Freeze backbone or most backbone blocks
- Train on same split
- Save LoRA checkpoint

## Week 8: LoRA evaluation and hard testing
- Compare baseline vs LoRA
- Test on Celeb-DF if available
- Test on custom fake samples
- Choose final deepfake model

## Week 9: Two-webcam capture extension
- Implement dual camera capture
- Pair frames by timestamp
- Save paired images and metadata
- Run detection on both streams

## Week 10: Dual-camera inference and fusion
- Run EfficientNet + ArcFace on both views
- Implement score fusion
- Add final dual-camera demo mode
- Measure latency and stability

## Week 11: Final evaluation
- Evaluate baseline/LoRA
- Evaluate single-camera and dual-camera demo
- Prepare tables and failure case analysis

## Week 12: Final packaging and report
- Clean scripts/configs
- Save final checkpoints
- Prepare demo instructions
- Finalize report and presentation

---

## 13) Deliverables

## 13.1 Must-have deliverables
- RetinaFace preprocessing pipeline
- EfficientNet-B0 baseline trained model
- ArcFace enrollment database and verification module
- Single-camera end-to-end demo
- Baseline evaluation report

## 13.2 Advanced deliverables
- LoRA/adapters fine-tuned EfficientNet
- Baseline vs LoRA comparison
- Hard-case testing on Celeb-DF/custom data

## 13.3 Demo extension deliverables
- Dual-webcam capture module
- Paired image storage with metadata
- Dual-camera inference mode
- Score fusion and final decision display

---

## 14) Risk Management

## 14.1 Dataset access delay
Mitigation:
- Start with available samples or smaller downloaded subsets
- Build preprocessing pipeline before full dataset arrives

## 14.2 EfficientNet overfitting
Mitigation:
- Use validation AUC
- Use augmentation carefully
- Keep test split isolated
- Avoid frame leakage across splits

## 14.3 ArcFace false accept/reject issues
Mitigation:
- Tune threshold on LFW and custom data
- Store multiple embeddings per user
- Use averaged templates

## 14.4 LoRA does not improve performance
Mitigation:
- Keep baseline as final model
- Use LoRA as comparison/advanced experiment

## 14.5 Dual-camera sync instability
Mitigation:
- Timestamp frames
- Drop unsynchronized pairs
- Keep single-camera fallback mode

---

## 15) Immediate Next Actions

1. Create the final folder structure
2. Download FaceForensics++ and LFW
3. Implement RetinaFace face extraction
4. Train EfficientNet-B0 baseline
5. Build ArcFace enrollment and verification
6. Integrate single-camera demo
7. Implement LoRA/adapters only after baseline is stable
8. Implement two-webcam capture and fusion only after LoRA/baseline comparison

---

## 16) Architecture Reference

The basic project architecture, module responsibilities, two-camera capture design, storage schema, and suggested directory structure are documented separately here:

- [`architecture.md`](architecture.md)

Use `architecture.md` as the implementation reference while using this `plan.md` as the execution roadmap.
