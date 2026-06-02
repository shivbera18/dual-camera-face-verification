# Plan Changes & Deviations

Tracks all deviations from `plan.md` made during actual setup and execution.

---

## 1. Dataset Changes (plan.md §4)

### 1.1 FaceForensics++ — replaced with Kaggle mirror

**Original plan:** Download FaceForensics++ c23 videos, extract faces using RetinaFace.

**What was done:** Used `dagnelies/deepfake-faces` from Kaggle — pre-extracted 224×224 face crops from FF++ with `metadata.csv` labels.

**Why:**
- FF++ requires Google Form approval (gated access, days of wait).
- Kaggle mirror has identical content (95,634 images, 100% label coverage via `metadata.csv`).
- Faces already extracted at 224×224 — skips the frame extraction preprocessing step.

**Impact:** No preprocessing needed for this dataset. `metadata.csv` maps each image to `REAL` or `FAKE` label and original video name.

**Location:** `data/raw/faceforensicspp/faces_224/` + `data/raw/faceforensicspp/metadata.csv`

---

### 1.2 Celeb-DF — replaced with two better datasets

**Original plan:** Celeb-DF as harder generalization test (gated, Google Form).

**What was done:**
- `xhlulu/140k-real-and-fake-faces` — 110,742 pre-split images (train/real: 40,742 | train/fake: 50,000 | test/real: 10,000 | test/fake: 10,000)
- `ciplab/real-and-fake-face-detection` — 2,041 high-quality face images (1,081 real + 960 fake)

**Why:** Both immediately accessible on Kaggle, already split into train/test, no extraction needed.

**Location:**
- `data/raw/celebdf/real_vs_fake/real-vs-fake/` (140k dataset)
- `data/raw/celebdf/real_and_fake_face/` (ciplab dataset)

---

### 1.3 Anti-spoofing dataset — added (not in original plan)

**Original plan:** No dedicated anti-spoofing dataset listed (planned to rely on FF++ deepfakes only).

**What was done:** Added `tapakah68/anti-spoofing` — 25 real-world videos across:
- `live_selfie/` — real live face
- `printouts/` — printed photo attack
- `cut-out printouts/` — cut-out photo attack
- `replay/` — screen replay attack

**Why:** Physical spoofing (print/replay) is distinct from deepfake detection. Useful for testing the full pipeline against real-world attacks, not just digital manipulations.

**Location:** `data/raw/custom/single_cam/`

---

### 1.4 LFW — download source changed

**Original plan:** `http://vis-www.cs.umass.edu/lfw/lfw.tgz`

**What was done:** Downloaded from `https://ndownloader.figshare.com/files/5976018` (mirror). Official UMass server returned DNS error from this machine.

**Pairs file:** Sourced from `davidsandberg/facenet` GitHub repo (6,000 pairs, 10-fold eval protocol).

**Location:** `data/raw/lfw/lfw/` (5,749 identities, 13,233 images) + `data/raw/lfw/pairs.txt`

---

## 2. Python Environment Changes (plan.md §12 Week 1)

**Original plan:** Used `requirements.txt` from old files with TensorFlow 2.13 + Python 3.9.

**What was done:** New `requirements.txt` using PyTorch 2.12 + Python 3.14 via `uv`.

**Why:** TensorFlow 2.13 does not support Python 3.14. PyTorch is the standard for EfficientNet/ArcFace in 2025. InsightFace ONNX runtime works with both.

**Key versions installed:**
- `torch==2.12.0`
- `torchvision`
- `insightface==1.0.1`
- `opencv-python==4.13.0`
- `albumentations`, `scikit-learn`, `pandas`, `tqdm`, `pyyaml`

---

## 3. Pre-trained Models

**Original plan:** Download models as needed.

**What was done:** Pre-downloaded InsightFace `buffalo_l` pack (5 ONNX files):
- `det_10g.onnx` — RetinaFace detection
- `1k3d68.onnx` — 3D landmark (68 points)
- `2d106det.onnx` — 2D landmark (106 points)
- `genderage.onnx` — gender/age
- `w600k_r50.onnx` — ArcFace ResNet-50 (face recognition)

**Location:** `~/.insightface/models/buffalo_l/`

---

## 4. Directory Structure Changes (plan.md §5.1)

**Original plan:** Directories listed as a bullet list.

**What was done:** All directories created. One addition:
- `data/raw/faceforensicspp/repo/` — FF++ GitHub repo cloned (contains download scripts, dataset splits JSON files needed for train/val/test split replication)

---

## 5. New Files Added (not in original plan)

| File | Purpose |
|---|---|
| `requirements.txt` | PyTorch-based deps, Python 3.14 compatible |
| `configs/dataset.yaml` | Dataset paths, split ratios, label mapping |
| `configs/model.yaml` | EfficientNet, ArcFace, RetinaFace config |
| `configs/pipeline.yaml` | Single-cam and dual-cam pipeline thresholds |
| `scripts/download_faceforensicspp.sh` | Ready-to-run script once FF++ form approved |
| `src/verify_setup.py` | Checks packages, dirs, LFW, InsightFace models |
| `.gitignore` | Excludes large data files, `.venv`, model binaries |

---

## 6. Dataset Count Summary

| Dataset | Images | Real | Fake | Source |
|---|---:|---:|---:|---|
| FF++ faces (Kaggle mirror) | 95,634 | 16,293 | 79,341 | `dagnelies/deepfake-faces` |
| 140k real vs fake | 110,742 | 50,742 | 60,000 | `xhlulu/140k-real-and-fake-faces` |
| ciplab real/fake | 2,041 | 1,081 | 960 | `ciplab/real-and-fake-face-detection` |
| Anti-spoofing | 25 videos | 9 real | 16 fake | `tapakah68/anti-spoofing` |
| LFW | 13,233 | 5,749 identities | — | figshare mirror |
| **Total images** | **221,650** | | | |

---

## 7. Remaining Plan Items (unchanged)

Everything in plan.md from Week 3 onward is unchanged:
- EfficientNet-B0 baseline training (Week 3–4)
- ArcFace enrollment + verification (Week 5)
- End-to-end single-cam demo (Week 6)
- LoRA/adapters experiment (Week 7–8)
- Dual-webcam capture (Week 9–10)
- Final evaluation + report (Week 11–12)
