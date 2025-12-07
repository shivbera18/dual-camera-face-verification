# Complete Dataset Guide for Dual-Camera Face Verification

## Overview

This guide ranks and details all datasets you'll need for training your face verification system with deepfake detection. Datasets are ranked by:
- **Ease of access** (free vs restricted)
- **Relevance** to your dual-camera setup
- **Quality** and size
- **Practical usability** for a BTech project

---

## Part 1: Face Anti-Spoofing Datasets

### Tier 1: HIGHLY RECOMMENDED (Start Here)

#### 1. CASIA-SURF (⭐⭐⭐⭐⭐ - BEST FOR YOUR PROJECT)

| Attribute | Details |
|-----------|---------|
| **Why #1** | Only dataset with RGB + Depth + IR - perfect for your dual-camera setup |
| **Size** | 21,000 videos, 1,000 subjects |
| **Modalities** | RGB, Depth (from Intel RealSense), Infrared |
| **Attack Types** | Flat print attacks, Curved print attacks, Cut photo attacks |
| **Resolution** | 640×480 |
| **Access** | Free after registration |
| **Link** | https://sites.google.com/view/face-anti-spoofing-challenge/dataset |
| **Paper** | "CASIA-SURF: A Large-scale Multi-modal Benchmark for Face Anti-Spoofing" (CVPR 2019) |

**How to Use:**
```python
# Dataset structure
CASIA-SURF/
├── Training/
│   ├── real/
│   │   ├── subject_001/
│   │   │   ├── color/  # RGB frames
│   │   │   ├── depth/  # Depth maps
│   │   │   └── ir/     # Infrared frames
│   └── fake/
│       ├── subject_001/
│       │   ├── color/
│       │   ├── depth/
│       │   └── ir/
├── Val/
└── Test/
```

**Training Strategy:**
- Use RGB + Depth for dual webcam setup
- Use RGB + IR for webcam + IR camera setup
- Pre-trained models available: https://github.com/SeuTao/CVPR19-Face-Anti-spoofing

---

#### 2. Replay-Attack Database (⭐⭐⭐⭐⭐ - EASIEST TO START)

| Attribute | Details |
|-----------|---------|
| **Why #2** | Simple, well-documented, great for learning |
| **Size** | 1,300 video clips, 50 subjects |
| **Modalities** | RGB only |
| **Attack Types** | Print (photo), Replay (video on screen) |
| **Resolution** | 320×240 |
| **Access** | Free - just sign license agreement |
| **Link** | https://www.idiap.ch/en/dataset/replayattack |
| **Paper** | "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing" (BIOSIG 2012) |

**How to Use:**
```python
# Dataset structure
replay-attack/
├── train/
│   ├── real/
│   │   └── client001_session01_webcam_authenticate_*.mov
│   └── attack/
│       ├── fixed/     # Fixed photo attacks
│       ├── hand/      # Hand-held photo attacks
│       └── highdef/   # High-def video replay
├── devel/  # Validation set
└── test/
```

**Training Strategy:**
- Perfect for initial model development
- Use for texture-based (LBP) classifier training
- Baseline accuracy to beat: 98% with simple LBP+SVM

---

#### 3. OULU-NPU (⭐⭐⭐⭐ - BEST FOR GENERALIZATION)

| Attribute | Details |
|-----------|---------|
| **Why #3** | Tests cross-device, cross-environment generalization |
| **Size** | 5,940 videos, 55 subjects |
| **Modalities** | RGB only |
| **Attack Types** | Print (2 printers), Replay (2 displays) |
| **Protocols** | 4 protocols testing different generalization scenarios |
| **Resolution** | 1080×1920 |
| **Access** | Free after registration |
| **Link** | https://sites.google.com/site/aboraborab/oulu-npu |
| **Paper** | "OULU-NPU: A Mobile Face Presentation Attack Database" (FG 2017) |

**4 Evaluation Protocols:**
1. **Protocol 1**: Cross-session (same conditions)
2. **Protocol 2**: Cross-attack (unseen attack types)
3. **Protocol 3**: Cross-device (unseen cameras)
4. **Protocol 4**: Cross-everything (hardest)

**Training Strategy:**
- Use Protocol 1 first (easiest)
- Graduate to Protocol 4 for robust model
- Good for testing if your model generalizes

---

### Tier 2: RECOMMENDED (Use for Better Performance)

#### 4. SiW (Spoof in the Wild) (⭐⭐⭐⭐)

| Attribute | Details |
|-----------|---------|
| **Size** | 4,478 videos, 165 subjects |
| **Modalities** | RGB |
| **Attack Types** | Print, Replay with variations (distance, pose, illumination) |
| **Special** | Includes live variations (expressions, poses) |
| **Access** | Request from MSU |
| **Link** | http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html |

---

#### 5. CASIA-FASD (⭐⭐⭐)

| Attribute | Details |
|-----------|---------|
| **Size** | 600 videos, 50 subjects |
| **Modalities** | RGB |
| **Attack Types** | Warped photo, Cut photo, Video replay |
| **Resolution** | 640×480 |
| **Access** | Free |
| **Link** | http://www.cbsr.ia.ac.cn/english/FASDB_V1.0.asp |

**Note:** Older dataset but still useful for baseline comparisons.

---

#### 6. MSU-MFSD (⭐⭐⭐)

| Attribute | Details |
|-----------|---------|
| **Size** | 440 videos, 35 subjects |
| **Modalities** | RGB |
| **Attack Types** | Print (2 types), Replay (2 types) |
| **Cameras** | Laptop webcam + Android phone |
| **Access** | Free |
| **Link** | https://www.cse.msu.edu/rgroups/biometrics/Publications/Databases/MSU_MFSD/ |

---

### Tier 3: ADVANCED (For Comprehensive Training)

#### 7. CelebA-Spoof (⭐⭐⭐⭐ - LARGEST)

| Attribute | Details |
|-----------|---------|
| **Size** | 625,537 images, 10,177 subjects |
| **Modalities** | RGB |
| **Attack Types** | Print, Replay, 3D Mask, Paper Mask |
| **Special** | Rich annotations (40 attributes, illumination, environment) |
| **Access** | Free |
| **Link** | https://github.com/ZhangYuanhan-AI/CelebA-Spoof |

**Best for:** Training deep learning models that need lots of data.

---

### Dataset Comparison Table (Anti-Spoofing)

| Rank | Dataset | Size | Multi-Modal | Difficulty | Best For |
|------|---------|------|-------------|------------|----------|
| 1 | CASIA-SURF | 21K | RGB+Depth+IR | Medium | Your project (multi-modal) |
| 2 | Replay-Attack | 1.3K | RGB | Easy | Learning, baseline |
| 3 | OULU-NPU | 5.9K | RGB | Hard | Generalization testing |
| 4 | SiW | 4.5K | RGB | Medium | Real-world variations |
| 5 | CelebA-Spoof | 625K | RGB | Medium | Deep learning training |
| 6 | CASIA-FASD | 600 | RGB | Easy | Quick experiments |
| 7 | MSU-MFSD | 440 | RGB | Easy | Cross-device testing |

---

## Part 2: Deepfake Detection Datasets

### Tier 1: MUST USE

#### 1. FaceForensics++ (⭐⭐⭐⭐⭐ - BEST STARTING POINT)

| Attribute | Details |
|-----------|---------|
| **Why #1** | Most widely used, well-documented, multiple manipulation types |
| **Size** | 1,000 original videos → 5,000 manipulated videos |
| **Manipulation Types** | DeepFakes, Face2Face, FaceSwap, NeuralTextures |
| **Quality Levels** | Raw, HQ (c23), LQ (c40) - tests compression robustness |
| **Access** | Free (sign agreement) |
| **Link** | https://github.com/ondyari/FaceForensics |
| **Paper** | "FaceForensics++: Learning to Detect Manipulated Facial Images" (ICCV 2019) |

**Dataset Structure:**
```
FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/  # Compressed videos
│           └── videos/
├── manipulated_sequences/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   └── NeuralTextures/
```

**Manipulation Types Explained:**
1. **DeepFakes**: Face swap using autoencoder
2. **Face2Face**: Facial reenactment (expressions transferred)
3. **FaceSwap**: Graphics-based face swap
4. **NeuralTextures**: Neural rendering manipulation

**Training Strategy:**
```python
# Recommended split
Train: 720 videos per manipulation type
Val: 140 videos per manipulation type  
Test: 140 videos per manipulation type

# Start with binary classification
Labels: 0 = Real, 1 = Fake

# Then try multi-class
Labels: 0 = Real, 1 = DeepFakes, 2 = Face2Face, 3 = FaceSwap, 4 = NeuralTextures
```

---

#### 2. Celeb-DF (⭐⭐⭐⭐⭐ - HIGHEST QUALITY FAKES)

| Attribute | Details |
|-----------|---------|
| **Why #2** | Most challenging - high quality deepfakes |
| **Size** | 590 real + 5,639 fake videos |
| **Subjects** | 59 celebrities |
| **Quality** | Very high - hard to detect visually |
| **Access** | Free |
| **Link** | https://github.com/yuezunli/celeb-deepfakeforensics |
| **Paper** | "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics" (CVPR 2020) |

**Why It's Challenging:**
- Reduced visual artifacts
- Better color matching
- Smoother face boundaries
- Tests if your model catches subtle cues

---

### Tier 2: RECOMMENDED FOR ROBUSTNESS

#### 3. DFDC (DeepFake Detection Challenge) (⭐⭐⭐⭐)

| Attribute | Details |
|-----------|---------|
| **Size** | 100,000+ videos |
| **Subjects** | 3,426 paid actors |
| **Variations** | Multiple ethnicities, ages, lighting conditions |
| **Access** | Free (Kaggle) |
| **Link** | https://www.kaggle.com/c/deepfake-detection-challenge |

**Note:** Very large - use subset for training.

---

#### 4. DeeperForensics-1.0 (⭐⭐⭐⭐)

| Attribute | Details |
|-----------|---------|
| **Size** | 60,000 videos |
| **Special** | Includes perturbations (compression, blur, noise) |
| **Access** | Free |
| **Link** | https://github.com/EndlessSora/DeeperForensics-1.0 |

**Best for:** Testing robustness to real-world degradations.

---

### Dataset Comparison Table (Deepfake)

| Rank | Dataset | Size | Difficulty | Best For |
|------|---------|------|------------|----------|
| 1 | FaceForensics++ | 5K videos | Medium | Learning, baseline |
| 2 | Celeb-DF | 6K videos | Hard | High-quality fake detection |
| 3 | DFDC | 100K videos | Medium | Large-scale training |
| 4 | DeeperForensics | 60K videos | Hard | Robustness testing |

---

## Part 3: Face Recognition Datasets

### For Training Face Embeddings

#### 1. CASIA-WebFace (⭐⭐⭐⭐⭐ - RECOMMENDED)

| Attribute | Details |
|-----------|---------|
| **Size** | 494,414 images, 10,575 subjects |
| **Access** | Free |
| **Link** | https://www.kaggle.com/datasets/debarghamitraroy/casia-webface |

**Best for:** Training face recognition from scratch.

---

#### 2. VGGFace2 (⭐⭐⭐⭐)

| Attribute | Details |
|-----------|---------|
| **Size** | 3.31 million images, 9,131 subjects |
| **Variations** | Pose, age, illumination, ethnicity |
| **Access** | Free |
| **Link** | https://github.com/ox-vgg/vgg_face2 |

---

### For Evaluation

#### 3. LFW (Labeled Faces in the Wild) (⭐⭐⭐⭐⭐ - MUST USE)

| Attribute | Details |
|-----------|---------|
| **Size** | 13,233 images, 5,749 subjects |
| **Purpose** | Standard benchmark for face verification |
| **Access** | Free |
| **Link** | http://vis-www.cs.umass.edu/lfw/ |

**Evaluation Protocol:**
- 6,000 face pairs (3,000 matched, 3,000 mismatched)
- 10-fold cross-validation
- Report accuracy (state-of-art: 99.8%+)

---

## Part 4: Recommended Dataset Strategy for Your Project

### Option A: Dual Webcam Setup

```
Training Pipeline:
1. Anti-Spoofing:
   - Primary: CASIA-SURF (use RGB + Depth channels)
   - Secondary: Replay-Attack (for texture features)
   
2. Deepfake Detection:
   - Primary: FaceForensics++ (all 4 manipulation types)
   - Secondary: Celeb-DF (for hard examples)
   
3. Face Recognition:
   - Use pre-trained ArcFace (no training needed)
   - Evaluate on LFW
```

### Option B: Webcam + IR Camera Setup

```
Training Pipeline:
1. Anti-Spoofing:
   - Primary: CASIA-SURF (use RGB + IR channels)
   - This is the ONLY public dataset with IR!
   
2. Deepfake Detection:
   - Same as Option A
   
3. Face Recognition:
   - Same as Option A
```

---

## Part 5: How to Download Each Dataset

### CASIA-SURF
```bash
# 1. Register at: https://sites.google.com/view/face-anti-spoofing-challenge/dataset
# 2. Download links will be emailed
# 3. Extract:
unzip CASIA-SURF.zip
```

### Replay-Attack
```bash
# 1. Go to: https://www.idiap.ch/en/dataset/replayattack
# 2. Sign license agreement
# 3. Download via provided script:
python download_replay_attack.py --output-dir ./data/replay-attack
```

### FaceForensics++
```bash
# 1. Clone repo
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# 2. Run download script (need to accept terms)
python download-FaceForensics.py . -d all -c c23 -t videos
```

### Celeb-DF
```bash
# 1. Go to: https://github.com/yuezunli/celeb-deepfakeforensics
# 2. Fill Google Form for access
# 3. Download via provided links
```

### Pre-trained Models (No Training Needed)

```python
# ArcFace - Face Recognition
pip install insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis()
app.prepare(ctx_id=0)  # Uses pre-trained model

# Silent-Face-Anti-Spoofing
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
# Pre-trained models included in repo

# DeepFace - Multiple models
pip install deepface
from deepface import DeepFace
# Automatically downloads models on first use
```

---

## Part 6: Data Augmentation Recommendations

### For Anti-Spoofing
```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### For Deepfake Detection
```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),  # Important!
    A.GaussianBlur(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## Part 7: Expected Performance Benchmarks

### Anti-Spoofing (What to Aim For)

| Dataset | Method | APCER | BPCER | ACER |
|---------|--------|-------|-------|------|
| Replay-Attack | LBP+SVM | 2.1% | 1.8% | 1.95% |
| Replay-Attack | CNN | 0.5% | 0.3% | 0.4% |
| OULU-NPU (P1) | CDCN | 0.4% | 0.0% | 0.2% |
| CASIA-SURF | Multi-modal | 0.1% | 0.1% | 0.1% |

**Metrics:**
- APCER: Attack Presentation Classification Error Rate (false accept)
- BPCER: Bona Fide Presentation Classification Error Rate (false reject)
- ACER: Average Classification Error Rate

### Deepfake Detection (What to Aim For)

| Dataset | Method | Accuracy | AUC |
|---------|--------|----------|-----|
| FaceForensics++ (c23) | XceptionNet | 95.7% | 0.98 |
| FaceForensics++ (c40) | XceptionNet | 86.9% | 0.93 |
| Celeb-DF | EfficientNet | 65.5% | 0.75 |

**Note:** Celeb-DF is much harder - 70%+ accuracy is good for a project.

### Face Recognition (What to Aim For)

| Dataset | Method | Accuracy |
|---------|--------|----------|
| LFW | ArcFace | 99.83% |
| LFW | FaceNet | 99.65% |

**Note:** Use pre-trained models - you'll get 99%+ easily.

---

## Quick Reference: What to Download First

### 🎯 LIGHTWEIGHT OPTION (Recommended for BTech Project) - ~8 GB Total

For a final year project, you DON'T need massive datasets. Here's a practical approach:

| Dataset | Size | Purpose |
|---------|------|---------|
| **Replay-Attack** | ~4 GB | Anti-spoofing training |
| **FaceForensics++ (faces only)** | ~3 GB | Deepfake detection |
| **LFW** | ~200 MB | Face recognition evaluation |
| **Pre-trained models** | ~500 MB | No training needed! |

**Total: ~8 GB** ✅

### Why This Works:

1. **Anti-Spoofing**: Replay-Attack is sufficient for learning and demo
   - 1,300 videos is plenty for training a good classifier
   - Your dual-camera depth analysis adds extra security anyway!

2. **Deepfake Detection**: Download only extracted face crops from FaceForensics++
   ```bash
   # Download faces only (not full videos)
   python download-FaceForensics.py . -d all -c c23 -t faces
   # This gives you ~3 GB instead of 500 GB!
   ```

3. **Face Recognition**: Use PRE-TRAINED models (ArcFace, FaceNet)
   - These already achieve 99%+ accuracy
   - No training needed = no large dataset needed
   - Just download the model weights (~100-200 MB)

### 📦 Even Lighter Option (~5 GB)

If storage is really tight:

| What | Size | Notes |
|------|------|-------|
| Replay-Attack (subset) | ~1 GB | Use only train + test splits |
| FaceForensics++ (DeepFakes only) | ~800 MB | One manipulation type is enough |
| LFW | ~200 MB | Essential for evaluation |
| Pre-trained models | ~500 MB | ArcFace + Anti-spoof model |
| Your own recordings | ~2 GB | Record your own test data! |

**Total: ~5 GB** ✅

### 🔥 Pro Tip: Record Your Own Data!

For a BTech project demo, your own recorded data is actually MORE impressive:

```
Your Custom Dataset:
├── real/
│   ├── person1_front.mp4
│   ├── person1_left.mp4
│   ├── person1_right.mp4
│   └── ... (10-20 people, 3-5 videos each)
├── spoof_photo/
│   ├── person1_photo_phone.mp4
│   ├── person1_photo_print.mp4
│   └── ...
├── spoof_video/
│   ├── person1_replay_phone.mp4
│   ├── person1_replay_laptop.mp4
│   └── ...
```

**Benefits:**
- Shows you understand the problem deeply
- Tests your specific camera setup
- Impressive for project evaluation
- ~1-2 GB for 20 subjects

---

## Practical Training Strategy (Low Storage)

### Step 1: Use Pre-trained Models Where Possible

```python
# Face Recognition - NO TRAINING NEEDED
from deepface import DeepFace
# Model downloads automatically (~100 MB)

# Anti-Spoofing - Use pre-trained, fine-tune on small data
# Clone Silent-Face-Anti-Spoofing (includes pre-trained model)
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
```

### Step 2: Fine-tune on Small Dataset

```python
# You only need ~1000 images to fine-tune!
# 500 real + 500 fake is enough for good results

# Training split for Replay-Attack:
# Train: 360 videos (180 real + 180 fake)
# Val: 160 videos
# Test: 160 videos
```

### Step 3: Augment Heavily

```python
# Data augmentation = more data without more storage!
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.ImageCompression(quality_lower=60, p=0.3),
])
# This effectively 10x your dataset!
```

---

## Storage Comparison

| Approach | Storage | Training Time | Accuracy |
|----------|---------|---------------|----------|
| Full datasets | 150+ GB | Days | 99%+ |
| **Lightweight (Recommended)** | **8 GB** | **Hours** | **95%+** |
| Minimal | 5 GB | Hours | 90%+ |
| Pre-trained only | 1 GB | Minutes | 85%+ |

**For a BTech project, 95% accuracy with 8 GB is perfect!**

---

## Download Commands (Lightweight)

### 1. Replay-Attack (~4 GB)
```bash
# Register at: https://www.idiap.ch/en/dataset/replayattack
# Download train + test only (skip devel if tight on space)
```

### 2. FaceForensics++ Faces Only (~3 GB)
```bash
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# Download ONLY face crops, not full videos
python download-FaceForensics.py ./data -d Deepfakes -c c23 -t faces
python download-FaceForensics.py ./data -d original -c c23 -t faces
```

### 3. LFW (~200 MB)
```bash
# Direct download
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz
```

### 4. Pre-trained Models
```python
# These download automatically on first use
pip install deepface insightface

# Or clone repos with included models
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
```

---

*This lightweight guide is specifically designed for a BTech final year project - practical, achievable, and impressive without needing a server farm!*
