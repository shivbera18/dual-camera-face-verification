# Quick Start Guide: Dual-Camera Face Verification System

## TL;DR - What You Need

### Hardware (~₹3,000)
- 2× Logitech C270 webcams (or similar 720p USB webcams)
- Rigid mounting bracket (6-10 cm apart)
- Computer with 8GB RAM

### Software (Free)
```bash
pip install opencv-python insightface tensorflow scikit-learn scikit-image
```

### Datasets (~8 GB)
1. **Replay-Attack** (4 GB) - Anti-spoofing training
2. **FaceForensics++** faces only (3 GB) - Deepfake detection
3. **LFW** (200 MB) - Face recognition evaluation

### Models (Pre-trained, No Training Needed!)
- **ArcFace**: Face recognition (auto-downloads via InsightFace)
- **RetinaFace**: Face detection (auto-downloads via InsightFace)

### What You'll Train
- **LBP+SVM**: Anti-spoofing classifier (~10 min training)
- **EfficientNet-B0**: Deepfake detector (~2-4 hours on GPU)

---

## Technology Choices at a Glance

| Component | Choice | Why | Alternative |
|-----------|--------|-----|-------------|
| **Stereo Depth** | OpenCV SGBM | Fast, built-in | RAFT-Stereo (slower) |
| **Face Detection** | RetinaFace | Most accurate | MTCNN (simpler) |
| **Anti-Spoofing** | Depth + LBP+SVM | Multi-modal | CNN (needs more data) |
| **Deepfake** | EfficientNet-B0 | Best speed/accuracy | XceptionNet (larger) |
| **Face Recognition** | ArcFace | State-of-the-art | FaceNet (older) |

---

## 4-Week Minimal Viable Product

### Week 1: Setup
- Buy cameras, build mount
- Calibrate stereo system
- Test depth maps

### Week 2: Core Features
- Face detection (RetinaFace)
- Depth-based liveness
- Basic pipeline

### Week 3: Recognition
- Install ArcFace (pre-trained)
- User enrollment
- Identity matching

### Week 4: Polish
- Record test data
- Tune thresholds
- Create demo

**Result:** Working face verification with depth-based anti-spoofing!

---

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Anti-Spoofing Accuracy | 95%+ | Depth + texture |
| Deepfake Detection | 93%+ | EfficientNet-B0 |
| Face Verification | 99%+ | Pre-trained ArcFace |
| Speed | 15-30 FPS | Real-time capable |

---

## Key Files to Read

1. **technical-specification.md** - Complete technical details (THIS IS THE MAIN DOCUMENT!)
2. **datasets-guide.md** - Dataset download and usage
3. **research.md** - Research papers and references
4. **requirements.md** - System requirements (formal spec)

---

## Installation (5 Minutes)

```bash
# Create environment
python -m venv face_env
source face_env/bin/activate  # Linux/Mac
# OR: face_env\Scripts\activate  # Windows

# Install everything
pip install opencv-python==4.8.1.78
pip install insightface==0.7.3
pip install tensorflow==2.13.0
pip install scikit-learn==1.3.0
pip install scikit-image==0.21.0
pip install numpy matplotlib

# Verify
python -c "import cv2, insightface, tensorflow; print('All good!')"
```

---

## First Steps

1. **Read technical-specification.md** (the complete guide)
2. **Buy hardware** (2 webcams + mount)
3. **Follow Week 1-2 roadmap** (calibration + depth)
4. **Download Replay-Attack dataset** (for anti-spoofing)
5. **Start coding!**

---

## Most Important Sections in Technical Spec

- **Section 2**: Hardware specifications (what to buy)
- **Section 3**: Stereo calibration (how to set up)
- **Section 5**: Anti-spoofing implementation (core feature)
- **Section 7**: Face recognition (ArcFace usage)
- **Section 12**: 12-week roadmap (step-by-step plan)

---

## Questions?

Refer to:
- **Section 14** (Troubleshooting) in technical-specification.md
- **Section 16** (Additional Resources) for learning materials
- Research papers in research.md

---

**Ready to start? Open `technical-specification.md` for the complete guide!**
