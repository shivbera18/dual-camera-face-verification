# Dual-Camera Face Verification System

> Biometric face verification using stereo vision for liveness detection and deepfake prevention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A BTech final year project implementing a complete face verification system with multi-modal anti-spoofing using dual webcams.

---

## 🎯 Key Features

- **Stereo Depth-Based Liveness**: 3D face depth analysis defeats photo/video attacks
- **Texture Anti-Spoofing**: LBP+SVM detects printed photos and screen displays  
- **Deepfake Detection**: EfficientNet-B0 identifies AI-generated faces
- **Face Verification**: ArcFace embeddings (99.83% accuracy on LFW)
- **Real-Time**: 15-30 FPS on standard hardware

---

## 📚 Documentation

**Start here based on your needs:**

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[PROJECT-REFERENCE-GUIDE.md](docs/PROJECT-REFERENCE-GUIDE.md)** ⭐ | Simple step-by-step with papers & datasets for each part | 15 min |
| **[PROJECT-OVERVIEW.md](docs/PROJECT-OVERVIEW.md)** | Complete system pipeline, all models, datasets, timeline | 30-45 min |
| **[LEARNING-ROADMAP.md](docs/LEARNING-ROADMAP.md)** | Learn from basics to implementation (3-4 months) | 20-30 min |
| **[technical-specification.md](docs/technical-specification.md)** | Detailed implementation guide with code examples | 2-3 hours |
| **[datasets-guide.md](docs/datasets-guide.md)** | Dataset downloads, rankings, training strategies | 30 min |
| **[research.md](docs/research.md)** | Research papers, implementations, resources | 1-2 hours |
| **[requirements.md](docs/requirements.md)** | Formal system requirements (EARS format) | 1 hour |

### 🚀 Quick Navigation

**Want simple step-by-step?** → Read [PROJECT-REFERENCE-GUIDE.md](docs/PROJECT-REFERENCE-GUIDE.md) ⭐  
**New to the project?** → Read [PROJECT-OVERVIEW.md](docs/PROJECT-OVERVIEW.md)  
**Need to learn basics?** → Follow [LEARNING-ROADMAP.md](docs/LEARNING-ROADMAP.md)  
**Ready to implement?** → Use [technical-specification.md](docs/technical-specification.md)  
**Looking for papers?** → Check [research.md](docs/research.md)

---

## 🔧 Hardware & Software

### Hardware (~₹3,000)
- 2× Logitech C270 webcams (720p)
- Rigid mounting bracket (6-10 cm baseline)
- Computer: i5/Ryzen 5, 8GB RAM

### Software
```bash
pip install opencv-python insightface tensorflow scikit-learn scikit-image
```

**Full setup:** See [technical-specification.md](docs/technical-specification.md) Section 9

---

## 📊 System Pipeline

```
Left Camera + Right Camera
    ↓
Stereo Calibration (OpenCV)
    ↓
Face Detection (RetinaFace)
    ↓
┌─────────────────┬──────────────────┐
│ Depth Analysis  │ Texture Analysis │ Anti-Spoofing
│ (SGBM)          │ (LBP + SVM)      │
└─────────────────┴──────────────────┘
    ↓
Deepfake Detection (EfficientNet-B0)
    ↓
Face Verification (ArcFace)
    ↓
ACCEPT / REJECT
```

**Detailed pipeline:** See [PROJECT-OVERVIEW.md](docs/PROJECT-OVERVIEW.md) Section 2

---

## 🧠 Models & Training

| Component | Model | Training? | Dataset | Time |
|-----------|-------|-----------|---------|------|
| Stereo Depth | OpenCV SGBM | ❌ No | - | - |
| Face Detection | RetinaFace | ❌ Pre-trained | - | - |
| Texture Anti-Spoof | LBP + SVM | ✅ Yes | Replay-Attack (4GB) | 10 min |
| Deepfake Detection | EfficientNet-B0 | ✅ Yes | FaceForensics++ (3GB) | 2-4 hrs |
| Face Recognition | ArcFace | ❌ Pre-trained | - | - |

**Only 2 models need training!** Total: ~3-5 hours with GPU

**Training details:** See [PROJECT-OVERVIEW.md](docs/PROJECT-OVERVIEW.md) Section 10

---

## 🗓️ Implementation Timeline

### 12-Week Full Implementation
- **Weeks 1-2:** Hardware setup & calibration
- **Weeks 3-5:** Stereo depth & face detection
- **Week 6:** Texture anti-spoofing training
- **Weeks 7-8:** Deepfake detector training
- **Week 9:** Face recognition integration
- **Weeks 10-12:** Integration, testing, documentation

### 4-Week MVP (Minimal Viable Product)
- **Week 1:** Hardware + calibration
- **Week 2:** Depth + face detection
- **Week 3:** Depth liveness + ArcFace
- **Week 4:** Integration + testing

**Detailed timeline:** See [PROJECT-OVERVIEW.md](docs/PROJECT-OVERVIEW.md) Section 12

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/shivbera18/dual-camera-face-verification.git
cd dual-camera-face-verification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Detailed setup:** See [technical-specification.md](docs/technical-specification.md) Section 9

---

## 🎯 Performance Targets

| Metric | Target |
|--------|--------|
| Anti-Spoofing Accuracy | >95% |
| Deepfake Detection Accuracy | >93% |
| Face Verification Accuracy | >99% |
| System Speed | 15-30 FPS |

**Benchmarks:** See [PROJECT-OVERVIEW.md](docs/PROJECT-OVERVIEW.md) Section 11

---

## 🎓 Academic Context

Implements research from:
- Stereo Vision Anti-Spoofing (Wang et al., ICPR 2015)
- LBP Anti-Spoofing (Määttä et al., BIOSIG 2012)
- FaceForensics++ (Rössler et al., ICCV 2019)
- ArcFace (Deng et al., CVPR 2019)

**All papers:** See [research.md](docs/research.md)

---

## 📊 Presentation

**15-slide presentation content** available in [Presentation/](Presentation/) folder:
- Introduction and problem statement
- RetinaFace, EfficientNet-B0, and LoRA explained
- Ready for PowerPoint/Google Slides

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

## 👥 Author

**Shiv Bera**  
BTech Final Year Project  
GitHub: [@shivbera18](https://github.com/shivbera18)

---

## 🙏 Acknowledgments

- OpenCV community
- InsightFace team
- FaceForensics++ creators
- All researchers whose work made this possible

---

**⭐ Star this repo if you find it useful!**
