# Dual-Camera Face Verification System

A biometric face verification system using stereo vision for liveness detection and deepfake prevention. Built as a BTech final year project.

## 🎯 Features

- **Stereo Depth-Based Liveness Detection**: Uses two webcams to compute 3D face depth, defeating photo and video attacks
- **Texture-Based Anti-Spoofing**: LBP analysis to detect printed photos and screen displays
- **Deepfake Detection**: EfficientNet-B0 model to identify AI-generated fake faces
- **Face Verification**: ArcFace embeddings for accurate identity matching
- **Real-Time Performance**: 15-30 FPS on standard hardware

## 🔧 Hardware Requirements

- 2× USB Webcams (720p minimum, Logitech C270 recommended)
- Rigid mounting bracket (6-10 cm baseline between cameras)
- Computer with:
  - CPU: Intel i5 / Ryzen 5 or better
  - RAM: 8 GB minimum (16 GB recommended)
  - GPU: Optional (NVIDIA GTX 1050+ for faster inference)
  - Storage: 20 GB for models and datasets

## 📦 Software Requirements

### Python Version
- Python 3.8, 3.9, or 3.10

### Dependencies
```bash
pip install -r requirements.txt
```

Main libraries:
- OpenCV 4.8+ (stereo vision, image processing)
- InsightFace 0.7+ (face detection, ArcFace)
- TensorFlow 2.13+ (deep learning models)
- scikit-learn 1.3+ (SVM classifier)
- scikit-image 0.21+ (LBP features)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/dual-camera-face-verification.git
cd dual-camera-face-verification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Camera Calibration

```bash
# Print checkerboard pattern (9×6, 25mm squares)
# Mount cameras 6-10 cm apart

# Run calibration
python calibration/calibrate.py

# Follow on-screen instructions to capture calibration images
```

### 3. Download Datasets (Optional - for training)

See `docs/datasets-guide.md` for detailed instructions.

**For anti-spoofing:**
- Replay-Attack dataset (~4 GB)

**For deepfake detection:**
- FaceForensics++ faces only (~3 GB)

### 4. Run Demo

```bash
# Basic face verification
python main.py --mode verify

# Enroll new user
python main.py --mode enroll --user-id john_doe

# Test liveness detection
python main.py --mode liveness-test
```

## 📁 Project Structure

```
dual-camera-face-verification/
├── calibration/              # Camera calibration scripts
│   ├── calibrate.py
│   └── calibration_params.json
├── models/                   # Trained models
│   ├── antispoofing_lbp.pkl
│   └── deepfake_detector.h5
├── data/                     # Data storage
│   ├── enrolled_users/
│   └── logs/
├── src/                      # Source code
│   ├── camera.py            # Camera capture & sync
│   ├── stereo.py            # Stereo depth computation
│   ├── face_detection.py    # Face detection module
│   ├── antispoofing.py      # Liveness detection
│   ├── deepfake_detection.py
│   ├── face_recognition.py
│   ├── enrollment.py
│   └── verification.py
├── train/                    # Training scripts
│   ├── train_antispoofing.py
│   └── train_deepfake.py
├── docs/                     # Documentation
│   ├── technical-specification.md
│   ├── datasets-guide.md
│   ├── research.md
│   └── requirements.md
├── tests/                    # Unit tests
├── requirements.txt
├── config.yaml
└── main.py
```

## 📊 Performance

| Component | Metric | Target |
|-----------|--------|--------|
| Anti-Spoofing | Accuracy | >95% |
| Deepfake Detection | Accuracy | >93% |
| Face Verification | Accuracy (LFW) | >99% |
| System | FPS | 15-30 |

## 🎓 Academic Context

This project implements research from multiple papers:
- **Stereo Vision Anti-Spoofing**: Wang et al. (ICPR 2015)
- **LBP Anti-Spoofing**: Määttä et al. (BIOSIG 2012)
- **Deepfake Detection**: Rössler et al. (ICCV 2019) - FaceForensics++
- **Face Recognition**: Deng et al. (CVPR 2019) - ArcFace

See `docs/research.md` for complete references.

## 📖 Documentation

- **[Technical Specification](docs/technical-specification.md)**: Complete technical details, model choices, implementation guide
- **[Datasets Guide](docs/datasets-guide.md)**: Dataset download instructions and rankings
- **[Research Papers](docs/research.md)**: All relevant research papers and resources
- **[Requirements](docs/requirements.md)**: Formal system requirements (EARS format)
- **[Quick Start](docs/QUICK-START.md)**: TL;DR version

## 🔬 Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Stereo Depth | OpenCV SGBM | Fast, accurate, built-in |
| Face Detection | RetinaFace | State-of-the-art accuracy |
| Anti-Spoofing | Depth + LBP+SVM | Multi-modal, robust |
| Deepfake Detection | EfficientNet-B0 | Best accuracy/speed ratio |
| Face Recognition | ArcFace | Highest accuracy (99.83% on LFW) |

## 🛠️ Training Your Own Models

### Anti-Spoofing Model

```bash
# Download Replay-Attack dataset first
python train/train_antispoofing.py \
    --dataset-path data/replay-attack \
    --output models/antispoofing_lbp.pkl
```

### Deepfake Detector

```bash
# Download FaceForensics++ dataset first
python train/train_deepfake.py \
    --dataset-path data/FaceForensics \
    --epochs 15 \
    --batch-size 32 \
    --output models/deepfake_detector.h5
```

## 🐛 Troubleshooting

**Poor depth maps?**
- Ensure cameras are rigidly mounted
- Recalibrate with more images
- Improve lighting conditions

**Face detection failing?**
- Check distance (40-80 cm optimal)
- Ensure good lighting
- Lower confidence threshold

**Slow performance?**
- Reduce camera resolution to 640×480
- Enable GPU acceleration
- Optimize SGBM parameters

See `docs/technical-specification.md` Section 14 for complete troubleshooting guide.

## 📝 License

MIT License - see LICENSE file for details

## 👥 Authors

- Your Name - BTech Final Year Project
- Institution: [Your College Name]
- Year: 2024-2025

## 🙏 Acknowledgments

- OpenCV community for stereo vision tools
- InsightFace team for ArcFace implementation
- FaceForensics++ dataset creators
- All researchers whose work made this possible

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**⭐ If you find this project useful, please star the repository!**
