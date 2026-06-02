# Project Reference Guide

A simple step-by-step guide with research papers and datasets for each component.

---

## 1. Stereo Camera Calibration

**What**: Calibrate two webcams to compute accurate depth from stereo images

**Key Papers**:
- Zhang, Z. (2000). "A flexible new technique for camera calibration" - IEEE TPAMI
  - *Why*: Standard method for camera calibration using checkerboard patterns

**Dataset**: Custom checkerboard images (9×6 corners, 25mm squares)
- *Why*: Need to calibrate your specific camera setup

---

## 2. Stereo Depth Computation

**What**: Calculate depth map from left and right camera images using disparity

**Key Papers**:
- Hirschmuller, H. (2008). "Stereo processing by semiglobal matching" - IEEE TPAMI
  - *Why*: SGBM algorithm is fast and accurate for real-time depth computation

**Dataset**: Custom stereo pairs from your dual-camera setup
- *Why*: Test depth accuracy on real faces vs photos/screens

---

## 3. Face Detection

**What**: Detect faces and extract 5 facial landmarks (eyes, nose, mouth corners)

**Key Papers**:
- Deng, J. et al. (2020). "RetinaFace: Single-shot multi-level face localisation" - CVPR
  - *Why*: 97% accuracy, provides landmarks for alignment, multi-scale detection

**Dataset**: WIDER FACE (for evaluation only, use pre-trained model)
- *Why*: Standard benchmark with 32,203 images, various scales and occlusions

---

## 4. Depth-Based Liveness Detection

**What**: Analyze depth features (range, nose prominence, variance) to detect flat attacks

**Key Papers**:
- Liu, Y. et al. (2018). "Learning deep models for face anti-spoofing" - CVPR
  - *Why*: Shows auxiliary depth supervision improves spoofing detection

**Dataset**: Replay-Attack Dataset (4 GB, 1,300 videos, 50 subjects)
- *Why*: Contains photo and video replay attacks in controlled/adverse lighting
- *Download*: https://www.idiap.ch/dataset/replayattack

---

## 5. Texture-Based Anti-Spoofing

**What**: Extract LBP features and train SVM to detect printing artifacts and moiré patterns

**Key Papers**:
- Määttä, J. et al. (2012). "Face spoofing detection from single images using LBP" - IET Biometrics
  - *Why*: LBP+SVM is simple, fast, and effective for texture-based spoofing detection

**Dataset**: Replay-Attack Dataset (same as above)
- *Why*: Train SVM on real vs spoof texture patterns

---

## 6. Multi-Modal Score Fusion

**What**: Combine depth score and texture score using weighted average

**Key Papers**:
- Zhang, S. et al. (2019). "CASIA-SURF: A dataset for multi-modal face anti-spoofing" - CVPR
  - *Why*: Demonstrates multi-modal fusion (RGB+Depth+IR) improves accuracy by 3-5%

**Dataset**: Replay-Attack Dataset
- *Why*: Validate that depth+texture fusion beats single modality

---

## 7. Deepfake Detection

**What**: Train EfficientNet-B0 to classify real vs AI-generated faces

**Key Papers**:
- Tan, M. & Le, Q. (2019). "EfficientNet: Rethinking model scaling" - ICML
  - *Why*: Compound scaling achieves best accuracy-efficiency trade-off (5.3M params)
- Rössler, A. et al. (2019). "FaceForensics++: Learning to detect manipulated faces" - ICCV
  - *Why*: Standard deepfake detection benchmark and methodology

**Dataset**: FaceForensics++ (3 GB face crops, 5,000 videos)
- *Why*: Contains 4 manipulation types (DeepFakes, Face2Face, FaceSwap, NeuralTextures)
- *Download*: https://github.com/ondyari/FaceForensics

---

## 8. LoRA Fine-Tuning

**What**: Apply Low-Rank Adaptation to reduce model size by 35× while maintaining accuracy

**Key Papers**:
- Hu, E. et al. (2021). "LoRA: Low-rank adaptation of large language models" - ICLR
  - *Why*: Reduces trainable parameters from 5.3M to 150K with <1% accuracy drop

**Dataset**: FaceForensics++ (for fine-tuning)
- *Why*: Adapt pre-trained EfficientNet to deepfake detection task efficiently

---

## 9. Face Verification

**What**: Extract 512-D embeddings and compute cosine similarity for identity matching

**Key Papers**:
- Deng, J. et al. (2019). "ArcFace: Additive angular margin loss" - CVPR
  - *Why*: State-of-the-art 99.83% accuracy on LFW, learns discriminative embeddings
- Schroff, F. et al. (2015). "FaceNet: A unified embedding for face recognition" - CVPR
  - *Why*: Pioneered triplet loss and metric learning for face verification

**Dataset**: Labeled Faces in the Wild (LFW) (200 MB, 13,233 images, 6,000 pairs)
- *Why*: Standard benchmark for face verification accuracy
- *Download*: http://vis-www.cs.umass.edu/lfw/

---

## 10. Temporal Consistency Check

**What**: Analyze landmark variance across frames to detect deepfake jitter

**Key Papers**:
- Sabir, E. et al. (2019). "Recurrent convolutional strategies for face manipulation detection" - CVPR
  - *Why*: Shows temporal inconsistencies are strong indicators of deepfakes

**Dataset**: FaceForensics++ (video sequences)
- *Why*: Test frame-to-frame consistency on manipulated videos

---

## Dataset Summary

| Dataset | Size | Purpose | Download |
|---------|------|---------|----------|
| **Replay-Attack** | 4 GB | Anti-spoofing training (photo/video attacks) | [Link](https://www.idiap.ch/dataset/replayattack) |
| **FaceForensics++** | 3 GB | Deepfake detection training | [Link](https://github.com/ondyari/FaceForensics) |
| **LFW** | 200 MB | Face verification benchmark | [Link](http://vis-www.cs.umass.edu/lfw/) |
| **WIDER FACE** | Optional | Face detection evaluation | [Link](http://shuoyang1213.me/WIDERFACE/) |
| **Custom Dataset** | 1-2 GB | Testing your specific camera setup | Record yourself |

**Total**: ~8 GB (manageable size)

---

## Why These Specific Datasets?

### Replay-Attack
- ✓ Contains both photo and video replay attacks
- ✓ 50 subjects with controlled and adverse lighting
- ✓ Standard benchmark for anti-spoofing (cited in 500+ papers)
- ✓ Manageable size (4 GB vs 50+ GB alternatives)

### FaceForensics++
- ✓ Multiple manipulation types (not just one deepfake method)
- ✓ High-quality (c23 compression) for realistic scenarios
- ✓ Face crops available (3 GB vs 500 GB full videos)
- ✓ Most cited deepfake detection benchmark (1000+ citations)

### LFW
- ✓ Industry standard for face verification (10,000+ citations)
- ✓ Unconstrained "in the wild" conditions
- ✓ Small size (200 MB) but comprehensive (13K images)
- ✓ Enables direct comparison with published methods

---

## Implementation Order

1. **Week 1-2**: Stereo calibration + depth computation
2. **Week 3**: Face detection integration (RetinaFace)
3. **Week 4-5**: Depth-based liveness + texture anti-spoofing
4. **Week 6**: Multi-modal fusion
5. **Week 7-8**: Deepfake detector training (EfficientNet + LoRA)
6. **Week 9**: Face verification integration (ArcFace)
7. **Week 10**: Temporal consistency + system integration
8. **Week 11**: Testing on custom dataset
9. **Week 12**: Documentation + presentation

---

## Quick Reference: Paper-to-Component Mapping

| Component | Primary Paper | Year | Venue |
|-----------|--------------|------|-------|
| Calibration | Zhang | 2000 | IEEE TPAMI |
| Stereo Depth | Hirschmuller (SGBM) | 2008 | IEEE TPAMI |
| Face Detection | RetinaFace | 2020 | CVPR |
| Anti-Spoofing | Määttä (LBP) | 2012 | IET Biometrics |
| Multi-Modal | CASIA-SURF | 2019 | CVPR |
| Deepfake Detection | EfficientNet | 2019 | ICML |
| Model Compression | LoRA | 2021 | ICLR |
| Face Verification | ArcFace | 2019 | CVPR |
| Temporal Analysis | Sabir | 2019 | CVPR |

---

## Additional Reading (Optional)

### Stereo Vision Fundamentals
- Scharstein, D. & Szeliski, R. (2002). "A taxonomy of stereo correspondence algorithms"

### Deep Learning for Face Recognition
- Taigman, Y. et al. (2014). "DeepFace: Closing the gap to human-level performance"

### Adversarial Robustness
- Qian, Y. et al. (2021). "Thinking in frequency: Face forgery detection by mining frequency-aware clues"

---

## Notes

- **Pre-trained models**: Use InsightFace library for RetinaFace and ArcFace (no training needed)
- **Transfer learning**: Start with ImageNet pre-trained EfficientNet-B0
- **GPU**: Recommended for training (2-4 hours), but CPU inference works (15-20 FPS)
- **Storage**: Keep 20 GB free for datasets + models + experiments

---

*Last Updated: December 2024*
