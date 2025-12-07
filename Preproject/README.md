# Pre-Project Report: Dual-Camera Face Verification System

## 📄 Document Overview

This folder contains the complete pre-project report for the Dual-Camera Face Verification System with Deepfake Detection.

### Files:

1. **preproject-report.tex** - Complete LaTeX source (805 lines, 7 chapters)
2. **REPORT-GUIDE.md** - Comprehensive guide for adding images and completing the report
3. **reference.tex** - Reference document for structure and formatting

---

## 📖 Report Structure

### Chapter 1: Introduction and Problem Statement (~120 lines)
- Motivation for biometric authentication
- Problem: Presentation attacks (photos, videos, masks)
- Problem: Deepfake attacks (AI-generated faces)
- Limitations of existing solutions
- Project objectives and expected contributions

### Chapter 2: Literature Review and Existing Solutions (~180 lines)
- Evolution of face recognition (classical to deep learning)
- Face anti-spoofing methods (texture, CNN, depth sensor)
- Deepfake detection methods (CNN, frequency, temporal)
- Gap analysis: Why existing solutions are insufficient

### Chapter 3: Proposed Solution and System Architecture (~150 lines)
- System overview (dual-camera stereo vision)
- Complete architecture (6 stages):
  1. Stereo camera calibration
  2. Synchronized frame acquisition
  3. Face detection and tracking (RetinaFace)
  4. Multi-modal anti-spoofing (Depth + Texture)
  5. Deepfake detection (EfficientNet-B0)
  6. Face verification (ArcFace)
- Why this approach is superior

### Chapter 4: Technical Methodology (~200 lines)
- **RetinaFace**: State-of-the-art face detection
  - Architecture overview
  - Why RetinaFace over alternatives (MTCNN, Haar, YOLO)
  - Integration in our system
  
- **EfficientNet-B0**: Compound scaling for efficiency
  - Compound scaling principle
  - Architecture details (MBConv blocks)
  - Why EfficientNet over XceptionNet/ResNet/MobileNet
  - Transfer learning strategy
  
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
  - Mathematical formulation: W = W₀ + BA
  - Parameter reduction analysis (35× reduction)
  - Why LoRA for edge deployment
  - Training strategy

### Chapter 5: Implementation Plan and Datasets (~100 lines)
- Hardware requirements (cameras, mounting, computing)
- Software stack (Python, OpenCV, TensorFlow, etc.)
- Datasets:
  - Replay-Attack (4 GB, anti-spoofing)
  - FaceForensics++ (3 GB, deepfake detection)
  - LFW (200 MB, evaluation)
  - Custom dataset (1-2 GB, testing)
- 12-week implementation timeline

### Chapter 6: Expected Results and Evaluation Metrics (~80 lines)
- Performance targets:
  - Anti-spoofing: >95% accuracy, <2% FAR
  - Deepfake detection: >93% accuracy
  - Face verification: >99% accuracy (LFW)
  - System: <200ms latency, >15 FPS
- Comparison with existing methods
- Evaluation methodology
- Ablation studies

### Chapter 7: Conclusion and Future Work (~50 lines)
- Summary of key innovations
- Expected contributions
- Future enhancements (short, medium, long-term)
- Societal impact and ethical considerations
- Final conclusion

### References (15 citations)
- Key papers: RetinaFace, EfficientNet, LoRA, ArcFace, FaceForensics++, etc.

---

## 🖼️ Adding Images (After Implementation)

### Images to Create:

1. **System Architecture Diagram** (Create now with Draw.io/PowerPoint)
2. **Hardware Setup Diagram** (Create now)
3. **Stereo Depth Visualization** (After calibration)
4. **RetinaFace Detection Example** (After face detection)
5. **Liveness Detection Comparison** (After anti-spoofing)
6. **EfficientNet Architecture** (Use existing diagram)
7. **LoRA Diagram** (Create now)
8. **Training Loss Curves** (After training)
9. **ROC Curves** (After evaluation)
10. **Comparison Table** (After evaluation)
11. **Confusion Matrix** (After evaluation)
12. **Model Size Comparison** (After optimization)

### How to Add Images in LaTeX:

```latex
% Uncomment and use this format:
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{images/figure-name.png}
\caption{Your caption here}
\label{fig:label-name}
\end{figure}
```

**See REPORT-GUIDE.md for detailed instructions and Python code to generate each image.**

---

## 📊 Tables Included

1. Software Dependencies (Table 5.1)
2. 12-Week Implementation Schedule (Table 5.2)
3. Anti-Spoofing Performance Targets (Table 6.1)
4. Deepfake Detection Performance Targets (Table 6.2)
5. Face Verification Performance Targets (Table 6.3)
6. System-Level Performance Targets (Table 6.4)
7. Comparison with State-of-the-Art (Table 6.5)

---

## 🔢 Key Mathematical Equations

1. **Depth from Disparity**: Depth = (Baseline × Focal Length) / Disparity
2. **LoRA Decomposition**: W = W₀ + BA
3. **EfficientNet Compound Scaling**: d = α^φ, w = β^φ, r = γ^φ
4. **Cosine Similarity**: similarity = (e₁ · e₂) / (||e₁|| × ||e₂||)
5. **Score Fusion**: s_liveness = w_d × s_depth + w_t × s_texture

---

## ✅ Compilation Instructions

### Prerequisites:
```bash
# Install LaTeX distribution
# Windows: MiKTeX or TeX Live
# Linux: sudo apt-get install texlive-full
# Mac: MacTeX
```

### Compile:
```bash
# Method 1: Using pdflatex
pdflatex preproject-report.tex
pdflatex preproject-report.tex  # Run twice for references

# Method 2: Using latexmk (recommended)
latexmk -pdf preproject-report.tex

# Method 3: Using Overleaf (online)
# Upload preproject-report.tex to Overleaf.com
```

### Output:
- **preproject-report.pdf** - Final compiled document

---

## 📏 Document Statistics

- **Total Lines**: 805
- **Total Pages**: ~25-30 (estimated)
- **Word Count**: ~8,000-9,000 words
- **Chapters**: 7
- **Tables**: 7
- **Equations**: 10+
- **References**: 15

**Similar length to reference.tex (704 lines)**

---

## 🎯 Key Highlights

### Technical Focus:
- **RetinaFace**: 97% accuracy, multi-scale detection, 5 facial landmarks
- **EfficientNet-B0**: 5.3M parameters, 0.39B FLOPs, compound scaling
- **LoRA**: 35× parameter reduction, <1% accuracy loss, 3.5MB model

### Performance Targets:
- Anti-spoofing: >95% accuracy
- Deepfake detection: >93% accuracy
- Face verification: >99% accuracy
- Real-time: 15-30 FPS
- Cost: ~₹3,000 (vs ₹15,000+ for depth sensors)

### Datasets:
- Replay-Attack: 4 GB, 1,300 videos
- FaceForensics++: 3 GB faces, 5,000 videos
- LFW: 200 MB, 13,233 images
- Total: ~8 GB (manageable!)

---

## 📝 Next Steps

1. ✅ **Report Complete** - All 7 chapters written
2. ⏭️ **Create Diagrams** - System architecture, hardware setup, LoRA
3. ⏭️ **Compile PDF** - Generate final document
4. ⏭️ **Review** - Check for typos, formatting
5. ⏭️ **Submit** - Submit pre-project report
6. ⏭️ **Implementation** - Follow 12-week timeline
7. ⏭️ **Add Results** - After implementation, add result images
8. ⏭️ **Final Report** - Expand to full project report

---

## 📚 Additional Resources

- **REPORT-GUIDE.md**: Detailed guide for images and structure
- **technical-specification.md**: Implementation details
- **PROJECT-OVERVIEW.md**: Complete project overview
- **LEARNING-ROADMAP.md**: Learning path from basics

---

**Document Status**: ✅ Complete and Ready for Compilation

**Last Updated**: December 2024

**Author**: Shivratan Bera (2022BECE103)
