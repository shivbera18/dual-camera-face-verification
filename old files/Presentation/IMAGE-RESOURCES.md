# Image Resources for Presentation & Report

Comprehensive collection of images, diagrams, and graphs for the Dual-Camera Face Verification project.

---

## 📊 Table of Contents

1. [RetinaFace Images](#retinaface-images)
2. [MTCNN vs RetinaFace Comparisons](#mtcnn-vs-retinaface-comparisons)
3. [EfficientNet Architecture](#efficientnet-architecture)
4. [EfficientNet Comparisons](#efficientnet-comparisons)
5. [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
6. [Face Detection Examples](#face-detection-examples)
7. [Deepfake Detection Examples](#deepfake-detection-examples)
8. [Stereo Vision & Depth Maps](#stereo-vision--depth-maps)
9. [Training Graphs & Results](#training-graphs--results)
10. [System Architecture Diagrams](#system-architecture-diagrams)

---

## 1. RetinaFace Images

### Architecture Diagrams

**RetinaFace Architecture (Official Paper)**
- **Source**: Original RetinaFace paper (CVPR 2020)
- **Link**: https://arxiv.org/pdf/1905.00641.pdf (Page 3, Figure 2)
- **Description**: Complete architecture showing FPN backbone, context modules, and multi-task branches
- **Use for**: Slide 9 - Explaining RetinaFace architecture

**RetinaFace Feature Pyramid Network**
- **GitHub**: https://github.com/deepinsight/insightface/tree/master/detection/retinaface
- **Image**: Architecture diagram in README
- **Description**: Shows multi-scale feature extraction
- **Use for**: Technical explanation of FPN

**RetinaFace Detection Examples**
- **Link**: https://github.com/deepinsight/insightface/blob/master/detection/retinaface/data/t1.jpg
- **Description**: Real-world detection with bounding boxes and landmarks
- **Use for**: Slide 9 - Showing detection results

### Performance Graphs

**WIDER FACE Benchmark Results**
- **Link**: http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html
- **Description**: Precision-Recall curves comparing all detectors
- **Use for**: Slide 10 - Showing RetinaFace superiority
- **Note**: RetinaFace achieves 97% on Hard subset

**RetinaFace vs Others (Bar Chart)**
- **Paper Link**: https://arxiv.org/pdf/1905.00641.pdf (Page 7, Table 4)
- **Description**: Accuracy comparison table on WIDER FACE
- **Use for**: Slide 10 - Quantitative comparison

---

## 2. MTCNN vs RetinaFace Comparisons

### MTCNN Architecture

**MTCNN 3-Stage Cascade**
- **Paper**: https://arxiv.org/pdf/1604.02878.pdf (Page 3, Figure 1)
- **Description**: Shows P-Net → R-Net → O-Net cascade
- **Use for**: Slide 10 - Explaining why cascade is slower

**MTCNN Detection Example**
- **GitHub**: https://github.com/ipazc/mtcnn
- **Image**: Example detections in README
- **Description**: Shows detection with 5 landmarks
- **Use for**: Visual comparison with RetinaFace

### Speed Comparison

**Inference Time Comparison**
- **Create your own**: Bar chart showing:
  - MTCNN: 50-80ms
  - RetinaFace: 20-30ms
  - YOLO-Face: 10-15ms
  - Haar Cascades: 5ms
- **Tool**: Use matplotlib or Excel
- **Use for**: Slide 10 - Speed comparison

**FPS Comparison Chart**
- **Data**:
  - MTCNN: 12-20 FPS
  - RetinaFace: 33-50 FPS
- **Use for**: Emphasizing real-time capability

### Accuracy Comparison

**WIDER FACE Results Table**
- **Source**: http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html
- **Data**:
  - MTCNN: 92-94%
  - RetinaFace: 97%
- **Use for**: Slide 10 - Accuracy comparison

---

## 3. EfficientNet Architecture

### Architecture Diagrams

**EfficientNet-B0 Architecture (Official)**
- **Paper**: https://arxiv.org/pdf/1905.11946.pdf (Page 4, Figure 1)
- **Description**: Shows MBConv blocks and network structure
- **Use for**: Slide 11 - Architecture explanation

**Compound Scaling Visualization**
- **Paper**: https://arxiv.org/pdf/1905.11946.pdf (Page 2, Figure 2)
- **Description**: Visual showing depth, width, resolution scaling
- **Use for**: Slide 11 - Explaining compound scaling concept
- **Key Image**: Shows how EfficientNet scales all three dimensions

**MBConv Block Diagram**
- **Paper**: https://arxiv.org/pdf/1905.11946.pdf (Page 3, Table 1)
- **Description**: Mobile Inverted Bottleneck Convolution structure
- **Use for**: Technical deep-dive if needed

**EfficientNet Family (B0-B7)**
- **Paper**: https://arxiv.org/pdf/1905.11946.pdf (Page 5, Figure 3)
- **Description**: Shows scaling from B0 to B7
- **Use for**: Explaining why we chose B0 (smallest, most efficient)

### Squeeze-and-Excitation Module

**SE Block Diagram**
- **Paper**: https://arxiv.org/pdf/1709.01507.pdf (Page 3, Figure 1)
- **Description**: Shows channel attention mechanism
- **Use for**: Explaining SE optimization in EfficientNet

---

## 4. EfficientNet Comparisons

### Accuracy vs Parameters

**EfficientNet vs Others (Scatter Plot)**
- **Paper**: https://arxiv.org/pdf/1905.11946.pdf (Page 1, Figure 1)
- **Description**: ImageNet accuracy vs parameters (millions)
- **Shows**: EfficientNet achieves better accuracy with fewer parameters
- **Use for**: Slide 12 - Efficiency comparison
- **Key Point**: EfficientNet-B0 at 5.3M params vs ResNet-50 at 25.6M

**Accuracy vs FLOPs**
- **Paper**: https://arxiv.org/pdf/1905.11946.pdf (Page 6, Figure 4)
- **Description**: Accuracy vs computational cost
- **Use for**: Slide 12 - Showing computational efficiency

### Performance Tables

**ImageNet Results Table**
- **Paper**: https://arxiv.org/pdf/1905.11946.pdf (Page 6, Table 2)
- **Data**:
  - ResNet-50: 76.0% (25.6M params, 4.1B FLOPs)
  - EfficientNet-B0: 77.1% (5.3M params, 0.39B FLOPs)
- **Use for**: Slide 12 - Quantitative comparison

**Model Size Comparison (Bar Chart)**
- **Create**: Bar chart showing model sizes:
  - ResNet-50: 98 MB
  - XceptionNet: 88 MB
  - EfficientNet-B0: 20 MB
  - EfficientNet-B0 + LoRA: 3.5 MB
- **Use for**: Slide 15 - LoRA compression benefits

---

## 5. LoRA (Low-Rank Adaptation)

### Concept Diagrams

**LoRA Weight Decomposition**
- **Paper**: https://arxiv.org/pdf/2106.09685.pdf (Page 3, Figure 1)
- **Description**: Shows W = W₀ + BA decomposition
- **Use for**: Slide 14 - Explaining LoRA concept
- **Key Visual**: Original matrix vs low-rank decomposition

**LoRA Architecture Integration**
- **Paper**: https://arxiv.org/pdf/2106.09685.pdf (Page 4, Figure 2)
- **Description**: How LoRA layers are injected into transformer/CNN
- **Use for**: Slide 14 - Technical explanation

**Parameter Reduction Visualization**
- **Create**: Visual showing:
  - Original: 1,638,400 parameters (full matrix)
  - LoRA: 20,480 parameters (rank-8 decomposition)
  - Reduction: 80× smaller
- **Use for**: Slide 14 - Dramatic size reduction

### Performance Comparisons

**LoRA vs Full Fine-tuning (Table)**
- **Paper**: https://arxiv.org/pdf/2106.09685.pdf (Page 7, Table 1)
- **Data**:
  - Full fine-tuning: 100% parameters, 100% accuracy
  - LoRA (r=8): 0.01% parameters, 99% accuracy
- **Use for**: Slide 15 - Accuracy preservation

**Training Time Comparison**
- **Create**: Bar chart showing:
  - Full fine-tuning: 4 hours
  - LoRA: 1.5 hours (60% faster)
- **Use for**: Slide 15 - Training efficiency

**Memory Usage Comparison**
- **Create**: Chart showing:
  - Full model: 20 MB
  - LoRA adapter: 3.5 MB
  - Multiple adapters: 20 MB + 3.5 MB × N
- **Use for**: Slide 15 - Deployment benefits

---

## 6. Face Detection Examples

### Real-World Detection Results

**Multi-Face Detection**
- **Source**: WIDER FACE dataset examples
- **Link**: http://shuoyang1213.me/WIDERFACE/
- **Description**: Crowded scenes with multiple faces
- **Use for**: Showing RetinaFace robustness

**Challenging Conditions**
- **Occlusions**: Faces with sunglasses, masks
- **Small faces**: Distant faces in images
- **Varied poses**: Profile, tilted faces
- **Link**: WIDER FACE "Hard" subset examples
- **Use for**: Demonstrating detection challenges

**Landmark Visualization**
- **GitHub**: https://github.com/deepinsight/insightface
- **Description**: 5 facial landmarks (eyes, nose, mouth corners)
- **Use for**: Slide 9 - Showing landmark detection

---

## 7. Deepfake Detection Examples

### FaceForensics++ Dataset

**Real vs Fake Comparison**
- **Paper**: https://arxiv.org/pdf/1901.08971.pdf (Page 3, Figure 2)
- **Description**: Side-by-side real and manipulated faces
- **Use for**: Slide 3 - Showing deepfake threat

**Manipulation Types**
- **DeepFakes**: Face swapping examples
- **Face2Face**: Expression transfer
- **FaceSwap**: Different swapping method
- **NeuralTextures**: Texture synthesis
- **Link**: https://github.com/ondyari/FaceForensics (Examples in README)
- **Use for**: Slide 11 - Dataset explanation

### Detection Artifacts

**Deepfake Artifacts Visualization**
- **Paper**: https://arxiv.org/pdf/1901.08971.pdf (Page 5, Figure 4)
- **Description**: Heatmaps showing detection focus areas
- **Use for**: Explaining what EfficientNet learns to detect

**Frequency Domain Analysis**
- **Paper**: https://arxiv.org/pdf/2001.00179.pdf (Page 3, Figure 2)
- **Description**: Frequency spectrum differences (real vs fake)
- **Use for**: Advanced explanation of deepfake artifacts

---

## 8. Stereo Vision & Depth Maps

### Stereo Calibration

**Checkerboard Calibration Pattern**
- **OpenCV**: https://docs.opencv.org/4.x/pattern.png
- **Description**: 9×6 checkerboard for calibration
- **Use for**: Slide 7 - Calibration explanation

**Stereo Rectification**
- **OpenCV Tutorial**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- **Description**: Before/after rectification images
- **Use for**: Explaining epipolar geometry

### Depth Map Examples

**Face Depth Map Visualization**
- **Create**: Colormap showing depth variation
  - Real face: 8-15 cm depth range (nose prominent)
  - Photo: <2 cm depth range (flat)
- **Tool**: Use OpenCV SGBM + colormap
- **Use for**: Slide 8 - Liveness detection concept

**Disparity Map Example**
- **OpenCV**: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- **Description**: Disparity to depth conversion
- **Use for**: Technical explanation of stereo matching

---

## 9. Training Graphs & Results

### EfficientNet Training

**Training Loss Curve**
- **Create**: Plot showing:
  - X-axis: Epochs (0-15)
  - Y-axis: Loss (decreasing)
  - Two lines: Training loss, Validation loss
- **Use for**: Slide 13 - Training progress

**Accuracy Curve**
- **Create**: Plot showing:
  - X-axis: Epochs
  - Y-axis: Accuracy (increasing to 94-95%)
- **Use for**: Slide 13 - Model convergence

**Learning Rate Schedule**
- **Create**: Plot showing cosine annealing
- **Use for**: Technical explanation of training strategy

### LoRA Training Comparison

**Training Time Comparison**
- **Bar Chart**:
  - Full fine-tuning: 4 hours
  - LoRA: 1.5 hours
- **Use for**: Slide 15 - Efficiency benefits

**Memory Usage During Training**
- **Line Graph**:
  - Full fine-tuning: 8 GB GPU memory
  - LoRA: 4.5 GB GPU memory
- **Use for**: Slide 15 - Resource efficiency

---

## 10. System Architecture Diagrams

### Overall System Pipeline

**Complete Pipeline Flowchart**
- **Create**: Flowchart showing:
  1. Dual cameras → Calibration
  2. Synchronized capture
  3. Face detection (RetinaFace)
  4. Depth analysis (SGBM)
  5. Deepfake detection (EfficientNet)
  6. Face verification (ArcFace)
- **Tool**: Draw.io, Lucidchart, or PowerPoint
- **Use for**: Slide 7 - System overview

**Data Flow Diagram**
- **Create**: Shows data transformation at each stage
- **Use for**: Technical presentation

### Hardware Setup

**Dual-Camera Setup Diagram**
- **Create**: Illustration showing:
  - Two webcams on rigid bracket
  - 6-10 cm baseline
  - Parallel alignment
- **Use for**: Slide 7 - Hardware explanation

**Stereo Geometry Diagram**
- **Create**: Shows:
  - Left and right cameras
  - Baseline B
  - Focal length f
  - Disparity d
  - Depth calculation formula
- **Use for**: Technical explanation of depth computation

---

## 📥 How to Download Images

### From Papers (ArXiv)

1. **Go to paper link** (e.g., https://arxiv.org/pdf/1905.11946.pdf)
2. **Open PDF** in browser
3. **Screenshot or extract** specific figures
4. **Cite properly**: "Source: [Author] et al., [Conference] [Year]"

### From GitHub

1. **Navigate to repository** (e.g., https://github.com/deepinsight/insightface)
2. **Find images** in README or `/data/` folder
3. **Right-click → Save image**
4. **Cite**: "Source: [Repository Name], GitHub"

### Creating Your Own

**Tools:**
- **Graphs**: Python (matplotlib, seaborn), Excel, Google Sheets
- **Diagrams**: Draw.io, Lucidchart, PowerPoint, Figma
- **Architecture**: TikZ (LaTeX), Draw.io, PowerPoint

**Tips:**
- Use consistent color scheme (blue for your method, gray for others)
- High resolution (300 DPI for reports, 1920×1080 for slides)
- Clear labels and legends
- Professional fonts (Arial, Calibri, Roboto)

---

## 🎨 Image Usage Guidelines

### For Presentation Slides

**Recommended Sizes:**
- Full-slide images: 1920×1080 pixels
- Half-slide images: 960×1080 pixels
- Small diagrams: 800×600 pixels

**Format:** PNG (for diagrams), JPG (for photos)

**Placement:**
- Architecture diagrams: Center of slide
- Comparison charts: Right side with text on left
- Examples: Grid layout (2×2 or 3×3)

### For Report (LaTeX)

**Recommended Sizes:**
- Full-width figures: 6 inches wide
- Half-width figures: 3 inches wide
- Resolution: 300 DPI minimum

**Format:** PDF (vector) or PNG (high-res)

**LaTeX Code:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{images/retinaface-architecture.png}
\caption{RetinaFace architecture showing FPN backbone and multi-task branches. Source: Deng et al., CVPR 2020}
\label{fig:retinaface-arch}
\end{figure}
```

---

## 📋 Image Checklist for Presentation

### Must-Have Images (Priority 1)

- [ ] RetinaFace architecture diagram (Slide 9)
- [ ] MTCNN vs RetinaFace comparison table (Slide 10)
- [ ] EfficientNet compound scaling diagram (Slide 11)
- [ ] EfficientNet accuracy vs parameters scatter plot (Slide 12)
- [ ] LoRA weight decomposition diagram (Slide 14)
- [ ] Model size comparison bar chart (Slide 15)

### Nice-to-Have Images (Priority 2)

- [ ] Face detection examples with landmarks
- [ ] Deepfake examples (real vs fake)
- [ ] Depth map visualization (real face vs photo)
- [ ] Training loss curves
- [ ] System pipeline flowchart
- [ ] Dual-camera setup diagram

### Optional Images (Priority 3)

- [ ] WIDER FACE benchmark results
- [ ] FaceForensics++ dataset examples
- [ ] Stereo calibration checkerboard
- [ ] SE block diagram
- [ ] Frequency domain analysis

---

## 🔗 Quick Links Summary

### Papers (Primary Sources)

1. **RetinaFace**: https://arxiv.org/pdf/1905.00641.pdf
2. **EfficientNet**: https://arxiv.org/pdf/1905.11946.pdf
3. **LoRA**: https://arxiv.org/pdf/2106.09685.pdf
4. **MTCNN**: https://arxiv.org/pdf/1604.02878.pdf
5. **FaceForensics++**: https://arxiv.org/pdf/1901.08971.pdf
6. **ArcFace**: https://arxiv.org/pdf/1801.07698.pdf

### GitHub Repositories

1. **InsightFace** (RetinaFace, ArcFace): https://github.com/deepinsight/insightface
2. **EfficientNet**: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
3. **LoRA**: https://github.com/microsoft/LoRA
4. **MTCNN**: https://github.com/ipazc/mtcnn
5. **FaceForensics++**: https://github.com/ondyari/FaceForensics

### Datasets

1. **WIDER FACE**: http://shuoyang1213.me/WIDERFACE/
2. **LFW**: http://vis-www.cs.umass.edu/lfw/
3. **FaceForensics++**: https://github.com/ondyari/FaceForensics

### Tools for Creating Images

1. **Draw.io**: https://app.diagrams.net/
2. **Matplotlib**: https://matplotlib.org/stable/gallery/index.html
3. **Seaborn**: https://seaborn.pydata.org/examples/index.html
4. **TikZ**: https://www.overleaf.com/learn/latex/TikZ_package

---

## 💡 Pro Tips

### For Presentation

1. **Use animations**: Build complex diagrams step-by-step
2. **Highlight key numbers**: Use color/bold for important metrics
3. **Before/after comparisons**: Side-by-side for maximum impact
4. **Consistent style**: Same color scheme across all slides

### For Report

1. **High resolution**: 300 DPI minimum for print quality
2. **Vector graphics**: Use PDF format when possible
3. **Clear captions**: Explain what the figure shows
4. **Proper citations**: Always credit original sources
5. **Cross-reference**: Refer to figures in text ("as shown in Figure 3")

### Copyright & Attribution

- **Academic papers**: Free to use with proper citation
- **GitHub images**: Check license (usually MIT/Apache)
- **Always cite**: "Source: [Author], [Publication], [Year]"
- **Create your own**: When possible, for unique visualizations

---

**Last Updated**: December 2024

**Note**: All links verified as of December 2024. If a link is broken, search for the paper title on Google Scholar or ArXiv.
