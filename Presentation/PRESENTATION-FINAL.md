# Dual-Camera Face Verification System
## Final Presentation (12 Slides)

---

## Slide 1: Title Slide

### Dual-Camera Face Verification System
**Stereo Vision-Based Liveness Detection with Deepfake Prevention**

**Shivratan Bera** (2022BECE103)  
Electronics and Communication Engineering  
National Institute of Technology, Srinagar  
**Guide:** Dr. Gausia Qazi  
December 2024

---

## Slide 2: Introduction

### Why Face Verification?

**Traditional Methods:**
- Passwords: Forgotten, stolen, phished
- OTPs: SIM swapping attacks
- Tokens: Lost, expensive

**Face Verification:**
- Non-intrusive, fast (15-50ms)
- Cannot be forgotten or lost
- Ubiquitous hardware

**📊 IMAGE: Real-world applications (Apple Face ID, banking, airports)**

---

## Slide 3: The Problem

### Security Vulnerabilities

**Presentation Attacks:**
- Photo attacks (80% success)
- Video replay attacks
- 3D masks

**Deepfake Attacks:**
- Face swapping
- Real-time reenactment
- Synthetic faces

**Core Issue:** Single-camera lacks depth → Cannot distinguish 2D from 3D

**📊 IMAGE: Attack examples (photo, video replay, deepfake)**

---

## Slide 4: Existing Solutions

### Anti-Spoofing Methods

**Depth Sensors (Intel RealSense, ToF)**
- ✅ High accuracy (>98%)
- ❌ Expensive hardware (₹15,000+)
- ❌ Not in consumer devices

**Texture-Based (LBP + SVM)**
- ✅ Fast, lightweight
- ❌ Low accuracy (85-90%)
- ❌ Fails on high-quality prints

**CNN-Based Anti-Spoofing**
- ✅ Better accuracy (92-95%)
- ❌ Poor generalization to new attacks
- ❌ Requires large labeled datasets

**Deepfake Detectors (XceptionNet)**
- ✅ Good on known methods (90-95%)
- ❌ Drops to 70-80% on unseen methods
- ❌ Adversarial arms race

**📊 IMAGE: Comparison table with checkmarks/crosses**

---

## Slide 5: Proposed Solution

### Dual-Camera Stereo Vision

**Components:**
1. Stereo Depth → Liveness detection
2. RetinaFace → Face detection
3. EfficientNet-B0 → Deepfake detection
4. LoRA → Model compression

**Cost:** ₹3,000 (vs ₹15,000 depth sensors)

**📊 IMAGE: System architecture flowchart**

---

## Slide 6: RetinaFace

| Detector | Accuracy | Speed | Landmarks |
|----------|----------|-------|-----------|
| MTCNN | 92-94% | 50-80ms | ✅ |
| YOLO-Face | 94-95% | 10-15ms | ❌ |
| **RetinaFace** | **97%** | **20-30ms** | ✅ |

**Why RetinaFace:**
- 97% accuracy (WIDER FACE)
- 3× faster than MTCNN
- 5 landmarks for alignment

**📊 IMAGE: RetinaFace architecture (FPN)**
**📊 IMAGE: Detection with 5 landmarks**

---

## Slide 7: EfficientNet-B0

### Compound Scaling

**Formula:**
```
depth:  d = α^φ
width:  w = β^φ
resolution: r = γ^φ
Constraint: α × β² × γ² ≈ 2
```

**Architecture:** MBConv + Squeeze-Excitation, 224×224 input

**📊 IMAGE: Compound scaling visualization**
**📊 IMAGE: EfficientNet-B0 architecture**

---

## Slide 8: EfficientNet - Comparison

| Model | Parameters | FLOPs | Accuracy |
|-------|------------|-------|----------|
| ResNet-50 | 25.6M | 4.1B | 76.0% |
| XceptionNet | 23M | 8.4B | - |
| MobileNetV2 | 3.5M | 0.3B | 72.0% |
| **EfficientNet-B0** | **5.3M** | **0.39B** | **77.1%** |

**5× fewer parameters, 10× fewer FLOPs, CPU capable**

**📊 IMAGE: Accuracy vs parameters scatter plot**

---

## Slide 9: Model Selection Rationale

### Why These Specific Models?

**RetinaFace Selection:**
- Compared: MTCNN, YOLO-Face, Haar Cascades
- Winner: 97% accuracy + 3× faster than MTCNN
- Provides 5 landmarks (essential for alignment)

**EfficientNet-B0 Selection:**
- Compared: ResNet-50, XceptionNet, MobileNetV2
- Winner: Best accuracy-efficiency trade-off
- 5× fewer parameters, 10× fewer FLOPs
- CPU inference capable

**LoRA Selection:**
- Problem: 20 MB model too large for edge
- Solution: 35× compression with <1% accuracy loss
- Enables deployment on resource-constrained devices

**📊 IMAGE: Decision tree or comparison matrix**

---

## Slide 10: LoRA

### Low-Rank Adaptation

**Concept:** W = W₀ + BA

```
Original: 1280×1280 = 1,638,400 params
LoRA: (1280×8) + (8×1280) = 20,480 params
Reduction: 80× per layer
```

**Full Model:** 20 MB → 3.5 MB (35× compression)

**📊 IMAGE: LoRA decomposition diagram**

---

## Slide 11: LoRA - Results

**Compression:** 35× (20 MB → 3.5 MB)
**Accuracy:** <1% drop (94.5% → 93.5%)

**Benefits:**
- Edge deployment
- Multiple adapters
- 60% faster training
- CPU inference: 15-20 FPS

**📊 IMAGE: Model size comparison**

---

## Slide 12: References

**Face Detection:**
1. Deng, J. et al. (2020). "RetinaFace." CVPR.
2. Zhang, K. et al. (2016). "MTCNN." IEEE SPL.

**Deepfake Detection:**
3. Tan, M. & Le, Q. (2019). "EfficientNet." ICML.
4. Rössler, A. et al. (2019). "FaceForensics++." ICCV.

**Model Compression:**
5. Hu, E. et al. (2021). "LoRA." ICLR 2022.

**Stereo Vision:**
6. Zhang, Z. (2000). "Camera calibration." IEEE TPAMI.
7. Hirschmuller, H. (2008). "SGBM." IEEE TPAMI.

**GitHub:** github.com/shivbera18/dual-camera-face-verification

---

## Thank You!

**Questions?**

Shivratan Bera (2022BECE103)  
2022bece103@nitsri.ac.in

---

**Total: 12 Slides | Time: 15-20 minutes**
