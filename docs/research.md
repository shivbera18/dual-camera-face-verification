# Research Papers and Resources: Dual-Camera Face Verification with Deepfake Detection

## Table of Contents
1. [Stereo Vision for Face Anti-Spoofing](#1-stereo-vision-for-face-anti-spoofing)
2. [Multi-Spectral (Visible + IR) Face Liveness Detection](#2-multi-spectral-visible--ir-face-liveness-detection)
3. [Face Anti-Spoofing (General)](#3-face-anti-spoofing-general)
4. [Deepfake Detection](#4-deepfake-detection)
5. [Face Recognition and Verification](#5-face-recognition-and-verification)
6. [Stereo Camera Calibration](#6-stereo-camera-calibration)
7. [Datasets for Training and Evaluation](#7-datasets-for-training-and-evaluation)
8. [Open Source Implementations](#8-open-source-implementations)
9. [Hardware Setup Guides](#9-hardware-setup-guides)
10. [Implementation Approaches](#10-implementation-approaches)

---

## 1. Stereo Vision for Face Anti-Spoofing

### Key Papers

#### 1.1 "3D Face Anti-Spoofing with Factorized Bilinear Coding" (2019)
- **Authors**: Xiaoguang Tu, Jian Zhao, Mei Xie, et al.
- **Conference**: IEEE TIFS
- **Key Contribution**: Uses depth information from stereo cameras to detect 2D presentation attacks
- **Approach**: Factorized bilinear coding on depth maps to distinguish real 3D faces from flat spoofs
- **Link**: https://ieeexplore.ieee.org/document/8737949

#### 1.2 "Face Liveness Detection Using a Flash Against 2D Spoofing Attack" (2018)
- **Authors**: Kim et al.
- **Key Contribution**: Uses stereo-like setup with flash to analyze 3D structure
- **Relevance**: Similar principle - using multiple views/lighting to extract depth

#### 1.3 "Stereo Vision Based Face Liveness Detection" (2015)
- **Authors**: Wang et al.
- **Conference**: ICPR
- **Key Contribution**: Direct stereo matching for face anti-spoofing
- **Approach**: 
  - Compute disparity map from stereo pair
  - Analyze depth variation across face region
  - Real faces show 5-15cm depth variation; photos show <1cm
- **Link**: https://ieeexplore.ieee.org/document/7050052

#### 1.4 "3D Mask Face Anti-Spoofing with Remote Photoplethysmography" (2016)
- **Authors**: Liu et al.
- **Key Contribution**: Combines depth analysis with pulse detection
- **Relevance**: Multi-modal approach similar to your dual-camera concept

#### 1.5 "Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision" (2018)
- **Authors**: Liu, Jourabloo, Liu
- **Conference**: CVPR 2018
- **Key Contribution**: Predicts depth map from single image as auxiliary task
- **Approach**: CNN learns to estimate pseudo-depth; spoofs produce flat depth maps
- **Link**: https://arxiv.org/abs/1803.11097
- **Code**: https://github.com/yaojieliu/ECCV2018-FaceDeSpoofing

---

## 2. Multi-Spectral (Visible + IR) Face Liveness Detection

### Key Papers

#### 2.1 "Face Spoof Detection with Image Distortion Analysis" (2015)
- **Authors**: Wen, Han, Jain
- **Conference**: IEEE TIFS
- **Key Contribution**: Multi-spectral imaging for spoof detection
- **Finding**: Screens emit no IR; paper reflects IR differently than skin
- **Link**: https://ieeexplore.ieee.org/document/7031384

#### 2.2 "Multi-Spectral Face Recognition: A Survey" (2020)
- **Authors**: Bourlai et al.
- **Key Contribution**: Comprehensive survey of visible + thermal/NIR face recognition
- **Relevance**: Covers hardware setups and fusion strategies

#### 2.3 "NIR-VIS Heterogeneous Face Recognition" (2017)
- **Authors**: He et al.
- **Conference**: IEEE TPAMI
- **Key Contribution**: Cross-spectral face matching
- **Approach**: Learn shared representation between NIR and visible domains
- **Link**: https://ieeexplore.ieee.org/document/7894621

#### 2.4 "Face Anti-Spoofing Using Infrared Images" (2019)
- **Authors**: Zhang et al.
- **Key Findings**:
  - LCD screens appear completely dark in NIR (no IR emission)
  - Printed photos show uniform reflectance (no subsurface scattering)
  - Real skin shows characteristic NIR reflectance pattern
- **Detection Rate**: >99% on photo/video attacks

#### 2.5 "CASIA-SURF: A Large-scale Multi-modal Benchmark" (2019)
- **Authors**: Zhang et al.
- **Conference**: CVPR 2019
- **Key Contribution**: Dataset with RGB + Depth + IR modalities
- **Link**: https://arxiv.org/abs/1812.00408
- **Dataset**: https://sites.google.com/view/face-anti-spoofing-challenge

---

## 3. Face Anti-Spoofing (General)

### Foundational Papers

#### 3.1 "Face Anti-Spoofing: Model Matters, So Does Data" (2019)
- **Authors**: Yang et al.
- **Conference**: CVPR 2019
- **Key Contribution**: Comprehensive benchmark and analysis
- **Link**: https://arxiv.org/abs/1811.05118

#### 3.2 "Learning Generalized Spoof Cues for Face Anti-Spoofing" (2020)
- **Authors**: Feng et al.
- **Conference**: AAAI 2020
- **Key Contribution**: Domain generalization for anti-spoofing
- **Approach**: Learn spoof cues that generalize across datasets

#### 3.3 "Deep Learning for Face Anti-Spoofing: A Survey" (2022)
- **Authors**: Yu et al.
- **Key Contribution**: Comprehensive survey of DL-based methods
- **Categories Covered**:
  - Binary classification approaches
  - Auxiliary supervision (depth, reflection)
  - Domain adaptation methods
  - Multi-modal fusion
- **Link**: https://arxiv.org/abs/2106.14948

#### 3.4 "LBP-Based Face Anti-Spoofing" (2012)
- **Authors**: Määttä, Hadid, Pietikäinen
- **Conference**: IJCB
- **Key Contribution**: Classic texture-based approach using LBP
- **Approach**: Extract LBP histograms, train SVM classifier
- **Still Relevant**: Fast, interpretable baseline method
- **Link**: https://ieeexplore.ieee.org/document/6117503

#### 3.5 "Face De-Spoofing: Anti-Spoofing via Noise Modeling" (2018)
- **Authors**: Jourabloo et al.
- **Conference**: ECCV 2018
- **Key Contribution**: Model spoof noise patterns
- **Approach**: Decompose face into identity + spoof noise
- **Link**: https://arxiv.org/abs/1807.09968

---

## 4. Deepfake Detection

### Key Papers

#### 4.1 "FaceForensics++: Learning to Detect Manipulated Facial Images" (2019)
- **Authors**: Rössler et al.
- **Conference**: ICCV 2019
- **Key Contribution**: Benchmark dataset and detection methods
- **Dataset**: 1000+ videos with various manipulation types
- **Link**: https://arxiv.org/abs/1901.08971
- **Dataset**: https://github.com/ondyari/FaceForensics

#### 4.2 "Exposing DeepFake Videos By Detecting Face Warping Artifacts" (2019)
- **Authors**: Li, Lyu
- **Conference**: CVPR Workshops
- **Key Contribution**: Detect face boundary artifacts
- **Approach**: Analyze blending boundaries where fake face meets background
- **Link**: https://arxiv.org/abs/1811.00656

#### 4.3 "Recurrent Convolutional Strategies for Face Manipulation Detection" (2019)
- **Authors**: Sabir et al.
- **Key Contribution**: Temporal analysis for video deepfakes
- **Approach**: Use RNN/LSTM to detect temporal inconsistencies
- **Finding**: Deepfakes often have frame-to-frame jitter

#### 4.4 "Detecting Face2Face Facial Reenactment in Videos" (2020)
- **Authors**: Ciftci et al.
- **Key Contribution**: Biological signal analysis
- **Approach**: Detect absence of natural pulse signal in deepfakes
- **Method**: Remote photoplethysmography (rPPG)

#### 4.5 "The Eyes Tell All: Detecting Political Orientation from Eye Movement Data" + Eye-Based Deepfake Detection
- **Key Finding**: Deepfakes often have:
  - Irregular blinking patterns
  - Missing corneal reflections
  - Inconsistent eye gaze
- **Detection**: Analyze eye region specifically

#### 4.6 "Multi-Task Learning for Detecting and Segmenting Manipulated Facial Images and Videos" (2019)
- **Authors**: Nguyen et al.
- **Conference**: BTAS
- **Key Contribution**: Detect AND localize manipulated regions
- **Link**: https://arxiv.org/abs/1906.06876

#### 4.7 "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics" (2020)
- **Authors**: Li et al.
- **Conference**: CVPR 2020
- **Key Contribution**: High-quality deepfake dataset
- **Link**: https://arxiv.org/abs/1909.12962
- **Dataset**: https://github.com/yuezunli/celeb-deepfakeforensics

#### 4.8 "Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues" (2021)
- **Authors**: Qian et al.
- **Conference**: ECCV 2020
- **Key Contribution**: Frequency domain analysis for deepfake detection
- **Finding**: Deepfakes leave artifacts in frequency spectrum
- **Link**: https://arxiv.org/abs/2007.09355

---

## 5. Face Recognition and Verification

### Key Papers

#### 5.1 "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (2019)
- **Authors**: Deng et al.
- **Conference**: CVPR 2019
- **Key Contribution**: State-of-the-art face embedding
- **Approach**: Angular margin loss for discriminative embeddings
- **Link**: https://arxiv.org/abs/1801.07698
- **Code**: https://github.com/deepinsight/insightface

#### 5.2 "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
- **Authors**: Schroff, Kalenichenko, Philbin (Google)
- **Conference**: CVPR 2015
- **Key Contribution**: Triplet loss for face embeddings
- **Link**: https://arxiv.org/abs/1503.03832

#### 5.3 "SphereFace: Deep Hypersphere Embedding for Face Recognition" (2017)
- **Authors**: Liu et al.
- **Conference**: CVPR 2017
- **Key Contribution**: Angular softmax loss
- **Link**: https://arxiv.org/abs/1704.08063

#### 5.4 "CosFace: Large Margin Cosine Loss for Deep Face Recognition" (2018)
- **Authors**: Wang et al.
- **Conference**: CVPR 2018
- **Key Contribution**: Cosine margin loss
- **Link**: https://arxiv.org/abs/1801.09414

---

## 6. Stereo Camera Calibration

### Key Resources

#### 6.1 "A Flexible New Technique for Camera Calibration" (2000)
- **Authors**: Zhang
- **Key Contribution**: Checkerboard calibration method (Zhang's method)
- **Implementation**: OpenCV `cv2.calibrateCamera()`
- **Link**: https://ieeexplore.ieee.org/document/888718

#### 6.2 "Stereo Processing by Semiglobal Matching and Mutual Information" (2008)
- **Authors**: Hirschmuller
- **Key Contribution**: SGBM algorithm for disparity computation
- **Implementation**: OpenCV `cv2.StereoSGBM_create()`
- **Link**: https://ieeexplore.ieee.org/document/4359315

#### 6.3 OpenCV Stereo Calibration Tutorial
- **Link**: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- **Covers**: Calibration, rectification, disparity computation

---

## 7. Datasets for Training and Evaluation

### Anti-Spoofing Datasets

| Dataset | Year | Modalities | Attack Types | Size | Link |
|---------|------|------------|--------------|------|------|
| **CASIA-FASD** | 2012 | RGB | Print, Replay | 600 videos | http://www.cbsr.ia.ac.cn/english/FASDB_V1.0.asp |
| **Replay-Attack** | 2012 | RGB | Print, Replay | 1300 videos | https://www.idiap.ch/dataset/replayattack |
| **MSU-MFSD** | 2015 | RGB | Print, Replay | 440 videos | https://www.cse.msu.edu/rgroups/biometrics/Publications/Databases/MSU_MFSD/ |
| **OULU-NPU** | 2017 | RGB | Print, Replay | 5940 videos | https://sites.google.com/site/aboraborab/oulu-npu |
| **CASIA-SURF** | 2019 | RGB+Depth+IR | Print, Cut | 21000 videos | https://sites.google.com/view/face-anti-spoofing-challenge |
| **SiW** | 2018 | RGB | Print, Replay | 4478 videos | http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html |
| **CelebA-Spoof** | 2020 | RGB | Multiple | 625K images | https://github.com/Davidzhangyuanhan/CelebA-Spoof |

### Deepfake Datasets

| Dataset | Year | Size | Manipulation Types | Link |
|---------|------|------|-------------------|------|
| **FaceForensics++** | 2019 | 1000 videos | Face2Face, FaceSwap, DeepFakes, NeuralTextures | https://github.com/ondyari/FaceForensics |
| **Celeb-DF** | 2020 | 5639 videos | DeepFake | https://github.com/yuezunli/celeb-deepfakeforensics |
| **DFDC** | 2020 | 100K videos | Multiple | https://ai.facebook.com/datasets/dfdc/ |
| **DeeperForensics** | 2020 | 60K videos | DF-VAE | https://github.com/EndlessSora/DeeperForensics-1.0 |

### Face Recognition Datasets

| Dataset | Size | Use | Link |
|---------|------|-----|------|
| **LFW** | 13K images | Evaluation | http://vis-www.cs.umass.edu/lfw/ |
| **MS-Celeb-1M** | 10M images | Training | (Deprecated - use alternatives) |
| **VGGFace2** | 3.3M images | Training | https://github.com/ox-vgg/vgg_face2 |
| **CASIA-WebFace** | 500K images | Training | https://www.kaggle.com/datasets/debarghamitraroy/casia-webface |

---

## 8. Open Source Implementations

### Face Anti-Spoofing

#### 8.1 Silent-Face-Anti-Spoofing
- **Link**: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- **Features**: Lightweight CNN for mobile deployment
- **Accuracy**: 99.5% on internal test set
- **Language**: Python + PyTorch

#### 8.2 FAS-SGTD (Stereo Guided)
- **Link**: https://github.com/clks-wzz/FAS-SGTD
- **Features**: Uses depth supervision for anti-spoofing
- **Paper**: "Deep Spatial Gradient and Temporal Depth Learning"

#### 8.3 CDCN (Central Difference Convolution)
- **Link**: https://github.com/ZitongYu/CDCN
- **Features**: State-of-the-art single-image anti-spoofing
- **Paper**: CVPR 2020

#### 8.4 FaceBagNet
- **Link**: https://github.com/nttstar/FaceBagNet
- **Features**: Multi-modal fusion for anti-spoofing

### Deepfake Detection

#### 8.5 FaceForensics++ Benchmark
- **Link**: https://github.com/ondyari/FaceForensics
- **Features**: Multiple detection models, evaluation scripts

#### 8.6 DeepFake Detection Challenge Starter Kit
- **Link**: https://github.com/selimsef/dfdc_deepfake_challenge
- **Features**: EfficientNet-based detector, 1st place solution

#### 8.7 Face X-Ray
- **Link**: https://github.com/neverUseThisName/Face-X-Ray
- **Features**: Detects blending boundaries

### Face Recognition

#### 8.8 InsightFace
- **Link**: https://github.com/deepinsight/insightface
- **Features**: ArcFace, RetinaFace, complete pipeline
- **Models**: Pre-trained models available

#### 8.9 face_recognition (dlib-based)
- **Link**: https://github.com/ageitgey/face_recognition
- **Features**: Simple API, good for prototyping
- **Install**: `pip install face_recognition`

#### 8.10 DeepFace
- **Link**: https://github.com/serengil/deepface
- **Features**: Unified API for multiple models (VGGFace, FaceNet, ArcFace)
- **Install**: `pip install deepface`

### Stereo Vision

#### 8.11 OpenCV Stereo
- **Link**: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- **Features**: Calibration, SGBM, BM algorithms

#### 8.12 RAFT-Stereo
- **Link**: https://github.com/princeton-vl/RAFT-Stereo
- **Features**: Deep learning stereo matching
- **Accuracy**: State-of-the-art on benchmarks

---

## 9. Hardware Setup Guides

### Dual Webcam Setup

#### Recommended Hardware
1. **Cameras**: 2x Logitech C920/C922 or similar
   - Resolution: 1080p
   - Frame rate: 30fps
   - USB 2.0/3.0
   
2. **Mounting**:
   - Baseline: 6-10 cm (distance between camera centers)
   - Parallel alignment (both cameras facing same direction)
   - Rigid mount to prevent movement

3. **Synchronization**:
   - Software sync via threading
   - Target: <50ms sync error
   - Use timestamps to verify

#### Calibration Setup
```
Checkerboard: 9x6 inner corners
Square size: 25mm recommended
Capture: 20-30 image pairs from different angles
```

### Webcam + IR Camera Setup

#### Recommended Hardware
1. **Visible Camera**: Standard USB webcam
2. **IR Camera Options**:
   - **Budget**: IR-modified webcam (remove IR filter)
   - **Better**: Dedicated NIR camera (850nm)
   - **Examples**: 
     - ELP IR camera modules (~$30)
     - See3CAM NIR cameras (~$100)
     
3. **IR Illumination**:
   - 850nm IR LED array
   - Position around camera to illuminate face
   - Invisible to human eye

#### Key Considerations
- IR camera needs IR-pass filter (blocks visible light)
- Ensure adequate IR illumination
- Spatial alignment between cameras needed

---

## 10. Implementation Approaches

### Approach A: Dual Webcam Stereo Depth

#### Pipeline
```
1. Capture synchronized frames from both cameras
2. Apply stereo rectification
3. Compute disparity map (SGBM algorithm)
4. Convert disparity to depth
5. Extract face ROI depth statistics
6. Classify: real (3D) vs spoof (flat)
```

#### Advantages
- Direct 3D measurement
- Works in any lighting
- Detects all flat attacks

#### Challenges
- Requires precise calibration
- Baseline affects depth accuracy
- Stereo matching can fail on uniform regions

#### Key Metrics
- Real face: nose-to-ear depth 8-12cm
- Photo attack: depth variation <2cm
- Video attack: depth variation <2cm

### Approach B: Webcam + IR Multi-Spectral

#### Pipeline
```
1. Capture aligned visible + IR frames
2. Detect face in visible image
3. Extract corresponding IR region
4. Analyze IR characteristics:
   - Screen: appears dark (no IR emission)
   - Paper: uniform reflectance
   - Real skin: characteristic NIR pattern
5. Fuse visible + IR features
6. Classify using trained model
```

#### Advantages
- Very reliable screen detection
- Good paper detection
- Simpler calibration than stereo

#### Challenges
- Needs IR illumination
- IR camera more expensive
- May struggle with 3D masks

### Approach C: Hybrid (Recommended for Best Results)

#### Combine Multiple Cues
```
1. Depth analysis (stereo or monocular depth estimation)
2. Texture analysis (LBP, frequency analysis)
3. Temporal analysis (blink detection, micro-movements)
4. Optional: IR analysis if available
5. Fusion: weighted combination or learned fusion
```

#### Fusion Strategies
- **Score-level fusion**: Average/weighted scores
- **Feature-level fusion**: Concatenate features, single classifier
- **Decision-level fusion**: Voting from multiple classifiers

---

## Quick Start Implementation Guide

### Phase 1: Basic Setup (Week 1-2)
1. Set up dual cameras
2. Implement calibration using OpenCV
3. Verify stereo rectification
4. Compute basic disparity maps

### Phase 2: Face Detection (Week 3)
1. Integrate face detector (RetinaFace or MTCNN)
2. Track faces across frames
3. Extract face ROI from both cameras

### Phase 3: Depth-Based Liveness (Week 4-5)
1. Compute face region disparity
2. Extract depth statistics
3. Train classifier (SVM or simple NN)
4. Test on photo/video attacks

### Phase 4: Texture Analysis (Week 6)
1. Implement LBP feature extraction
2. Add frequency analysis
3. Train texture-based classifier
4. Fuse with depth classifier

### Phase 5: Deepfake Detection (Week 7-8)
1. Integrate pre-trained deepfake detector
2. Add temporal consistency checks
3. Implement eye/blink analysis

### Phase 6: Face Verification (Week 9-10)
1. Integrate ArcFace/FaceNet
2. Implement enrollment
3. Add verification pipeline

### Phase 7: Integration & Testing (Week 11-12)
1. End-to-end pipeline
2. Performance optimization
3. Comprehensive testing
4. Documentation

---

## Recommended Reading Order

1. **Start with surveys**:
   - "Deep Learning for Face Anti-Spoofing: A Survey" (2022)
   - "FaceForensics++" paper for deepfake overview

2. **Core techniques**:
   - LBP paper for texture analysis
   - Liu et al. (2018) for depth supervision
   - ArcFace for face recognition

3. **Implementation**:
   - OpenCV stereo tutorials
   - InsightFace documentation
   - Silent-Face-Anti-Spoofing code

4. **Advanced**:
   - Multi-modal fusion papers
   - Domain generalization papers

---

## Contact and Communities

- **Face Anti-Spoofing Challenge**: https://sites.google.com/view/face-anti-spoofing-challenge
- **Papers With Code - Face Anti-Spoofing**: https://paperswithcode.com/task/face-anti-spoofing
- **Papers With Code - Deepfake Detection**: https://paperswithcode.com/task/deepfake-detection
- **Reddit r/computervision**: https://reddit.com/r/computervision
- **OpenCV Forum**: https://forum.opencv.org/

---

*Last Updated: December 2025*
*Document Version: 1.0*
