# Technical Specification: Dual-Camera Face Verification System

## Document Overview

This document provides complete technical details for implementing a stereo vision-based face verification system with deepfake detection. Every technology choice is explained with alternatives considered and reasons for selection.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Hardware Specifications](#2-hardware-specifications)
3. [Stereo Vision & Depth Estimation](#3-stereo-vision--depth-estimation)
4. [Face Detection](#4-face-detection)
5. [Face Anti-Spoofing (Liveness Detection)](#5-face-anti-spoofing-liveness-detection)
6. [Deepfake Detection](#6-deepfake-detection)
7. [Face Recognition & Verification](#7-face-recognition--verification)
8. [Datasets](#8-datasets)
9. [Software Stack](#9-software-stack)
10. [Model Training Strategy](#10-model-training-strategy)
11. [Performance Benchmarks](#11-performance-benchmarks)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. System Architecture Overview

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DUAL-CAMERA FACE VERIFICATION SYSTEM                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐  ┌──────────┐                                                 │
│  │ Camera 1 │  │ Camera 2 │   HARDWARE LAYER                                │
│  │  (Left)  │  │  (Right) │                                                 │
│  └────┬─────┘  └────┬─────┘                                                 │
│       │             │                                                        │
│       ▼             ▼                                                        │
│  ┌─────────────────────────┐                                                │
│  │   Stereo Calibration    │  ← Checkerboard calibration (one-time)        │
│  │   & Rectification       │                                                │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────────┐                                                │
│  │   Synchronized Frame    │  ← Threading + timestamp matching              │
│  │      Acquisition        │                                                │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────────┐                                                │
│  │    Face Detection       │  ← RetinaFace / MTCNN                          │
│  │    (Both Frames)        │                                                │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│       ┌──────┴──────┐                                                       │
│       ▼             ▼                                                        │
│  ┌─────────┐  ┌──────────────┐                                              │
│  │ Stereo  │  │   Texture    │                                              │
│  │  Depth  │  │   Analysis   │   ANTI-SPOOFING LAYER                        │
│  │Analysis │  │    (LBP)     │                                              │
│  └────┬────┘  └──────┬───────┘                                              │
│       │              │                                                       │
│       ▼              ▼                                                       │
│  ┌─────────────────────────┐                                                │
│  │   Deepfake Detection    │  ← EfficientNet-B0 + Temporal Analysis         │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────────┐                                                │
│  │   Score Fusion &        │  ← Weighted combination of all scores          │
│  │   Liveness Decision     │                                                │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│              ▼ (If LIVE)                                                     │
│  ┌─────────────────────────┐                                                │
│  │   Face Embedding        │  ← ArcFace (ResNet-50 backbone)                │
│  │   Extraction            │                                                │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────────┐                                                │
│  │   Identity Matching     │  ← Cosine similarity with enrolled templates   │
│  └───────────┬─────────────┘                                                │
│              │                                                               │
│              ▼                                                               │
│  ┌─────────────────────────┐                                                │
│  │   ACCEPT / REJECT       │  ← Final decision with confidence scores       │
│  └─────────────────────────┘                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | Primary Technology | Backup Option | Output |
|--------|-------------------|---------------|--------|
| Calibration | OpenCV stereoCalibrate | - | Calibration matrices |
| Frame Capture | OpenCV VideoCapture + Threading | - | Synchronized frame pairs |
| Face Detection | RetinaFace | MTCNN | Bounding boxes, landmarks |
| Depth Analysis | OpenCV SGBM | RAFT-Stereo | Depth map, depth score |
| Texture Analysis | LBP + SVM | CNN-based | Spoof probability |
| Deepfake Detection | EfficientNet-B0 | XceptionNet | Fake probability |
| Face Embedding | ArcFace (InsightFace) | FaceNet | 512-D vector |
| Identity Matching | Cosine Similarity | Euclidean Distance | Match score |

---

## 2. Hardware Specifications

### 2.1 Camera Selection

#### RECOMMENDED: Logitech C270 / C310

| Specification | Value | Why It Matters |
|--------------|-------|----------------|
| Resolution | 720p (1280×720) | Sufficient for face detection |
| Frame Rate | 30 fps | Smooth real-time processing |
| Focus | Fixed | No autofocus hunting issues |
| Price | ₹1,200-1,500 each | Budget-friendly |
| Availability | High | Easy to replace |

#### Why NOT Higher Resolution?
- 1080p = 4x more pixels = 4x slower processing
- 720p is standard for face recognition research
- Most pre-trained models expect 224×224 or 112×112 input anyway

#### Alternative Options

| Camera | Price | Pros | Cons |
|--------|-------|------|------|
| Generic 720p USB | ₹500-800 | Cheapest | Variable quality |
| Logitech C270 | ₹1,200 | Reliable, consistent | Basic |
| Logitech C920 | ₹6,000 | 1080p, better low-light | Overkill, expensive |
| ELP USB Camera Module | ₹1,500 | Compact, no housing | Needs mounting |

### 2.2 Stereo Mounting Setup

```
┌─────────────────────────────────────────────────────────┐
│                    MOUNTING BRACKET                      │
│                                                          │
│    ┌─────────┐                      ┌─────────┐         │
│    │ Camera  │◄────── 6-10 cm ─────►│ Camera  │         │
│    │  LEFT   │       (Baseline)      │  RIGHT  │         │
│    └─────────┘                      └─────────┘         │
│                                                          │
│    ◄──────────────── 20-25 cm ─────────────────►        │
│                   (Total width)                          │
└─────────────────────────────────────────────────────────┘
```

#### Baseline Distance: 6-10 cm

**Why this range?**

| Baseline | Depth Range | Accuracy | Use Case |
|----------|-------------|----------|----------|
| 3 cm | 20-50 cm | Low | Too close, poor depth |
| **6-10 cm** | **30-100 cm** | **Good** | **Face verification (ideal)** |
| 15 cm | 50-200 cm | Medium | Room scanning |
| 30 cm | 100-500 cm | High | Outdoor/robotics |

**Formula:** `Depth = (focal_length × baseline) / disparity`

- Larger baseline = better depth accuracy at distance
- But too large = face may not be visible in both cameras
- 6-10 cm is optimal for face at 40-80 cm distance

### 2.3 Mounting Options

| Option | Cost | Difficulty | Stability |
|--------|------|------------|-----------|
| Cardboard + tape | ₹50 | Easy | Poor |
| Wooden plank + screws | ₹200 | Medium | Good |
| 3D printed bracket | ₹500 | Medium | Excellent |
| Aluminum profile | ₹800 | Hard | Excellent |

**Recommendation:** Start with wooden plank, upgrade to 3D printed if needed.

---

## 3. Stereo Vision & Depth Estimation

### 3.1 Calibration Process

#### What We Need to Compute

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Intrinsic Matrix (Left) | K₁ | Focal length, principal point |
| Intrinsic Matrix (Right) | K₂ | Focal length, principal point |
| Distortion Coefficients | D₁, D₂ | Lens distortion correction |
| Rotation Matrix | R | Rotation between cameras |
| Translation Vector | T | Position offset between cameras |
| Rectification Transforms | R₁, R₂ | Align image planes |
| Projection Matrices | P₁, P₂ | 3D to 2D mapping |

#### Calibration Checkerboard

```
Recommended: 9×6 inner corners, 25mm square size

┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ ■ │   │ ■ │   │ ■ │   │ ■ │   │ ■ │   │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │ ■ │   │ ■ │   │ ■ │   │ ■ │   │ ■ │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ ■ │   │ ■ │   │ ■ │   │ ■ │   │ ■ │   │
...
```

**Print on A4 paper, mount on flat rigid surface (cardboard/foam board)**

#### Calibration Code

```python
import cv2
import numpy as np
import json

class StereoCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=25.0):
        """
        checkerboard_size: (columns, rows) of inner corners
        square_size: size of each square in mm
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0)...
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration points
        self.obj_points = []  # 3D points in real world
        self.img_points_left = []  # 2D points in left image
        self.img_points_right = []  # 2D points in right image
        
    def add_calibration_pair(self, img_left, img_right):
        """Add a pair of calibration images"""
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, self.checkerboard_size, None
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, self.checkerboard_size, None
        )
        
        if ret_left and ret_right:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            self.obj_points.append(self.objp)
            self.img_points_left.append(corners_left)
            self.img_points_right.append(corners_right)
            return True
        return False

    def calibrate(self, image_size):
        """Perform stereo calibration"""
        # Individual camera calibration first
        ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self.obj_points, self.img_points_left, image_size, None, None
        )
        ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self.obj_points, self.img_points_right, image_size, None, None
        )
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self.obj_points,
            self.img_points_left,
            self.img_points_right,
            K1, D1, K2, D2,
            image_size,
            criteria=criteria,
            flags=flags
        )
        
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        
        # Compute rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, image_size, cv2.CV_32FC1
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, image_size, cv2.CV_32FC1
        )
        
        return {
            'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2,
            'R': R, 'T': T, 'R1': R1, 'R2': R2,
            'P1': P1, 'P2': P2, 'Q': Q,
            'map1_left': map1_left, 'map2_left': map2_left,
            'map1_right': map1_right, 'map2_right': map2_right,
            'reprojection_error': ret
        }
    
    def save_calibration(self, calib_data, filepath):
        """Save calibration to JSON file"""
        save_data = {}
        for key, value in calib_data.items():
            if isinstance(value, np.ndarray):
                save_data[key] = value.tolist()
            else:
                save_data[key] = value
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
```

### 3.2 Disparity/Depth Computation

#### SELECTED: Semi-Global Block Matching (SGBM)

**Why SGBM over other methods?**

| Method | Speed | Accuracy | Complexity | Our Choice |
|--------|-------|----------|------------|------------|
| Block Matching (BM) | Fast | Low | Simple | ❌ Too noisy |
| **SGBM** | **Medium** | **Good** | **Medium** | **✅ Best balance** |
| RAFT-Stereo | Slow | Excellent | High | ❌ Needs GPU |
| PSMNet | Slow | Excellent | High | ❌ Overkill |

#### SGBM Parameters Explained

```python
class DepthEstimator:
    def __init__(self, calib_data):
        self.calib_data = calib_data
        
        # SGBM Parameters - TUNED FOR FACE DETECTION
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,        # Minimum disparity value
            numDisparities=64,     # Range of disparity (must be divisible by 16)
            blockSize=5,           # Matched block size (odd number, 3-11)
            P1=8 * 3 * 5 ** 2,    # Penalty for disparity changes of 1
            P2=32 * 3 * 5 ** 2,   # Penalty for disparity changes > 1
            disp12MaxDiff=1,       # Max allowed difference in left-right check
            uniquenessRatio=10,    # Margin for best match uniqueness (%)
            speckleWindowSize=100, # Max size of smooth disparity regions
            speckleRange=32,       # Max disparity variation within region
            preFilterCap=63,       # Truncation value for prefiltered pixels
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Full 8-direction SGBM
        )

- **Too small (<5 cm)**: Poor depth accuracy at 50cm distance
- **Too large (>12 cm)**: Harder to find corresponding points, occlusion issues
- **6-10 cm**: Optimal for face at 40-80 cm from cameras
- **Human eye baseline**: ~6.5 cm (we're mimicking human stereo vision)

#### Mounting Requirements

**Critical: Rigid mounting is essential!**

| Requirement | Why | Solution |
|-------------|-----|----------|
| No movement | Any shift breaks calibration | Solid bracket (wood/metal/3D print) |
| Parallel alignment | Cameras must face same direction | Use level during mounting |
| Same height | Vertical alignment needed | Measure carefully |
| Stable base | Vibration affects depth | Heavy base or desk clamp |

**DIY Mounting Options:**
1. **Wooden plank** (20cm × 8cm × 2cm) with camera screws
2. **3D printed bracket** (STL files available online)
3. **Metal L-bracket** from hardware store
4. **Cardboard prototype** (for initial testing only)

### 2.3 Computer Requirements

| Component | Minimum | Recommended | Why |
|-----------|---------|-------------|-----|
| CPU | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 | Stereo matching is CPU-intensive |
| RAM | 8 GB | 16 GB | Multiple models in memory |
| GPU | None (CPU only) | NVIDIA GTX 1050+ | 10x faster inference |
| Storage | 20 GB | 50 GB | Models + datasets |
| USB Ports | 2× USB 2.0 | 2× USB 3.0 | Bandwidth for dual cameras |
| OS | Windows 10/11, Ubuntu 20.04+ | Ubuntu 22.04 | Better OpenCV support on Linux |

**GPU Acceleration:**
- **Without GPU**: ~5-10 FPS (acceptable for demo)
- **With GPU**: ~30 FPS (real-time)
- **For BTech project**: CPU-only is fine!

---

## 3. Stereo Vision & Depth Estimation

### 3.1 Calibration Method

**Technology: Zhang's Checkerboard Calibration**

**Why Zhang's Method?**
- Industry standard since 2000
- Built into OpenCV (`cv2.calibrateCamera`)
- Requires only printed checkerboard pattern
- Accurate and robust

**Alternative Considered:**
- **Charuco board**: More accurate but harder to print
- **Self-calibration**: Unreliable for beginners
- **Decision**: Zhang's method - proven and simple



#### Calibration Checkerboard Specifications

```
┌─────────────────────────────────────────┐
│  ■ □ ■ □ ■ □ ■ □ ■ □   Checkerboard    │
│  □ ■ □ ■ □ ■ □ ■ □ ■                   │
│  ■ □ ■ □ ■ □ ■ □ ■ □   - 9×6 inner     │
│  □ ■ □ ■ □ ■ □ ■ □ ■     corners       │
│  ■ □ ■ □ ■ □ ■ □ ■ □   - 25mm squares  │
│  □ ■ □ ■ □ ■ □ ■ □ ■   - Print on A4   │
│  ■ □ ■ □ ■ □ ■ □ ■ □   - Mount on      │
│                           cardboard     │
└─────────────────────────────────────────┘
```

**Specifications:**
- **Pattern**: 9×6 inner corners (10×7 squares)
- **Square size**: 25mm × 25mm
- **Print**: High-quality laser printer on A4 paper
- **Mount**: Glue on flat cardboard for rigidity
- **Download**: OpenCV provides pattern generators

**Calibration Process:**
1. Capture 20-30 image pairs of checkerboard
2. Move checkerboard to different positions/angles
3. Ensure good coverage of image area
4. OpenCV detects corners automatically
5. Computes intrinsic + extrinsic parameters

#### OpenCV Functions Used

```python
# Calibration pipeline
cv2.findChessboardCorners()      # Detect corners
cv2.cornerSubPix()               # Refine corner locations
cv2.calibrateCamera()            # Compute intrinsics (per camera)
cv2.stereoCalibrate()            # Compute extrinsics (between cameras)
cv2.stereoRectify()              # Compute rectification transforms
cv2.initUndistortRectifyMap()    # Create lookup tables
cv2.remap()                      # Apply rectification (fast)
```

### 3.2 Disparity Computation

**Technology: Semi-Global Block Matching (SGBM)**

**Why SGBM?**


| Method | Accuracy | Speed | Complexity | Choice |
|--------|----------|-------|------------|--------|
| Block Matching (BM) | Low | Very Fast | Simple | ❌ Too noisy |
| **SGBM** | **High** | **Fast** | **Medium** | **✅ BEST** |
| RAFT-Stereo (DL) | Very High | Slow | Complex | ❌ Overkill |

**SGBM Advantages:**
- Built into OpenCV (`cv2.StereoSGBM_create()`)
- Good accuracy on faces (smooth surfaces)
- Real-time capable (30+ FPS)
- No training required
- Robust to lighting changes

**SGBM Parameters (Tuned for Faces):**

```python
stereo = cv2.StereoSGBM_create(
    minDisparity=0,           # Start of disparity search
    numDisparities=64,        # Range of disparity (must be divisible by 16)
    blockSize=5,              # Matching block size (odd number, 3-11)
    P1=8 * 3 * 5**2,         # Smoothness penalty (small differences)
    P2=32 * 3 * 5**2,        # Smoothness penalty (large differences)
    disp12MaxDiff=1,         # Left-right consistency check
    uniquenessRatio=10,      # Uniqueness threshold (5-15)
    speckleWindowSize=100,   # Speckle filter window
    speckleRange=32,         # Speckle filter range
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Best quality mode
)
```

**Parameter Tuning Guide:**
- `numDisparities`: Higher = more depth range, slower
- `blockSize`: Larger = smoother but less detail
- `P1, P2`: Control smoothness (P2 > P1 always)
- Tune on your specific camera setup!

### 3.3 Depth Conversion

**Formula: Disparity to Depth**

```
Depth (mm) = (Baseline × Focal_Length) / Disparity

Where:
- Baseline: Distance between cameras (mm)
- Focal_Length: From calibration (pixels)
- Disparity: From SGBM (pixels)
```

**Example Calculation:**
```
Baseline = 80 mm
Focal_Length = 700 pixels (typical for 720p webcam)
Disparity = 10 pixels

Depth = (80 × 700) / 10 = 5,600 mm = 5.6 meters
```

**Depth Range for Face Detection:**
- **Optimal**: 40-80 cm from cameras
- **Minimum**: 30 cm (too close = poor disparity)
- **Maximum**: 150 cm (too far = low disparity accuracy)

---

## 4. Face Detection

### 4.1 Primary Method: RetinaFace

**Why RetinaFace?**

| Method | Speed | Accuracy | Landmarks | Choice |
|--------|-------|----------|-----------|--------|
| Haar Cascades | Very Fast | Low | No | ❌ Outdated |
| HOG + SVM | Fast | Medium | No | ❌ Not robust |
| MTCNN | Medium | High | Yes (5 points) | ⚠️ Backup |
| **RetinaFace** | **Fast** | **Very High** | **Yes (5 points)** | **✅ PRIMARY** |
| YOLOv8-Face | Very Fast | High | No | ❌ No landmarks |



**RetinaFace Advantages:**
- State-of-the-art accuracy (WIDER FACE benchmark winner)
- Detects faces at multiple scales
- Provides 5 facial landmarks (eyes, nose, mouth corners)
- Fast inference (~30ms per frame on CPU)
- Pre-trained models available

**Implementation:**

```python
# Using InsightFace library (includes RetinaFace)
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Detect faces
faces = app.get(image)
# Returns: bounding box, landmarks, confidence
```

**Alternative: MTCNN (Backup)**

If RetinaFace has issues:
```python
from mtcnn import MTCNN

detector = MTCNN()
faces = detector.detect_faces(image)
```

**Why MTCNN as backup?**
- Simpler installation (pure Python)
- More lightweight
- Still provides landmarks
- Slightly slower but more stable

### 4.2 Face Alignment

**Purpose**: Normalize face orientation for better recognition

**Method**: Similarity transform using eye landmarks

```python
# Align face to canonical position
def align_face(image, landmarks):
    # Get eye positions
    left_eye = landmarks[0]   # Left eye
    right_eye = landmarks[1]  # Right eye
    
    # Compute angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Compute center and scale
    eye_center = ((left_eye[0] + right_eye[0]) / 2,
                  (left_eye[1] + right_eye[1]) / 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    
    # Apply transformation
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return aligned
```

**Why alignment matters:**
- Improves face recognition accuracy by 5-10%
- Standardizes input for neural networks
- Reduces pose variation

---

## 5. Face Anti-Spoofing (Liveness Detection)

### 5.1 Depth-Based Liveness (Primary Method)

**Technique: 3D Face Depth Analysis**

**Core Principle:**
- Real face: 3D structure with depth variation
- Photo/Screen: Flat surface with minimal depth

**Depth Features Extracted:**



| Feature | Real Face | Photo/Video | Threshold |
|---------|-----------|-------------|-----------|
| **Depth Range** | 8-15 cm | <2 cm | >5 cm = LIVE |
| **Nose Prominence** | 2-3 cm forward | ~0 cm | >1.5 cm = LIVE |
| **Depth Variance** | High (σ > 15mm) | Low (σ < 5mm) | σ > 10mm = LIVE |
| **Depth Continuity** | Smooth gradient | Uniform/noisy | Smooth = LIVE |

**Implementation:**

```python
def analyze_face_depth(disparity_map, face_bbox):
    # Extract face region from disparity
    x, y, w, h = face_bbox
    face_disparity = disparity_map[y:y+h, x:x+w]
    
    # Convert to depth
    depth_map = (baseline * focal_length) / (face_disparity + 1e-6)
    
    # Feature 1: Depth range
    depth_range = np.max(depth_map) - np.min(depth_map)
    
    # Feature 2: Nose prominence (center region)
    center_y, center_x = h//2, w//2
    nose_region = depth_map[center_y-10:center_y+10, center_x-10:center_x+10]
    nose_depth = np.min(nose_region)  # Closest point
    face_avg_depth = np.median(depth_map)
    nose_prominence = face_avg_depth - nose_depth
    
    # Feature 3: Depth variance
    depth_variance = np.std(depth_map)
    
    # Feature 4: Depth continuity (gradient smoothness)
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    continuity_score = 1.0 / (np.mean(gradient_magnitude) + 1e-6)
    
    # Compute liveness score
    liveness_score = 0.0
    if depth_range > 50:  # >5cm
        liveness_score += 0.4
    if nose_prominence > 15:  # >1.5cm
        liveness_score += 0.3
    if depth_variance > 10:
        liveness_score += 0.2
    if continuity_score > 0.5:
        liveness_score += 0.1
    
    return liveness_score  # 0.0 to 1.0
```

**Decision Threshold:**
- Score > 0.7: LIVE (accept)
- Score 0.4-0.7: UNCERTAIN (retry)
- Score < 0.4: SPOOF (reject)

### 5.2 Texture-Based Anti-Spoofing (Secondary Method)

**Technique: Local Binary Patterns (LBP)**

**Why LBP?**

| Method | Accuracy | Speed | Training Data | Choice |
|--------|----------|-------|---------------|--------|
| **LBP + SVM** | **95%** | **Very Fast** | **Small** | **✅ PRIMARY** |
| CNN (ResNet) | 98% | Medium | Large | ⚠️ Backup |
| Frequency Analysis | 92% | Fast | None | ⚠️ Complementary |

**LBP Advantages:**
- Detects texture differences (paper vs skin)
- Detects moiré patterns (screen displays)
- Fast computation (~5ms per face)
- Works with small training data
- Interpretable features



**LBP Implementation:**

```python
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# LBP parameters
radius = 1
n_points = 8 * radius
method = 'uniform'  # Rotation invariant

def extract_lbp_features(face_image):
    # Convert to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method)
    
    # Compute histogram
    n_bins = n_points + 2  # uniform patterns + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    return hist

# Training
X_train = []  # LBP features
y_train = []  # Labels (0=real, 1=fake)

for image, label in training_data:
    features = extract_lbp_features(image)
    X_train.append(features)
    y_train.append(label)

# Train SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)

# Inference
def predict_spoof(face_image):
    features = extract_lbp_features(face_image)
    features_scaled = scaler.transform([features])
    spoof_prob = svm.predict_proba(features_scaled)[0][1]
    return spoof_prob  # 0.0 to 1.0
```

**Training Data Required:**
- 500 real face images
- 500 spoof images (photos + screens)
- Source: Replay-Attack dataset

**Expected Performance:**
- Accuracy: 95-98% on Replay-Attack
- False Accept Rate: <2%
- False Reject Rate: <3%

### 5.3 CNN-Based Anti-Spoofing (Optional Enhancement)

**Model: MobileNetV2 (Lightweight CNN)**

**Why MobileNetV2?**
- Lightweight (14 MB model)
- Fast inference (20ms on CPU)
- Good accuracy (97%+)
- Pre-trained on ImageNet

**Implementation:**

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained base
base_model = MobileNetV2(weights='imagenet', include_top=False, 
                         input_shape=(224, 224, 3))

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary: real/fake

model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tune on anti-spoofing data
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=val_data)
```

**When to use CNN vs LBP:**
- **Use LBP**: Limited data, need speed, interpretability
- **Use CNN**: Have GPU, want highest accuracy, large dataset

---

## 6. Deepfake Detection

### 6.1 Primary Model: EfficientNet-B0

**Why EfficientNet-B0?**



| Model | Params | Speed | Accuracy (FF++) | Choice |
|-------|--------|-------|-----------------|--------|
| ResNet-50 | 25M | Medium | 94% | ❌ Larger |
| **EfficientNet-B0** | **5M** | **Fast** | **95%** | **✅ BEST** |
| XceptionNet | 23M | Medium | 96% | ⚠️ Backup |
| EfficientNet-B4 | 19M | Slow | 97% | ❌ Overkill |

**EfficientNet-B0 Advantages:**
- Excellent accuracy-to-size ratio
- Fast inference (30ms on CPU, 5ms on GPU)
- Pre-trained on ImageNet
- Easy to fine-tune
- Industry standard for deepfake detection

**Architecture:**

```
Input (224×224×3)
    ↓
EfficientNet-B0 Backbone (pre-trained)
    ↓
Global Average Pooling
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(1, Sigmoid) → Deepfake probability
```

**Implementation:**

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained EfficientNet-B0
base_model = EfficientNetB0(weights='imagenet', include_top=False,
                            input_shape=(224, 224, 3))

# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Train on FaceForensics++
model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

**Training Strategy:**
1. **Phase 1**: Train on FaceForensics++ (all manipulation types)
2. **Phase 2**: Fine-tune on Celeb-DF (harder examples)
3. **Data Augmentation**: Compression, blur, noise (simulate real-world)

### 6.2 Temporal Consistency Check

**Technique: Frame-to-Frame Analysis**

**Why temporal analysis?**
- Deepfakes often have frame jitter
- Inconsistent facial landmarks across frames
- Unnatural motion patterns

**Implementation:**

```python
from collections import deque

class TemporalDeepfakeDetector:
    def __init__(self, window_size=10):
        self.landmark_history = deque(maxlen=window_size)
        self.score_history = deque(maxlen=window_size)
    
    def analyze_temporal_consistency(self, landmarks, deepfake_score):
        self.landmark_history.append(landmarks)
        self.score_history.append(deepfake_score)
        
        if len(self.landmark_history) < 5:
            return 0.5  # Not enough history
        
        # Check landmark stability
        landmark_array = np.array(self.landmark_history)
        landmark_variance = np.var(landmark_array, axis=0)
        stability_score = 1.0 / (np.mean(landmark_variance) + 1e-6)
        
        # Check score consistency
        score_variance = np.var(self.score_history)
        consistency_score = 1.0 - min(score_variance * 10, 1.0)
        
        # Combined temporal score
        temporal_score = 0.6 * stability_score + 0.4 * consistency_score
        
        return temporal_score
```

### 6.3 Eye Blink Detection (Optional)

**Technique: Eye Aspect Ratio (EAR)**

**Why blink detection?**
- Early deepfakes had no blinking
- Modern deepfakes have irregular blink patterns
- Real humans blink 15-20 times per minute



```python
def eye_aspect_ratio(eye_landmarks):
    # Compute vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Compute horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Blink detection
EAR_THRESHOLD = 0.2
blink_counter = 0
total_blinks = 0

for frame in video:
    landmarks = detect_landmarks(frame)
    left_ear = eye_aspect_ratio(landmarks['left_eye'])
    right_ear = eye_aspect_ratio(landmarks['right_eye'])
    avg_ear = (left_ear + right_ear) / 2.0
    
    if avg_ear < EAR_THRESHOLD:
        blink_counter += 1
    else:
        if blink_counter >= 2:  # Blink detected
            total_blinks += 1
        blink_counter = 0

# Expected: 15-20 blinks per minute
# Suspicious: <5 or >30 blinks per minute
```

---

## 7. Face Recognition & Verification

### 7.1 Face Embedding Model: ArcFace

**Why ArcFace?**

| Model | Accuracy (LFW) | Embedding Size | Speed | Choice |
|-------|----------------|----------------|-------|--------|
| FaceNet | 99.65% | 128-D | Fast | ⚠️ Good |
| **ArcFace** | **99.83%** | **512-D** | **Fast** | **✅ BEST** |
| CosFace | 99.73% | 512-D | Fast | ⚠️ Similar |
| VGGFace2 | 99.13% | 2048-D | Slow | ❌ Outdated |

**ArcFace Advantages:**
- State-of-the-art accuracy
- Robust to pose, lighting, age variations
- Pre-trained models available (no training needed!)
- Fast inference (10ms per face)
- 512-D embeddings (good balance)

**Implementation (Using InsightFace):**

```python
from insightface.app import FaceAnalysis

# Initialize
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Extract embedding
def get_face_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        return None
    
    # Get largest face
    face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
    
    # Embedding is already computed
    embedding = face.embedding  # 512-D vector
    
    # Normalize (important!)
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding
```

**Pre-trained Model:**
- Model: `buffalo_l` (large, most accurate)
- Backbone: ResNet-100
- Training data: MS1MV3 (5.2M images)
- Download: Automatic via InsightFace

### 7.2 Similarity Metric: Cosine Similarity

**Why Cosine Similarity?**

| Metric | Range | Interpretation | Choice |
|--------|-------|----------------|--------|
| Euclidean Distance | [0, ∞] | Smaller = more similar | ❌ Scale-dependent |
| **Cosine Similarity** | **[-1, 1]** | **Larger = more similar** | **✅ BEST** |
| Dot Product | [-∞, ∞] | Unnormalized | ❌ Not robust |

**Formula:**

```
cosine_similarity = (A · B) / (||A|| × ||B||)

Where:
- A, B: Face embeddings (512-D vectors)
- · : Dot product
- ||·||: L2 norm
```

**Implementation:**

```python
def cosine_similarity(embedding1, embedding2):
    # Assuming embeddings are already normalized
    similarity = np.dot(embedding1, embedding2)
    return similarity

# Verification
def verify_identity(probe_embedding, enrolled_embedding, threshold=0.6):
    similarity = cosine_similarity(probe_embedding, enrolled_embedding)
    
    if similarity >= threshold:
        return True, similarity  # Match
    else:
        return False, similarity  # No match
```

**Threshold Selection:**

| Threshold | False Accept Rate | False Reject Rate | Use Case |
|-----------|-------------------|-------------------|----------|
| 0.4 | 1% | 10% | Low security |
| **0.6** | **0.1%** | **5%** | **Balanced (recommended)** |
| 0.7 | 0.01% | 15% | High security |



### 7.3 User Enrollment

**Strategy: Multi-shot Enrollment**

**Why multiple images?**
- Captures pose variation
- Averages out noise
- More robust template
- Better generalization

**Enrollment Process:**

```python
def enroll_user(user_id, num_shots=5):
    embeddings = []
    
    print(f"Enrolling user {user_id}...")
    print("Please look at the camera from different angles")
    
    for i in range(num_shots):
        print(f"Capture {i+1}/{num_shots}...")
        
        # Capture frame
        frame = capture_frame()
        
        # Check liveness
        if not check_liveness(frame):
            print("Liveness check failed. Please try again.")
            continue
        
        # Extract embedding
        embedding = get_face_embedding(frame)
        if embedding is not None:
            embeddings.append(embedding)
        
        time.sleep(1)  # Wait between captures
    
    if len(embeddings) < 3:
        print("Enrollment failed. Not enough valid captures.")
        return False
    
    # Average embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    # Store in database
    save_enrollment(user_id, avg_embedding)
    
    print(f"User {user_id} enrolled successfully!")
    return True
```

**Storage Format (JSON):**

```json
{
  "user_id": "user_001",
  "name": "John Doe",
  "enrollment_date": "2024-12-07T10:30:00",
  "embedding": [0.123, -0.456, 0.789, ...],  // 512 values
  "metadata": {
    "num_samples": 5,
    "enrollment_device": "dual_webcam_v1"
  }
}
```

**Security: Encryption**

```python
from cryptography.fernet import Fernet

# Generate key (store securely!)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt embedding
embedding_bytes = embedding.tobytes()
encrypted = cipher.encrypt(embedding_bytes)

# Decrypt for verification
decrypted = cipher.decrypt(encrypted)
embedding = np.frombuffer(decrypted, dtype=np.float32)
```

---

## 8. Datasets

### 8.1 Dataset Summary

| Purpose | Dataset | Size | Usage |
|---------|---------|------|-------|
| **Anti-Spoofing Training** | Replay-Attack | 4 GB | Train LBP/CNN classifier |
| **Deepfake Detection** | FaceForensics++ | 3 GB (faces) | Train EfficientNet |
| **Face Recognition Eval** | LFW | 200 MB | Benchmark ArcFace |
| **Custom Data** | Self-recorded | 2 GB | Test your setup |

**Total Storage: ~10 GB**

### 8.2 Replay-Attack Dataset

**Purpose**: Train anti-spoofing models

**Statistics:**
- 1,300 videos
- 50 subjects
- Attack types: Print (photo), Replay (video on screen)
- Conditions: Controlled, adverse lighting

**Download:**
```bash
# Register at: https://www.idiap.ch/en/dataset/replayattack
# Follow email instructions to download
```

**Data Split:**
- Train: 360 videos (180 real + 180 attack)
- Dev: 360 videos
- Test: 480 videos

**Usage:**
```python
# Load Replay-Attack
train_real = load_videos('replay-attack/train/real/')
train_attack = load_videos('replay-attack/train/attack/')

# Extract frames and train LBP classifier
for video in train_real:
    frames = extract_frames(video)
    for frame in frames:
        features = extract_lbp_features(frame)
        X_train.append(features)
        y_train.append(0)  # Real

for video in train_attack:
    frames = extract_frames(video)
    for frame in frames:
        features = extract_lbp_features(frame)
        X_train.append(features)
        y_train.append(1)  # Fake
```

### 8.3 FaceForensics++ Dataset

**Purpose**: Train deepfake detector

**Statistics:**
- 1,000 original videos
- 4,000 manipulated videos (4 types)
- Manipulation types: DeepFakes, Face2Face, FaceSwap, NeuralTextures

**Download (Faces Only):**
```bash
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# Download only face crops (saves space)
python download-FaceForensics.py ./data \
    -d DeepFakes Face2Face FaceSwap NeuralTextures original \
    -c c23 \
    -t faces
```

**Data Split:**
- Train: 720 videos per type
- Val: 140 videos per type
- Test: 140 videos per type

**Usage:**
```python
# Data generator for training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    'FaceForensics/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # real vs fake
)
```

### 8.4 LFW Dataset

**Purpose**: Evaluate face recognition accuracy

**Statistics:**
- 13,233 images
- 5,749 people
- 1,680 people with 2+ images

**Download:**
```bash
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz
```

**Evaluation Protocol:**
```python
# LFW pairs for verification
pairs = load_lfw_pairs('pairs.txt')

correct = 0
total = len(pairs)

for pair in pairs:
    img1, img2, is_same = pair
    
    emb1 = get_face_embedding(img1)
    emb2 = get_face_embedding(img2)
    
    similarity = cosine_similarity(emb1, emb2)
    prediction = similarity >= 0.6
    
    if prediction == is_same:
        correct += 1

accuracy = correct / total
print(f"LFW Accuracy: {accuracy * 100:.2f}%")
# Target: >99% with ArcFace
```



### 8.5 Custom Dataset (Recommended!)

**Purpose**: Test your specific dual-camera setup

**What to record:**

```
custom_dataset/
├── real/
│   ├── person1/
│   │   ├── frontal_01.mp4
│   │   ├── left_angle_01.mp4
│   │   ├── right_angle_01.mp4
│   │   └── varying_distance_01.mp4
│   ├── person2/
│   └── ... (10-20 people)
│
├── spoof_photo/
│   ├── person1_phone_display.mp4
│   ├── person1_printed_photo.mp4
│   └── ...
│
├── spoof_video/
│   ├── person1_laptop_replay.mp4
│   ├── person1_tablet_replay.mp4
│   └── ...
│
└── spoof_mask/  (optional)
    └── person1_paper_mask.mp4
```

**Recording Guidelines:**
- Duration: 10-15 seconds per video
- Resolution: 720p
- Lighting: Varied (bright, dim, mixed)
- Distance: 40-80 cm from cameras
- Subjects: Friends, family, classmates (with permission!)

**Why custom data is valuable:**
- Tests your specific camera setup
- Demonstrates real-world performance
- Impressive for project evaluation
- Helps tune thresholds

---

## 9. Software Stack

### 9.1 Core Libraries

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **Python** | 3.8-3.10 | Programming language | `python.org` |
| **OpenCV** | 4.8+ | Computer vision, stereo | `pip install opencv-python` |
| **NumPy** | 1.24+ | Numerical computing | `pip install numpy` |
| **InsightFace** | 0.7+ | Face detection, ArcFace | `pip install insightface` |
| **TensorFlow** | 2.13+ | Deep learning | `pip install tensorflow` |
| **scikit-learn** | 1.3+ | ML algorithms (SVM) | `pip install scikit-learn` |
| **scikit-image** | 0.21+ | Image processing (LBP) | `pip install scikit-image` |

### 9.2 Optional Libraries

| Library | Purpose | Installation |
|---------|---------|--------------|
| **MTCNN** | Backup face detector | `pip install mtcnn` |
| **dlib** | Face landmarks | `pip install dlib` |
| **matplotlib** | Visualization | `pip install matplotlib` |
| **Pillow** | Image I/O | `pip install Pillow` |

### 9.3 Complete Installation Script

```bash
# Create virtual environment
python -m venv face_verification_env
source face_verification_env/bin/activate  # Linux/Mac
# OR
face_verification_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install core libraries
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install insightface==0.7.3
pip install tensorflow==2.13.0
pip install scikit-learn==1.3.0
pip install scikit-image==0.21.0

# Install optional libraries
pip install mtcnn==0.1.1
pip install matplotlib==3.7.2
pip install Pillow==10.0.0

# For GPU support (optional)
pip install tensorflow-gpu==2.13.0

# Verify installation
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import insightface; print('InsightFace: OK')"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

### 9.4 Project Structure

```
dual-camera-face-verification/
├── calibration/
│   ├── calibrate.py              # Stereo calibration script
│   ├── calibration_params.json   # Saved calibration
│   └── checkerboard_images/      # Calibration images
│
├── models/
│   ├── arcface/                  # ArcFace model (auto-downloaded)
│   ├── antispoofing_lbp.pkl      # Trained LBP+SVM model
│   ├── antispoofing_cnn.h5       # Trained CNN model (optional)
│   └── deepfake_detector.h5      # Trained EfficientNet model
│
├── data/
│   ├── enrolled_users/           # User embeddings database
│   ├── logs/                     # Verification logs
│   └── test_videos/              # Test data
│
├── src/
│   ├── camera.py                 # Camera capture & sync
│   ├── stereo.py                 # Stereo depth computation
│   ├── face_detection.py         # Face detection module
│   ├── antispoofing.py           # Liveness detection
│   ├── deepfake_detection.py     # Deepfake detector
│   ├── face_recognition.py       # Face embedding & matching
│   ├── enrollment.py             # User enrollment
│   ├── verification.py           # Main verification pipeline
│   └── utils.py                  # Helper functions
│
├── train/
│   ├── train_antispoofing.py     # Train anti-spoofing model
│   └── train_deepfake.py         # Train deepfake detector
│
├── notebooks/
│   ├── 01_calibration.ipynb      # Interactive calibration
│   ├── 02_depth_analysis.ipynb   # Depth visualization
│   └── 03_evaluation.ipynb       # Performance evaluation
│
├── requirements.txt              # Python dependencies
├── config.yaml                   # System configuration
└── main.py                       # Main application
```

---

## 10. Model Training Strategy

### 10.1 Anti-Spoofing Model Training

**Model: LBP + SVM**

**Training Pipeline:**

```python
# Step 1: Load Replay-Attack dataset
from sklearn.model_selection import train_test_split

X_train, y_train = load_replay_attack_data('train')
X_val, y_val = load_replay_attack_data('devel')
X_test, y_test = load_replay_attack_data('test')

# Step 2: Extract LBP features
X_train_lbp = [extract_lbp_features(img) for img in X_train]
X_val_lbp = [extract_lbp_features(img) for img in X_val]
X_test_lbp = [extract_lbp_features(img) for img in X_test]

# Step 3: Normalize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_lbp)
X_val_scaled = scaler.transform(X_val_lbp)
X_test_scaled = scaler.transform(X_test_lbp)

# Step 4: Train SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}

svm = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1)
svm.fit(X_train_scaled, y_train)

print(f"Best parameters: {svm.best_params_}")

# Step 5: Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = svm.predict(X_test_scaled)
y_prob = svm.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test AUC: {auc:.4f}")

# Step 6: Save model
import joblib
joblib.dump(svm, 'models/antispoofing_lbp.pkl')
joblib.dump(scaler, 'models/antispoofing_scaler.pkl')
```

**Expected Results:**
- Training time: 5-10 minutes (CPU)
- Test accuracy: 95-98%
- Model size: <5 MB



### 10.2 Deepfake Detection Model Training

**Model: EfficientNet-B0**

**Training Pipeline:**

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'FaceForensics/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'FaceForensics/val/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Step 2: Build model
base_model = EfficientNetB0(weights='imagenet', include_top=False,
                            input_shape=(224, 224, 3))

# Freeze early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Step 3: Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Step 4: Train
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/deepfake_detector_best.h5',
            monitor='val_auc',
            save_best_only=True
        )
    ]
)

# Step 5: Evaluate
test_generator = val_datagen.flow_from_directory(
    'FaceForensics/test/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

results = model.evaluate(test_generator)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1] * 100:.2f}%")
print(f"Test AUC: {results[2]:.4f}")

# Step 6: Save final model
model.save('models/deepfake_detector.h5')
```

**Training Configuration:**
- Batch size: 32
- Epochs: 15 (with early stopping)
- Learning rate: 1e-4
- Optimizer: Adam
- Training time: 2-4 hours (GPU), 12-24 hours (CPU)

**Expected Results:**
- Test accuracy: 93-96%
- Test AUC: 0.96-0.98
- Model size: ~20 MB

### 10.3 Training Tips

**Data Augmentation Best Practices:**

```python
# For anti-spoofing
antispoofing_augmentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    horizontal_flip=True,
    # Important: Add compression artifacts
    preprocessing_function=lambda x: add_compression(x, quality=random.randint(60, 100))
)

# For deepfake detection
deepfake_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    # Critical: Simulate compression
    preprocessing_function=lambda x: add_compression(x, quality=random.randint(40, 100))
)
```

**Handling Class Imbalance:**

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Use in training
model.fit(
    train_generator,
    class_weight=class_weight_dict,
    ...
)
```

---

## 11. Performance Benchmarks

### 11.1 Target Performance Metrics

**Anti-Spoofing (Liveness Detection):**

| Metric | Target | Excellent | Acceptable |
|--------|--------|-----------|------------|
| Accuracy | >95% | >98% | >90% |
| APCER (False Accept) | <2% | <0.5% | <5% |
| BPCER (False Reject) | <3% | <1% | <5% |
| Speed (FPS) | >20 | >30 | >10 |

**Deepfake Detection:**

| Metric | Target | Excellent | Acceptable |
|--------|--------|-----------|------------|
| Accuracy | >93% | >96% | >90% |
| AUC | >0.95 | >0.98 | >0.90 |
| Precision | >90% | >95% | >85% |
| Recall | >90% | >95% | >85% |

**Face Verification:**

| Metric | Target | Excellent | Acceptable |
|--------|--------|-----------|------------|
| Accuracy (LFW) | >99% | >99.5% | >98% |
| FAR @ 0.1% FRR | <0.01% | <0.001% | <0.1% |
| Speed (FPS) | >25 | >30 | >15 |

**System-Level:**

| Metric | Target | Notes |
|--------|--------|-------|
| End-to-end latency | <200ms | From capture to decision |
| Throughput | >15 FPS | Real-time performance |
| Memory usage | <2 GB | Including all models |
| CPU usage | <80% | On recommended hardware |

### 11.2 Evaluation Metrics Explained

**APCER (Attack Presentation Classification Error Rate):**
```
APCER = (Number of attack samples classified as real) / (Total attack samples)
```
Lower is better. Measures false acceptance of spoofs.

**BPCER (Bona Fide Presentation Classification Error Rate):**
```
BPCER = (Number of real samples classified as attack) / (Total real samples)
```
Lower is better. Measures false rejection of genuine users.

**ACER (Average Classification Error Rate):**
```
ACER = (APCER + BPCER) / 2
```
Overall anti-spoofing performance metric.

**AUC (Area Under ROC Curve):**
- Measures classifier's ability to distinguish classes
- Range: 0.5 (random) to 1.0 (perfect)
- >0.95 is excellent for deepfake detection

### 11.3 Benchmark Testing Protocol

```python
def evaluate_system(test_data):
    results = {
        'liveness': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'deepfake': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'verification': {'correct': 0, 'total': 0},
        'latency': []
    }
    
    for sample in test_data:
        start_time = time.time()
        
        # Run full pipeline
        liveness_result = check_liveness(sample)
        deepfake_result = check_deepfake(sample)
        verification_result = verify_identity(sample)
        
        latency = time.time() - start_time
        results['latency'].append(latency)
        
        # Update metrics
        update_confusion_matrix(results, sample.ground_truth, 
                               liveness_result, deepfake_result, 
                               verification_result)
    
    # Compute final metrics
    metrics = compute_metrics(results)
    return metrics
```

---

## 12. Implementation Roadmap

### 12-Week Implementation Plan



#### Week 1-2: Hardware Setup & Calibration

**Tasks:**
- [ ] Acquire 2× webcams (Logitech C270 or similar)
- [ ] Build/acquire mounting bracket (6-10 cm baseline)
- [ ] Print checkerboard pattern (9×6, 25mm squares)
- [ ] Implement calibration script using OpenCV
- [ ] Capture 20-30 calibration image pairs
- [ ] Verify calibration quality (reprojection error <0.5 pixels)
- [ ] Save calibration parameters

**Deliverable:** Working stereo camera setup with calibration file

**Code to implement:**
```python
# calibration/calibrate.py
- capture_calibration_images()
- calibrate_stereo_cameras()
- save_calibration_params()
- test_rectification()
```

#### Week 3: Stereo Depth Computation

**Tasks:**
- [ ] Implement synchronized frame capture
- [ ] Apply stereo rectification
- [ ] Implement SGBM disparity computation
- [ ] Convert disparity to depth
- [ ] Visualize depth maps
- [ ] Tune SGBM parameters for faces

**Deliverable:** Real-time depth map visualization

**Code to implement:**
```python
# src/stereo.py
- StereoCamera class
- capture_synchronized_frames()
- compute_disparity()
- disparity_to_depth()
- visualize_depth()
```

#### Week 4: Face Detection & Tracking

**Tasks:**
- [ ] Install InsightFace library
- [ ] Implement RetinaFace detection
- [ ] Add face tracking across frames
- [ ] Implement stereo face correspondence
- [ ] Extract face ROI from both cameras
- [ ] Test on various poses and distances

**Deliverable:** Robust face detection in stereo frames

**Code to implement:**
```python
# src/face_detection.py
- FaceDetector class
- detect_faces()
- track_faces()
- find_stereo_correspondence()
- extract_face_roi()
```

#### Week 5: Depth-Based Liveness Detection

**Tasks:**
- [ ] Implement face depth analysis
- [ ] Extract depth features (range, variance, etc.)
- [ ] Set thresholds for real vs spoof
- [ ] Test with photos and videos
- [ ] Tune parameters for your setup
- [ ] Measure accuracy on test data

**Deliverable:** Working depth-based liveness detector

**Code to implement:**
```python
# src/antispoofing.py
- DepthLivenessDetector class
- analyze_face_depth()
- extract_depth_features()
- compute_liveness_score()
```

#### Week 6: Texture-Based Anti-Spoofing

**Tasks:**
- [ ] Download Replay-Attack dataset
- [ ] Implement LBP feature extraction
- [ ] Train SVM classifier
- [ ] Evaluate on test set
- [ ] Integrate with depth-based method
- [ ] Implement score fusion

**Deliverable:** Trained LBP+SVM anti-spoofing model

**Code to implement:**
```python
# train/train_antispoofing.py
- load_replay_attack_data()
- extract_lbp_features()
- train_svm_classifier()
- evaluate_model()

# src/antispoofing.py
- TextureLivenessDetector class
- fuse_liveness_scores()
```

#### Week 7-8: Deepfake Detection

**Tasks:**
- [ ] Download FaceForensics++ dataset (faces only)
- [ ] Implement data loader and augmentation
- [ ] Build EfficientNet-B0 model
- [ ] Train on FaceForensics++
- [ ] Evaluate on test set
- [ ] Implement temporal consistency check
- [ ] Integrate into pipeline

**Deliverable:** Trained deepfake detection model

**Code to implement:**
```python
# train/train_deepfake.py
- load_faceforensics_data()
- build_efficientnet_model()
- train_model()
- evaluate_model()

# src/deepfake_detection.py
- DeepfakeDetector class
- detect_deepfake()
- temporal_consistency_check()
```

#### Week 9: Face Recognition & Verification

**Tasks:**
- [ ] Set up ArcFace (InsightFace)
- [ ] Implement embedding extraction
- [ ] Implement cosine similarity matching
- [ ] Test on LFW dataset
- [ ] Implement user enrollment
- [ ] Create enrollment database

**Deliverable:** Working face verification system

**Code to implement:**
```python
# src/face_recognition.py
- FaceRecognizer class
- extract_embedding()
- cosine_similarity()
- verify_identity()

# src/enrollment.py
- enroll_user()
- save_enrollment()
- load_enrollments()
```

#### Week 10: Integration & Pipeline

**Tasks:**
- [ ] Integrate all modules into main pipeline
- [ ] Implement decision fusion logic
- [ ] Add logging and error handling
- [ ] Create configuration file
- [ ] Implement GUI (optional)
- [ ] Test end-to-end system

**Deliverable:** Complete integrated system

**Code to implement:**
```python
# src/verification.py
- VerificationPipeline class
- run_verification()
- aggregate_scores()
- make_decision()

# main.py
- Main application loop
- Command-line interface
```

#### Week 11: Testing & Optimization

**Tasks:**
- [ ] Record custom test dataset
- [ ] Comprehensive testing (all attack types)
- [ ] Measure performance metrics
- [ ] Optimize for speed (if needed)
- [ ] Fix bugs and edge cases
- [ ] Tune thresholds for best performance

**Deliverable:** Tested and optimized system

#### Week 12: Documentation & Presentation

**Tasks:**
- [ ] Write project report
- [ ] Create presentation slides
- [ ] Record demo video
- [ ] Prepare code documentation
- [ ] Create README with setup instructions
- [ ] Prepare for project defense

**Deliverable:** Complete project documentation

### Quick Start (Minimal Viable Product - 4 Weeks)

If you need a working demo quickly:

**Week 1:** Hardware + Calibration
**Week 2:** Depth computation + Face detection
**Week 3:** Depth-based liveness + Pre-trained ArcFace
**Week 4:** Integration + Testing

This gives you:
- ✅ Stereo depth-based liveness detection
- ✅ Face verification (using pre-trained ArcFace)
- ✅ Working end-to-end demo
- ❌ No texture-based anti-spoofing (can add later)
- ❌ No deepfake detection (can add later)

---

## 13. Configuration File Example

**config.yaml:**

```yaml
# System Configuration
system:
  name: "Dual-Camera Face Verification System"
  version: "1.0"
  
# Camera Settings
cameras:
  left_id: 0
  right_id: 1
  resolution: [1280, 720]
  fps: 30
  baseline_mm: 80  # Distance between cameras
  
# Calibration
calibration:
  file: "calibration/calibration_params.json"
  checkerboard_size: [9, 6]
  square_size_mm: 25
  
# Face Detection
face_detection:
  model: "retinaface"
  confidence_threshold: 0.9
  min_face_size: 80
  
# Liveness Detection
liveness:
  depth_threshold: 0.7
  texture_threshold: 0.6
  fusion_weights:
    depth: 0.6
    texture: 0.4
  
# Deepfake Detection
deepfake:
  model_path: "models/deepfake_detector.h5"
  threshold: 0.5
  temporal_window: 10
  
# Face Recognition
face_recognition:
  model: "arcface"
  embedding_size: 512
  similarity_threshold: 0.6
  
# Enrollment
enrollment:
  num_shots: 5
  min_valid_shots: 3
  database_path: "data/enrolled_users/"
  
# Logging
logging:
  enabled: true
  log_dir: "data/logs/"
  log_level: "INFO"
  max_log_size_mb: 100
```

---

## 14. Troubleshooting Guide

### Common Issues & Solutions

**Issue 1: Poor Stereo Calibration**
- **Symptom:** High reprojection error (>1.0 pixel)
- **Solution:** 
  - Recapture calibration images with better coverage
  - Ensure checkerboard is flat and well-lit
  - Use more images (30-40)
  - Check camera synchronization

**Issue 2: Noisy Depth Maps**
- **Symptom:** Speckled, inconsistent depth
- **Solution:**
  - Tune SGBM parameters (increase P1, P2)
  - Improve lighting (avoid shadows)
  - Use speckle filter
  - Ensure cameras are rigidly mounted

**Issue 3: Face Detection Failures**
- **Symptom:** Faces not detected or inconsistent
- **Solution:**
  - Check lighting conditions
  - Ensure face is 40-80 cm from cameras
  - Lower confidence threshold
  - Try MTCNN as alternative

**Issue 4: Low Anti-Spoofing Accuracy**
- **Symptom:** Photos/videos accepted as real
- **Solution:**
  - Retrain with more diverse attack data
  - Adjust liveness thresholds
  - Ensure depth computation is accurate
  - Add texture-based method

**Issue 5: Slow Performance**
- **Symptom:** <10 FPS
- **Solution:**
  - Reduce camera resolution to 640×480
  - Use smaller SGBM numDisparities
  - Enable GPU acceleration
  - Optimize code (use threading)

---

## 15. Summary & Recommendations

### Technology Stack Summary

| Component | Technology | Why |
|-----------|-----------|-----|
| **Cameras** | 2× Logitech C270 | Affordable, reliable |
| **Calibration** | OpenCV Zhang's method | Industry standard |
| **Depth** | OpenCV SGBM | Fast, accurate |
| **Face Detection** | RetinaFace | State-of-the-art |
| **Anti-Spoofing** | Depth + LBP+SVM | Multi-modal, robust |
| **Deepfake** | EfficientNet-B0 | Best accuracy/speed |
| **Face Recognition** | ArcFace | Highest accuracy |
| **Language** | Python 3.8+ | Rich ecosystem |

### Key Success Factors

1. **Rigid Camera Mounting**: Most critical for depth accuracy
2. **Good Calibration**: Foundation of stereo vision
3. **Multi-Modal Fusion**: Depth + Texture > Either alone
4. **Pre-trained Models**: Use ArcFace, don't train from scratch
5. **Custom Testing**: Record your own test data
6. **Iterative Tuning**: Adjust thresholds for your setup

### Expected Project Outcomes

**Technical Achievements:**
- Working stereo vision system
- 95%+ anti-spoofing accuracy
- 93%+ deepfake detection accuracy
- 99%+ face verification accuracy
- Real-time performance (15-30 FPS)

**Academic Value:**
- Novel application of stereo vision to biometrics
- Multi-modal security approach
- Practical implementation of research papers
- Strong BTech final year project

**Demonstration:**
- Live demo with dual cameras
- Depth map visualization
- Attack detection (photos, videos)
- User enrollment and verification
- Performance metrics dashboard

---

## 16. Additional Resources

### Recommended Reading Order

1. **Start:** OpenCV stereo calibration tutorial
2. **Then:** RetinaFace paper (face detection)
3. **Next:** LBP anti-spoofing paper (Määttä et al.)
4. **Advanced:** ArcFace paper (face recognition)
5. **Optional:** EfficientNet paper (deepfake detection)

### Online Courses

- **Computer Vision:** Coursera - "First Principles of Computer Vision"
- **Deep Learning:** fast.ai - "Practical Deep Learning"
- **OpenCV:** PyImageSearch tutorials

### Communities

- **Stack Overflow:** opencv, computer-vision tags
- **Reddit:** r/computervision, r/MachineLearning
- **GitHub:** Search for "face anti-spoofing" repos

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Author:** Technical Specification for BTech Final Year Project  
**Status:** Complete and Ready for Implementation

