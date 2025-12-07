# Image Guide for Final Presentation
## Essential Images Only - Slide by Slide

---

## Slide 2: Introduction
**📊 IMAGE NEEDED: Real-world applications**

**What to show:**
- Apple Face ID logo/screenshot
- Banking app with face authentication
- Airport automated gate

**Source:**
- Google Images: "Apple Face ID", "face recognition banking", "airport biometric gate"
- Create collage: 3 images side-by-side

**Size:** 1920×600 pixels (full width, half height)

---

## Slide 3: The Problem
**📊 IMAGE NEEDED: Attack examples**

**What to show:**
- Photo attack: Phone displaying a face photo
- Video replay: Tablet playing a video
- Deepfake: Side-by-side real vs fake face

**Sources:**
1. **Photo/Video attacks:** 
   - Replay-Attack dataset: https://www.idiap.ch/dataset/replayattack
   - Example images in dataset documentation

2. **Deepfake:**
   - FaceForensics++ paper: https://arxiv.org/pdf/1901.08971.pdf
   - Page 3, Figure 2 (Real vs Fake comparison)

**Layout:** 3 images in a row with labels

**Size:** 1920×600 pixels

---

## Slide 4: Existing Solutions
**📊 IMAGE NEEDED: Accuracy vs cost bar chart**

**What to show:**
- X-axis: Methods (Depth Sensors, Texture-Based, CNN, Deepfake Detectors)
- Y-axis 1 (left): Accuracy (%)
- Y-axis 2 (right): Cost (₹)
- Two bar series: Blue for accuracy, Orange for cost

**Data:**
```
Depth Sensors: 98%, ₹15,000
Texture-Based: 87%, ₹500
CNN Anti-Spoofing: 93%, ₹5,000
Deepfake Detectors: 92%, ₹5,000
```

**Create using:** Excel, Python (matplotlib), or PowerPoint

**Size:** 1200×800 pixels

---

## Slide 5: Proposed Solution
**📊 IMAGE NEEDED: System architecture flowchart**

**What to show:**
```
[Left Camera] ──┐
                ├──> [Stereo Depth] ──> [Liveness Detection]
[Right Camera] ─┘                              ↓
                                        [RetinaFace]
                                              ↓
                                        [EfficientNet-B0]
                                              ↓
                                           [LoRA]
                                              ↓
                                        [Accept/Reject]
```

**Create using:** Draw.io, PowerPoint, or Lucidchart

**Colors:**
- Cameras: Gray boxes
- Processing: Blue boxes
- Decision: Green/Red

**Size:** 1200×900 pixels

**Template:** https://app.diagrams.net/ (use flowchart shapes)

---

## Slide 6: RetinaFace

### IMAGE 1: RetinaFace Architecture
**📊 IMAGE NEEDED: FPN backbone architecture**

**Source:** 
- RetinaFace paper: https://arxiv.org/pdf/1905.00641.pdf
- **Page 3, Figure 2** (Architecture diagram)

**What it shows:**
- Feature Pyramid Network (FPN)
- Multi-scale feature extraction
- Multi-task branches (bbox, landmarks, classification)

**Download:** Screenshot from PDF, crop to architecture only

**Size:** 1400×700 pixels

---

### IMAGE 2: Detection Example
**📊 IMAGE NEEDED: Face detection with 5 landmarks**

**Source:**
- InsightFace GitHub: https://github.com/deepinsight/insightface/tree/master/detection/retinaface
- Example images in `/data/` folder
- OR use: https://github.com/deepinsight/insightface/blob/master/detection/retinaface/data/t1.jpg

**What it shows:**
- Detected face with bounding box
- 5 landmarks marked: 2 eyes, nose, 2 mouth corners
- Confidence score

**Size:** 800×800 pixels

---

## Slide 7: EfficientNet-B0

### IMAGE 1: Compound Scaling
**📊 IMAGE NEEDED: Compound scaling visualization**

**Source:**
- EfficientNet paper: https://arxiv.org/pdf/1905.11946.pdf
- **Page 2, Figure 2** (Compound scaling illustration)

**What it shows:**
- Three dimensions: Depth (d), Width (w), Resolution (r)
- Visual showing all three scaling together
- Comparison: Single dimension vs compound scaling

**Download:** Screenshot from PDF, Page 2, Figure 2

**Size:** 1200×600 pixels

---

### IMAGE 2: EfficientNet-B0 Architecture
**📊 IMAGE NEEDED: Network architecture diagram**

**Source:**
- EfficientNet paper: https://arxiv.org/pdf/1905.11946.pdf
- **Page 4, Table 1** (Architecture details)
- OR **Page 3, Figure 1** (Network structure)

**What it shows:**
- MBConv blocks
- 7 stages
- Input (224×224) to output
- Squeeze-and-Excitation modules

**Alternative:** Create simplified diagram showing:
```
Input (224×224)
    ↓
MBConv Block 1
    ↓
MBConv Block 2
    ↓
...
    ↓
MBConv Block 7
    ↓
Global Pooling
    ↓
Output
```

**Size:** 800×1000 pixels (vertical)

---

## Slide 8: EfficientNet Comparison

**📊 IMAGE NEEDED: Accuracy vs Parameters scatter plot**

**Source:**
- EfficientNet paper: https://arxiv.org/pdf/1905.11946.pdf
- **Page 1, Figure 1** (ImageNet accuracy vs parameters)

**What it shows:**
- X-axis: Parameters (millions)
- Y-axis: ImageNet Top-1 Accuracy (%)
- Points for: ResNet-50, XceptionNet, MobileNetV2, EfficientNet-B0
- EfficientNet-B0 should be in top-left (high accuracy, low parameters)

**Download:** Screenshot from PDF, Page 1, Figure 1

**OR Create using Python:**
```python
import matplotlib.pyplot as plt

models = ['ResNet-50', 'XceptionNet', 'MobileNetV2', 'EfficientNet-B0']
params = [25.6, 23, 3.5, 5.3]  # millions
accuracy = [76.0, 75.0, 72.0, 77.1]  # %

plt.scatter(params, accuracy, s=200)
for i, model in enumerate(models):
    plt.annotate(model, (params[i], accuracy[i]))
plt.xlabel('Parameters (M)')
plt.ylabel('ImageNet Top-1 Accuracy (%)')
plt.title('Model Efficiency Comparison')
plt.grid(True)
plt.savefig('efficientnet_comparison.png', dpi=300)
```

**Size:** 1200×800 pixels

---

## Slide 9: Training Strategy

### IMAGE 1: Training Pipeline
**📊 IMAGE NEEDED: Transfer learning flowchart**

**What to show:**
```
[ImageNet Pre-trained EfficientNet-B0]
              ↓
    [Remove 1000-class head]
              ↓
    [Add Binary Classifier]
              ↓
      [Freeze Early Layers]
              ↓
     [Fine-tune Later Layers]
              ↓
    [Train on FaceForensics++]
```

**Create using:** PowerPoint or Draw.io

**Colors:**
- Pre-trained: Blue
- Modifications: Orange
- Training: Green

**Size:** 1000×800 pixels

---

### IMAGE 2: Real vs Fake Examples
**📊 IMAGE NEEDED: FaceForensics++ dataset examples**

**Source:**
- FaceForensics++ paper: https://arxiv.org/pdf/1901.08971.pdf
- **Page 3, Figure 2** (Real and manipulated faces)
- Shows 4 manipulation types

**What it shows:**
- Original (real) face
- DeepFakes version
- Face2Face version
- FaceSwap version
- NeuralTextures version

**Layout:** 5 images in a row (1 real + 4 fakes)

**Size:** 1920×400 pixels

**Alternative source:**
- GitHub: https://github.com/ondyari/FaceForensics
- Example images in README

---

## Slide 10: LoRA

**📊 IMAGE NEEDED: LoRA decomposition diagram**

**Source:**
- LoRA paper: https://arxiv.org/pdf/2106.09685.pdf
- **Page 3, Figure 1** (LoRA architecture)

**What it shows:**
- Original weight matrix W₀ (frozen)
- Low-rank matrices B and A (trainable)
- Formula: W = W₀ + BA
- Visual showing matrix dimensions

**Download:** Screenshot from PDF, Page 3, Figure 1

**Alternative:** Create simplified diagram:
```
┌─────────────┐
│   W₀        │  (Frozen, d×k)
│  (Original) │
└─────────────┘
       +
┌───┐   ┌───┐
│ B │ × │ A │  (Trainable, d×r and r×k)
└───┘   └───┘
       =
┌─────────────┐
│   W         │  (Final weights)
└─────────────┘
```

**Size:** 1200×600 pixels

---

## Slide 11: LoRA Results

**📊 IMAGE NEEDED: Model size comparison**

**What to show:**
- Bar chart comparing model sizes
- X-axis: Model type
- Y-axis: Size (MB)

**Data:**
```
Original EfficientNet-B0: 20 MB
EfficientNet-B0 + LoRA: 3.5 MB
```

**Create using:** Excel or Python

**Python code:**
```python
import matplotlib.pyplot as plt

models = ['Original\nEfficientNet-B0', 'With LoRA']
sizes = [20, 3.5]
colors = ['#ff6b6b', '#4ecdc4']

plt.bar(models, sizes, color=colors, width=0.6)
plt.ylabel('Model Size (MB)')
plt.title('LoRA Compression: 35× Reduction')
plt.ylim(0, 25)

# Add value labels on bars
for i, v in enumerate(sizes):
    plt.text(i, v + 0.5, f'{v} MB', ha='center', fontweight='bold')

plt.savefig('lora_compression.png', dpi=300, bbox_inches='tight')
```

**Size:** 1000×700 pixels

---

## Quick Download Checklist

### From Papers (ArXiv PDFs):

✅ **Slide 6:** RetinaFace paper, Page 3, Figure 2
✅ **Slide 7:** EfficientNet paper, Page 2, Figure 2 (compound scaling)
✅ **Slide 7:** EfficientNet paper, Page 4, Table 1 (architecture)
✅ **Slide 8:** EfficientNet paper, Page 1, Figure 1 (scatter plot)
✅ **Slide 9:** FaceForensics++ paper, Page 3, Figure 2 (examples)
✅ **Slide 10:** LoRA paper, Page 3, Figure 1 (decomposition)

### From GitHub:

✅ **Slide 6:** InsightFace detection examples
✅ **Slide 9:** FaceForensics++ examples

### Create Yourself:

✅ **Slide 2:** Real-world applications collage (Google Images)
✅ **Slide 3:** Attack examples collage
✅ **Slide 4:** Accuracy vs cost bar chart (Excel/Python)
✅ **Slide 5:** System architecture flowchart (Draw.io)
✅ **Slide 9:** Training pipeline flowchart (PowerPoint)
✅ **Slide 11:** Model size comparison bar chart (Python)

---

## Image Placement in Slides

### PowerPoint Tips:

1. **Full-width images:** Place at top or bottom, leave space for title
2. **Side-by-side:** Text on left (40%), image on right (60%)
3. **Centered:** For architecture diagrams, center with title above
4. **Comparison images:** Use 2-3 column layout

### Recommended Layout:

**Slide 2:** Image at bottom (full width)
**Slide 3:** 3 images in row at center
**Slide 4:** Chart on right, table on left
**Slide 5:** Flowchart centered
**Slide 6:** Architecture top, detection example bottom
**Slide 7:** Scaling left, architecture right
**Slide 8:** Scatter plot centered
**Slide 9:** Pipeline top, examples bottom
**Slide 10:** Diagram centered
**Slide 11:** Bar chart centered

---

## File Naming Convention

Save images as:
- `slide02_applications.png`
- `slide03_attacks.png`
- `slide04_comparison.png`
- `slide05_architecture.png`
- `slide06_retinaface_arch.png`
- `slide06_retinaface_detection.png`
- `slide07_compound_scaling.png`
- `slide07_efficientnet_arch.png`
- `slide08_accuracy_params.png`
- `slide09_training_pipeline.png`
- `slide09_faceforensics_examples.png`
- `slide10_lora_decomposition.png`
- `slide11_model_size.png`

---

## Image Quality Standards

**Resolution:** Minimum 1920×1080 for full-slide images
**DPI:** 300 for print, 150 for screen presentation
**Format:** PNG (for diagrams), JPG (for photos)
**File size:** <2 MB per image
**Colors:** Use consistent color scheme (blue, orange, green)

---

## Total Images Needed: 13

**From papers:** 6 images
**From GitHub:** 2 images
**Create yourself:** 5 images

**Estimated time:** 2-3 hours to collect and create all images

---

**Last Updated:** December 2024
