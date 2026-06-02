# Datasets Used

All datasets downloaded to `data/` (git-ignored). Re-download using links below.

---

## 1. FF++ Faces (Deepfake Detection — Training)

| | |
|---|---|
| **Kaggle** | https://www.kaggle.com/datasets/dagnelies/deepfake-faces |
| **Local path** | `data/raw/faceforensicspp/faces_224/` |
| **Labels** | `data/raw/faceforensicspp/metadata.csv` |
| **Size** | ~433 MB, 95,634 images |
| **Content** | Pre-extracted 224×224 face crops from FaceForensics++ (c23). 16,293 real + 79,341 fake. Label column: `REAL` / `FAKE`. |

```bash
python -m kaggle datasets download -d dagnelies/deepfake-faces -p data/raw/faceforensicspp --unzip
```

---

## 2. 140k Real vs Fake Faces (Deepfake Detection — Train + Test)

| | |
|---|---|
| **Kaggle** | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces |
| **Local path** | `data/raw/celebdf/real_vs_fake/real-vs-fake/` |
| **Size** | ~3.8 GB, 110,742 images |
| **Content** | Pre-split train/test. StyleGAN fakes vs real faces. `train/real`: 40,742 · `train/fake`: 50,000 · `test/real`: 10,000 · `test/fake`: 10,000 |

```bash
python -m kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw/celebdf --unzip
```

---

## 3. Real and Fake Face Detection — ciplab (Validation)

| | |
|---|---|
| **Kaggle** | https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection |
| **Local path** | `data/raw/celebdf/real_and_fake_face/` |
| **Size** | ~431 MB, 2,041 images |
| **Content** | High-quality GAN-generated fakes vs real. `training_real/`: 1,081 · `training_fake/`: 960. Good for validation/hard-negative testing. |

```bash
python -m kaggle datasets download -d ciplab/real-and-fake-face-detection -p data/raw/celebdf --unzip
```

---

## 4. Anti-Spoofing Dataset (Physical Attack Testing)

| | |
|---|---|
| **Kaggle** | https://www.kaggle.com/datasets/tapakah68/anti-spoofing |
| **Local path** | `data/raw/custom/single_cam/` |
| **Size** | ~1 GB, 25 videos |
| **Content** | Real-world attack videos. `live_selfie/`: real · `printouts/`: printed photo attack · `cut-out printouts/`: cut-out attack · `replay/`: screen replay attack. |

```bash
python -m kaggle datasets download -d tapakah68/anti-spoofing -p data/raw/custom/single_cam --unzip
```

---

## 5. LFW — Labeled Faces in the Wild (ArcFace Evaluation)

| | |
|---|---|
| **Official** | http://vis-www.cs.umass.edu/lfw/ |
| **Mirror used** | https://ndownloader.figshare.com/files/5976018 |
| **Pairs file** | https://raw.githubusercontent.com/davidsandberg/facenet/master/data/pairs.txt |
| **Local path** | `data/raw/lfw/lfw/` |
| **Size** | ~172 MB, 13,233 images, 5,749 identities |
| **Content** | Standard face verification benchmark. 6,000 pairs (3,000 matched + 3,000 mismatched) for threshold tuning. |

```bash
curl -L https://ndownloader.figshare.com/files/5976018 -o data/raw/lfw/lfw.tgz
tar -xzf data/raw/lfw/lfw.tgz -C data/raw/lfw/
curl -L https://raw.githubusercontent.com/davidsandberg/facenet/master/data/pairs.txt -o data/raw/lfw/pairs.txt
```

---

## 6. InsightFace buffalo_l (Pre-trained Models)

| | |
|---|---|
| **Source** | Auto-downloaded by InsightFace on first use |
| **Local path** | `~/.insightface/models/buffalo_l/` |
| **Size** | ~190 MB |
| **Models** | `det_10g.onnx` (RetinaFace), `w600k_r50.onnx` (ArcFace R50), `1k3d68.onnx` (3D landmarks), `2d106det.onnx` (2D landmarks), `genderage.onnx` |

```python
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  # downloads automatically if missing
```

---

## Re-download All

```bash
# Requires Kaggle credentials at ~/.kaggle/access_token
python -m kaggle datasets download -d dagnelies/deepfake-faces       -p data/raw/faceforensicspp      --unzip
python -m kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw/celebdf              --unzip
python -m kaggle datasets download -d ciplab/real-and-fake-face-detection -p data/raw/celebdf          --unzip
python -m kaggle datasets download -d tapakah68/anti-spoofing         -p data/raw/custom/single_cam    --unzip

curl -L https://ndownloader.figshare.com/files/5976018 -o data/raw/lfw/lfw.tgz
tar -xzf data/raw/lfw/lfw.tgz -C data/raw/lfw/
curl -L https://raw.githubusercontent.com/davidsandberg/facenet/master/data/pairs.txt -o data/raw/lfw/pairs.txt
```

---

## Dataset Summary

| # | Dataset | Images | Real | Fake | Purpose |
|---|---|---:|---:|---:|---|
| 1 | FF++ faces | 95,634 | 16,293 | 79,341 | EfficientNet training |
| 2 | 140k real vs fake | 110,742 | 50,742 | 60,000 | EfficientNet train + test |
| 3 | ciplab real/fake | 2,041 | 1,081 | 960 | Validation / hard negatives |
| 4 | Anti-spoofing | 25 videos | 9 real | 16 fake | Physical attack testing |
| 5 | LFW | 13,233 | 5,749 IDs | — | ArcFace threshold tuning |
| 6 | buffalo_l | 5 models | — | — | RetinaFace + ArcFace inference |
| | **Total images** | **221,650** | | | |
