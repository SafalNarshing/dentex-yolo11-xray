<div align="center">

# 🦷 DentEx-YOLO11 Dental Disease Detection

**Automated detection and segmentation of dental diseases in panoramic X-rays using YOLO11n-seg trained on the DentEx dataset.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-FF6B35?logo=data:image/svg+xml;base64,)](https://ultralytics.com)
[![Dataset](https://img.shields.io/badge/Dataset-DentEx-green)](https://github.com/ibrahimethemhamamci/DentEx)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![WandB](https://img.shields.io/badge/Tracking-Weights%20%26%20Biases-orange?logo=weightsandbiases)](https://wandb.ai)
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle%20GPU-20BEFF?logo=kaggle)](https://kaggle.com)

Kaggle Notebook URL: [Click Here](https://www.kaggle.com/code/safalnarshing/annotation-model)

<br/>
<img src="https://github.com/SafalNarshing/dentex-yolo11-xray/blob/f6f347a8f4161d4e135d8f1939362b53e5d2a8bc/outputs/images/result_1.png" alt="Dental disease detection result on panoramic X-ray" width="860"/>

*YOLO11n-seg detecting Caries, Deep Caries, Periapical Lesions and Impacted Teeth on a panoramic dental X-ray*


</div>

---

## Table of Contents

- [Overview](#-overview)
- [Detected Disease Classes](#-detected-disease-classes)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Analysis — Why the Scores Are Moderate](#-analysis--why-the-scores-are-moderate)
- [Visualisations](#-visualisations)
- [Future Improvements](#-future-improvements)
- [Citation](#-citation)
- [License](#-license)

---

## Overview

This project applies **YOLO11n-seg** (instance segmentation) to automatically detect and localise four clinically significant dental conditions in panoramic radiographs (OPGs). The model outputs both **bounding boxes** and **segmentation masks** for each detected lesion, making it suitable for computer-aided diagnosis (CAD) systems.

The training pipeline uses the [DentEx](https://github.com/ibrahimethemhamamci/DentEx) challenge dataset, which provides triple-level hierarchical annotations: **quadrant → tooth enumeration → disease classification**. Only the disease-level annotations (`category_id_3`) are used for this task.

---

## Detected Disease Classes

| ID | Disease | Clinical Significance |
|----|---------|----------------------|
| `0` | **Caries** | Early-stage tooth decay |
| `1` | **Deep Caries** | Advanced decay approaching the pulp |
| `2` | **Periapical Lesion** | Infection/abscess at the root apex |
| `3` | **Impacted Tooth** | Tooth unable to fully erupt |

---

## Dataset

### DentEx Dataset Structure

```
Dentex/
├── test_data/                          # 250 unlabelled images (held-out)
├── training_data/
│   ├── quadrant/                       # 693 images — quadrant-only labels
│   ├── quadrant_enumeration/           # 634 images — quadrant + tooth ID labels
│   ├── quadrant_enumeration_disease/   # 705 images — full triple labels ✅ used
│   └── unlabeled/                      # 1,571 images — no annotations
└── validation_data/
    └── validation_data/
        └── quadrant_enumeration_disease/
            ├── xrays/                  # 50 validation images ✅ used
            └── validation_triple.json  # Official COCO-format annotations
```

### Split Summary

| Split | Images | Annotations | Notes |
|-------|--------|-------------|-------|
| **Train** | 705 | ~2,700+ instances | `quadrant_enumeration_disease` only |
| **Validation** | **50** | ~182 instances | Official DentEx val split |
| **Test** | 250 | — | No labels released |
| **Unlabelled** | 1,571 | — | Not used in this run |

> ⚠️ **The official validation set contains only 50 images** — a critically small evaluation set that directly affects metric reliability (see [Analysis](#-analysis--why-the-scores-are-moderate)).

### Disease Distribution (Validation Split)

| Class | GT Instances |
|-------|-------------|
| Caries | 40 |
| Deep Caries | 101 |
| Periapical Lesion | 9 |
| Impacted Tooth | 32 |
| **Total** | **182** |

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| **Base model** | `yolo11n-seg` (nano — instance segmentation) |
| **Parameters** | ~2.9 M |
| **Input resolution** | 640 × 640 |
| **Task** | Instance segmentation (box + mask) |
| **Backbone** | C3k2 + SPPF + C2PSA |
| **Head** | Segmentation head with prototype masks |
| **Training hardware** | Kaggle T4 GPU (16 GB) |
| **Experiment tracking** | Weights & Biases (WandB) |

### Training Configuration

```python
model.train(
    data     = "dentex.yaml",
    epochs   = 100,
    imgsz    = 640,
    batch    = 16,
    patience = 20,          # early stopping
    optimizer= "AdamW",
    lr0      = 0.001,
    augment  = True,        # mosaic, flipud, fliplr, hsv
    project  = "runs",
    name     = "dentex_yolo11n_seg",
)
```
---

## Results

> Evaluated on the **official DentEx validation split** (50 images, 182 GT instances) at `conf=0.25`, `IoU=0.50`.

### Bounding Box Metrics

| Metric | Score |
|--------|-------|
| **mAP@50** | 0.4478 (44.8%) |
| **mAP@50-95** | 0.2856 (28.6%) |
| Precision | 0.6145 (61.4%) |
| Recall | 0.5410 (54.1%) |
| F1 | 0.5754 (57.5%) |
| FP Rate (FDR) | 0.3855 (38.6%) |
| FN Rate (Miss) | 0.4590 (45.9%) |

### Segmentation Mask Metrics

| Metric | Score |
|--------|-------|
| **Mask mAP@50** | 0.3981 (39.8%) |
| **Mask mAP@50-95** | 0.2557 (25.6%) |
| Mask Precision | 0.5567 (55.7%) |
| Mask Recall | 0.5054 (50.5%) |
| Mask F1 | 0.5298 (53.0%) |

### Per-Class Error Breakdown

| Class | GT | Missed (FN) | Miss Rate | Notes |
|-------|----|-------------|-----------|-------|
| **Caries** | 40 | 6 | **15.0%** ✅ | Best performing |
| **Deep Caries** | 101 | 69 | **68.3%** ❌ | Hardest class |
| **Periapical Lesion** | 9 | 5 | **55.6%** ⚠️ | Very few GT instances |
| **Impacted Tooth** | 32 | 13 | **40.6%** ⚠️ | Moderate difficulty |

```
  Precision    0.6145   61.4%  ██████████████████
  Recall       0.5410   54.1%  ████████████████
  F1           0.5754   57.5%  █████████████████
  mAP@50       0.4478   44.8%  █████████████
  mAP@50-95    0.2856   28.6%  ████████
  FP rate      0.3855   38.6%  ███████████
  FN rate      0.4590   45.9%  █████████████
```

---

## Analysis — Why the Scores Are Moderate

The results are consistent with what is expected given the experimental constraints. This section provides an honest breakdown.

### 1. Critically Small Dataset

The DentEx disease-labelled training set contains only **705 images** — far below the tens of thousands typically required to train a robust object detector for medical imaging. The validation set of **50 images** further limits score reliability:

- A single missed image can swing mAP by several percentage points.
- Classes with very few GT instances (e.g. Periapical Lesion: only **9** in validation) produce statistically unstable AP scores.
- For comparison, general-purpose YOLO benchmarks use validation sets of 5,000+ images (COCO).

### 2. Nano Model on a Hard Task

`yolo11n` is the **smallest variant** in the YOLO11 family (~2.9 M parameters), designed for speed and edge deployment. Dental radiograph analysis is a high-difficulty task because:

- Lesions are often **small, low-contrast** and visually similar to healthy tissue.
- **Deep Caries vs Caries** share very similar radiographic appearance — even radiologists sometimes disagree.
- The background (bone, soft tissue) creates noisy context that confuses a small model.

A larger variant (`yolo11s`, `yolo11m`) would likely improve mAP by 8–15 points with the same data.

### 3. Severe Class Imbalance

| Class | Train instances (est.) | Val instances |
|-------|------------------------|---------------|
| Caries | high | 40 |
| **Deep Caries** | dominant | **101** |
| Periapical Lesion | rare | **9** |
| Impacted Tooth | moderate | 32 |

Deep Caries dominates both splits. The model sees far more Deep Caries during training yet still achieves the **worst miss rate (68.3%)**, indicating the class itself is intrinsically hard (subtle visual difference from early Caries).

### 4. No Unlabelled Data Used

DentEx provides 1,571 unlabelled images that could be leveraged with **semi-supervised learning** or **pseudo-labelling** — none of this was done in the current run, leaving a large portion of available data unused.

### 5. Single-Scale Training at 640px

Panoramic X-rays are wide-aspect-ratio images (~2800×1400px). Resizing to 640×640 **crops or compresses** the image significantly, causing small lesions near the edges to be lost or distorted.

### Summary Table

| Factor | Impact | Mitigation |
|--------|--------|------------|
| 705 labelled training images | High negative | Collect more data / pseudo-label |
| 50-image validation set | High (metric noise) | Use cross-validation |
| Nano model (2.9M params) | Medium negative | Upgrade to yolo11s/m |
| Class imbalance (Deep Caries) | High negative | Weighted loss / oversampling |
| 640px input (wide X-rays) | Medium negative | Rectangular inference |
| No semi-supervised learning | Medium negative | Pseudo-label 1.5K unlabelled |

---

## Visualisations

### Training Curves (WandB)

<div align="center">
<img src="https://github.com/SafalNarshing/dentex-yolo11-xray/blob/f6f347a8f4161d4e135d8f1939362b53e5d2a8bc/outputs/images/training_graphs.png" alt="WandB training metrics — box loss, cls loss, mAP over epochs" width="860"/>
</div>

<br/>

### Confusion Matrix

<div align="center">
<img src="https://github.com/SafalNarshing/dentex-yolo11-xray/blob/f6f347a8f4161d4e135d8f1939362b53e5d2a8bc/outputs/images/confusion_matrix.png" alt="Confusion matrix showing TP FP FN breakdown across disease classes" width="600"/>
</div>

> Rows = Ground Truth, Columns = Predicted. The **Background column** represents missed detections (FN). Note how Deep Caries has the largest spillage into the Background column, confirming the 68.3% miss rate.

<br/>

### Validation — Prediction vs Ground Truth

<table align="center">
  <tr>
    <th>Ground Truth</th>
    <th>Model Prediction</th>
  </tr>
  <tr>
    <td><img src="https://github.com/SafalNarshing/dentex-yolo11-xray/blob/f6f347a8f4161d4e135d8f1939362b53e5d2a8bc/outputs/images/val_batch0_labels.jpg" alt="Ground truth annotations" width="420"/></td>
    <td><img src="https://github.com/SafalNarshing/dentex-yolo11-xray/blob/f6f347a8f4161d4e135d8f1939362b53e5d2a8bc/outputs/images/val_batch0_pred.jpg" alt="Model predictions" width="420"/></td>
  </tr>
</table>

*Left: ground-truth polygons from `validation_triple.json`. Right: model predictions at `conf=0.25, IoU=0.50`.*

---

## 🚀 Future Improvements

| Priority | Improvement | Expected Gain |
|----------|-------------|---------------|
| 🔴 High | Use larger model (`yolo11s` or `yolo11m`) | +8–15 mAP points |
| 🔴 High | Semi-supervised pseudo-labelling on 1,571 unlabelled images | +5–10 mAP |
| 🔴 High | Rectangular inference (`rect=True`) for wide-aspect X-rays | Reduce crop loss |
| 🟡 Medium | Weighted loss for Deep Caries / Periapical Lesion | Reduce 68% miss rate |
| 🟡 Medium | Test-time augmentation (TTA) | +2–4 mAP |
| 🟡 Medium | Cross-validation on 5 folds (small dataset) | Reliable metric estimation |
| 🟢 Low | Export to ONNX / TensorRT for clinical deployment | Speed |
| 🟢 Low | Integrate quadrant + tooth-number heads (full DentEx task) | Multi-task |


---

## 📖 Citation

If you use this work, please cite both this repository and the original DentEx dataset:

```bibtex
@misc{dentex-yolo11-2026,
  title        = {DentEx-YOLO11: Dental Disease Detection and Segmentation on Panoramic X-rays},
  author       = {Safal Narshing Shrestha},
  year         = {2026},
  howpublished = {\url{https://github.com/SafalNarshing/dentex-yolo11-xray}},
  note         = {YOLO11n-seg trained on the DentEx dataset}
}

@article{hamamci2023dentex,
  title   = {DentEx: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-Rays},
  author  = {Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and
             Yuksel, Atif Emre and Gultekin, Sadullah and Ozdemir, Serife Damla and
             Rosa, Mustafa and others},
  journal = {arXiv preprint arXiv:2305.19112},
  year    = {2023}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The **DentEx dataset** is subject to its own terms. Please refer to the [DentEx repository](https://github.com/ibrahimethemhamamci/DentEx) for dataset licensing and usage conditions before using it in commercial or clinical settings.

---

<div align="center">

If this project helped you, please consider giving it a ⭐

</div>
