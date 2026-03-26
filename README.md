

# 🧬 VIVEC-U: Medical Segmentation Model
**V**ariable **I**ntensity **V**ascular & **E**pithelial **C**haracterization — **Ultra**

VIVEC-U is a specialized deep learning architecture engineered for high-precision medical image segmentation. While based on the YOLOv8-seg framework, it utilizes a heavily modified backbone reminiscent of **DFEM-NET** to handle the unique challenges of the **BUSI (Breast Ultrasound Images)** dataset.

> **Note:** VIVEC-U is currently optimized for GPU-accelerated inference. A CPU-optimized version, **VIVEC-n** (Normal), based on the BhaskarNET architecture, is currently in development.

---

## 🏗 Architecture Overview

The core of VIVEC-U lies in three custom-engineered modules designed to enhance multi-scale feature extraction and boundary precision:

| Module | Purpose | Mechanism |
| :--- | :--- | :--- |
| **Scalseq** | **Multi-Scale Fusion** | Aggregates features from P3, P4, and P5 layers using channel-attention gating to prioritize relevant lesion scales. |
| **TuaBottleneck** | **Feature Extraction** | Employs a bottleneck structure with **GELU** activation for smoother gradient flow and superior non-linear representation. |
| **Zoomcat** | **Spatial Context** | Processes input at three parallel scales (0.5x, 1x, 2x) to capture both global context and fine micro-calcification edges. |

### Aliasing Injection
To leverage the `ultralytics` training engine while using custom logic, these modules are injected via class aliasing:
* `GhostConv` $\rightarrow$ **Scalseq**
* `GhostBottleneck` $\rightarrow$ **TuaBottleneck**
* `Focus` $\rightarrow$ **Zoomcat**

---

## 📊 Performance Metrics

Evaluated on a standardized random split of the BUSI dataset. Results distinguish between **Global** performance and **Conditional** performance (metrics calculated only on successful detections).

### Clinical Evaluation
* **Lesion Recall:** 92.31%
* **Mean DICE (Conditional):** **0.8061** 🔥
* **Mean IoU (Conditional):** **0.7223**
* **Mean HD95 (Conditional):** **58.01 pixels** ✅

> **Metric Insight:** The **HD95** (95th percentile Hausdorff Distance) measures boundary alignment. In ultrasound, where edges are often "fuzzy," a conditional HD95 of ~58px indicates high reliability for surgical margin estimation.

---

## 💻 Hardware Benchmarking (CPU Baseline)
*Tested on Intel Core i7 / Lenovo Laptop*

| Metric | Result |
| :--- | :--- |
| **Mean Latency** | 2813.03 ms |
| **Throughput** | 0.36 FPS |
| **Configuration** | imgsz=640, retina_masks=True |

*Current CPU performance is suitable for static diagnostic analysis. For real-time applications, a CUDA-enabled GPU is recommended.*

---

## 🚀 Training Configuration

* **Epochs:** 100
* **Image Size:** 640x640
* **Optimizer:** AdamW
* **Augmentation:** Mosaic (0.5), Mixup (0.0), Albumentations (CLAHE, Blur, ToGray)
* **Precision Settings:** * `retina_masks=True`: Ensures high-res mask boundaries.
    * `overlap_mask=False`: Prevents mask bleeding during training.

---

## 🛠 Usage & Requirements

### Installation
```bash
pip install ultralytics torch torchvision opencv-python tqdm scipy
```

### Inference
1.  **Evaluation:** Use the `run_conditional_eval()` function in `vivecbusiv3.ipynb` to generate clinical reports.
2.  **Refinement:** A 10-epoch "Edge Sharpening" script is available to increase `box` gain and disable mosaic noise for final polishing.

---
