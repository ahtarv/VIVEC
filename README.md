## README: Vivec-U SOTA Segmentation Model

This repository contains the implementation and experimental results for **Vivec-U**, a specialized deep learning architecture designed for high-precision medical image segmentation, specifically targeting the **BUSI (Breast Ultrasound Images) dataset**.

---

## 🏗 Architecture Overview

Vivec-U is built upon the YOLOv8-segmentation framework but incorporates three primary custom modules to enhance multi-scale feature extraction and boundary precision:

| Module | Purpose | Mechanism |
| :--- | :--- | :--- |
| **Scalseq** | Multi-Scale Fusion | Aggregates features from P3, P4, and P5 layers using a channel-attention gating mechanism to prioritize relevant scales. |
| **TuaBottleneck** | Deep Feature Extraction | Utilizes a bottleneck structure with **GELU** activation for smoother gradient flow and better non-linear representation. |
| **Zoomcat** | Spatial Contextualization | Processes input at three parallel scales (interpolated down, original, and interpolated up) and concatenates them to capture both global context and fine edges. |

### Aliasing Injection
To integrate with the `ultralytics` engine, these modules are injected as aliases into the standard YOLO task parser (e.g., `GhostConv` maps to `Scalseq`), allowing the model to benefit from the optimized training pipeline while utilizing custom SOTA logic.

---

## 📊 Performance Metrics

The model was evaluated on a standardized random split of the BUSI dataset. Results are categorized into "Global" (all images) and "Conditional" (only images where a lesion was detected).

### Clinical Evaluation Results
* **Recall:** 92.31% (Lesion Detection Rate)
* **Mean DICE (Conditional):** **0.8061** 🔥
* **Mean IoU (Conditional):** **0.7223**
* **Mean HD95 (Conditional):** **58.01 pixels** ✅

> **Note:** The **HD95** (95th percentile Hausdorff Distance) serves as a key clinical metric for boundary precision, measuring how closely the predicted mask edge aligns with the ground truth.

---

## 🚀 Training Configuration

The model was trained using a "SOTA Ablation" setting with the following hyperparameters:
* **Epochs:** 100
* **Image Size:** 640x640
* **Optimizer:** AdamW (auto-selected)
* **Augmentation:** Mosaic (0.5), Mixup (0.0), Albumentations (Blur, MedianBlur, CLAHE, ToGray)
* **Specialized Settings:** * `retina_masks=True`: For high-resolution mask boundaries.
    * `overlap_mask=False`: To ensure distinct mask separation during primary training.

---

## 📂 Project Structure

* `vivec_net.yaml`: The architectural definition of the Vivec-U backbone and head.
* `busi.yaml`: Dataset configuration pointing to the BUSI golden split.
* `vivecbusiv3.ipynb`: The primary research notebook containing training logs and evaluators.
* `runs/`: Directory containing weights (`best.pt`, `last.pt`) and training validation plots.

---

## 🛠 Usage

To run the clinical evaluator or fine-tune the model:

1.  **Dependencies:** `pip install ultralytics torch torchvision opencv-python tqdm scipy`
2.  **Evaluation:** Use the `run_conditional_eval()` function in the provided notebook to generate DICE and HD95 reports.
3.  **Refinement:** A 10-epoch refinement script is included to sharpen boundaries by increasing `box` gain and disabling mosaic augmentation (`close_mosaic=10`).

---

**Would you like me to generate a summary of the training loss curves or the specific layer-by-layer parameter count for this architecture?**
