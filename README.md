
# Multimodal Fusion (MMF) for Thoracic Disease Classification

A novel Multimodal Fusion (MMF) framework designed to bridge the semantic gap between chest X-ray images and clinical narratives for robust, interpretable thoracic disease classification. 

The architecture leverages a **Swin Transformer** visual encoder and a **Bio_ClinicalBERT** textual encoder, tightly integrated via a **Hierarchical Bidirectional Co-Attention** mechanism to dynamically ground language concepts into spatial radiological regions.

---

## 📂 Repository Structure

The `src/` directory contains all python modules required for the end-to-end pipeline:

*   **Encoders:** Model architectures for the Swin Transformer (`vision_encoder.py`) and Bio_ClinicalBERT (`text_encoder.py`).
*   **Fusion Strategies:** Modular implementations for all three fusion techniques (early_fusion.py, mid_fusion.py, and late_fusion.py).
*   **Preprocessing:** Data cleaning, tokenization, and medical image transformation pipelines.(data_preprocessing.py)
*   **Interpretability:** Scripts for generating **Grad-CAM** heatmaps and **Saliency Maps** to explain model decisions.
*   **Execution:** Core scripts for training and evaluation pipelines.

---

## 🚀 Training

Train any of the three fusion strategies evaluated in the paper by passing the `--fusion` flag to the training script.

### Early Fusion
Combines modality features at an early stage before deep representation learning.
```bash
python src/train.py --fusion early
```

### Mid Fusion (Best-Performing BioFuse Model)
Leverages the Hierarchical Bidirectional Co-Attention mechanism to dynamically fuse latent features.
```bash
python src/train.py --fusion mid
```
### Late Fusion
Aggregates independent modality predictions at the final decision layer.
```bash
python src/train.py --fusion late
```
---
## Evaluation & Interpretability
Generate metrics, Grad-CAM heatmaps, and saliency maps from a trained checkpoint:
```bash
python src/evaluate.py --checkpoint path/to/model.pt
```
---

python src/evaluation.py --checkpoint path/to/model.pt
