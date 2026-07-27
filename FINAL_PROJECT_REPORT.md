# Project Report: AI-Driven Classification of Retinopathy of Prematurity (ROP)

**Student Name:** [Your Name]  
**Topic:** Medical Image Classification for ROP Detection  
**Hardware Used:** Lenovo Legion 7i Pro (RTX 5070 Ti Blackwell GPU, 32GB RAM)  
**Frameworks:** PyTorch Nightly, CUDA 12.8, TorchVision  

---

## 1. Project Overview
The goal of this project is to develop a high-accuracy Deep Learning model to classify retinal fundus images into two categories: **ROP** (Retinopathy of Prematurity) and **Normal**. ROP is a leading cause of childhood blindness, and early, automated detection can significantly improve clinical outcomes.

## 2. The Challenge: Hardware & Data
- **Hardware Compatibility:** The NVIDIA RTX 5070 Ti (Blackwell) is a next-generation GPU. Standard versions of PyTorch and TensorFlow were incompatible. We solved this by implementing the **PyTorch Nightly** build with CUDA 12.8 support.
- **Data Limitations:** The dataset consisted of only ~185 usable images. Small datasets are notoriously difficult for Deep Learning models as they lead to "overfitting" (memorizing the data instead of learning patterns).

## 3. Technical Methodology

### A. Model Architecture: EfficientNet-B3
We selected **EfficientNet-B3** as our backbone. Unlike standard CNNs, EfficientNet uses "Compound Scaling" to balance depth, width, and resolution. This makes it exceptionally good at identifying tiny, fine features like the twisted blood vessels characteristic of ROP.

### B. Transfer Learning
To overcome the small dataset, we used **Transfer Learning**. Instead of starting with a "blank" model, we used a model pre-trained on the ImageNet-1K dataset (millions of images). This provided the model with a baseline "visual intelligence" which we then fine-tuned for retinal imagery.

### C. Advanced Data Augmentation
To artificially expand our dataset, we implemented a real-time augmentation pipeline:
- **Random Rotations & Flips:** Helping the model recognize ROP regardless of image orientation.
- **Color & Lighting Jitter:** Accounting for variations in camera exposure and lighting conditions in clinical settings.
- **High-Resolution Processing:** Training at **512x512** resolution to ensure fine vascular details were preserved.

### D. Multi-Phase Training
1. **Phase 1 (Warmup):** Frozen the main backbone and trained only the classification "head" to stabilize the model.
2. **Phase 2 (Fine-tuning):** Unfrozen the entire network and used a very low learning rate (`1e-5`) to carefully adapt the weights to the ROP dataset.

## 4. Results & Performance
The final model was evaluated on a held-out validation set (~20% of the data).

| Metric | Result |
| :--- | :--- |
| **Total Accuracy** | **92%** |
| **Precision (ROP)** | **100%** |
| **F1-Score** | **0.91** |
| **AUC-ROC** | **0.99** |

**Key Finding:** The model achieved **100% Precision for ROP**, meaning every time the model identified ROP, it was correct. It also achieved a **100% Recall for Normal images**, correctly identifying every healthy patient.

## 5. Conclusion
By transitioning from a custom-built CNN to a state-of-the-art Transfer Learning pipeline, we successfully reached professional-grade accuracy levels. The system demonstrates that with advanced architectures like EfficientNet and proper hardware optimization, high-accuracy medical diagnosis is possible even with limited datasets.
