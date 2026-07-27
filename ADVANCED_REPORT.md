# Project Report: Advanced AI-Driven Classification of Retinopathy of Prematurity (ROP)

---

## 1. Project Overview
This project presents an advanced deep learning solution for the automated detection of Retinopathy of Prematurity (ROP) using retinal fundus images. Moving beyond initial prototypes, we developed a high-performance pipeline capable of identifying ROP with industrial-grade accuracy.

The system utilizes a state-of-the-art **EfficientNet-B3** architecture, optimized for the latest NVIDIA hardware, to provide clinicians with rapid, confidence-based diagnostic support.

---

## 2. Dataset Specifications
*   **Dataset Name:** HVDROPDB
*   **Source:** Mendeley Data (High-resolution preterm infant retinal images)
*   **Categories:** 
    *   **Normal:** Healthy retinal development (Label = 0)
    *   **ROP:** Retinopathy of Prematurity detected (Label = 1)
*   **Resolution:** 2048 × 2048 pixels
*   **Processing Pipeline:**
    *   **Resolution Scaling:** 512 × 512 pixels (Optimized for feature extraction vs. memory).
    *   **Normalization:** ImageNet-based Mean/Std scaling.
    *   **Format:** RGB 3-channel tensors.

---

## 3. Hardware & Environment Configuration
To achieve maximum training speed and model depth, the project utilized a next-generation hardware stack:
*   **System:** Lenovo Legion 7i Pro
*   **GPU:** NVIDIA RTX 5070 Ti (12 GB VRAM, Blackwell Architecture)
*   **Architecture Support:** **PyTorch Nightly (CUDA 12.8)**
    *   *Note: This specific framework version was selected to unlock support for the sm_120 Blackwell kernels, resolving compatibility issues found in stable builds.*

---

## 4. Technical Methodology: From Prototype to Production

### A. The Evolution of the Solution
| Phase | Approach | Outcome |
| :--- | :--- | :--- |
| **Initial** | Custom CuPy-based CNN | Functional but limited accuracy (~65%) and weak feature learning. |
| **Advanced** | **PyTorch + EfficientNet-B3** | **High Accuracy (92%)** with professional metrics and GPU optimization. |

### B. EfficientNet-B3 Architecture
We replaced standard CNN layers with the EfficientNet-B3 backbone. This architecture uses "Compound Scaling" to optimize the depth, width, and resolution of the network simultaneously. This is critical for medical imaging, as it allows the model to detect both global retinal structures and micro-vascular abnormalities.

### C. Advanced Training Strategy
*   **Transfer Learning:** Initialized with ImageNet-V1 weights to leverage pre-existing visual intelligence.
*   **Two-Phase Fine-Tuning:**
    *   **Phase 1:** Classifier head warmup (Backbone frozen).
    *   **Phase 2:** Full backbone fine-tuning at a low learning rate (1e-5).
*   **Real-time Augmentation:** Implemented random horizontal/vertical flips, rotations, and color jitter to artificially expand the dataset and prevent overfitting.

---

## 5. Confidence-Based Inference
A key feature of the final system is **Probabilistic Inference**. Instead of a simple "Yes/No" output, the model provides a confidence score for its prediction.
*   **Example Output:** `Result: Normal | Confidence: 98.19%`
*   **Impact:** This allows medical professionals to prioritize cases where the model has lower confidence for manual review.

---

## 6. Performance & Results
The model demonstrated exceptional generalization capabilities on the validation set.

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.86 | 1.00 | 0.93 | 19 |
| **ROP** | **1.00** | 0.83 | 0.91 | 18 |
| **Accuracy** | | | **0.92** | 37 |

### Key Metrics
*   **Overall Accuracy:** **92%**
*   **ROP Precision:** **1.00** (Zero False Positives for ROP)
*   **AUC-ROC Score:** **0.99** (Near-perfect separation of classes)

---

## 7. Visualization & Interpretability
The system generates:
1.  **Training Curves:** Visualizing the convergence of loss and accuracy.
2.  **Confusion Matrix:** Detailing the specific classification performance for each category.
3.  **Inference Visuals:** Displaying the input image alongside the predicted class and confidence percentage.

---

## 8. Conclusion
The transition from a custom CNN prototype to a professional PyTorch pipeline using EfficientNet-B3 successfully addressed the challenges of hardware compatibility and small dataset size. The final model provides a reliable, high-precision tool for ROP classification, achieving a 92% accuracy rate and a near-perfect AUC-ROC of 0.99. This project validates the efficacy of transfer learning and advanced augmentation in the domain of medical AI.
