# 🚀 Project Roadmap: Next Frontiers in Crop Classification

This document tracks the strategic initiatives and experimental ideas to push the performance of our signed-laplacian-based crop classification system.

---

## 🔬 Research & Experimentation Pipeline

### 1️⃣ Inference Strategy: Hard Thresholding (❌ Unsuccessful)
> *Observation: Hard thresholding didn't improve results. This is likely because some edge weights fall near $d^*$, and pulling them away from this boundary point doesn't help.*
- [x] Implement a **Hard Threshold** mechanism for inference.
- [x] Compare performance metrics (Precision, Recall, F1) against the current `modified_sigmoid` approach.
- [x] Analyze if the hard threshold reduces noise in the Laplacian weights.

### 2️⃣ Best-Case Performance Analysis (Oracle Experiment) (⚠️ Not Informative)
> *Observation: The experiment wasn't informative as we achieved 100% accuracy, which is expected when using GT for the Laplacian. It doesn't provide new insights into the gap.*
- [x] Define the **Signed Laplacian** directly from the **Ground Truth** labels for each binary classifier.
- [x] Run validation to see the "Best Scenario" performance.
- [x] Use this baseline to quantify the gap between our current feature extractor and an "Ideal" extractor.

### 3️⃣ Random Forest 2.0: Feature Enrichment
> *Objective: Improve the non-neural classification component by feeding it richer information.*
- [ ] Include additional spectral bands or seasonal indices.
- [ ] Integrate **topological features** or eigenvector-derived statistics.
- [ ] Hyperparameter tuning for the enhanced Random Forest model.

### 4️⃣ Robust Sign Correction Strategy
> *Objective: Solve the inherent sign ambiguity in Laplacian eigenvectors more reliably.*
- [ ] Refine the current `correct_pred_sign` logic.
- [ ] Explore alternative anchoring methods (e.g., using small, high-confidence labeled subsets).
- [ ] Test the stability of sign correction across different crop types and geographical variations.

### 5️⃣ Eigen Solver Initiatives: Starting Vector Optimization
> *Objective: Improve the convergence and efficiency of our differentiable eigen solvers by optimizing the initial vector.*
- [ ] Implement **Fixed Vector Initialization**: Use a constant vector (e.g., all ones) to ensure deterministic starting points across iterations.
- [ ] Implement **Learned Starting Vector**: Use a learnable parameter or a small network to predict a "warm-start" vector for the eigen solver.
- [ ] Evaluate the impact of initialization on convergence speed and gradient stability.

### 6️⃣ End-to-End Pipeline: GraphSpectralNet
> *Objective: Train the feature extractor and eigen solver as a single unified system.*
- [x] Create `GraphSpectralNet` to integrate CNN, Sparse Graph Construction, and Differentiable Eigen Solver.
- [x] Implement **Signed Laplacian Loss** following the standard quadratic form $x^T L_s x$.
- [x] Pre-process high-resolution patches into **56x56** and **32x32** subpatch datasets for efficient training.
- [ ] Optimize the trade-off between **Rayleigh** and **Power Method** solvers during backpropagation.

---

## 🛠 Status Dashboard

| Idea | Priority | Complexity | Status |
| :--- | :---: | :---: | :---: |
| Hard Thresholding | 🟡 Med | 🟢 Low | ❌ Failed |
| GT Signed Laplacian | 🔴 High | 🟡 Med | ⚠️ N/A |
| RF Feature Expansion | 🟡 Med | 🟡 Med | 🕒 Pending |
| Sign Correction | 🔴 High | 🔴 High | 🏁 In Progress |
| Eigen Solver Init | 🟡 Med | 🟡 Med | 🕒 Pending |
| End-to-End Tuning | 🔴 High | 🔴 High | 🏁 In Progress |

---

> *"The best way to predict the future is to build it."* 
> — **The Crop-Classification Team**
