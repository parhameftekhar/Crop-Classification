# Research Findings: Spectral Graph Network for Crop Classification

This document tracks key insights and numerical findings discovered during the development and fine-tuning of the end-to-end spectral pipeline.

---

## 1. Rayleigh Solver Performance vs. Laplacian Formulation

### **Observation**
We observed a significant discrepancy in the performance of the **Rayleigh Minimization (Gradient Descent)** solver when switching between the **Standard Laplacian** and the **Signed Laplacian**.

### **Standard Laplacian Findings**
When using the **Standard Laplacian** ($D = \sum W$), the **Rayleigh Minimizer** with 200 iterations and an initial step size of 0.1 proves to be highly effective. It produces results nearly identical to the ground-truth eigenvalues achieved by the non-differentiable **SciPy Lanczos (eigsh)** solver. This confirms that for stable graphs, gradient descent on the Rayleigh quotient is an efficient and accurate differentiable solver.

### **Signed Laplacian Findings**
When switching to the **Signed Laplacian** ($D_{abs} = \sum |W|$), the **Rayleigh Minimizer** failed to converge to a meaningful solution with the same hyperparameters. 
*   Unlike the Standard version, the Signed Laplacian results in much larger diagonal values ($D_{ii}$ becomes a sum of absolute weights). 
*   This makes the standard 0.1 step size unstable for the Rayleigh solver, likely due to overshooting the minimum.
*   **Verification:** To confirm the issue was the solver and not the theory, we verified the Signed Laplacian using the **Lanczos (SciPy)** solver and the **Shifted Power Method**. Both produced reasonable and stable accuracy (approx. 0.75 F1-score), proving that the Signed Laplacian is mathematically sound but requires a much more carefully tuned step size when using a Rayleigh-based gradient approach.

### **Conclusion**
For **Standard Laplacians**, the Rayleigh solver is robust and achieves parity with SciPy's Lanczos. For **Signed Laplacians**, a different step-size strategy (or a scale-independent solver like the Shifted Power Method) is necessary to maintain numerical stability.

---

## 2. Scaling Strategy: Large Batch Size vs. Gradient Accumulation

### **Observation**
During the implementation of the batched `GraphSpectralNet`, we compared two methods for processing high-resolution subpatches:
1.  **Block-Diagonal Batching:** Constructing a single $BN \times BN$ giant sparse matrix for $B$ patches.
2.  **Sequential Accumulation:** Processing batch size of 1 with $B$ gradient accumulation steps.

### **Numerical Insight**
Surprisingly, we found that **Gradient Accumulation** is more efficient for our spectral pipeline:
*   **Memory Overhead:** Constructing the "giant" block-diagonal sparse matrix in COO format creates significant indexing overhead. As $B$ increases, the management of the global index tensor ($2 \times E_{total}$) becomes more memory-intensive than processing patches sequentially.
*   **Solver Convergence Time:** The eigen solvers (Power Method/Rayleigh) spend more time per iteration on the giant matrix. Sequential processing of smaller matrices allows for better memory locality and avoids the overhead of managing massive sparse tensors in PyTorch.
*   **Parallelization Gap:** While GPUs excel at parallelism, the specific sparse structure of our local graphs (window size 30) doesn't benefit from being merged into a single large system. The overhead of the "giant" sparse index management outweighs the throughput gains.

### **Strategic Decision**
We will prioritize **Gradient Accumulation** with `batch_size: 1` as the primary scaling strategy. This maintains a low memory footprint while achieving the desired effective batch size for stable gradient descent.

> [!IMPORTANT]
> **Note on Implementation:** The current block-diagonal batched implementation in `GraphSpectralNet` and `EigenSolver` is a first-pass prototype. While it functions correctly, there is significant potential for optimization in how the global indices are managed to reduce the overhead mentioned above. Future iterations may explore more efficient sparse handling or custom CUDA kernels to bridge the performance gap.
