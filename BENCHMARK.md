# Benchmark Results

## Crop 176 (Target Crop)

**Table 1: Performance comparison of different eigen-solvers (F1-score / Accuracy) across various image resolutions for Crop 176 (computed on the validation set).**

| Solver \ Resolution | 32 × 32 | 56 × 56 | 112 × 112 |
| :--- | :---: | :---: | :---: |
| Rayleigh (No Training) | - | - | - |
| Rayleigh (With Training) | - | - | - |
| Lanczos | 73.73 / 87.53 | 73.59 / 87.28 | 72.41 / 86.45 |

**Table 2: Relationship between image resolution and the number of iterations of the Rayleigh solver (no training). Metrics are reported as F1-score / Accuracy.**

| Iterations \ Resolution | 32 × 32 | 56 × 56 | 112 × 112 |
| :---: | :---: | :---: | :---: |
| 10 | 0.6304 / 0.7967 | 0.6527 / 0.8119 | 0.6812 / 0.8335 |
| 100 | 0.6885 / 0.8422 | 0.6744 / 0.8294 | 0.7137 / 0.8573 |
| 200 | 0.6918 / 0.8448 | 0.6757 / 0.8305 | 0.7185 / 0.8608 |

**Table 3: Relationship between image resolution and the number of iterations of the Rayleigh solver (with training). Metrics are reported as F1-score / Accuracy.**

| Iterations \ Resolution | 32 × 32 | 56 × 56 | 112 × 112 |
| :---: | :---: | :---: | :---: |
| 10 | - | - | - |
| 100 | 0.7119 / 0.8513 | - | - |
| 200 | - | - | - |
