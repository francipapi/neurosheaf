
# Phase 3 Sheaf Pipeline — Validation Targets & Acceptance Criteria  
**Version:** 1.1  **Date:** 2025-07-11

_Updated for a 12‑core CPU‑only Mac with 32 GB RAM._

---

## 2 Quantitative Acceptance Criteria (updated rows in **bold**)  

| ID | Metric | Target | Rationale |
|----|--------|--------|-----------|
| **Q‑M01** | **Runtime (ResNet‑50, batch 256)** | **≤ 15 min (12‑core CPU)** | Tuned to desktop performance |
| **Q‑M02** | **Peak resident memory (same run)** | **≤ 8 GB** | Leaves > 20 GB headroom |
| Q‑N03 | Restriction map residual | < 0.05 synthetic; < 0.10 real | unchanged |
| Q‑N04 | Metric‑compat. error | < 1e‑12 | unchanged |
| Q‑L05 | Laplacian symmetry | ≤ 1e‑10 | unchanged |
| Q‑L06 | Laplacian PSD | ≥ −1e‑9 | unchanged |
| Q‑L07 | Sparsity (path n=200) | nnz/N² ≤ 1 % | unchanged |
| Q‑S08 | Harmonics vs comps | equal | unchanged |
| Q‑P09 | Persistence diagram stability | 1‑W ≤ 1e‑2 | unchanged |
| Q‑C10 | Coverage | ≥ 90 % | unchanged |
| Q‑S11 | Static analysis | 0 errors, ≤ 10 warnings | unchanged |

Other sections remain identical; any mention of GPU has been removed.

---

### End of Document
