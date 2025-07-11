
# Phase 3 Real‑Network Test — Ground‑Truth Metrics  
**Version:** 1.0  **Date:** 2025-07-11

| Metric | Expected | Unit |
|--------|----------|------|
| Nodes in poset | **79** | |
| Edges in poset | **85** | |
| Σ stalk dimension | **≈ 9 300** | float |
| Laplacian nnz | **≈ 278 000** | ints |
| Harmonic dimension | **4** | count |
| Restriction residual max | **0.048** | ratio |
| Symmetry error | **< 1e‑11** | ∞‑norm |
| λ_min(Δ) | **≥ −8 × 10⁻¹⁰** | value |
| Pipeline runtime | **≤ 300 s** | seconds |
| Peak RAM | **≤ 3.8 GB** | GiB |

All numbers obtained on **macOS 14, M2 Pro 12‑core CPU, 32 GB RAM**, using PyTorch 2.2 (CPU) and SciPy 1.13.  Allow ±10 % variance on different CPUs.

*Reference log stored at* `ground_truth/run_log_m2pro_seed0.txt`.
