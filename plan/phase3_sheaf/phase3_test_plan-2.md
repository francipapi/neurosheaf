
# Phase 3 Sheaf Construction & Laplacian — Comprehensive Validation Test Plan  
**Version:** 1.1  **Date:** 2025-07-11

_This revision adapts the plan for a **CPU‑only macOS workstation** (12 cores, 32 GB RAM). GPU‑specific steps have been removed or down‑scoped._

---

## 1 Purpose & Scope  
Same as v1.0, but all performance targets are recalibrated for a high‑end desktop CPU environment.

---

## 2 Test Environment  

| Item | Specification |
|------|---------------|
| **CPU** | 12‑core Apple Silicon / Intel (AVX2) |
| **RAM** | 32 GB |
| **OS** | macOS 14 (Sonoma) |
| **Python** | 3.11 |
| **PyTorch** | 2.2 (CPU build) |
| **NetworkX** | 3.x |
| **SciPy** | 1.13 |
| **pytest** | 8.x + `pytest‑xdist` & `pytest‑benchmark` |
| **Optional** | `hypothesis`, `psutil` |

> **Note** – `xdist ‑n auto` will spawn up to 12 workers (one per logical core).

---

## 3 Datasets & Models  

| Category | Data / Model | Purpose |
|----------|--------------|---------|
| **Synthetic IID** | Gaussian activations (𝑛 = 128…2048, 𝑑 = 16…512) | Scaled down for RAM |
| **Toy graphs** | Path, star, complete DAGs (≤ 10 nodes) | Analytic eigenvalues |
| **Reference NN** | `torchvision.models.resnet18` | Skip‑connection handling |
| | Tiny Transformer (2 encoder blocks) | Attention handling |
| **Production** | `torchvision.models.resnet50`, **batch 256** | Feasible within 32 GB |

Synthetic data remain deterministic under `torch.manual_seed(0)`.

---

## 4 Test Matrix (Key Updates Highlighted)  

| ID | Feature Under Test | Level | Data Set | Assertion(s) |
|----|-------------------|-------|----------|--------------|
| **unchanged** | U‑P01 … U‑L05 | | | |
| I‑C06 | End‑to‑end sheaf build | integration | ResNet‑18 | unchanged |
| I‑C07 | Sheaf + laplacian pipeline | integration | Transformer | unchanged |
| **P‑M08** | Memory footprint | perf | ResNet‑50 × 256 | **RSS ≤ 8 GB** |
| **P‑T09** | Throughput | perf | ResNet‑50 × 256 | **End‑to‑end ≤ 15 min** |
| S‑E10 | Numerical stability | stress | Gaussian rank‑deficient | unchanged |
| S‑O11 | Large‑depth path | stress | 200‑layer MLP | unchanged |

Rows not shown are identical to v1.0.

---

## 5 Detailed Test Procedures (delta)  

### 5.6 Performance Benchmarks (revised)

*Command*  
```bash
pytest tests/phase3_sheaf/validation/test_performance.py --benchmark-json=perf.json -n auto
```  
`--benchmark-autosave` is enabled; CPU wall‑clock time is compared to 15‑minute threshold. RSS gathered via `psutil`.

---

## 6 Tooling & Automation  

GPU markers have been removed. GitHub Actions CI matrix is now **cpu/{unit,integ,perf}**.

Coverage, lint, and dashboard steps remain unchanged.

---

## 7 Deliverables & Traceability  

No change except updated performance artefacts.

---

### End of Document
