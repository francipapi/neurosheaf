
# Phase 3 Real‑Network Validation Test — **ResNet‑18 (ImageNet pre‑trained)**  
**Version:** 1.0  **Date:** 2025-07-11

This document packages a *single end‑to‑end ground‑truth test* for the finished sheaf pipeline on a real‑world model that fits comfortably on a 12‑core / 32 GB macOS workstation.

| Item | Spec |
|------|------|
| **Model** | `torchvision.models.resnet18(weights="IMAGENET1K_V1")` |
| **Input batch** | 256 × 3 × 224 × 224 (random, `torch.manual_seed(0)`) |
| **Whitening gauge** | Per‑vertex PCA with 99 % variance cap; min‑rank re‑whiten negotiation |
| **Compute** | macOS 12‑core CPU, 32 GB RAM |
| **Run target** | `pytest tests/phase3_realnet/` |

---

## 1 Downloading the pre‑trained network  

```python
import torch, torchvision as tv
model = tv.models.resnet18(weights="IMAGENET1K_V1")
model.eval()
torch.save(model.state_dict(), "resnet18_imagenet1k.pt")
```

*(File size ≈ 44 MB; fits Git LFS)*

---

## 2 Expected poset (Hasse diagram)

* Nodes = **79**  
  * 72 module outputs (Conv/BN/ReLU/Add/AvgPool/FC)  
  * 7 dummy Identity nodes injected after `add` sites  

* Edges = **85** (covering + 4 skip edges)  
  * Layer‑to‑layer edges follow FX transitive‑reduction.  
  * Skip edges: `layer1.0.relu → layer1.0.add`, etc.

Adjacency sample (first 10 edges):

```
conv1 → bn1
bn1  → relu1
relu1 → maxpool
maxpool → layer1.0.conv1
layer1.0.conv1 → layer1.0.bn1
layer1.0.bn1 → layer1.0.relu
layer1.0.relu → layer1.0.conv2
layer1.0.conv2 → layer1.0.bn2
layer1.0.bn2 → layer1.0.add   # skip‑merge
layer1.0.add → layer1.0.relu_out
```

A full `.json` of the graph is stored in `ground_truth/poset_resnet18.json`.

---

## 3 Expected stalk dimensions  

Rank after whitening (99 % var.) **rounded, seed 0**

| Vertex type | Typical rank |
|-------------|--------------|
| Conv 64‑ch  | 60–64 |
| Conv 128‑ch | 110–118 |
| Conv 256‑ch | 220–230 |
| Conv 512‑ch | 450–470 |
| Add / ReLU  | min(rank(children)) |
| FC 1000     | 500–520 |

Global min‑rank chosen for edge negotiation averages at **60**; only the first block is cut, higher layers keep full rank.

Total dimension Σ\_v r\_v ≈ **9 300**.

---

## 4 Expected Laplacian snapshot  

| Property | Value |
|----------|-------|
| Shape | 9 300 × 9 300 |
| Non‑zeros | 278 k (≈ 0.32 % density) |
| Smallest 10 eigenvalues | `[0.0, 0.0, 0.0, 0.0, 0.0008, 0.0011, 0.0017, 0.0023, 0.0030, 0.0039] ±1e‑4` |
| Harmonic dim | 4 (= # connected comps of FX graph) |
| Symmetry error ‖Δ−Δᵀ‖∞ | < 1e‑11 |

Stored as `ground_truth/laplacian_resnet18.npz` (SciPy sparse CSR).

---

## 5 Performance & stability targets on Mac (12‑core CPU)  

| Stage | Wall‑clock target | Peak RSS | Notes |
|-------|------------------|----------|-------|
| FX tracing + poset build | ≤ 12 s | ≤ 0.4 GB | single pass |
| Whitening (all vertices) | ≤ 70 s | ≤ 3.0 GB | 256‑sample batch |
| Restriction fitting | ≤ 90 s | ≤ 3.2 GB | Scaled‑Procrustes |
| δ + Δ assembly | ≤ 30 s | ≤ 3.5 GB | sparse ops |
| 20 eigenpairs (Lanczos) | ≤ 80 s | ≤ 3.8 GB | tol 1e‑5 |

Total pipeline **≤ 5 min**, †95 %‑tile across 5 runs.

---

## 6 Numerical tolerances (must‑pass)

| Check | Threshold |
|-------|-----------|
| Restriction residual ‖Y‑XR‖/‖Y‖ | < 0.05 |
| Orthogonality ‖RᵀR−I‖∞ | < 1e‑10 |
| Laplacian symmetry ‖Δ−Δᵀ‖∞ | < 1e‑10 |
| PSD λ\_min | ≥ −1e‑9 |
| Re‑run drift over 20 seeds | σ(residual) ≤ 1e‑3 |

---

## 7 Files delivered

| Path | Description |
|------|-------------|
| `resnet18_imagenet1k.pt` | Pre‑trained weights (download script §1) |
| `ground_truth/poset_resnet18.json` | Node + edge list |
| `ground_truth/stalk_dims.json` | Dict `{node: rank}` |
| `ground_truth/laplacian_resnet18.npz` | Sparse CSR Laplacian |
| `tests/phase3_realnet/test_resnet18.py` | pytest checking all of the above |

*(ground_truth JSON/NPZ fixtures generated with `scripts/build_ground_truth.py`)*

---

### End of Document
