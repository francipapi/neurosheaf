# Cellular‑Sheaf Pipeline — Acceptance Criteria

This checklist gives **quantitative pass/fail thresholds** that an implementation must satisfy to be considered *mathematically sound* and production‑ready.  All tests assume stalks have been whitened (or kernels quotiented) unless explicitly noted.

---

## 1  Stalk Quality

| # | Test | Statistic | Accept if | Rationale |
|---|------|-----------|----------|-----------|
| 1 | *Rank adequacy* | `rank(K_v)/min(n,d_v)` | **≥ 0.95** | avoid degenerate inner products |
| 2 | *Condition number* | `cond(K_v)` | **≤ 1 × 10⁵** | keeps whitening stable |

## 2  Edge Maps

| # | Test | Statistic | Accept if | Rationale |
|---|------|-----------|----------|-----------|
| 3 | Orthogonality | `‖Q_eᵀ Q_e – I‖_F / √n` | ≤ 1 × 10⁻⁶ | ensures Q_e ∈ O(n) |
| 4 | Fit residual | `ρ_e := ‖E_e‖_F / ‖K_w‖_F` | ≤ 5 × 10⁻² | Procrustes must match K_w within 5 % |
| 5 | Metric‑compatibility | `μ_e := ‖R_eᵀ K_w R_e – K_v‖_F / ‖K_v‖_F` | ≤ 5 × 10⁻² | sheaf axiom (approx.) |

## 3  Functoriality

| # | Test | Statistic | Accept if | Rationale |
|---|------|-----------|----------|-----------|
| 6 | Composition error (u→v→w) | `‖R_uw – R_vw R_uv‖_F / ‖R_uw‖_F` | ≤ 1 × 10⁻² | compatibility of composites |
| 7 | Path consistency (parallel routes) | `‖R_uw^{(p)} – R_uw^{(q)}‖_F / ‖R_uw^{(p)}‖_F` | ≤ 1 × 10⁻² | uniqueness of image |

## 4  Laplacian Checks

| # | Test | Statistic | Accept if | Rationale |
|---|------|-----------|----------|-----------|
| 8 | Symmetry | `‖Δ – Δᵀ‖_F / ‖Δ‖_F` | ≤ 1 × 10⁻¹² | numerical sanity |
| 9 | Positive‑semidefinite | `λ_min(Δ)` | ≥ –1 × 10⁻⁸ · ‖Δ‖₂ | small FP negatives OK |
| 10 | Betti‑0 | `dim ker Δ` | = # components | e.g. 1 for connected poset |

## 5  Filtration & Masking

| # | Test | Description | Accept if | Rationale |
|---|------|-------------|-----------|-----------|
| 11 | Edge monotonicity | `E(τ_i) ⊇ E(τ_j)` for τ_i < τ_j | always true | definition of filtration |
| 12 | Laplacian reuse | Tests 8–9 on every `Δ(τ)` | same tolerances | validates masking scheme |

## 6  Exact Metrics (Whitened Coords)

| # | Test | Statistic | Accept if | Rationale |
|---|------|-----------|----------|-----------|
| 13 | Exact orthogonality | `‖Q̃_eᵀ Q̃_e – I‖_F` | ≤ 1 × 10⁻¹² | machine‑zero |
| 14 | Exact metric‑compat. | `‖R̃_eᵀ R̃_e – I‖_F` | ≤ 1 × 10⁻¹² | axiom holds symbolically |

## 7  Performance Targets*  *(ResNet‑50, batch 1024 on A100)*

| # | Test | Accept if | Notes |
|---|------|-----------|-------|
| 15 | Peak GPU memory | ≤ 3 GB | end‑to‑end run |
| 16 | Wall‑clock time | ≤ 5 min | end‑to‑end run |

> *Skip §7 if hardware differs; keep proportional targets.*

---

### Failure Diagnosis

| Fails | Typical cause | Quick fix |
|-------|---------------|-----------|
| 3–5 | scale blow‑up, rank‑deficiency | whiten stalks; tighten `ρ_max` |
| 6–7 | residuals accumulate | drop worst edges or raise `ρ_max` |
| 8–9 | coding bug, mask leakage | audit Laplacian builder & cache |
| 10 | β₀ off by –1 | still‑broken compatibility ⇒ Δ shifted |
| 11 | non‑monotone edges | weight formula not strictly positive |

---

Pass **criteria 1–14** (or **1–16** incl. perf) and the pipeline is deemed *production‑grade*.

