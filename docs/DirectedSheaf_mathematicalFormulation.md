# Persistent Directed Spectral Sheaf Pipeline (v4)

*A fully-worked mathematical exposition incorporating directed cellular sheaves for asymmetric network analysis, with explicit assumptions, known caveats, and practical patches.*

---

## 0  Notation and Setup

* **Network graph**: A directed graph $G=(V,E)$ extracted by *torch.fx* (or a fallback tracer). Vertices represent modules plus dedicated **input**/**output** nodes. 
* **Sample size**: A batch of $n\le 4096$ i.i.d. inputs is fixed once per run.
* **Field**: The base field is $\mathbb{C}$ for stalks and restriction maps, with real representations for computation. Hermitian transpose $(·)^*$ denotes conjugate transpose.
* **Directionality parameter**: $q \in \mathbb{R}$, typically $q = 1/4$, controls the strength of directional encoding.

Throughout, we order vertices and edges once and for all; block matrices inherit this ordering.

---

## 1  Stalks (Local Data Spaces)

For every vertex $v\in V$:

1. Collect the raw (uncentered) activation matrix
   $$X_v \in \mathbb{R}^{n\times d_v}.$$

2. Form the (uncentered) Gram matrix
   $$K_v = X_v X_v^{\top} \in \mathbb{R}^{n\times n}.$$

3. The stalk at $v$ is initially the real inner-product space
   $$\mathcal{V}_v := \bigl(\mathbb{R}^{n}, \langle u,w\rangle_{K_v}=u^{\top}K_v w\bigr).$$

### Whitening Transformation (MANDATORY)

A thin spectral factorization $K_v=U_v \Sigma_v U_v^{\top}$ (with $\Sigma_v \in \mathbb{R}^{r_v \times r_v}$ positive diagonal) defines the whitening map:
$$W_v=\Sigma_v^{-1/2}U_v^{\top}:\mathbb{R}^{n}\twoheadrightarrow\mathbb{R}^{r_v}.$$

**Critical**: The whitened stalk $\tilde{\mathcal{V}}_v=(\mathbb{R}^{r_v},\langle\cdot,\cdot\rangle_I)$ uses the standard inner product. All subsequent computations occur in whitened coordinates.

### Complex Extension for Directed Sheaves

For the directed sheaf construction, we extend whitened stalks to complex vector spaces:
$$\tilde{\mathcal{F}}(v) := \mathbb{C}^{r_v} = \mathbb{R}^{r_v} \otimes_{\mathbb{R}} \mathbb{C}.$$

This extension is necessary to accommodate complex-valued restriction maps encoding directionality.

---

## 2  Directed Cellular Sheaf and Restriction Maps

### 2.1 Directed Cellular Sheaf Definition

A **Directed Cellular Sheaf** on $G=(V,E)$ with adjacency matrix $A$ consists of:

1. **Directional encoding matrix**: 
   $$T^{(q)} := \exp(i 2\pi q (A - A^{\top}))$$
   where $(T^{(q)})_{uv} = \exp(i 2\pi q (A - A^{\top})_{uv})$.

2. **Vertex stalks**: $\tilde{\mathcal{F}}(v) = \mathbb{C}^{r_v}$ for each $v \in V$.

3. **Edge stalks**: $\tilde{\mathcal{F}}(e) = \mathbb{C}^{r_{t(e)}}$ for each $e \in E$ (where $t(e)$ is the target of edge $e$).

4. **Restriction maps**: For edge $e=(u,v)$:
   - From source: $\tilde{\mathcal{F}}_{u \triangleleft e}: \tilde{\mathcal{F}}(u) \to \tilde{\mathcal{F}}(e)$
   - From target: $\tilde{\mathcal{F}}_{v \triangleleft e}: \tilde{\mathcal{F}}(v) \to \tilde{\mathcal{F}}(e)$

### 2.2 Computing Restriction Maps

For each edge $e=(u,v)$:

1. **Real-valued base map** via scaled Procrustes in whitened coordinates:
   $$(s_e, Q_e) = \arg\min_{s>0, Q \in \mathbb{R}^{r_v \times r_u}} \|s W_v K_v K_u^{\top} W_u^{\top} - W_v K_w K_v^{\top} W_v^{\top}\|_F$$
   subject to $Q^{\top}Q = I_{r_u}$ (orthogonality constraint).

2. **Directed restriction maps**:
   - Source map (real): $\tilde{\mathcal{F}}_{u \triangleleft e} = s_e Q_e$
   - Target map (complex): $\tilde{\mathcal{F}}_{v \triangleleft e} = T^{(q)}_{uv} I_{r_v}$

For undirected edges ${u,v}$, we have $T^{(q)}_{uv} = 1$, so both maps remain real.

> **Key insight**: Directionality is encoded in the complex phase of the target restriction map.

---

## 3  Directed Cochain Complex and Laplacian

### 3.1 Directed Coboundary Operator

The space of 0-cochains: $C^0(G; \tilde{\mathcal{F}}) = \bigoplus_{v\in V} \tilde{\mathcal{F}}(v) \cong \mathbb{C}^{\sum_v r_v}$

The directed coboundary operator $\tilde{\delta}: C^0 \to C^1$ acts on edge $e=(u,v)$ as:
$$(\tilde{\delta} f)_e = \tilde{\mathcal{F}}_{v \triangleleft e} f_v - \tilde{\mathcal{F}}_{u \triangleleft e} f_u = T^{(q)}_{uv} f_v - s_e Q_e f_u$$

### 3.2 Directed Sheaf Laplacian

The Directed Sheaf Laplacian is:
$$\boxed{\mathcal{L}^{\tilde{\mathcal{F}}} = \tilde{\delta}^* \tilde{\delta}}$$

This is a **Hermitian** operator (not just symmetric) with block structure:

* **Diagonal block** (vertex $v$, size $r_v \times r_v$):
  $$\mathcal{L}^{\tilde{\mathcal{F}}}_{vv} = \sum_{e=(v,w)} s_e^2 Q_e^{\top} Q_e + \sum_{e=(u,v)} |T^{(q)}_{uv}|^2 I_{r_v}$$

* **Off-diagonal blocks** (for edge $e=(u,v)$):
  $$\mathcal{L}^{\tilde{\mathcal{F}}}_{uv} = -s_e Q_e^{\top} \overline{T^{(q)}_{uv}}$$
  $$\mathcal{L}^{\tilde{\mathcal{F}}}_{vu} = -T^{(q)}_{uv} s_e Q_e$$

**Mathematical Properties**:
- ✅ **Hermitian**: $(\mathcal{L}^{\tilde{\mathcal{F}}})^* = \mathcal{L}^{\tilde{\mathcal{F}}}$
- ✅ **Positive Semi-Definite**: All eigenvalues are real and non-negative
- ✅ **Reduces to undirected case**: When $G$ is undirected, $T^{(q)}_{uv} = 1$ and we recover the real symmetric Laplacian

---

## 4  Real Representation for Computation

Since GPU libraries work with real matrices, we use the standard complex-to-real embedding:

For a complex matrix block $Z = X + iY$, the real representation is:
$$\mathrm{Re}(Z) = \begin{pmatrix} X & -Y \\ Y & X \end{pmatrix} \in \mathbb{R}^{2d \times 2d}$$

This doubles all dimensions but preserves spectral properties:
- Eigenvalues of $Z$ appear as conjugate pairs in $\mathrm{Re}(Z)$
- Hermitian matrices map to symmetric matrices
- Positive definiteness is preserved

### Example: For $q=1/4$ and directed edge $(u,v)$

Since $T^{(q)}_{uv} = i$, the off-diagonal block becomes:
$$\mathcal{L}^{\tilde{\mathcal{F}}}_{uv} = -i s_e Q_e^{\top} \mapsto \begin{pmatrix} 0 & s_e Q_e^{\top} \\ -s_e Q_e^{\top} & 0 \end{pmatrix}$$

---

## 5  Edge-Weight Filtration

Edge weights remain $w_e = s_e$ (scale factors from Procrustes fit). For threshold $\tau \geq 0$:
$$E(\tau) = \{e \in E : w_e > \tau\}$$

We maintain a static Laplacian and apply boolean masks:
$$\mathcal{L}^{\tilde{\mathcal{F}}}(\tau) = M(\tau) \odot \mathcal{L}^{\tilde{\mathcal{F}}}$$

**Critical**: The masking must be applied to the real representation, zeroing out both the real and imaginary parts (all four blocks in the $2 \times 2$ real representation) for filtered edges.

---

## 6  Spectral Persistence with Complex Structure

### 6.1 Spectral Decomposition

For the Hermitian Laplacian $\mathcal{L}^{\tilde{\mathcal{F}}}(\tau)$:
- All eigenvalues $\lambda_i(\tau)$ are **real** and non-negative
- Eigenvectors are complex but come in conjugate pairs in the real representation
- The harmonic space $\mathcal{H}^0(\tau) = \ker \mathcal{L}^{\tilde{\mathcal{F}}}(\tau)$ has real dimension

### 6.2 Persistence Tracking

The persistence pipeline remains largely unchanged:

1. **Compute spectra** at thresholds $\tau_0 < \tau_1 < \cdots < \tau_T$
2. **Cluster** near-degenerate eigenvalues (using real eigenvalues)
3. **Match subspaces** using principal angles between complex eigenspaces
4. **Track births/deaths** to create persistence diagrams

**Key difference**: When computing principal angles between complex subspaces, use the standard Hermitian inner product.

---

## 7  Implementation Considerations

### 7.1 Memory and Computation

- **Stalk dimension**: Each complex stalk of dimension $r_v$ requires $2r_v$ real dimensions
- **Laplacian size**: Total real dimension is $2\sum_v r_v$
- **Eigensolvers**: Use real symmetric eigensolvers on the real representation

### 7.2 Numerical Stability

- **Condition number**: Monitor $\kappa(\mathcal{L}^{\tilde{\mathcal{F}}})$ in the real representation
- **Phase consistency**: Ensure consistent phase conventions across the pipeline
- **Tolerance adjustments**: May need tighter tolerances due to complex arithmetic

---

## 8  Mathematical Validation Checklist

1. **Hermitian structure**: Verify $\|\mathcal{L}^{\tilde{\mathcal{F}}} - (\mathcal{L}^{\tilde{\mathcal{F}}})^*\|_F < 10^{-12}$
2. **Real spectrum**: Confirm all eigenvalues have imaginary part $< 10^{-12}$
3. **Reduction to undirected**: Check that setting $q=0$ recovers the original pipeline
4. **Persistence stability**: Wasserstein distance between diagrams under resampling

---

## 9  Summary of Key Changes from v3

| Component | v3 (Undirected) | v4 (Directed) |
|-----------|-----------------|---------------|
| Stalks | Real $\mathbb{R}^{r_v}$ | Complex $\mathbb{C}^{r_v}$ |
| Restriction maps | Real orthogonal | Complex with phase |
| Laplacian | Symmetric PSD | Hermitian PSD |
| Computation | Direct | Real embedding (2× size) |
| Eigenvalues | Real | Real (from Hermitian) |
| Persistence | Standard | Standard (real eigenvalues) |

---

## 10  Theoretical Guarantees

**Theorem**: The directed sheaf Laplacian $\mathcal{L}^{\tilde{\mathcal{F}}}$ satisfies:
1. $\mathcal{L}^{\tilde{\mathcal{F}}} \succeq 0$ (positive semi-definite)
2. $\dim \ker \mathcal{L}^{\tilde{\mathcal{F}}}$ equals the number of weakly connected components
3. The spectrum is real and contained in $[0, 2\lambda_{\max}]$ where $\lambda_{\max}$ depends on the restriction maps

**Corollary**: Persistence diagrams are well-defined and stable under perturbations.

---

## References

* S. Fiorini et al. – *Sheaves Reloaded: A Directional Awakening* (2025)
* J. Hansen & R. Ghrist – *Toward a Spectral Theory of Cellular Sheaves* (2019)
* Original pipeline documentation v3

---

© 2025 Persistent Directed Sheaf Lab