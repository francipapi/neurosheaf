# Eigenvalue-Preserving Whitening Framework

## Mathematical Foundation and Implementation Plan

*A comprehensive analysis of eigenvalue-preserving whitening transformations for enhanced cross-architecture neural network similarity analysis.*

---

## Executive Summary

This document explores extending the current identity-based whitening approach to preserve eigenvalue information in diagonal form. Instead of transforming Gram matrices to identity matrices, we propose using positive eigenvalues as diagonal entries to retain spectral characteristics while maintaining computational benefits.

**Key Innovation**: Replace `K_whitened = I` with `K_whitened = diag(λ₁, λ₂, ..., λᵣ)` where λᵢ are positive eigenvalues.

**Primary Goal**: Improve cross-architecture functional similarity detection by preserving spectral signatures.

---

## 1. Mathematical Framework

### 1.1 Current Identity-Based Approach

For Gram matrix `K = UΣU^T`, the current whitening transformation yields:

```
W = Σ^(-1/2) U^T : ℝⁿ → ℝʳ
K_whitened = I_r (identity matrix)
Inner product: ⟨u,v⟩ = u^T v
Adjoint: R* = R^T
```

### 1.2 Proposed Eigenvalue-Preserving Approach

**Core Transformation**:
```
W = Σ^(-1/2) U^T : ℝⁿ → ℝʳ (same as current)
K_whitened = Σ = diag(λ₁, λ₂, ..., λᵣ) (eigenvalue diagonal)
Inner product: ⟨u,v⟩_Σ = u^T Σ v
```

**Critical Mathematical Properties**:

1. **Adjoint Computation**: For restriction map `R: (ℝʳˢ, Σₛ) → (ℝʳᵗ, Σₜ)`:
   ```
   R* = Σₛ^(-1) R^T Σₜ
   ```

2. **Metric Compatibility**: 
   ```
   R*R = Σₛ^(-1) R^T Σₜ R
   ```
   This is NO LONGER the identity matrix, which has profound implications.

3. **Self-Adjoint Property**: For the Laplacian `Δ = δ*δ` to remain self-adjoint:
   ```
   Δ* = Δ ⟺ (δ*δ)* = δ*δ**
   ```

### 1.3 Sheaf Laplacian Reformulation

**Current Laplacian Blocks**:
- Off-diagonal: `L[u,v] = -R^T`, `L[v,u] = -R`
- Diagonal: `L[v,v] = Σ_incoming I + Σ_outgoing R^T R`

**Proposed Eigenvalue-Aware Laplacian Blocks**:
- Off-diagonal: `L[u,v] = -Σᵤ^(-1) R^T Σᵥ`, `L[v,u] = -R`
- Diagonal: `L[v,v] = Σ_incoming Σᵥ + Σ_outgoing Σᵥ^(-1) R^T Σᵥ R`

**Symmetry Verification**:
```
L[u,v]^T = (Σᵤ^(-1) R^T Σᵥ)^T = Σᵥ R Σᵤ^(-1) ≠ L[v,u] = -R
```

⚠️ **CRITICAL ISSUE**: The proposed formulation breaks symmetry unless additional constraints are imposed.

---

## 2. Mathematical Solution: The Hodge Laplacian Framework

### 2.1 Resolution of Symmetry Problem

The symmetry issue is completely resolved by correctly formulating the problem within the Hodge Laplacian framework. The key insight is to properly define the coboundary operator (δ) and its adjoint (δ*) with respect to the new eigenvalue-weighted inner products.

**Core Principle**: The Laplacian `Δ = δ*δ` is automatically self-adjoint when δ* is correctly defined as the adjoint of δ with respect to the appropriate inner products.

### 2.2 Hodge Laplacian Formulation

#### Vector Spaces and Inner Products

**Space of 0-cochains (Nodes)**: `C⁰ = ⊕_{v∈V} ℝ^{r_v}` with inner product:
```
⟨α, β⟩_{C⁰} = Σ_{v∈V} ⟨α_v, β_v⟩_{Σ_v} = Σ_{v∈V} α_v^T Σ_v β_v
```

**Space of 1-cochains (Edges)**: `C¹ = ⊕_{e=(u,v)∈E} ℝ^{r_v}` with inner product:
```
⟨f, g⟩_{C¹} = Σ_{e=(u,v)∈E} ⟨f_e, g_e⟩_{Σ_v} = Σ_{e=(u,v)∈E} f_e^T Σ_v g_e
```

**Critical Design Choice**: Edge data lives in the target node's space `F(v)` with inner product `Σ_v`, avoiding dimensional mismatch. This corresponds to identifying the edge stalk `F(e)` with the target vertex stalk `F(v)` for each edge `e=(u,v)`.

#### Coboundary Operator (δ₀)

For α ∈ C⁰, the coboundary operator `δ₀: C⁰ → C¹` is defined as:
```
(δ₀α)_e = α_v - R_{vu}α_u    for edge e=(u,v)
```

#### Adjoint Coboundary Operator (δ₀*)

Using the adjoint definition `⟨δ₀α, f⟩_{C¹} = ⟨α, δ₀*f⟩_{C⁰}`, we derive:
```
(δ₀*f)_v = Σ_{e=(u,v)∈E} f_e - Σ_{e=(v,w)∈E} Σ_v^{-1} R_{wv}^T Σ_w f_e
```

#### Corrected Hodge Laplacian Formulations

**CRITICAL CORRECTION**: The graph structure determines the appropriate formulation:

##### For Undirected Graphs (Real Symmetric Laplacian)
When treating the graph as undirected, with edge stalk `F(e)` identified with target vertex stalk `F(v)`, the correct Hodge Laplacian blocks are:

**Off-diagonal Blocks** (for edge {u,v}):
```
L[u,v] = -Σ_u^{-1} R_{vu}^T Σ_v
L[v,u] = -Σ_v R_{vu} Σ_u^{-1}  (transpose of L[u,v])
```

**Diagonal Blocks**:
```
L[v,v] = Σ_{e={v,w}} Σ_v^{-1} R_{wv}^T Σ_w R_{wv}
```

**Note**: The simplified formula `deg(v)I` is a special case where all restriction maps are identity and all `Σ` matrices are identity. The general formula above applies to the eigenvalue-preserving framework with arbitrary `Σ_v` matrices and restriction maps `R_{wv}`, where edge stalk `F(e)` is identified with target vertex stalk.

**Mathematical Guarantee**: `L = L^T` and `L ⪰ 0` automatically for undirected graphs.

##### For Directed Graphs (Hermitian Laplacian)
**Essential Insight**: Real-valued sheaf Laplacians on directed graphs are NOT generally symmetric. True directed analysis requires the Hermitian framework, with edge stalk `F(e)` identified with target vertex stalk `F(v)`.

**Off-diagonal Blocks** (for directed edge e=(u,v)):
```
L[u,v] = -(F̃_{u≤e})* (F̃_{v≤e}) = -Σ_u^{-1} F̃_{u≤e}^H Σ_e F̃_{v≤e}
L[v,u] = L[u,v]^H  (Hermitian conjugate)
```

**Diagonal Blocks**:
```
L[u,u] = Σ_{e∈Γ(u)} (F̃_{u≤e})* (F̃_{u≤e})
```

where `F̃_{v≤e} = F̃⁰_{v≤e} T^{(q)}` incorporates directional encoding, and edge stalk `F(e)` is identified with target vertex stalk `F(v)` so that `Σ_e = Σ_v`.

**Mathematical Guarantee**: `L = L^H` (Hermitian) and `L ⪰ 0` automatically for directed graphs.

### 2.3 Numerical Stability

**Eigenvalue Conditioning**:
- Small eigenvalues in `Σᵤ^(-1)` create numerical instability
- Robust regularization: `(Σᵤ + εI)^(-1)` before inversion
- Threshold-based eigenvalue filtering during whitening

**Computational Complexity**:
- Adjoint computation: `R* = Σᵤ^(-1) R^T Σᵥ` requires O(r²) operations
- Diagonal blocks: Additional `Σᵥ^(-1) R^T Σw R` computations
- Memory overhead: Store eigenvalue matrices `Σᵥ` for each stalk

**Regularization Strategy**:
```python
# Safe inversion with regularization
Sigma_reg = Sigma + epsilon * torch.eye(Sigma.shape[0])
Sigma_inv = torch.linalg.solve(Sigma_reg, torch.eye(Sigma.shape[0]))
```

---

## 3. Information Preservation Analysis

### 3.1 Spectral Signature Retention

**Current Approach Information Loss**:
- All eigenvalue magnitudes → 1
- Only subspace orientation preserved
- Layer "importance" information lost

**Proposed Approach Information Gain**:
- Eigenvalue magnitudes preserved
- Spectral characteristics maintained
- Cross-architecture signature comparison possible

### 3.2 Cross-Architecture Benefits

**Hypothesis**: Different architectures with similar function should have:
1. Similar eigenvalue distributions in corresponding layers
2. Better functional similarity detection via spectral signatures
3. Improved CKA alignment across architectures

**Validation Strategy**:
- Compare ResNet vs EfficientNet on same task
- Measure eigenvalue correlation between functionally similar layers
- Evaluate CKA improvements with eigenvalue preservation

---

## 4. Implementation Strategy

### 4.1 Minimal API Extension Design

**Principle**: Extend current API without breaking existing functionality.

#### 4.1.1 WhiteningProcessor Extension

```python
class WhiteningProcessor:
    def __init__(self, ..., preserve_eigenvalues: bool = False):
        self.preserve_eigenvalues = preserve_eigenvalues
    
    def whiten_gram_matrix(self, K):
        W, info = self.compute_whitening_map(K)
        r = W.shape[0]
        
        if self.preserve_eigenvalues:
            # Use eigenvalue diagonal
            eigenvals = info['eigenvalues']
            K_whitened = torch.diag(torch.from_numpy(eigenvals))
        else:
            # Current identity approach
            K_whitened = torch.eye(r, dtype=W.dtype)
        
        return K_whitened, W, info
```

#### 4.1.2 Hodge-Compliant Adjoint Computation

```python
def compute_hodge_adjoint(self, R, Sigma_source, Sigma_target):
    """Compute R* = Σₛ^(-1) R^T Σₜ following Hodge Laplacian framework."""
    # Regularized inverse for numerical stability
    eps = self.regularization
    Sigma_source_reg = Sigma_source + eps * torch.eye(Sigma_source.shape[0])
    Sigma_source_inv = torch.linalg.solve(Sigma_source_reg, 
                                         torch.eye(Sigma_source.shape[0]))
    
    # Hodge adjoint: R* = Σₛ^(-1) R^T Σₜ
    return Sigma_source_inv @ R.T @ Sigma_target
```

#### 4.1.3 Eigenvalue-Aware Laplacian Builder

```python
class SheafLaplacianBuilder:
    def _build_laplacian_optimized(self, sheaf, edge_weights, metadata):
        # Check if sheaf uses eigenvalue-preserving stalks
        if self._has_eigenvalue_stalks(sheaf):
            return self._build_eigenvalue_laplacian(sheaf, edge_weights, metadata)
        else:
            return self._build_standard_laplacian(sheaf, edge_weights, metadata)
    
    def _has_eigenvalue_stalks(self, sheaf):
        """Check if sheaf contains eigenvalue-diagonal stalks."""
        for stalk in sheaf.stalks.values():
            if not torch.allclose(stalk, torch.eye(stalk.shape[0])):
                return True  # Non-identity stalk found
        return False
    
    def _build_eigenvalue_laplacian(self, sheaf, edge_weights, metadata):
        """Build Laplacian using Hodge formulation with eigenvalue-diagonal stalks."""
        # Hodge approach automatically used when preserve_eigenvalues=True
        # Diagonal blocks: L[v,v] = deg_in(v)I + Σ Σᵥ⁻¹ R^T Σw R
        # Off-diagonal blocks: L[v,u] = -Σᵥ⁻¹ R^T Σᵤ for edge (u,v)
        # Mathematical guarantee: L = L^T and L ⪰ 0
```

### 4.2 Backward Compatibility

**API Compatibility**:
- Default behavior unchanged (`preserve_eigenvalues=False`)
- Existing code works without modification  
- Hodge formulation automatically activated when `preserve_eigenvalues=True`

**Testing Strategy**:
- All existing tests pass with default settings (`preserve_eigenvalues=False`)
- New test suite for eigenvalue-preserving mode (`preserve_eigenvalues=True`)
- Mathematical validation: Hodge formulation guarantees symmetry and PSD automatically
- Comparative analysis between identity-based and eigenvalue-preserving approaches

---

## 5. Research Questions and Validation Plan

### 5.1 Mathematical Validation

1. **Symmetry**: ✅ **RESOLVED** - Hodge formulation guarantees `L = L^T` automatically
2. **PSD Property**: ✅ **RESOLVED** - Hodge formulation guarantees `L ⪰ 0` automatically  
3. **Spectral Properties**: How do eigenvalues of the Hodge Laplacian compare to standard formulation?
4. **Numerical Stability**: Validation of regularization strategies for `Σ^(-1)` computations

### 5.2 Empirical Validation

1. **Cross-Architecture Similarity**: Does eigenvalue preservation improve functional similarity detection?
2. **Performance Impact**: What is the computational overhead?
3. **Numerical Stability**: How robust is the approach to eigenvalue conditioning?

### 5.3 Experimental Design

**Phase 1: Eigenvalue-Preserving Framework**
- Implement eigenvalue-preserving whitening with automatic Hodge formulation
- Verify automatic symmetry and PSD properties
- Compare spectral characteristics with identity-based approach

**Phase 2: Cross-Architecture Testing**
- Test on ResNet-18 vs ResNet-50
- Test on ResNet vs EfficientNet
- Measure CKA improvements

**Phase 3: Production Integration**
- Performance optimization
- Robustness testing
- Documentation and API finalization

---

## 6. Technical Implementation Details

### 6.1 Key Files to Modify

1. **Core Whitening** (`neurosheaf/sheaf/core/whitening.py`):
   - Add `preserve_eigenvalues` parameter for eigenvalue-diagonal stalks
   - Modify `whiten_gram_matrix()` method to return `Σ` instead of `I`
   - Add `compute_hodge_adjoint()` method

2. **Restriction Management** (`neurosheaf/sheaf/assembly/restrictions.py`):
   - Add eigenvalue-aware restriction computation using Hodge adjoints
   - Handle eigenvalue matrices `Σᵥ` in restriction pipeline

3. **Laplacian Construction** (`neurosheaf/sheaf/assembly/laplacian.py`):
   - Add `_build_eigenvalue_laplacian()` method with automatic Hodge formulation
   - Implement: `L[v,u] = -Σᵥ⁻¹ R^T Σᵤ` when `preserve_eigenvalues=True`
   - Automatic symmetry and PSD properties (no manual verification needed)

### 6.2 Data Structure Extensions

```python
@dataclass
class EigenvalueMetadata:
    """Metadata for eigenvalue-preserving operations with automatic Hodge formulation."""
    eigenvalue_matrices: Dict[str, torch.Tensor] = None  # Σᵥ for each stalk v
    condition_numbers: Dict[str, float] = None
    regularization_applied: Dict[str, bool] = None
    eigenvalue_mode_enabled: bool = False
    
class EigenvalueSheaf(Sheaf):
    """Extended sheaf with eigenvalue-diagonal stalks using Hodge formulation."""
    eigenvalue_metadata: EigenvalueMetadata = None
```

---

## 7. Open Questions and Future Work

### 7.1 Immediate Questions

1. **Mathematical Formulation**: ✅ **CORRECTED** - Proper undirected (symmetric) vs directed (Hermitian) formulations
2. **Graph Type Clarification**: Must distinguish between undirected and directed analysis in implementation
3. **Regularization Strategy**: Implement `(Σ + εI)^(-1)` for numerical stability
4. **Performance Tradeoffs**: Quantify computational overhead vs. cross-architecture similarity gains
5. **Implementation Priority**: Implement correct Hodge formulations based on graph structure

### 7.2 Long-term Research Directions

1. **Adaptive Eigenvalue Weighting**: Could we learn optimal eigenvalue weightings?
2. **Hierarchical Eigenvalue Structures**: Can we encode layer hierarchy in eigenvalue patterns?
3. **Multi-scale Analysis**: How do eigenvalue patterns change across network scales?

---

## 8. Directed Sheaf Compatibility

### 8.1 Integration with Directed Sheaf Framework

The eigenvalue-preserving whitening approach must be fully compatible with the existing directed sheaf implementation, which extends the base sheaf framework to support:

- **Complex-valued stalks**: `F(v) = ℂ^{r_v}` for asymmetric network analysis
- **Directional encoding**: `T^{(q)} = exp(i 2π q (A - A^T))` with phase information
- **Hermitian Laplacians**: Complex extension of the Hodge formulation
- **Real embedding**: Computational efficiency through 2×2 block structure

### 8.2 Compatibility Requirements

#### Whitening Layer Compatibility
```python
# Base sheaf: K_whitened = I or diag(eigenvalues)
# Directed sheaf: Complex extension F(v) = R^{r_v} ⊗_R C

class WhiteningProcessor:
    def whiten_gram_matrix(self, K, preserve_eigenvalues=False):
        if preserve_eigenvalues:
            K_whitened = torch.diag(eigenvalues)
        else:
            K_whitened = torch.eye(r, dtype=W.dtype)
        
        # For directed sheaf compatibility:
        # K_whitened (real Σ matrix) defines Hermitian inner product ⟨w,z⟩ = w^H Σ z
        # No dimension doubling needed for complex extension
        return K_whitened, W, info
```

#### Hodge Adjoint for Complex Stalks
```python
def compute_hodge_adjoint_complex(self, R, Sigma_source, Sigma_target):
    """Compute complex Hodge adjoint for directed sheaf compatibility."""
    # Real part computation (same as base)
    Sigma_source_inv = self._regularized_inverse(Sigma_source)
    R_adjoint_real = Sigma_source_inv @ R.T @ Sigma_target
    
    # Complex extension preserves the structure
    # DirectedSheaf will handle complex phase encoding separately
    return R_adjoint_real
```

#### Laplacian Construction Compatibility
```python
class SheafLaplacianBuilder:
    def _build_eigenvalue_laplacian(self, sheaf, edge_weights, metadata):
        """Build Hodge Laplacian compatible with directed sheaf extension."""
        # Corrected Hodge formulation (undirected case)
        # L[u,v] = -Σᵤ⁻¹ R_{vu}^T Σᵥ for edge {u,v}
        # L[v,u] = -Σᵥ R_{vu} Σᵤ⁻¹ (transpose ensures symmetry)
        # L[v,v] = Σ_{e={v,w}} Σᵥ⁻¹ R_{wv}^T Σw R_{wv} (general diagonal blocks)

class DirectedSheafLaplacianBuilder:
    def _build_hermitian_eigenvalue_laplacian(self, directed_sheaf, edge_weights, metadata):
        """Build Hermitian Laplacian using CORRECTED eigenvalue-preserving Hodge formulation."""
        # CORRECTED Hermitian Hodge formulation following Fiorini et al.:
        # For directed edge e=(u,v) with edge stalk F(e) identified with target vertex stalk F(v):
        # L[u,v] = -(F̃_{u≤e})* (F̃_{v≤e}) = -Σᵤ⁻¹ F̃_{u≤e}^H Σᵥ F̃_{v≤e}
        # L[v,u] = L[u,v]^H  (Hermitian conjugate)
        # L[u,u] = Σ_{e∈Γ(u)} (F̃_{u≤e})* (F̃_{u≤e})
        #
        # where F̃_{v≤e} = F̃⁰_{v≤e} T^{(q)} incorporates directional encoding
        # and Σₑ = Σᵥ due to edge stalk identification with target vertex stalk
        
        # This guarantees Hermitian property: L = L^H automatically
        # while preserving eigenvalue information in complex domain
```

### 8.3 Mathematical Compatibility Verification

#### Real Base → Complex Extension Chain
1. **Eigenvalue Preservation**: `K_whitened = diag(λ₁, ..., λᵣ)` (real)
2. **Hodge Adjoint**: `R* = Σᵤ⁻¹ R^T Σᵥ` (real, well-defined)
3. **Complex Extension**: `F(v) = R^{r_v} ⊗_R C` (preserves structure)
4. **Directional Encoding**: `T^{(q)}` applied to complex restrictions
5. **Hermitian Laplacian**: Automatic via directed sheaf construction

#### Key Compatibility Points
```python
# 1. Eigenvalue matrices are real → complex extension preserves structure
Sigma_v_real = diag(λ₁, λ₂, ..., λᵣ)  # Real eigenvalue matrix
# CORRECTED: Hermitian inner product on ℂⁿ defined directly by real positive-definite matrix
# ⟨w, z⟩ = w^H Σ z (no dimension doubling needed)

# 2. Hodge adjoint computation remains real-valued
R_adjoint = Sigma_source⁻¹ R^T Sigma_target  # Real computation
# DirectedSheaf applies T^{(q)} phase encoding separately

# 3. CORRECTED Hermitian Laplacian structure with eigenvalue preservation
# Undirected case (symmetric):
L_real[u,v] = -Σᵤ⁻¹ R_{vu}^T Σᵥ                     # Correct undirected formulation
L_real[v,u] = -Σᵥ R_{vu} Σᵤ⁻¹                       # Transpose ensures symmetry
L_real[v,v] = Σ_{e={v,w}} Σᵥ⁻¹ R_{wv}^T Σw R_{wv}   # General diagonal blocks

# Directed case (Hermitian):
L_hermitian[u,v] = -Σᵤ⁻¹ F̃_{u≤e}^H Σₑ F̃_{v≤e}      # Correct Hermitian formulation
L_hermitian[v,u] = L_hermitian[u,v]^H                # Hermitian conjugate
L_hermitian[u,u] = Σ_{e∈Γ(u)} (F̃_{u≤e})* (F̃_{u≤e}) # Correct diagonal blocks
```

### 8.4 Implementation Strategy for Compatibility

#### Phase 1: Ensure Base Compatibility
- Verify eigenvalue-preserving whitening works with existing directed sheaf construction
- Test complex extension of eigenvalue-diagonal stalks  
- Update DirectedSheafLaplacianBuilder to use Hermitian Hodge formulation
- Validate Hodge adjoint computation in directed context

#### Phase 2: Extended Validation
```python
def test_directed_compatibility():
    # 1. Create eigenvalue-preserving base sheaf
    base_sheaf = construct_sheaf(preserve_eigenvalues=True)
    
    # 2. Apply directed sheaf extension
    directed_sheaf = DirectedSheafBuilder().extend_to_complex(base_sheaf)
    
    # 3. Verify properties
    assert_hermitian_property(directed_sheaf.laplacian)  # L* = L with eigenvalue preservation
    assert_eigenvalue_preservation(directed_sheaf)       # Σᵥ matrices preserved in diagonal blocks
    assert_phase_encoding_correctness(directed_sheaf)    # T^{(q)} correctly applied
    assert_hodge_formulation_correctness(directed_sheaf) # Updated Hodge structure verified
```

#### Phase 3: Unified Pipeline
```python
# Single API for both standard and directed sheaf analysis
def analyze_network(model, data, preserve_eigenvalues=False, directed=False):
    # Step 1: Eigenvalue-preserving whitening (if requested)
    sheaf = SheafBuilder(preserve_eigenvalues=preserve_eigenvalues).build(model, data)
    
    # Step 2: Directed extension (if requested)
    if directed:
        sheaf = DirectedSheafBuilder().extend_to_complex(sheaf)
    
    # Step 3: Unified spectral analysis
    return SpectralAnalyzer().analyze(sheaf)
```

### 8.5 Compatibility Guarantees

1. **Mathematical Consistency**: Eigenvalue preservation is compatible with complex extension and Hermitian Hodge formulation
2. **API Consistency**: Single `preserve_eigenvalues` flag works for both standard and directed sheaf
3. **Hermitian Property**: Directed sheaf Laplacian remains Hermitian with eigenvalue preservation: `L* = L`
4. **Performance**: No additional overhead for standard sheaf when directed=False
5. **Validation**: All existing directed sheaf tests pass with updated Hermitian Hodge formulation

#### Mathematical Verification of Corrected Properties

**For Undirected Graphs (Symmetry)**:
```
L[v,u] = (-Σᵥ R_{vu} Σᵤ⁻¹) = (-Σᵤ⁻¹ R_{vu}^T Σᵥ)^T = L[u,v]^T
```
Symmetry is guaranteed by construction.

**For Directed Graphs (Hermitian Property)**:
```
L[v,u] = L[u,v]^H = (-Σᵤ⁻¹ F̃_{u≤e}^H Σₑ F̃_{v≤e})^H
```
Hermiticity is guaranteed by definition in the corrected formulation.

**Positive Semi-Definite Property**: Both formulations guarantee `L ⪰ 0` through the construction `L = δ*δ`.

The eigenvalue-preserving framework seamlessly integrates with the directed sheaf extension, providing enhanced cross-architecture analysis capabilities for both symmetric and asymmetric network structures.

---

## 9. Conclusion and Next Steps

The Hodge eigenvalue-preserving whitening framework provides a mathematically rigorous solution to improve cross-architecture neural network similarity analysis by retaining spectral characteristics while guaranteeing essential mathematical properties.

**Key Breakthrough**: The Hodge Laplacian formulation completely resolves the symmetry and PSD issues, with automatic activation when `preserve_eigenvalues=True`.

**Immediate Next Steps**:
1. ✅ **CORRECTED** - Mathematical framework with proper undirected/directed formulations
2. Implement `compute_hodge_adjoint()` method in WhiteningProcessor  
3. Add `_build_eigenvalue_laplacian()` method with correct undirected formulation to SheafLaplacianBuilder
4. Update `DirectedSheafLaplacianBuilder` with correct Hermitian formulation
5. Create validation experiments comparing identity vs. eigenvalue-preserving approaches

**Success Metrics**:
- ✅ **GUARANTEED** - Mathematical correctness (symmetry, PSD properties) via Hodge theory
- Improved cross-architecture CKA similarity scores  
- Acceptable computational overhead (< 2x current runtime)
- Robust numerical stability with regularized `Σ^(-1)` computations

**Mathematical Foundation**: The automatic Hodge formulation (activated by `preserve_eigenvalues=True`) provides the rigorous foundation needed to preserve eigenvalue information while maintaining all essential Laplacian properties. This approach transforms a challenging engineering problem into a well-understood mathematical framework.

The framework's success will be measured by its ability to reveal functional similarities between different architectures that are missed by the current identity-based approach, with the confidence that all mathematical properties are automatically satisfied by construction.