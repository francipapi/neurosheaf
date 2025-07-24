# Gromov-Wasserstein Sheaf Construction Implementation Plan

## Executive Summary

This document outlines the implementation of a new sheaf construction method based on Gromov-Wasserstein (GW) optimal transport. The approach constructs sheaves directly from neural network activations using metric-measure spaces and GW couplings, providing a theoretically grounded alternative to the existing Procrustes-based approach.

### Key Benefits
- **Architecture-agnostic**: Works across networks with different layer dimensions
- **Metric preservation**: Preserves intrinsic geometric structure via GW distances
- **Theoretical foundation**: Based on optimal transport theory with provable properties
- **Seamless integration**: Extends existing infrastructure without disruption

## Mathematical Foundation

### 1. Core Concepts

#### Metric-Measure Stalks
For each layer $i$ with activations $X_i = \{x_i^1, \ldots, x_i^{n_i}\} \subset \mathbb{R}^{d_i}$:
- **Cost matrix**: $C_i[k,\ell] = 1 - \frac{\langle x_i^k, x_i^\ell \rangle}{\|x_i^k\| \|x_i^\ell\|}$ (cosine distance)
- **Measure**: $p_i = (1/n_i, \ldots, 1/n_i)$ (uniform distribution)
- **Stalk**: $\mathcal{F}(i) = (X_i, C_i, p_i)$

#### Gromov-Wasserstein Coupling
For edge $e = (i \to j)$, compute entropic GW coupling:
```
π_{j→i} = argmin_{π∈Π(p_j,p_i)} ∑_{k,ℓ,k',ℓ'} |C_j[k,k'] - C_i[ℓ,ℓ']|² π[k,ℓ]π[k',ℓ'] - ε H(π)
```
where $\Pi(p_j, p_i)$ enforces marginal constraints: $\pi \mathbf{1} = p_j$ and $\pi^T \mathbf{1} = p_i$.

#### Restriction Maps
The backward restriction map: $\rho_{j \to i} = \pi_{j \to i}^T : \mathbb{R}^{n_j} \to \mathbb{R}^{n_i}$

**Important**: When $p_i, p_j$ are uniform, $\pi^T$ is column-stochastic. For non-uniform measures, $\rho_{j \to i}$ preserves $p_i$-weighted averages: $\rho_{j \to i} \mathbf{1}_{n_j} = \mathbf{1}_{n_i}$ only when measures are uniform.

### 2. Sheaf Laplacian Construction

The sheaf Hodge Laplacian $L = \delta^T \delta$ has block structure:

**Diagonal blocks** ($L_{ii}$):
```
L_{ii} = (∑_{e∈in(i)} I_{n_i}) + ∑_{e=(i→j)∈out(i)} ρ_{j→i}^T ρ_{j→i}
```

**Off-diagonal blocks** ($L_{ij}$ for $i \neq j$):
```
L_{ij} = {
  -ρ_{j→i}^T  if (i→j) ∈ E
  -ρ_{i→j}    if (j→i) ∈ E
  0           otherwise
}
```

## Implementation Architecture

### Phase 1: Core GW Components (Week 1)

#### 1.1 GW Computer Module
**Location**: `neurosheaf/sheaf/core/gromov_wasserstein.py`

```python
class GromovWassersteinComputer:
    """Core GW computation engine with caching and GPU support."""
    
    def __init__(self, config: GWConfig):
        self.epsilon = config.epsilon
        self.max_iter = config.max_iter
        self.tolerance = config.tolerance
        self.backend = self._setup_backend(config.use_gpu)
        
    def compute_cosine_cost_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine distances with numerical stability."""
        
    def compute_gw_coupling(self, C_source: torch.Tensor, C_target: torch.Tensor,
                           p_source: Optional[torch.Tensor] = None,
                           p_target: Optional[torch.Tensor] = None) -> GWResult:
        """
        Solve entropic GW problem with convergence diagnostics.
        
        Returns:
        --------
        GWResult containing:
            - coupling: π_{j→i} transport plan
            - cost: scalar GW distortion value
            - log: convergence information
        """
```

#### 1.2 Configuration
**Location**: `neurosheaf/sheaf/config.py`

```python
@dataclass
class GWConfig:
    """Configuration for GW-based sheaf construction."""
    epsilon: float = 0.1              # Entropic regularization
    max_iter: int = 1000             # Maximum iterations
    tolerance: float = 1e-9          # Convergence tolerance
    quasi_sheaf_tolerance: float = 0.1  # ε-sheaf validation threshold
    use_gpu: bool = True             # GPU acceleration
    cache_cost_matrices: bool = True  # Memory-time tradeoff
    validate_couplings: bool = True   # Runtime validation
    uniform_measures: bool = True     # Use uniform p_i (vs importance sampling)
    weighted_inner_product: bool = False  # Use p_i-weighted L2 inner products
```

### Phase 2: Sheaf Assembly Integration (Week 1-2)

#### 2.1 GW Restriction Manager
**Location**: `neurosheaf/sheaf/assembly/gw_builder.py`

```python
class GWRestrictionManager:
    """Manages GW-based restriction map computation with validation."""
    
    def __init__(self, gw_computer: GromovWassersteinComputer,
                 config: GWConfig):
        self.gw_computer = gw_computer
        self.config = config
        self._cost_cache = {}  # Cache expensive cost matrices
        self._gw_results = {}  # Store both couplings and costs
        
    def compute_all_restrictions(self, 
                               activations: Dict[str, torch.Tensor],
                               poset: nx.DiGraph) -> Tuple[RestrictionMaps, GWCosts]:
        """
        Compute all restriction maps with parallel edge processing.
        
        Returns:
        --------
        restrictions : Dict[Tuple[str, str], torch.Tensor]
            Restriction maps ρ_{j→i} = π_{j→i}^T
        gw_costs : Dict[Tuple[str, str], float]
            GW distortion costs for edge weights
        """
        
    def validate_quasi_sheaf_property(self, 
                                    restrictions: RestrictionMaps,
                                    poset: nx.DiGraph) -> ValidationReport:
        """Validate ε-sheaf approximation quality."""
```

#### 2.2 Updated SheafBuilder
**Location**: Modify `neurosheaf/sheaf/assembly/builder.py`

```python
class SheafBuilder:
    """Extended to support multiple restriction methods."""
    
    SUPPORTED_METHODS = ['scaled_procrustes', 'gromov_wasserstein', 'whitened_procrustes']
    
    def __init__(self, ..., restriction_method: str = 'scaled_procrustes'):
        if restriction_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {restriction_method}")
        self.restriction_method = restriction_method
        
    def build_from_activations(self, ...) -> Sheaf:
        """Route to appropriate construction method."""
        if self.restriction_method == 'gromov_wasserstein':
            return self._build_gw_sheaf(activations, poset)
        else:
            return self._build_procrustes_sheaf(...)  # Existing logic
```

### Phase 3: Laplacian Assembly (Week 2)

#### 3.1 GW-Specific Laplacian Builder
**Location**: `neurosheaf/sheaf/assembly/gw_laplacian.py`

```python
class GWLaplacianBuilder:
    """Efficient block-structured Laplacian assembly."""
    
    def build_laplacian(self, sheaf: Sheaf, 
                       sparse: bool = True) -> Union[torch.Tensor, torch.sparse.Tensor]:
        """Construct L = δ^T δ using block formula with proper inner products."""
        
    def build_coboundary(self, sheaf: Sheaf) -> torch.sparse.Tensor:
        """
        Construct sparse coboundary operator δ.
        
        For weighted measures, incorporates p_i-weighted inner product:
        δ* = P^{-1/2} δ^T P^{1/2} where P = diag(p_weights)
        """
        
    def extract_edge_weights(self, sheaf: Sheaf) -> Dict[Tuple[str, str], float]:
        """
        Extract GW costs as edge weights for persistence.
        
        Important: GW costs represent metric distortion (lower = better match)
        Unlike Procrustes norms (higher = stronger connection)
        """
```

#### 3.2 Comprehensive Spectral Module Integration

The spectral module requires careful integration at multiple levels to properly support GW-based sheaves.

##### 3.2.1 Updated Laplacian Builder Interface
**Location**: Modify `neurosheaf/sheaf/assembly/laplacian.py`

```python
class SheafLaplacianBuilder:
    """Extended to support multiple construction methods."""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        
    def build_laplacian(self, sheaf: Sheaf, 
                       sparse: bool = True) -> LaplacianResult:
        """Route to appropriate Laplacian construction."""
        if sheaf.metadata.get('construction_method') == 'gromov_wasserstein':
            return self._build_gw_laplacian(sheaf, sparse)
        else:
            return self._build_standard_laplacian(sheaf, sparse)
            
    def _build_gw_laplacian(self, sheaf: Sheaf, sparse: bool) -> LaplacianResult:
        """
        Build Laplacian using GW-specific block formula.
        
        Handles weighted inner products for non-uniform measures:
        - If uniform measures: standard Euclidean inner product
        - If weighted: use p_i-weighted L2 inner product in cochain spaces
        """
```

##### 3.2.2 Static Laplacian Unified Integration
**Location**: Update `neurosheaf/spectral/static_laplacian_unified.py`

```python
class UnifiedStaticLaplacian:
    """Extended to handle GW-based restrictions."""
    
    def _extract_edge_info(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Extract edge information with GW awareness."""
        edges_info = {}
        construction_method = sheaf.metadata.get('construction_method', 'standard')
        
        for edge, restriction in sheaf.restrictions.items():
            if construction_method == 'gromov_wasserstein':
                # GW restrictions have different properties
                edge_weight = self._compute_gw_edge_weight(restriction, sheaf, edge)
                restriction_data = self._extract_gw_restriction(restriction)
            else:
                # Standard restriction processing
                edge_weight = torch.norm(restriction, 'fro').item()
                restriction_data = restriction
                
            edges_info[edge] = {
                'restriction': restriction_data,
                'weight': edge_weight,
                'source_dim': restriction.shape[1],
                'target_dim': restriction.shape[0],
                'construction_method': construction_method  # For threshold logic
            }
            
        return edges_info
        
    def _compute_gw_edge_weight(self, restriction: torch.Tensor, 
                               sheaf: Sheaf, edge: Tuple[str, str]) -> float:
        """Extract GW cost as edge weight from sheaf metadata."""
        # GW costs stored during construction
        gw_costs = sheaf.metadata.get('gw_costs', {})
        return gw_costs.get(edge, torch.norm(restriction, 'fro').item())
        
    def _create_edge_threshold_func(self, construction_method: str) -> Callable:
        """Create appropriate threshold function based on construction method."""
        if construction_method == 'gromov_wasserstein':
            # GW: Include edges with cost ≤ threshold (increasing complexity)
            # Small costs = good matches = added first
            return lambda weight, param: weight <= param
        else:
            # Standard: Include edges with weight ≥ threshold (decreasing complexity)  
            # Large weights = strong connections = kept longest
            return lambda weight, param: weight >= param
```

##### 3.2.3 Persistent Spectral Analyzer Updates
**Location**: Update `neurosheaf/spectral/persistent.py`

```python
class PersistentSpectralAnalyzer:
    """Extended to handle GW-specific persistence."""
    
    def __init__(self, sheaf: Sheaf, **kwargs):
        super().__init__(sheaf, **kwargs)
        self.construction_method = sheaf.metadata.get('construction_method', 'standard')
        
    def _generate_filtration_params(self, edge_weights: Dict[Tuple[str, str], float],
                                  num_steps: int = 50) -> List[float]:
        """Generate filtration parameters appropriate for construction method."""
        if self.construction_method == 'gromov_wasserstein':
            # GW costs have different scale and distribution
            return self._generate_gw_filtration_params(edge_weights, num_steps)
        else:
            return self._generate_standard_filtration_params(edge_weights, num_steps)
            
    def _generate_gw_filtration_params(self, edge_weights: Dict, 
                                     num_steps: int) -> List[float]:
        """
        Generate INCREASING filtration for GW costs (typically in [0, 2] for cosine).
        
        GW Filtration Logic:
        - Start with NO edges (only isolated stalks)
        - Add edges with cost ≤ threshold as threshold increases
        - Small costs = good matches = added first
        - Results in INCREASING complexity (opposite of Procrustes)
        """
        if not edge_weights:
            return [0.0]
        
        # GW costs represent metric distortion (lower = better match)
        min_cost = min(edge_weights.values())
        max_cost = max(edge_weights.values())
        
        # Use log-scale for better resolution of small costs
        # INCREASING filtration: start small, grow to include more edges
        if min_cost > 0:
            return np.logspace(np.log10(min_cost), np.log10(max_cost), num_steps).tolist()
        else:
            return np.linspace(min_cost, max_cost, num_steps).tolist()
```

##### 3.2.4 Edge Weight Propagation
**Location**: Create `neurosheaf/spectral/edge_weights.py`

```python
@dataclass
class EdgeWeightExtractor:
    """Unified edge weight extraction for different sheaf types."""
    
    def extract_weights(self, sheaf: Sheaf) -> Dict[Tuple[str, str], float]:
        """Extract edge weights based on construction method."""
        method = sheaf.metadata.get('construction_method', 'standard')
        
        if method == 'gromov_wasserstein':
            return self._extract_gw_weights(sheaf)
        elif method == 'scaled_procrustes':
            return self._extract_procrustes_weights(sheaf)
        else:
            return self._extract_standard_weights(sheaf)
            
    def _extract_gw_weights(self, sheaf: Sheaf) -> Dict[Tuple[str, str], float]:
        """Extract GW costs as weights."""
        # Primary source: stored GW costs
        gw_costs = sheaf.metadata.get('gw_costs', {})
        
        # Fallback: compute from restriction properties
        weights = {}
        for edge, restriction in sheaf.restrictions.items():
            if edge in gw_costs:
                weights[edge] = gw_costs[edge]
            else:
                # Fallback to operator norm for stability
                weights[edge] = torch.linalg.norm(restriction, ord=2).item()
                
        return weights
```

##### 3.2.5 Subspace Tracker Integration
**Location**: Update `neurosheaf/spectral/tracker.py`

```python
class SubspaceTracker:
    """Extended to handle different restriction representations."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self._method_handlers = {
            'gromov_wasserstein': self._track_gw_subspaces,
            'standard': self._track_standard_subspaces
        }
        
    def track_across_filtration(self, spectral_data: List[Dict],
                              construction_method: str = 'standard') -> Dict:
        """
        Route to appropriate tracking method.
        
        Note: Both GW and standard methods use increasing parameter sequences,
        but with different threshold semantics:
        - Standard: weight >= param (decreasing complexity)
        - GW: weight <= param (increasing complexity)
        
        The tracker handles both correctly using birth <= death validation.
        """
        handler = self._method_handlers.get(construction_method, 
                                           self._track_standard_subspaces)
        return handler(spectral_data)
        
    def _track_gw_subspaces(self, spectral_data: List[Dict]) -> Dict:
        """
        Track subspaces for GW-based filtration.
        
        GW-specific considerations:
        - Parameters increase, complexity increases (opposite of standard)
        - Birth/death semantics: birth < death still valid
        - Early parameters → sparse graphs → many small eigenvalues
        - Later parameters → dense graphs → fewer small eigenvalues
        """
        # Reuse standard tracking logic - birth <= death validation works
        # because we still use increasing parameter sequences
        return self._track_standard_subspaces(spectral_data)
```

#### 3.3 Key Spectral Integration Challenges

##### 3.3.1 Edge Weight Semantics
- **Standard sheaves**: Edge weights are Frobenius norms of restrictions (higher = stronger connection)
- **GW sheaves**: Edge weights are GW costs (higher = greater metric distortion)
- **Impact**: Filtration direction may need to be reversed for persistence

##### 3.3.2 Numerical Properties
- **GW restrictions**: Column-stochastic matrices (different from orthogonal Procrustes)
- **Laplacian structure**: Different block formula requires careful assembly
- **Conditioning**: GW Laplacians may have different numerical properties

##### 3.3.3 Metadata Propagation
The sheaf must carry sufficient metadata for spectral analysis:
```python
sheaf.metadata = {
    'construction_method': 'gromov_wasserstein',
    'gw_costs': {edge: cost for edge, cost in gw_costs.items()},  # Scalar distortion
    'gw_couplings': {edge: coupling for edge, coupling in gw_couplings.items()},  # Full π matrices
    'gw_config': gw_config.to_dict(),
    'quasi_sheaf_tolerance': validation_report.max_violation,
    'edge_weight_type': 'metric_distortion',  # vs 'correlation' for standard
    'measure_type': 'uniform',  # or 'importance_sampled'
    'inner_product_weights': p_weights if non_uniform else None
}
```

### Phase 4: High-Level API (Week 2-3)

#### 4.1 API Integration
**Location**: Update `neurosheaf/api.py`

```python
class NeurosheafAnalyzer:
    """Extended with GW method support."""
    
    def analyze(self, 
                model: torch.nn.Module,
                data: torch.Tensor,
                method: str = 'procrustes',
                gw_config: Optional[GWConfig] = None,
                **kwargs) -> AnalysisResult:
        """
        Analyze neural network with specified sheaf construction method.
        
        Parameters:
        -----------
        method : str
            'procrustes' (default), 'gromov_wasserstein', or 'whitened_procrustes'
        gw_config : GWConfig, optional
            Configuration for GW method (uses defaults if None)
        """
```

### Phase 5: Testing and Validation (Week 3)

#### 5.1 Unit Tests
**Location**: `tests/test_gw_sheaf.py`

```python
class TestGWComponents:
    """Test core GW functionality."""
    
    def test_cosine_cost_matrix_properties(self):
        """Verify cost matrix is symmetric, non-negative, zero diagonal."""
        
    def test_gw_coupling_column_stochastic(self):
        """Verify coupling satisfies measure constraints."""
        
    def test_numerical_stability_edge_cases(self):
        """Test handling of zero vectors, identical inputs, etc."""

class TestGWIntegration:
    """Test integration with existing pipeline."""
    
    def test_api_compatibility(self):
        """Verify GW method works through high-level API."""
        
    def test_persistence_analysis_compatibility(self):
        """Ensure GW sheaves work with existing spectral analysis."""
```

#### 5.2 Validation Suite
**Location**: `tests/validation/test_gw_mathematical_properties.py`

```python
class TestQuasiSheafProperties:
    """Validate mathematical correctness."""
    
    def test_approximate_functoriality(self):
        """Test ||ρ_{k→i} - ρ_{j→i} ∘ ρ_{k→j}||_F ≤ η."""
        
    def test_laplacian_positive_semidefinite(self):
        """Verify L ⪰ 0 numerically."""
        
    def test_spectral_stability(self):
        """Test eigenvalue continuity under perturbations."""
```

#### 5.3 Spectral Module Integration Tests
**Location**: `tests/test_gw_spectral_integration.py`

```python
class TestGWSpectralIntegration:
    """Test GW sheaves work correctly with spectral analysis."""
    
    def test_laplacian_block_structure(self):
        """Verify GW Laplacian follows equations (5.1) and (5.2)."""
        # Create GW sheaf
        # Extract blocks and verify formula
        
    def test_edge_weight_extraction(self):
        """Test GW costs are correctly extracted as edge weights."""
        # Build GW sheaf with known costs
        # Verify weights match GW costs, not Frobenius norms
        
    def test_filtration_parameter_generation(self):
        """Test appropriate INCREASING filtration for GW costs."""
        # GW costs in [0, 2] for cosine distance
        # Verify min-to-max parameter generation (increasing complexity)
        # Verify log-scale generation for small costs
        
    def test_persistence_with_gw_weights(self):
        """End-to-end persistence analysis with GW sheaf."""
        # Create GW sheaf
        # Run persistence analysis with INCREASING complexity filtration
        # Verify threshold function: weight <= param
        # Verify barcodes correspond to metric distortion connectivity
        
    def test_mixed_architecture_comparison(self):
        """Test spectral comparison across different architectures."""
        # Create GW sheaves for networks with different dimensions
        # Compare spectral properties
        # Verify meaningful similarity metrics
```

#### 5.4 Integration Regression Tests
**Location**: `tests/test_gw_regression.py`

```python
class TestGWRegression:
    """Ensure GW integration doesn't break existing functionality."""
    
    def test_procrustes_method_unchanged(self):
        """Verify default Procrustes behavior is preserved."""
        
    def test_api_backward_compatibility(self):
        """Test existing API calls work without modification."""
        
    def test_persistence_backward_compatibility(self):
        """Ensure existing persistence analysis works."""
```

## Integration Points

### 1. Minimal Disruption Strategy

1. **Backward Compatibility**: Default behavior unchanged (Procrustes method)
2. **Opt-in Design**: GW method requires explicit `method='gromov_wasserstein'`
3. **Shared Infrastructure**: Reuses existing Sheaf, Laplacian, and persistence code
4. **Metadata Propagation**: Construction method stored in sheaf.metadata

### 2. Key Integration Files

```
neurosheaf/
├── api.py                              # Add method parameter
├── sheaf/
│   ├── assembly/
│   │   ├── builder.py                 # Add routing logic
│   │   ├── laplacian.py              # Extend for GW support
│   │   ├── gw_builder.py              # NEW: GW-specific assembly
│   │   └── gw_laplacian.py           # NEW: Block Laplacian builder
│   ├── core/
│   │   └── gromov_wasserstein.py     # NEW: Core GW computations
│   └── config.py                      # Add GWConfig
└── spectral/
    ├── static_laplacian_unified.py    # Update edge extraction & weights
    ├── persistent.py                  # Update filtration generation
    ├── tracker.py                     # Handle GW subspace tracking
    └── edge_weights.py                # NEW: Unified weight extraction
```

### 3. Dependencies

Add to `requirements.txt`:
```
pot>=0.9.0  # Python Optimal Transport library
```

## Performance Considerations

### 1. Computational Complexity

- **Cost matrix**: O(n²d) per layer
- **GW coupling**: O(n²m² log(nm)) per edge
- **Laplacian assembly**: O(|E| · max(n_i)²)

### 2. Memory Optimization

```python
class CostMatrixCache:
    """LRU cache for expensive cost matrices."""
    
    def __init__(self, max_size_gb: float = 2.0):
        self.cache = OrderedDict()
        self.max_bytes = max_size_gb * 1e9
        
    def get_or_compute(self, layer_id: str, 
                      activations: torch.Tensor,
                      compute_fn: Callable) -> torch.Tensor:
        """Cache-aware cost matrix computation."""
```

### 3. Parallelization

- Edge-level parallelism for GW computations
- Batch processing for multiple networks
- GPU acceleration via POT backend

## Validation Metrics

### 1. Quasi-Sheaf Quality
- Maximum functoriality violation: $\max_{i<j<k} \|\rho_{k \to i} - \rho_{j \to i} \circ \rho_{k \to j}\|_F$
- Average violation across all paths
- Percentage of paths within tolerance

### 2. Numerical Quality
- Coupling constraint satisfaction: $\max_e \|\pi_e \mathbf{1} - p_{\text{target}}\|_\infty$
- Laplacian conditioning: $\kappa(L) = \lambda_{\max} / \lambda_{\min}^{+}$
- Spectral gap stability

### 3. Performance Metrics
- Wall time vs Procrustes method
- Memory usage comparison
- Scalability with network size

## Migration Guide

### For Users

```python
# Existing code (unchanged)
analyzer = NeurosheafAnalyzer()
result = analyzer.analyze(model, data)

# Using GW method
result = analyzer.analyze(model, data, method='gromov_wasserstein')

# With custom configuration
gw_config = GWConfig(epsilon=0.01, max_iter=2000)
result = analyzer.analyze(model, data, 
                         method='gromov_wasserstein',
                         gw_config=gw_config)
```

### For Developers

1. **Adding new restriction methods**:
   - Implement restriction manager in `sheaf/assembly/`
   - Add method name to `SheafBuilder.SUPPORTED_METHODS`
   - Update routing logic in `build_from_activations`

2. **Extending GW functionality**:
   - Subclass `GromovWassersteinComputer` for variants
   - Override `compute_gw_coupling` for custom solvers

## Timeline

- **Week 1**: Core GW components and configuration
- **Week 2**: Sheaf assembly integration and Laplacian
- **Week 3**: API integration and testing
- **Week 4**: Performance optimization and documentation

## Mathematical Considerations and Gotchas

### 1. Entropy Regularization Effects
- **Issue**: Entropic GW produces smoother couplings than unregularized optimal transport
- **Impact**: Spectral properties may shift with regularization parameter ε
- **Solution**: Document stability analysis with respect to ε; provide guidelines for ε selection
- **Implementation**: Store ε in metadata; offer stability diagnostics

### 2. Quasi-Sheaf Approximation
- **Issue**: Violations of exact functoriality: ||ρ_{k→i} - ρ_{j→i} ∘ ρ_{k→j}||_F ≤ η
- **Impact**: Small violations are mathematically acceptable due to spectral continuity
- **Solution**: Emphasize that quasi-sheaf property is sufficient for meaningful comparisons
- **Implementation**: Report violation statistics; set reasonable tolerance thresholds

### 3. Edge Weight Semantics and Filtration Direction
- **Issue**: GW costs represent metric distortion (lower = better match) vs Procrustes norms (higher = stronger connection)
- **Critical Impact**: Requires **INCREASING complexity filtration** for GW sheaves
- **GW Filtration Logic**: 
  - Start with NO edges (isolated stalks)
  - Add edges with cost ≤ threshold as threshold increases
  - Small costs = good matches = added first
  - Results in increasing connectivity (opposite of standard Procrustes filtration)
- **Implementation**: 
  - Use `weight <= param` threshold function for GW
  - Use `weight >= param` threshold function for standard
  - Generate parameters from min to max cost for GW
  - Document filtration semantics clearly in visualizations

### 4. Weighted Inner Products
- **Issue**: Non-uniform measures p_i require weighted L2 inner products
- **Impact**: Laplacian adjoint δ* = P^{-1/2} δ^T P^{1/2} where P = diag(p_weights)
- **Solution**: Adjust cochain inner products consistently throughout
- **Implementation**: Flag weighted vs standard mode; ensure dtype/device consistency

### 5. Multiple Parallel Edges
- **Issue**: Different data subsets may create parallel edges between same nodes
- **Impact**: Each edge contributes a separate row to coboundary δ
- **Solution**: Sum contributions properly in Laplacian assembly
- **Implementation**: Handle edge multiplicity in block construction

### 6. Numerical Stability
- **Issue**: Gram terms ρ^T ρ require careful dtype/device handling
- **Solution**: Ensure consistent tensor properties to avoid silent copies
- **Implementation**: Device-aware tensor operations; activation norm caching

### 7. Persistence Interpretation and Filtration Semantics
- **Issue**: Persistence is over edge sets (changing δ, hence L) not vertex sets
- **Impact**: Barcodes represent connectivity changes, not topological features
- **GW-Specific**: 
  - **Increasing complexity**: Early parameters → sparse → many components
  - **Birth/death interpretation**: Still valid with birth < death
  - **Semantic difference**: Birth = "good match threshold", Death = "connectivity lost"
- **Solution**: Clear documentation of what persistence measures for each construction method
- **Implementation**: Method-specific visualization labels and interpretation guides

## Success Criteria

1. **Functional**: GW method produces valid sheaves passing all tests
2. **Performance**: <2x slowdown vs Procrustes for typical networks
3. **Quality**: Quasi-sheaf tolerance <0.1 for standard architectures
4. **Integration**: Zero breaking changes to existing API
5. **Mathematical**: Proper handling of all identified gotchas with documentation