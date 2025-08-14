# Production-Ready Global Section (H⁰) Tracking with GW Transport

## Executive Summary

This document provides a **production-hardened** implementation plan for global section tracking in the Gromov-Wasserstein neurosheaf pipeline. Built on mathematically correct transport-informed persistence, this plan prioritizes numerical stability, clean APIs, comprehensive testing, and observability for deployment at scale.

## 0) Guiding Principles (Engineering for λ ≈ 10⁻¹²)

### **Core Design Philosophy**
- **Decide in σ-space, not λ-space**: Work with SVD of whitened coboundary δ̃, use hysteretic thresholds tied to ||δ̃||₂ and √ε
- **Subspaces near 0, vectors away from 0**: Treat kernel as subspace; never label individual "zero" eigenvectors
- **Transport then project**: GW couplings → linear transport T → whiten to T̃ → project onto kernels for F_t
- **Rank decisions via RRQR**: Births/deaths using column-pivoted QR with tolerance; avoid Hungarian matching on near-zero bases
- **Everything float64**: Deterministic seeds, log thresholds and norms every step

### **Numerical Robustness Standards**
- **No matrix inverses**: Use Cholesky + triangular solves or element-wise operations
- **Mass floors everywhere**: Clip tiny weights to prevent division by zero
- **Two-step confirmation**: Prevent noise-induced flickering in persistence diagrams
- **Hysteretic thresholds**: Scale-aware, adaptive to operator norms
- **Comprehensive logging**: Every numerical decision is recorded and verifiable

---

## 1) Numerical Constants and Formulas (Anti-Flake)

### **Data Types and Precision**
```python
# MANDATORY: Use float64 throughout for stability at machine precision
DTYPE = torch.float64
EPS = np.finfo(np.float64).eps  # ≈ 2.2e-16
```

### **Robust Threshold System**
```python
# Spectral norm (carry from previous step if expensive)
S_t = ||δ̃(t)||₂

# Zero detection hysteresis (in σ-space)
τ_in = c_in × √ε × S_t     # Enter zero state, c_in = 100
τ_out = 3 × τ_in           # Exit zero state (prevents flickering)

# RRQR persistence threshold (on Y = F_t @ Q_t)
τ_keep = c_keep × √ε × ||Y||₂   # Keep generators, c_keep = 200

# Two-step confirmation for:
# (1) Kernel dimension changes
# (2) Death events
```

### **Whitening Operations (Safe Implementation)**
```python
# Diagonal metrics (common case)
if G₀ = diag(a), G₁ = diag(w):
    δ̃ = D_w^(1/2) @ δ @ D_a^(-1/2)    # Element-wise operations
    T̃ = D_b^(-1/2) @ π^T @ D_a^(-1/2)  # No matrix inverses

# General SPD metrics (full case)
if G₀, G₁ arbitrary SPD:
    L₀ = cholesky(G₀), L₁ = cholesky(G₁)
    δ̃ = L₁ @ δ @ solve_triangular(L₀)  # Use triangular solves
    T̃ = L_{t+1} @ T @ solve_triangular(L_t)

# Mass floors (prevent singular matrices)
masses ← max(masses, 10⁻¹² × median(masses))
```

---

## 2) Clean Module Architecture

### **Package Structure**
```
neurosheaf/
├── spectral/
│   ├── global_sections.py     # Whitening, SVD slice, kernel basis, hysteresis
│   ├── transport.py           # GW π → T → T̃ (balanced & unbalanced)
│   ├── h0_persistence.py      # RRQR update loop, births/deaths, confirmation
│   └── utils_numerical.py     # Spectral norm estimation, RRQR, logging
├── io/
│   ├── types.py               # Typed protocols for builders (δ, G₀/G₁, π)
│   └── config.py              # Dataclasses for thresholds, margins, dtype
└── tests/
    ├── test_kernels.py        # Correctness & stability tests
    ├── test_transport.py      # π → T → T̃ contracts and properties
    └── test_persistence.py    # Barcode sanity, degenerate cases
```

### **Configuration Management**
```python
@dataclass(frozen=True)
class H0Config:
    """Production configuration for H⁰ persistence tracking."""
    
    # Numerical precision
    dtype: str = "float64"
    seed: int = 123
    deterministic: bool = True
    
    # Threshold parameters
    c_in: float = 100.0          # Zero entry threshold multiplier
    gap: float = 3.0             # Hysteresis gap (τ_out = gap × τ_in)
    c_keep: float = 200.0        # RRQR keep threshold multiplier
    margin: int = 12             # Extra singular values to compute
    
    # Stability controls
    confirm_steps: int = 2       # Two-step confirmation window
    mass_floor_factor: float = 1e-12  # Floor for tiny masses
    max_cholesky_retries: int = 3     # Retries with increased regularization
    
    # Performance tuning
    dense_threshold: int = 5000  # Switch to sparse methods above this size
    spectral_norm_iterations: int = 8  # Power iteration steps
    cache_cholesky: bool = True  # Cache factorizations between steps
    
    # Observability
    log_level: int = logging.INFO
    save_diagnostics: bool = True
    validate_certificates: bool = True
```

---

## 3) Core APIs (Stable Contracts)

### **3.1 Global Sections Module**
```python
# neurosheaf/spectral/global_sections.py

def whiten_coboundary_robust(delta: torch.Tensor, 
                           G0: torch.Tensor, 
                           G1: torch.Tensor,
                           cfg: H0Config) -> WhiteningResult:
    """
    Whiten coboundary operator using numerically stable methods.
    
    Returns δ̃, L₀, L₁ using Cholesky + triangular solves (no matrix inverses).
    Includes automatic regularization retry if Cholesky fails.
    
    Mathematical: δ̃ = G₁^(1/2) @ δ @ G₀^(-1/2)
    Implementation: Uses triangular solves for stability
    """

def kernel_basis_with_hysteresis(delta_tilde: torch.Tensor,
                               S_prev: float,
                               labels_prev: Optional[List[str]],
                               cfg: H0Config) -> KernelResult:
    """
    Compute orthonormal kernel basis using hysteretic σ-space classification.
    
    Algorithm:
    1. Compute r = prev_nullity + cfg.margin smallest singular values
    2. Apply hysteretic labeling: σ ≤ τ_in → ZERO, σ ≥ τ_out → NONZERO
    3. Extract kernel basis V⁰ from ZERO-labeled right singular vectors
    4. Verify residual: ||δ̃ @ V⁰||₂ ≤ 10√ε × S_t
    
    Returns: V⁰ (orthonormal), k (nullity), labels, S_t, diagnostics
    """

def validate_kernel_certificate(delta_tilde: torch.Tensor,
                              V0: torch.Tensor, 
                              S_t: float,
                              cfg: H0Config) -> CertificateResult:
    """
    Validate kernel basis with numerical certificate.
    
    Certificate: ||δ̃ @ V⁰||₂ / S_t ≤ 10√ε
    """
```

### **3.2 Transport Module**
```python
# neurosheaf/spectral/transport.py

def construct_transport_from_gw_coupling(Pi: torch.Tensor,
                                       node_masses_t: torch.Tensor,
                                       node_masses_tp1: torch.Tensor,
                                       coupling_type: str,
                                       cfg: H0Config) -> TransportResult:
    """
    Build linear transport map from GW coupling with numerical safeguards.
    
    Balanced GW: T = D_b^(-1) @ π^T
    Unbalanced GW: Use realized marginals ã_t = π @ 1, b̃_{t+1} = π^T @ 1
    
    Includes mass floor application and safe division handling.
    """

def create_whitened_transport(T: torch.Tensor,
                            G0_prev: torch.Tensor,
                            G0_next: torch.Tensor,
                            cfg: H0Config) -> torch.Tensor:
    """
    Create metric-aware whitened transport T̃ for numerical stability.
    
    General: T̃ = L_{t+1} @ T @ L_t^(-1) (triangular solves)
    Diagonal: T̃ = D_b^(-1/2) @ π^T @ D_a^(-1/2) (element-wise)
    
    Returns transport map suitable for kernel projection.
    """

def validate_transport_properties(T_tilde: torch.Tensor,
                                Pi: torch.Tensor,
                                masses_t: torch.Tensor,
                                masses_tp1: torch.Tensor) -> TransportValidation:
    """
    Validate transport map properties and marginal consistency.
    """
```

### **3.3 H⁰ Persistence Module**
```python
# neurosheaf/spectral/h0_persistence.py

def compute_induced_map_on_kernels(V0_prev: torch.Tensor,
                                 V0_next: torch.Tensor,
                                 T_tilde: torch.Tensor) -> torch.Tensor:
    """
    Compute transport-induced map on global section spaces.
    
    F_t = V⁰_{t+1}^T @ T̃ @ V⁰_t
    
    This is the core map used for RRQR persistence updates.
    """

def rrqr_persistence_update(F: torch.Tensor,
                          Q_prev: torch.Tensor,
                          filtration_param: float,
                          cfg: H0Config) -> RRQRResult:
    """
    Update persistence generators using rank-revealing QR decomposition.
    
    Algorithm:
    1. Push alive generators: Y = F @ Q_prev
    2. Column-pivoted QR: Y @ P = Q̂ @ R  
    3. Keep columns with |R_jj| ≥ c_keep × √ε × ||Y||₂
    4. Apply two-step confirmation for deaths
    
    Returns: Q_next, keep_mask, deaths, births
    """

def run_h0_persistence_pipeline(filtration_data: List[FiltrationStep],
                              builders: Dict[str, Any],
                              cfg: H0Config) -> PersistenceResult:
    """
    Complete H⁰ persistence pipeline with transport-informed evolution.
    
    For each filtration step t:
    1. Whiten coboundary: δ̃_t = whiten_coboundary_robust(...)
    2. Compute kernel: V⁰_t, k_t = kernel_basis_with_hysteresis(...)
    3. If t > 0:
       a. Build transport: T̃_t = construct_transport_from_gw_coupling(...)
       b. Induced map: F_t = compute_induced_map_on_kernels(...)
       c. Update persistence: Q_t, deaths = rrqr_persistence_update(...)
       d. Detect births: births = k_t - Q_t.shape[1]
       e. Apply two-step confirmation
    4. Record events and diagnostics
    
    Returns: H⁰ persistence intervals, Betti curve, diagnostics
    """
```

---

## 4) Performance and Scaling Guidelines

### **4.1 Algorithm Selection by Problem Size**
```python
def select_svd_method(matrix_shape: Tuple[int, int], 
                     nnz: int,
                     r: int,
                     cfg: H0Config) -> str:
    """
    Automatic algorithm selection based on problem characteristics.
    
    Small dense (n ≤ 5k): torch.linalg.svd + slice
    Large/sparse: scipy.sparse.linalg.svds(which="SM") with warm start
    Very large: Randomized SVD with subspace iteration
    """
    
    n, m = matrix_shape
    density = nnz / (n * m) if n * m > 0 else 1.0
    
    if max(n, m) <= cfg.dense_threshold:
        return "dense_full"
    elif density < 0.1:
        return "sparse_iterative"  
    else:
        return "randomized_subspace"
```

### **4.2 Caching and Warm Starts**
```python
class PersistenceCache:
    """Cache for expensive computations across filtration steps."""
    
    def __init__(self, cfg: H0Config):
        self.cholesky_factors = {}  # Cache L₀(t), L₁(t) 
        self.spectral_norms = {}    # Cache ||δ̃||₂ estimates
        self.kernel_subspaces = {}  # Previous kernel for warm starts
        self.cfg = cfg
    
    def get_cholesky_factors(self, step: int, G0: torch.Tensor, G1: torch.Tensor):
        """Cached Cholesky factorization with automatic invalidation."""
        
    def warm_start_svd(self, step: int, prev_subspace: torch.Tensor):
        """Use previous step's kernel subspace to accelerate SVD."""
```

---

## 5) Observability and Diagnostics

### **5.1 Structured Logging Per Step**
```python
@dataclass
class StepDiagnostics:
    """Comprehensive diagnostics for each filtration step."""
    
    step: int
    filtration_param: float
    
    # Spectral properties
    spectral_norm: float          # S_t = ||δ̃||₂
    kernel_dimension: int         # k_t = dim(ker(δ̃))
    kernel_residual: float        # ||δ̃ @ V⁰||₂ / S_t
    
    # Thresholds used
    tau_in: float                 # Zero entry threshold
    tau_out: float                # Zero exit threshold  
    tau_keep: float               # RRQR keep threshold
    
    # Persistence events
    n_births: int
    n_deaths: int
    n_confirmed_deaths: int       # After two-step confirmation
    dimension_change: bool
    
    # Transport properties (if applicable)
    transport_norm: Optional[float]      # ||T̃||₂
    induced_map_rank: Optional[int]      # rank(F_t)
    mass_floor_applied: bool
    
    # Numerical certificates
    cholesky_retries: int
    certificate_passed: bool
    computational_time: float

def log_step_diagnostics(diagnostics: StepDiagnostics, logger: logging.Logger):
    """Structured logging with JSON format for analysis."""
    logger.info(json.dumps(asdict(diagnostics), indent=2))
```

### **5.2 Numerical Certificates**
```python
def compute_numerical_certificates(delta_tilde: torch.Tensor,
                                 V0: torch.Tensor,
                                 Y: torch.Tensor,
                                 Q_hat: torch.Tensor,
                                 R: torch.Tensor,
                                 P: torch.Tensor,
                                 cfg: H0Config) -> CertificateReport:
    """
    Comprehensive numerical validation of computed results.
    
    Certificates:
    1. Kernel residual: ||δ̃ @ V⁰||₂ / ||δ̃||₂ ≤ 10√ε
    2. RRQR consistency: ||Y - Q̂ @ R @ P^T||_F / ||Y||_F ≤ √ε
    3. Orthogonality: ||V⁰^T @ V⁰ - I||_F ≤ 10√ε
    4. Transport marginal preservation (if applicable)
    """
```

### **5.3 Deterministic Execution**
```python
def setup_deterministic_execution(cfg: H0Config):
    """
    Configure deterministic execution for reproducible results.
    
    Sets all random seeds, enables deterministic algorithms where available,
    warns about nondeterministic operations.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## 6) Robust Error Handling

### **6.1 Numerical Failure Recovery**
```python
def robust_cholesky_with_fallback(G: torch.Tensor, 
                                cfg: H0Config) -> Tuple[torch.Tensor, bool]:
    """
    Cholesky decomposition with automatic regularization on failure.
    
    Strategy:
    1. Try standard Cholesky
    2. If fails, add diagonal regularization: G + ε×I
    3. Retry with progressively larger ε
    4. If still fails, fall back to diagonal approximation
    """
    regularization = 0.0
    
    for attempt in range(cfg.max_cholesky_retries):
        try:
            if attempt > 0:
                regularization = cfg.mass_floor_factor * (10 ** attempt)
                G_reg = G + regularization * torch.eye(G.shape[0], dtype=G.dtype)
            else:
                G_reg = G
                
            L = torch.linalg.cholesky(G_reg)
            return L, attempt > 0  # Return whether regularization was used
            
        except torch.linalg.LinAlgError:
            continue
    
    # Ultimate fallback: diagonal approximation
    logger.warning("Cholesky failed completely, using diagonal fallback")
    return torch.diag(torch.sqrt(torch.diag(G))), True
```

### **6.2 Transport Noise Mitigation**
```python
def stabilize_transport_if_needed(T_tilde: torch.Tensor,
                                masses_t: torch.Tensor,
                                masses_tp1: torch.Tensor,
                                noise_threshold: float = 0.1) -> torch.Tensor:
    """
    Optional transport stabilization for extremely noisy couplings.
    
    If transport exhibits excessive noise, blend with uniform coupling:
    T̃_α = (1-α) × T̃ + α × (√b @ √a^T) / (||√b|| × ||√a||)
    """
    # Detect noise via condition number or spectral radius
    if torch.linalg.cond(T_tilde) > 1/noise_threshold:
        alpha = 0.05  # Small stabilization factor
        
        # Uniform transport baseline
        u = torch.sqrt(masses_tp1) / torch.norm(torch.sqrt(masses_tp1))
        v = torch.sqrt(masses_t) / torch.norm(torch.sqrt(masses_t))
        uniform_transport = torch.outer(u, v)
        
        # Convex combination
        T_stabilized = (1 - alpha) * T_tilde + alpha * uniform_transport
        
        logger.info(f"Applied transport stabilization with α = {alpha}")
        return T_stabilized
    
    return T_tilde
```

### **6.3 Edge Case Handling Matrix**
```python
EDGE_CASE_HANDLERS = {
    'zero_marginals': apply_mass_floor_before_division,
    'singular_cholesky': robust_cholesky_with_fallback, 
    'transport_explosion': stabilize_transport_if_needed,
    'dimension_flicker': apply_two_step_confirmation,
    'numerical_overflow': scale_and_retry_computation,
    'empty_kernel': handle_trivial_kernel_case,
    'full_rank_coupling': handle_degenerate_transport
}
```

---

## 7) Comprehensive Testing Framework

### **7.1 Correctness Tests**
```python
class TestNumericalCorrectness:
    """Test mathematical correctness of global section tracking."""
    
    def test_hodge_decomposition_consistency(self):
        """Verify dim(ker(δ̃)) = dim(ker(L̃)) on small examples."""
        
    def test_graph_component_counting(self):
        """Test β₀ matches known component counts on simple graphs."""
        
    def test_transport_identity_cases(self):
        """With π = a ⊗ b^T (fully mixed), verify F_t collapses spans correctly."""
        
    def test_kernel_certificates(self):
        """Verify ||δ̃ @ V⁰||₂ ≤ certificate_threshold × ||δ̃||₂."""
        
    def test_rrqr_rank_consistency(self):
        """Verify RRQR rank detection matches theoretical expectations."""

class TestNumericalStability:
    """Test stability under various numerical stress conditions."""
    
    def test_threshold_parameter_sweep(self):
        """Sweep c_in ∈ [50, 200], c_keep ∈ [100, 400]: verify stable barcodes."""
        
    def test_gw_coupling_perturbation(self):
        """Add small noise to π_t: verify intervals don't flicker (two-step confirm)."""
        
    def test_extreme_eigenvalue_cases(self):
        """Construct cases with λ_min ≈ 10⁻¹² and verify robust σ-space classification."""
        
    def test_mass_floor_effectiveness(self):
        """Verify mass floors prevent division by zero and maintain stability."""
        
    def test_cholesky_fallback_recovery(self):
        """Test recovery from near-singular G₀, G₁ matrices."""

class TestPerformanceScaling:
    """Test computational performance and memory scaling."""
    
    def test_sparse_vs_dense_crossover(self):
        """Verify optimal algorithm selection at different problem sizes."""
        
    def test_cache_effectiveness(self):
        """Measure speedup from Cholesky factor and subspace caching."""
        
    def test_memory_bounded_execution(self):
        """Ensure memory usage stays bounded (no quadratic blow-ups)."""
```

### **7.2 Regression Test Suite**
```python
def create_regression_test_suite():
    """
    Generate comprehensive test cases covering:
    
    1. Known pathological cases (near-singular matrices, extreme eigenvalues)
    2. Validated reference results on standard test graphs
    3. Performance benchmarks on representative problem sizes
    4. Error recovery scenarios (Cholesky failures, transport instability)
    """
    
    test_cases = [
        # Correctness benchmarks
        ('erdos_renyi_n100_p0.1', validate_component_count_evolution),
        ('lattice_2d_10x10_percolation', validate_percolation_threshold),
        ('complete_bipartite_k33', validate_known_topology),
        
        # Numerical stress tests  
        ('near_singular_laplacian', test_extreme_conditioning),
        ('machine_precision_eigenvalues', test_sigma_space_robustness),
        ('noisy_gw_coupling', test_transport_stability),
        
        # Performance benchmarks
        ('sparse_scale_n10k', benchmark_sparse_performance),
        ('dense_scale_n1k', benchmark_dense_performance),
        ('cache_effectiveness', benchmark_cached_vs_uncached)
    ]
    
    return test_cases
```

---

## 8) CLI Interface and Output Format

### **8.1 Command Line Interface**
```bash
# Basic usage
neurosheaf-h0 --config h0_config.yaml --filtration filtration.json --output results/

# Advanced usage with custom thresholds
neurosheaf-h0 \
    --config h0_config.yaml \
    --filtration filtration.json \
    --output results/ \
    --c-in 150.0 \
    --c-keep 250.0 \
    --confirm-steps 3 \
    --dtype float64 \
    --deterministic \
    --save-diagnostics
```

### **8.2 Output Format Specification**
```python
# H⁰ persistence intervals
# File: h0_intervals.parquet
Schema = {
    'birth_idx': Int32,           # Birth step index
    'death_idx': Int32,           # Death step index (null for infinite)
    'birth_param': Float64,       # Birth filtration parameter
    'death_param': Float64,       # Death filtration parameter (null for infinite)
    'lifetime': Float64,          # death_param - birth_param
    'confirmed': Boolean,         # Passed two-step confirmation
    'transport_informed': Boolean # Used transport evolution (vs birth at step 0)
}

# Betti curve evolution  
# File: betti_curve.parquet
Schema = {
    'step': Int32,               # Filtration step
    'param': Float64,            # Filtration parameter
    'beta0': Int32,              # β₀ = dim(H⁰) at this step
    'kernel_dim': Int32,         # Raw kernel dimension before persistence update
    'confirmed_dim': Int32       # Confirmed dimension after two-step validation
}

# Per-step diagnostics
# File: diagnostics.jsonl (JSON Lines format)
Schema = StepDiagnostics  # As defined in Section 5.1
```

### **8.3 Visualization Outputs**
```python
def generate_standard_plots(results: PersistenceResult, output_dir: Path):
    """Generate standard visualization suite."""
    
    # Betti curve: β₀(t) evolution
    plot_betti_curve(
        results.betti_curve,
        save_path=output_dir / "betti_curve.png",
        title="H⁰ Betti Number Evolution"
    )
    
    # Persistence diagram: (birth, death) scatter
    plot_h0_persistence_diagram(
        results.intervals,
        save_path=output_dir / "h0_persistence.png", 
        title="H⁰ Persistence Diagram"
    )
    
    # Diagnostics dashboard: thresholds, norms, certificates over time
    plot_diagnostics_dashboard(
        results.diagnostics,
        save_path=output_dir / "diagnostics_dashboard.png"
    )
    
    # Transport evolution (if available)
    if results.transport_data:
        plot_transport_evolution(
            results.transport_data,
            save_path=output_dir / "transport_evolution.png"
        )
```

---

## 9) Production Deployment Checklist

### **9.1 Integration with Existing GW Pipeline** 
- [ ] **Drop-in replacement**: Replace `_generate_persistence_diagrams()` in `PersistentSpectralAnalyzer`
- [ ] **Backward compatibility**: Maintain existing eigenvalue-based tracking as fallback
- [ ] **Metadata compatibility**: Extract GW couplings and masses from existing sheaf metadata
- [ ] **Performance parity**: Ensure comparable or better performance vs current method

### **9.2 Numerical Validation Checklist**
- [ ] **Certificate validation**: All kernel residuals ≤ 10√ε × ||δ̃||₂  
- [ ] **Threshold robustness**: Stable results across c_in ∈ [50, 200], c_keep ∈ [100, 400]
- [ ] **Transport stability**: <5% barcode changes under 10% coupling perturbation
- [ ] **Two-step confirmation**: No spurious bars shorter than 2 filtration steps
- [ ] **Deterministic execution**: Identical results across runs with same seeds

### **9.3 Performance Validation**
- [ ] **Memory bounded**: <3GB for ResNet50-scale problems (maintained from original target)
- [ ] **Time bounded**: <5 minutes for complete analysis (maintained from original target)  
- [ ] **Scaling verified**: Sublinear memory growth, optimal algorithm selection
- [ ] **Cache effectiveness**: >50% speedup on repeated similar-size problems

### **9.4 Testing and Quality Assurance**
- [ ] **Unit tests**: >95% code coverage with meaningful assertions
- [ ] **Integration tests**: End-to-end pipeline tests on representative examples
- [ ] **Regression tests**: Automated comparison against validated reference results
- [ ] **Performance benchmarks**: Continuous monitoring of computational efficiency
- [ ] **Numerical stress tests**: Robustness under extreme conditions

---

## 10) Key Engineering Improvements vs Original Plan

### **10.1 Numerical Robustness Hardening**
- **Replaced** absolute λ thresholds with **hysteretic σ-space** thresholds tied to operator scale
- **Added** comprehensive mass floor system to prevent division by zero
- **Implemented** robust Cholesky with automatic regularization retry
- **Introduced** two-step confirmation to prevent noise-induced flickering

### **10.2 Clean Architecture and APIs**
- **Designed** stable contracts with typed protocols and dataclasses
- **Separated** concerns: whitening, transport, persistence into focused modules
- **Implemented** comprehensive configuration management with sensible defaults
- **Added** extensive caching and performance optimization hooks

### **10.3 Production-Grade Observability**
- **Structured logging** with JSON format for analysis and debugging
- **Numerical certificates** for every major computation with automated validation
- **Deterministic execution** with full reproducibility controls
- **Comprehensive diagnostics** tracking all thresholds, norms, and decisions

### **10.4 Extensive Testing Framework**  
- **Multi-layered testing**: correctness, stability, performance, regression
- **Pathological case coverage**: extreme eigenvalues, singular matrices, transport noise
- **Automated benchmarking**: continuous performance monitoring with alerts
- **Reference validation**: comparison against known mathematical results

This production-ready plan transforms the theoretical foundation into a robust, deployable system that can handle the numerical challenges of global section tracking at machine precision while providing the observability and reliability required for production neural network analysis.