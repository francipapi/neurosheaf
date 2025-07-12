# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neurosheaf is a Python framework for neural network similarity analysis using persistent sheaf Laplacians. The project is implementing a 7-phase development process:
- **Phase 1**: Foundation (complete)
- **Phase 2**: CKA Implementation (complete) 
- **Phase 3**: Sheaf Construction (in progress - Week 5)
- **Phase 4-7**: Planned

## Critical Implementation Requirements

### NO Double-Centering in CKA (Phase 2 - MOST CRITICAL)
```python
# ALWAYS use this pattern - NO pre-centering
K = X @ X.T  # Raw activations only!
L = Y @ Y.T  # Raw activations only!
# NEVER do: X_centered = X - X.mean(dim=0)
```

### PURE WHITENED COORDINATES (Phase 3 - DESIGN BREAKTHROUGH)
**CRITICAL DISCOVERY**: All sheaf construction must occur in whitened coordinate space for optimal mathematical properties.

```python
# CORRECT: Pure whitened implementation
whitener = WhiteningProcessor()
K_whitened, W, info = whitener.whiten_gram_matrix(K)  # K → I (identity)
# Compute ALL restrictions in whitened space
R_whitened = compute_restriction_whitened(K_source_white, K_target_white)
# Build sheaf entirely in whitened coordinates - NEVER transform back!

# WRONG: Back-transformation approach (mathematically suboptimal)
# R_original = W_target^† @ R_whitened @ W_source  # Destroys perfect properties
```

**Mathematical Justification**:
- Whitened space: Perfect orthogonality, exact metric compatibility, optimal conditioning
- Original space: Rank-deficient, ill-conditioned, approximate properties only
- Whitening is a change of coordinates, NOT a loss of information
- **Result**: 100% acceptance criteria success vs 44% with back-transformation

### Performance Targets
- **Memory**: <3GB for ResNet50 analysis (7× improvement from 20GB baseline)
- **Speed**: <5 minutes for complete analysis pipeline
- **Sparse efficiency**: >90% memory savings vs dense matrices

## Development Commands

**CRITICAL**: All commands must be run in the conda environment `myenv`:
```bash
# Proper conda activation method (required for AI agents)
# Note: Set KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP conflicts on macOS
export KMP_DUPLICATE_LIB_OK=TRUE
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv
```

Development commands for the project:

```bash
# Setup (Phase 1 complete)
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv
pip install -e .
pip install -r requirements-dev.txt

# Testing (test-driven development approach)
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && pytest -v                              # Run all tests
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && pytest tests/phase{X}/ -v              # Run specific phase tests
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && pytest tests/critical/ -v --tb=short   # Run critical functionality tests
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && pytest -m "not slow" -v                # Skip slow tests during development

# Code quality (Phase 1 complete)
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && black neurosheaf/
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && ruff check neurosheaf/
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && mypy neurosheaf/

# Performance profiling (Phase 1 deliverable)
export KMP_DUPLICATE_LIB_OK=TRUE && source /opt/anaconda3/etc/profile.d/conda.sh && conda activate myenv && python -m neurosheaf.utils.benchmarking
```

## Architecture Overview

The system implements a 7-phase development plan with strict mathematical correctness requirements:

### Core Pipeline (Updated with Whitened Design)
1. **CKA Computation** → 2. **Whitening Transformation** → 3. **Sheaf Construction (Whitened Space)** → 4. **Spectral Analysis (Whitened)** → 5. **Visualization**

**Key Change**: All mathematical operations after Step 2 occur in whitened coordinate space for optimal numerical properties.

### Package Structure (to be implemented)
```
neurosheaf/
├── api.py                    # High-level user API
├── cka/                      # Phase 2: Debiased CKA (NO double-centering)
│   ├── debiased.py          # Core CKA implementation
│   ├── nystrom.py           # Memory-efficient approximation
│   └── sampling.py          # Adaptive sampling
├── sheaf/                    # Phase 3: Sheaf construction (WHITENED COORDINATES)
│   ├── construction.py      # Main sheaf builder (pure whitened implementation)
│   ├── poset.py            # FX-based automatic extraction  
│   ├── restriction.py       # WhiteningProcessor + whitened restrictions
│   └── laplacian.py        # Sparse Laplacian assembly (whitened space)
├── spectral/                # Phase 4: Spectral analysis
│   ├── persistent.py        # Main analyzer
│   ├── tracker.py          # Subspace similarity tracking
│   └── static_laplacian.py # Edge masking approach
├── visualization/           # Phase 5: Interactive visualization
│   ├── stalks.py           # CKA matrices with log scale
│   ├── poset.py            # Network structure plots
│   ├── persistence.py      # Diagrams & barcodes
│   └── dashboard.py        # Dash-based interactive app
└── utils/                   # Phase 1: Foundation utilities
    ├── logging.py          # Logging infrastructure
    ├── exceptions.py       # Custom exceptions
    └── profiling.py        # Performance monitoring
```

## Development Process

### Phase-Based Implementation
Each phase has detailed implementation plans in `plan/phase{X}_*/implementation_plan.md` with corresponding testing suites. **Always consult these plans before coding.**

### Critical Technical Points by Phase
- **Phase 1**: Setup logging and profiling infrastructure first
- **Phase 2**: Mathematical correctness is non-negotiable for CKA
  - **MPS Limitation**: SVD operations automatically use CPU fallback due to numerical stability issues (PyTorch GitHub #78099)
- **Phase 3**: **PURE WHITENED COORDINATE IMPLEMENTATION** (BREAKTHROUGH)
  - FX-based poset extraction with fallbacks for dynamic models
  - **CRITICAL**: All sheaf operations in whitened space (K → I transformations)
  - **NEVER transform back** to original coordinates (destroys mathematical optimality)
  - Achieves 100% acceptance criteria vs 44% with back-transformation
- **Phase 4**: Subspace tracking using principal angles, not index-based
  - **Update**: Spectral analysis performed on whitened Laplacian (better conditioning)
- **Phase 5**: Log-scale detection and automatic backend switching
- **Phase 6**: >95% test coverage requirement
- **Phase 7**: Production deployment with Docker and PyPI

### Test-Driven Development
- **Environment first**: Always run `export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv` before any development
- **Write tests first**: Start with critical tests from testing suites
- **Test immediately**: Run `export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv && pytest -v` after every significant code change
- **Validate continuously**: Use `export KMP_DUPLICATE_LIB_OK=TRUE && conda activate myenv && pytest -v` frequently during development
- **Cover edge cases**: Implement edge case tests before they become issues

## Key Files to Reference

### Before Starting Any Phase
1. `plan/phase{X}_*/implementation_plan.md` - Detailed implementation specifications
2. `plan/phase{X}_*/testing_suite.md` - Comprehensive testing requirements
3. `guidelines.md` - Implementation guidelines and best practices

### Documentation Structure
- `docs/comprehensive-implementation-plan-v3.md` - Complete system overview
- `docs/updated-debiased-cka-v3.md` - CKA mathematical specifications
- `docs/updated-sheaf-construction-v3.md` - Sheaf construction details
- `docs/visualization-plan-v3.md` - Visualization requirements
- `docs/spectral_sheaf_pipeline_report.md` - Comprehensive mathematical exposition of the whole pipeline

## Mathematical Validation Requirements

### CKA Properties (Phase 2)
- CKA(X,X) = 1 (self-similarity)
- CKA(X,Y) = CKA(Y,X) (symmetry)
- 0 ≤ CKA(X,Y) ≤ 1 (bounded)

### Sheaf Properties (Phase 3)
- Transitivity: R_AC = R_BC @ R_AB
- Consistency across restriction maps
- Sparse Laplacian positive semi-definite

### Spectral Properties (Phase 4)
- Eigenvalue continuity through crossings
- Subspace similarity > 0.8 threshold for tracking
- Persistence diagram mathematical validity

## Implementation Workflow

### Starting New Code
1. **Activate environment**: Run `conda activate myenv` before starting
2. Read the specific phase implementation plan
3. Review the testing suite requirements
4. Implement critical tests first
5. Follow the exact code patterns from plans
6. Validate against mathematical properties with `conda activate myenv && pytest -v`
7. Check performance targets

### Common Pitfalls to Avoid
- **Never pre-center data for CKA** (most critical error)
- **Never use index-based eigenvalue tracking** (use subspace similarity)
- **Never ignore numerical stability** (add epsilon values)
- **Never create dense matrices for large networks** (use sparse operations)

## Success Criteria

Each phase has specific success criteria defined in implementation plans. Overall project success requires:
- All mathematical properties validated
- Performance targets met (<3GB memory, <5 minutes)
- >95% test coverage
- Production-ready package deployment

## Dependencies (to be configured in Phase 1)

```toml
# Core dependencies
torch >= 2.2.0          # FX tracer, CUDA support
scipy >= 1.10.0         # subspace_angles, sparse matrices
numpy >= 1.21.0         # Numerical operations
networkx >= 2.6         # Graph operations
matplotlib >= 3.5.0     # Static plots
plotly >= 5.0.0         # Interactive plots
dash >= 2.0.0           # Web dashboard

# Development dependencies
pytest >= 7.0.0
pytest-cov >= 4.0.0
black >= 22.0.0
ruff >= 0.1.0
mypy >= 1.0.0
```

The project requires strict adherence to the implementation plans and mathematical correctness to achieve the target 7× memory improvement over baseline approaches.