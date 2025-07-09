# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neurosheaf is a Python framework for neural network similarity analysis using persistent sheaf Laplacians. The project is in **planning phase** - no code has been implemented yet. All implementation plans and guidelines are prepared for a 7-phase development process.

## Critical Implementation Requirements

### NO Double-Centering in CKA (Phase 2 - MOST CRITICAL)
```python
# ALWAYS use this pattern - NO pre-centering
K = X @ X.T  # Raw activations only!
L = Y @ Y.T  # Raw activations only!
# NEVER do: X_centered = X - X.mean(dim=0)
```

### Performance Targets
- **Memory**: <3GB for ResNet50 analysis (500× improvement from 1.5TB baseline)
- **Speed**: <5 minutes for complete analysis pipeline
- **Sparse efficiency**: >90% memory savings vs dense matrices

## Development Commands

Since the project is in planning phase, these commands should be implemented as part of Phase 1:

```bash
# Setup (to be implemented in Phase 1)
pip install -e .
pip install -r requirements-dev.txt

# Testing (test-driven development approach)
pytest -v                              # Run all tests
pytest tests/phase{X}/ -v              # Run specific phase tests
pytest tests/critical/ -v --tb=short   # Run critical functionality tests
pytest -m "not slow" -v                # Skip slow tests during development

# Code quality (to be configured in Phase 1)
black neurosheaf/
ruff check neurosheaf/
mypy neurosheaf/

# Performance profiling (Phase 1 deliverable)
python -m neurosheaf.utils.benchmarking
```

## Architecture Overview

The system implements a 7-phase development plan with strict mathematical correctness requirements:

### Core Pipeline
1. **CKA Computation** → 2. **Sheaf Construction** → 3. **Spectral Analysis** → 4. **Visualization**

### Package Structure (to be implemented)
```
neurosheaf/
├── api.py                    # High-level user API
├── cka/                      # Phase 2: Debiased CKA (NO double-centering)
│   ├── debiased.py          # Core CKA implementation
│   ├── nystrom.py           # Memory-efficient approximation
│   └── sampling.py          # Adaptive sampling
├── sheaf/                    # Phase 3: Sheaf construction
│   ├── construction.py      # Main sheaf builder
│   ├── poset.py            # FX-based automatic extraction
│   ├── restriction.py       # Scaled Procrustes maps
│   └── laplacian.py        # Sparse Laplacian assembly
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
- **Phase 3**: FX-based poset extraction with fallbacks for dynamic models
- **Phase 4**: Subspace tracking using principal angles, not index-based
- **Phase 5**: Log-scale detection and automatic backend switching
- **Phase 6**: >95% test coverage requirement
- **Phase 7**: Production deployment with Docker and PyPI

### Test-Driven Development
- **Write tests first**: Start with critical tests from testing suites
- **Test immediately**: Run tests after every significant code change
- **Validate continuously**: Use `pytest -v` frequently during development
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
1. Read the specific phase implementation plan
2. Review the testing suite requirements
3. Implement critical tests first
4. Follow the exact code patterns from plans
5. Validate against mathematical properties
6. Check performance targets

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

The project requires strict adherence to the implementation plans and mathematical correctness to achieve the target 500× memory improvement over baseline approaches.