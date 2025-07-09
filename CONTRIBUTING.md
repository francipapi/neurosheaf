# Contributing to Neurosheaf

Thank you for your interest in contributing to Neurosheaf! This document provides guidelines and information for contributors.

## 🎯 Project Overview

Neurosheaf is a Python framework for neural network similarity analysis using persistent sheaf Laplacians. The project follows a structured 7-phase development approach with strict mathematical correctness requirements.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- PyTorch 2.0+
- Basic knowledge of neural networks and topological data analysis

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/neurosheaf.git
   cd neurosheaf
   ```

2. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Setup Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Installation**
   ```bash
   make test
   ```

## 📝 Development Process

### Phase-Based Development

Neurosheaf follows a 7-phase development process:

1. **Phase 1**: Foundation (✅ Complete)
2. **Phase 2**: CKA Implementation (🔄 Current)
3. **Phase 3**: Sheaf Construction (📅 Planned)
4. **Phase 4**: Spectral Analysis (📅 Planned)
5. **Phase 5**: Visualization (📅 Planned)
6. **Phase 6**: Testing & Documentation (📅 Planned)
7. **Phase 7**: Deployment (📅 Planned)

### Before Contributing

1. **Review the Current Phase**: Check `plan/phase{X}_*/implementation_plan.md`
2. **Understand Requirements**: Read `guidelines.md` for critical requirements
3. **Check Issues**: Look for open issues or create a new one
4. **Discuss**: For major changes, discuss in an issue first

## 🔧 Code Standards

### Code Quality

- **Formatting**: Use `black` and `isort`
- **Linting**: Pass `flake8` checks
- **Type Hints**: Use type hints for all functions
- **Documentation**: Document all public APIs

### Testing Requirements

- **Coverage**: Maintain >95% test coverage
- **Test Types**: Unit, integration, and performance tests
- **Test-Driven**: Write tests before implementation when possible

### Critical Requirements

#### 1. NO Double-Centering in CKA (Phase 2)
```python
# CORRECT: Use raw activations
K = X @ X.T  # No centering!
L = Y @ Y.T  # No centering!

# WRONG: Don't pre-center
X_centered = X - X.mean(dim=0)  # DON'T DO THIS!
```

#### 2. Mathematical Correctness
- All algorithms must match reference implementations
- Validate mathematical properties in tests
- Use appropriate numerical precision

#### 3. Memory Efficiency
- Target: <3GB for ResNet50 analysis
- Use sparse operations where possible
- Implement adaptive sampling

## 📋 Contribution Types

### Bug Reports

Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Minimal code example

### Feature Requests

Use the feature request template and include:
- Clear description of the feature
- Use case and motivation
- Implementation suggestions
- Compatibility considerations

### Code Contributions

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement Changes**
   - Follow code standards
   - Add comprehensive tests
   - Update documentation

3. **Run Quality Checks**
   ```bash
   make format
   make lint
   make type-check
   make test
   ```

4. **Submit Pull Request**
   - Use the PR template
   - Include clear description
   - Reference related issues

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Individual component tests
├── integration/    # Component interaction tests
└── performance/    # Benchmarking and profiling tests
```

### Test Naming

```python
def test_cka_no_double_centering():
    """Test CKA uses raw activations without pre-centering."""
    
def test_subspace_tracking_eigenvalue_crossings():
    """Test subspace tracker handles eigenvalue crossings correctly."""
```

### Test Categories

Use pytest markers:
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.gpu`: GPU-required tests
- `@pytest.mark.phase1`: Phase-specific tests

### Running Tests

```bash
# All tests
make test

# Specific categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v

# With coverage
make test-coverage

# Specific phase
pytest tests/ -m phase1
```

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use numpy-style docstrings
- **Type Hints**: All functions and methods
- **Comments**: Explain complex logic, not obvious code

### Example Docstring

```python
def compute_cka_matrix(
    activations: Dict[str, torch.Tensor],
    method: str = "debiased"
) -> torch.Tensor:
    """Compute CKA similarity matrix between activations.
    
    Parameters
    ----------
    activations : Dict[str, torch.Tensor]
        Dictionary mapping layer names to activation tensors
    method : str, default="debiased"
        CKA computation method ("debiased" or "standard")
        
    Returns
    -------
    torch.Tensor
        CKA similarity matrix of shape (n_layers, n_layers)
        
    Raises
    ------
    ValidationError
        If activations have incompatible shapes
    ComputationError
        If CKA computation fails
        
    Examples
    --------
    >>> activations = {"layer1": torch.randn(100, 512), 
    ...                "layer2": torch.randn(100, 256)}
    >>> cka_matrix = compute_cka_matrix(activations)
    >>> cka_matrix.shape
    torch.Size([2, 2])
    """
```

## 🔍 Code Review Process

### For Contributors

1. **Self-Review**: Check your code before submitting
2. **Clear Description**: Explain changes and motivation
3. **Tests**: Ensure all tests pass
4. **Responsive**: Address feedback promptly

### For Reviewers

1. **Constructive**: Provide helpful feedback
2. **Thorough**: Check code quality and tests
3. **Timely**: Review within 48 hours when possible
4. **Respectful**: Maintain professional tone

## 🚨 Common Pitfalls

### Mathematical Errors
- ❌ Pre-centering data for CKA computation
- ❌ Index-based eigenvalue tracking
- ❌ Ignoring numerical stability

### Performance Issues
- ❌ Dense matrices for large networks
- ❌ Loading entire datasets into memory
- ❌ Inefficient GPU memory usage

### Code Quality Issues
- ❌ Missing tests for new features
- ❌ Hardcoded values and magic numbers
- ❌ Poor error handling

## 📊 Performance Standards

### Memory Usage
- CPU: Monitor with `psutil`
- GPU: Monitor with `torch.cuda.memory_allocated()`
- Target: <3GB for ResNet50

### Speed Requirements
- Full analysis: <5 minutes
- CKA computation: <30 seconds
- Laplacian assembly: <1 minute

### Profiling Tools

```python
from neurosheaf.utils.profiling import profile_memory, profile_time

@profile_memory(memory_threshold_mb=1000.0)
@profile_time(time_threshold_seconds=30.0)
def your_function():
    # Implementation
    pass
```

## 🎨 Visualization Guidelines

### Plot Standards
- **Consistent styling**: Use project color scheme
- **Clear labels**: Axes, titles, legends
- **Scalable**: Support different data sizes
- **Interactive**: Use Plotly for complex visualizations

### Dashboard Requirements
- **Responsive**: Works on different screen sizes
- **Fast**: <1 second load time
- **Intuitive**: Clear navigation and controls

## 🔒 Security Guidelines

### Data Handling
- Never log sensitive information
- Validate all inputs
- Use secure temporary files
- Clean up resources properly

### Dependencies
- Keep dependencies up to date
- Use `pip audit` for security checks
- Minimize external dependencies

## 📅 Release Process

### Version Numbering
- Follow semantic versioning (SemVer)
- Phase releases: 0.1.0, 0.2.0, etc.
- Patch releases: 0.1.1, 0.1.2, etc.

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks pass
- [ ] Security audit complete
- [ ] Changelog updated

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Report inappropriate behavior

### Communication
- Use clear, professional language
- Stay on topic in discussions
- Search before asking questions
- Be patient with beginners

## 📞 Getting Help

### Documentation
- README.md: Project overview
- guidelines.md: Implementation guidelines
- plan/: Phase-specific plans

### Support Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and discussions
- Email: contact@neurosheaf.org

### Mentorship
New contributors are welcome! We provide:
- Code review and feedback
- Guidance on best practices
- Help with setup and development
- Pairing sessions for complex features

## 🏆 Recognition

### Contributors
All contributors are recognized in:
- README.md acknowledgments
- Release notes
- Documentation credits

### Types of Contributions
- Code contributions
- Bug reports and testing
- Documentation improvements
- Performance optimizations
- Community support

## 📈 Roadmap

### Current Priorities
1. Phase 2: CKA Implementation
2. Performance optimization
3. Documentation improvements
4. Community building

### Future Plans
- Integration with popular ML frameworks
- Cloud deployment options
- Extended architecture support
- Advanced visualization features

---

Thank you for contributing to Neurosheaf! Your contributions help advance the understanding of neural network similarity and topological analysis.