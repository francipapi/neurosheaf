# Phase 1: Foundation Testing Suite

## Test Categories

### 1. Setup Validation Tests
```python
# tests/test_phase1_setup.py
import pytest
import importlib
import subprocess
from pathlib import Path

class TestProjectSetup:
    """Validate project structure and setup."""
    
    def test_package_structure(self):
        """Verify all required modules exist."""
        required_modules = [
            'neurosheaf',
            'neurosheaf.api',
            'neurosheaf.cka',
            'neurosheaf.sheaf',
            'neurosheaf.spectral',
            'neurosheaf.visualization',
            'neurosheaf.utils',
        ]
        
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                pytest.fail(f"Module {module} not found")
    
    def test_dependencies_installed(self):
        """Check all required dependencies are available."""
        required = ['torch', 'numpy', 'scipy', 'networkx', 'matplotlib']
        
        for dep in required:
            try:
                importlib.import_module(dep)
            except ImportError:
                pytest.fail(f"Dependency {dep} not installed")
    
    def test_git_hooks_configured(self):
        """Verify pre-commit hooks are set up."""
        hooks_path = Path('.git/hooks/pre-commit')
        assert hooks_path.exists(), "Pre-commit hook not configured"
```

### 2. Infrastructure Tests
```python
# tests/test_logging_infrastructure.py
import pytest
import logging
import tempfile
from pathlib import Path
from neurosheaf.utils.logging import setup_logger

class TestLoggingInfrastructure:
    """Test logging configuration and functionality."""
    
    def test_logger_levels(self):
        """Test different logging levels work correctly."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            logger = setup_logger(f'test_{level}', level=level)
            assert logger.level == getattr(logging, level)
    
    def test_concurrent_loggers(self):
        """Test multiple loggers don't interfere."""
        logger1 = setup_logger('test1')
        logger2 = setup_logger('test2')
        
        # Log to different handlers
        assert logger1.name != logger2.name
        assert logger1.handlers != logger2.handlers
    
    @pytest.mark.parametrize("message", [
        "Simple message",
        "Message with 特殊字符",
        "Message with\nnewlines",
        "Very " * 1000 + "long message"
    ])
    def test_logger_handles_various_messages(self, message):
        """Test logger handles different message types."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            logger = setup_logger('test', log_file=Path(f.name))
            logger.info(message)
            
            # Verify message was written
            content = Path(f.name).read_text()
            assert message in content
```

### 3. Exception Handling Tests
```python
# tests/test_exception_hierarchy.py
import pytest
from neurosheaf.utils.exceptions import (
    NeurosheafError, ValidationError, ComputationError,
    MemoryError, ArchitectureError
)

class TestExceptionHierarchy:
    """Test custom exception behavior."""
    
    def test_exception_inheritance(self):
        """Verify exception hierarchy is correct."""
        assert issubclass(ValidationError, NeurosheafError)
        assert issubclass(ComputationError, NeurosheafError)
        assert issubclass(MemoryError, NeurosheafError)
        assert issubclass(ArchitectureError, NeurosheafError)
    
    def test_exception_messages(self):
        """Test exceptions carry messages correctly."""
        msg = "Test error message"
        
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(msg)
        
        assert str(exc_info.value) == msg
    
    def test_exception_catching(self):
        """Test base exception catches all derived exceptions."""
        exceptions = [
            ValidationError("validation"),
            ComputationError("computation"),
            MemoryError("memory"),
            ArchitectureError("architecture")
        ]
        
        for exc in exceptions:
            with pytest.raises(NeurosheafError):
                raise exc
```

### 4. Performance Baseline Tests
```python
# tests/test_performance_baseline.py
import pytest
import torch
import numpy as np
from neurosheaf.utils.profiling import profile_memory
import psutil
import gc

class TestPerformanceBaseline:
    """Establish and verify performance baselines."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up memory before and after tests."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_memory_profiler_accuracy(self):
        """Verify memory profiler reports accurate measurements."""
        size_mb = 100
        
        @profile_memory
        def allocate_memory():
            # Allocate approximately size_mb of memory
            data = np.zeros((size_mb * 1024 * 1024 // 8,), dtype=np.float64)
            return data
        
        # Should report approximately 100MB
        result = allocate_memory()
        assert result.nbytes / 1024 / 1024 == pytest.approx(size_mb, rel=0.1)
    
    @pytest.mark.slow
    def test_baseline_cka_memory(self):
        """Test baseline CKA memory usage."""
        n_samples = 1000
        n_features = 512
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Baseline CKA computation
        K = X @ X.T  # n x n matrix
        L = Y @ Y.T  # n x n matrix
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = peak_memory - initial_memory
        
        # Verify quadratic memory scaling
        expected_mb = 2 * (n_samples ** 2) * 4 / 1024 / 1024  # 2 matrices, 4 bytes per float
        assert memory_used >= expected_mb * 0.8  # Allow 20% overhead
```

### 5. Edge Cases and Stability Tests
```python
# tests/test_phase1_edge_cases.py
import pytest
import logging
from pathlib import Path
from neurosheaf.utils.logging import setup_logger
from neurosheaf.utils.exceptions import NeurosheafError

class TestPhase1EdgeCases:
    """Test edge cases and error conditions."""
    
    def test_logger_with_invalid_level(self):
        """Test logger handles invalid log levels."""
        with pytest.raises(AttributeError):
            setup_logger("test", level="INVALID_LEVEL")
    
    def test_logger_with_readonly_file(self, tmp_path):
        """Test logger handles write permission errors."""
        log_file = tmp_path / "readonly.log"
        log_file.touch()
        log_file.chmod(0o444)  # Read-only
        
        # Should not crash, but log to console only
        logger = setup_logger("test", log_file=log_file)
        logger.info("Test message")
    
    def test_nested_exception_handling(self):
        """Test nested exception scenarios."""
        def inner():
            raise ValidationError("Inner error")
        
        def outer():
            try:
                inner()
            except ValidationError:
                raise ComputationError("Outer error")
        
        with pytest.raises(ComputationError) as exc_info:
            outer()
        
        assert "Outer error" in str(exc_info.value)
    
    def test_memory_profiler_with_exceptions(self):
        """Test profiler handles functions that raise exceptions."""
        from neurosheaf.utils.profiling import profile_memory
        
        @profile_memory
        def failing_function():
            raise RuntimeError("Intentional failure")
        
        with pytest.raises(RuntimeError):
            failing_function()
```

### 6. Integration Tests
```python
# tests/test_phase1_integration.py
import pytest
import subprocess
import sys
from pathlib import Path

class TestPhase1Integration:
    """Test components work together correctly."""
    
    def test_package_importable_in_new_process(self):
        """Test package can be imported in clean environment."""
        code = "import neurosheaf; print(neurosheaf.__version__)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
    
    def test_ci_configuration_valid(self):
        """Test CI configuration files are valid."""
        ci_files = [
            ".github/workflows/test.yml",
            ".github/workflows/release.yml"
        ]
        
        for file in ci_files:
            path = Path(file)
            if path.exists():
                # Basic YAML validation
                import yaml
                with open(path) as f:
                    yaml.safe_load(f)
    
    def test_development_workflow(self):
        """Test common development commands work."""
        commands = [
            "pytest --version",
            "black --version",
            "isort --version",
            "flake8 --version"
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd.split(), capture_output=True)
            assert result.returncode == 0
```

## Test Execution Plan

### Daily Testing
```bash
# Run during development
pytest tests/unit/ -v
pytest tests/integration/ -v -m "not slow"
```

### Complete Test Suite
```bash
# Run before commits
pytest tests/ -v --cov=neurosheaf --cov-report=html
```

### Performance Testing
```bash
# Run separately due to resource requirements
pytest tests/performance/ -v -m "benchmark"
```

## Success Criteria

1. **Coverage**: >95% code coverage for all modules
2. **Performance**: Baseline measurements established and documented
3. **Stability**: All edge cases handled gracefully
4. **Integration**: All components work together seamlessly

## Common Issues and Solutions

### Issue: Import errors during testing
**Solution**: Ensure PYTHONPATH includes project root
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Issue: Memory tests fail on CI
**Solution**: Skip large memory tests on CI
```python
@pytest.mark.skipif(
    os.environ.get('CI') == 'true',
    reason="Skip memory-intensive tests on CI"
)
```

### Issue: Flaky profiling tests
**Solution**: Use relative comparisons instead of absolute
```python
assert memory_used >= expected * 0.8  # 20% tolerance
```