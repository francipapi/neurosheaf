"""Shared test fixtures for Neurosheaf test suite.

This module provides common fixtures and utilities used across all tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import logging
import os
import sys
from unittest.mock import Mock, patch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import neurosheaf components
from neurosheaf.utils.logging import setup_logger, shutdown_logging
from neurosheaf.utils.profiling import get_profile_manager


@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Set up test environment."""
    # Set test environment variables
    os.environ["NEUROSHEAF_TEST_MODE"] = "true"
    os.environ["NEUROSHEAF_LOG_LEVEL"] = "DEBUG"
    
    # Disable GPU for tests unless explicitly requested
    if "NEUROSHEAF_TEST_GPU" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    yield
    
    # Clean up
    if "NEUROSHEAF_TEST_MODE" in os.environ:
        del os.environ["NEUROSHEAF_TEST_MODE"]


@pytest.fixture(scope="function", autouse=True)
def cleanup_logging():
    """Clean up logging after each test."""
    yield
    shutdown_logging()


@pytest.fixture(scope="function", autouse=True)
def cleanup_profiling():
    """Clean up profiling after each test."""
    yield
    get_profile_manager().clear_results()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for testing."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.touch()
    yield temp_file


@pytest.fixture
def logger():
    """Create a test logger."""
    logger = setup_logger("test", level="DEBUG")
    yield logger
    shutdown_logging()


@pytest.fixture
def mock_torch():
    """Mock torch for tests that don't require actual PyTorch."""
    with patch('neurosheaf.utils.profiling.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.memory_allocated.return_value = 0
        mock_torch.cuda.max_memory_allocated.return_value = 0
        yield mock_torch


@pytest.fixture
def mock_torch_with_gpu():
    """Mock torch with GPU support."""
    with patch('neurosheaf.utils.profiling.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100MB
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 200  # 200MB
        mock_torch.cuda.reset_peak_memory_stats.return_value = None
        mock_torch.cuda.empty_cache.return_value = None
        yield mock_torch


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "small_matrix": [[1, 2], [3, 4]],
        "large_matrix": [[i * j for j in range(100)] for i in range(100)],
        "test_string": "test_data",
        "test_dict": {"key1": "value1", "key2": "value2"},
        "test_list": [1, 2, 3, 4, 5],
    }


@pytest.fixture
def memory_threshold():
    """Default memory threshold for testing."""
    return 100.0  # 100MB


@pytest.fixture
def time_threshold():
    """Default time threshold for testing."""
    return 5.0  # 5 seconds


class TestCase:
    """Base test case class with common utilities."""
    
    def assert_memory_usage(self, usage_mb, threshold_mb):
        """Assert memory usage is within threshold."""
        assert usage_mb <= threshold_mb, f"Memory usage {usage_mb:.2f}MB exceeds threshold {threshold_mb:.2f}MB"
    
    def assert_execution_time(self, time_seconds, threshold_seconds):
        """Assert execution time is within threshold."""
        assert time_seconds <= threshold_seconds, f"Execution time {time_seconds:.2f}s exceeds threshold {threshold_seconds:.2f}s"
    
    def assert_file_exists(self, file_path):
        """Assert file exists."""
        assert Path(file_path).exists(), f"File {file_path} does not exist"
    
    def assert_directory_exists(self, dir_path):
        """Assert directory exists."""
        assert Path(dir_path).is_dir(), f"Directory {dir_path} does not exist"


@pytest.fixture
def test_case():
    """Provide TestCase instance for tests."""
    return TestCase()


# Skip decorators for conditional tests
skip_if_no_torch = pytest.mark.skipif(
    not pytest.importorskip("torch", minversion="1.0"),
    reason="PyTorch not available"
)

skip_if_no_gpu = pytest.mark.skipif(
    not (pytest.importorskip("torch", minversion="1.0") and 
         pytest.importorskip("torch").cuda.is_available()),
    reason="CUDA not available"
)

skip_if_no_scipy = pytest.mark.skipif(
    not pytest.importorskip("scipy", minversion="1.0"),
    reason="SciPy not available"
)

skip_if_no_networkx = pytest.mark.skipif(
    not pytest.importorskip("networkx", minversion="2.0"),
    reason="NetworkX not available"
)


# Custom pytest plugins
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", 
        "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers",
        "benchmark: marks tests as benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    if config.getoption("--runxfail"):
        # Remove skip markers when running xfail tests
        for item in items:
            if "skip" in item.keywords:
                item.keywords.pop("skip")


def pytest_runtest_setup(item):
    """Setup for each test."""
    # Add test name to environment for debugging
    os.environ["PYTEST_CURRENT_TEST"] = item.nodeid


def pytest_runtest_teardown(item):
    """Teardown for each test."""
    # Clean up test environment variable
    if "PYTEST_CURRENT_TEST" in os.environ:
        del os.environ["PYTEST_CURRENT_TEST"]


# Custom assertions
def assert_tensor_equal(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert tensors are equal within tolerance."""
    try:
        import torch
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    except ImportError:
        # Fallback to numpy comparison
        import numpy as np
        assert np.allclose(tensor1, tensor2, rtol=rtol, atol=atol)


def assert_matrix_shape(matrix, expected_shape):
    """Assert matrix has expected shape."""
    actual_shape = getattr(matrix, 'shape', None)
    if actual_shape is None:
        actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"


# Register custom assertions
pytest.assert_tensor_equal = assert_tensor_equal
pytest.assert_matrix_shape = assert_matrix_shape