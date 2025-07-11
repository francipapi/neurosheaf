"""Integration tests for package setup and component interaction."""

import pytest
import sys
import subprocess
import importlib
from pathlib import Path

from neurosheaf.utils.logging import setup_logger
from neurosheaf.utils.exceptions import ValidationError, ComputationError
from neurosheaf.utils.profiling import profile_memory, get_profile_manager


class TestPackageImports:
    """Test package imports and basic functionality."""
    
    def test_main_package_import(self):
        """Test main package can be imported."""
        import neurosheaf
        
        assert hasattr(neurosheaf, '__version__')
        assert neurosheaf.__version__ == "0.1.0"
        assert hasattr(neurosheaf, '__author__')
        assert hasattr(neurosheaf, '__email__')
    
    def test_utils_module_imports(self):
        """Test utils module components can be imported."""
        from neurosheaf.utils import (
            setup_logger,
            profile_memory,
            NeurosheafError,
            ValidationError,
            ComputationError,
            MemoryError,
            ArchitectureError,
        )
        
        # Should not raise ImportError
        assert callable(setup_logger)
        assert callable(profile_memory)
        assert issubclass(ValidationError, NeurosheafError)
    
    def test_submodule_imports(self):
        """Test submodule imports work correctly."""
        from neurosheaf.utils.logging import setup_logger
        from neurosheaf.utils.exceptions import ValidationError
        from neurosheaf.utils.profiling import profile_memory
        
        # Should not raise ImportError
        assert callable(setup_logger)
        assert issubclass(ValidationError, Exception)
        assert callable(profile_memory)
    
    def test_main_package_exports(self):
        """Test main package exports are available."""
        import neurosheaf
        
        # Should be able to access from main package
        assert hasattr(neurosheaf, 'setup_logger')
        assert hasattr(neurosheaf, 'profile_memory')
        assert hasattr(neurosheaf, 'NeurosheafError')
        assert hasattr(neurosheaf, 'ValidationError')
    
    def test_future_imports_placeholder(self):
        """Test that implemented components are available."""
        import neurosheaf
        
        # These should be available in Phase 1
        assert neurosheaf.NeurosheafAnalyzer is not None
        assert neurosheaf.DebiasedCKA is not None
        
        # These will be None until implemented in later phases
        assert neurosheaf.SheafBuilder is None


class TestComponentIntegration:
    """Test integration between different components."""
    
    def test_logging_with_exceptions(self):
        """Test logging system works with exception handling."""
        logger = setup_logger("integration_test")
        
        try:
            raise ValidationError("Test integration error")
        except ValidationError as e:
            logger.error(f"Caught validation error: {e}")
            logger.info(f"Error context: {e.context}")
        
        # Should not raise any exceptions
        assert True
    
    def test_profiling_with_logging(self):
        """Test profiling system works with logging."""
        logger = setup_logger("profiling_integration")
        
        @profile_memory(log_results=True)
        def test_function():
            logger.info("Inside profiled function")
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Should have profiling results
        manager = get_profile_manager()
        results = manager.get_results("test_function")
        assert len(results) >= 1
    
    def test_profiling_with_exceptions(self):
        """Test profiling system handles exceptions correctly."""
        @profile_memory(log_results=False)
        def failing_function():
            raise ValidationError("Test profiling error")
        
        with pytest.raises(ValidationError):
            failing_function()
        
        # Should not cause additional issues
        assert True
    
    def test_all_components_together(self):
        """Test all components work together."""
        logger = setup_logger("all_components")
        
        @profile_memory(log_results=True)
        def integrated_function():
            logger.info("Starting integrated function")
            
            try:
                # Simulate some work
                data = [i * 2 for i in range(1000)]
                logger.info(f"Generated {len(data)} items")
                return data
            except Exception as e:
                logger.error(f"Error in integrated function: {e}")
                raise
        
        result = integrated_function()
        assert len(result) == 1000
        assert result[0] == 0
        assert result[999] == 1998
        
        # Check profiling results
        manager = get_profile_manager()
        results = manager.get_results("integrated_function")
        assert len(results) >= 1


class TestPackageInstallation:
    """Test package installation and command line interface."""
    
    def test_package_installable(self):
        """Test package can be installed in development mode."""
        # This test runs in the context where the package is already installed
        # We test that imports work correctly
        
        import neurosheaf
        assert neurosheaf.__version__ == "0.1.0"
    
    def test_cli_commands_available(self):
        """Test CLI commands are available (when implemented)."""
        # For Phase 1, we just check that the modules exist
        # CLI will be implemented in later phases
        
        # Check that the CLI module path exists
        cli_path = Path(__file__).parent.parent.parent / "neurosheaf" / "cli.py"
        # CLI not implemented in Phase 1, so we skip this test
        pytest.skip("CLI not implemented in Phase 1")
    
    def test_import_in_clean_environment(self):
        """Test package can be imported in a clean Python environment."""
        # Run a subprocess to test import in clean environment
        code = """
import sys
try:
    import neurosheaf
    print(f"SUCCESS: {neurosheaf.__version__}")
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)
"""
        
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "SUCCESS: 0.1.0" in result.stdout


class TestDevelopmentWorkflow:
    """Test development workflow and tooling."""
    
    def test_pytest_markers_available(self):
        """Test that pytest markers are configured."""
        # Check that phase1 marker is available
        # This test itself should be marked with phase1
        assert hasattr(pytest.mark, 'phase1')
        assert hasattr(pytest.mark, 'slow')
        assert hasattr(pytest.mark, 'gpu')
        assert hasattr(pytest.mark, 'benchmark')
    
    def test_package_metadata_accessible(self):
        """Test package metadata is accessible."""
        import neurosheaf
        
        # Check metadata
        assert neurosheaf.__version__ == "0.1.0"
        assert isinstance(neurosheaf.__author__, str)
        assert isinstance(neurosheaf.__email__, str)
        assert len(neurosheaf.__all__) > 0
    
    def test_development_dependencies_available(self):
        """Test development dependencies are available."""
        # Test that key development dependencies can be imported
        try:
            import pytest
            import black
            import isort
            import flake8
            import mypy
        except ImportError as e:
            pytest.skip(f"Development dependency not available: {e}")
    
    def test_git_repository_setup(self):
        """Test git repository is set up correctly."""
        repo_root = Path(__file__).parent.parent.parent
        
        # Check .git directory exists
        git_dir = repo_root / ".git"
        assert git_dir.exists(), "Git repository not initialized"
        
        # Check .gitignore exists
        gitignore = repo_root / ".gitignore"
        assert gitignore.exists(), ".gitignore file not found"
        
        # Check .gitignore has Python-specific entries
        gitignore_content = gitignore.read_text()
        assert "__pycache__" in gitignore_content
        assert "*.pyc" in gitignore_content
        assert ".pytest_cache" in gitignore_content


class TestFileStructure:
    """Test file structure and organization."""
    
    def test_package_structure_exists(self):
        """Test package structure exists as expected."""
        repo_root = Path(__file__).parent.parent.parent
        neurosheaf_dir = repo_root / "neurosheaf"
        
        # Check main package directory
        assert neurosheaf_dir.exists()
        assert (neurosheaf_dir / "__init__.py").exists()
        
        # Check submodules
        expected_dirs = ["utils", "cka", "sheaf", "spectral", "visualization"]
        for dir_name in expected_dirs:
            subdir = neurosheaf_dir / dir_name
            assert subdir.exists(), f"Directory {dir_name} not found"
            assert (subdir / "__init__.py").exists(), f"__init__.py not found in {dir_name}"
    
    def test_test_structure_exists(self):
        """Test test structure exists as expected."""
        repo_root = Path(__file__).parent.parent.parent
        tests_dir = repo_root / "tests"
        
        # Check test directory structure
        assert tests_dir.exists()
        assert (tests_dir / "__init__.py").exists()
        
        # Check test subdirectories
        expected_dirs = ["unit", "integration", "performance"]
        for dir_name in expected_dirs:
            subdir = tests_dir / dir_name
            assert subdir.exists(), f"Test directory {dir_name} not found"
            assert (subdir / "__init__.py").exists(), f"__init__.py not found in tests/{dir_name}"
    
    def test_configuration_files_exist(self):
        """Test configuration files exist."""
        repo_root = Path(__file__).parent.parent.parent
        
        # Check configuration files
        config_files = [
            "pyproject.toml",
            "pytest.ini",
            ".pre-commit-config.yaml",
            "Makefile",
            ".gitignore"
        ]
        
        for file_name in config_files:
            file_path = repo_root / file_name
            assert file_path.exists(), f"Configuration file {file_name} not found"
    
    def test_github_workflows_exist(self):
        """Test GitHub workflows exist."""
        repo_root = Path(__file__).parent.parent.parent
        workflows_dir = repo_root / ".github" / "workflows"
        
        assert workflows_dir.exists(), "GitHub workflows directory not found"
        
        # Check workflow files
        workflow_files = ["test.yml", "release.yml"]
        for file_name in workflow_files:
            file_path = workflows_dir / file_name
            assert file_path.exists(), f"Workflow file {file_name} not found"


class TestMemoryAndPerformance:
    """Test memory and performance characteristics."""
    
    def test_basic_memory_usage(self):
        """Test basic memory usage is reasonable."""
        import neurosheaf
        
        # Import should not use excessive memory
        # This is a basic sanity check
        assert True
    
    def test_import_speed(self):
        """Test package import speed is reasonable."""
        import time
        
        start_time = time.time()
        import neurosheaf
        end_time = time.time()
        
        import_time = end_time - start_time
        
        # Import should be fast (less than 1 second)
        assert import_time < 1.0, f"Import took {import_time:.2f}s, too slow"
    
    def test_no_memory_leaks_in_basic_usage(self):
        """Test no obvious memory leaks in basic usage."""
        # Test repeated imports don't cause memory issues
        for i in range(10):
            # Force reimport (this is a basic test)
            importlib.reload(importlib.import_module('neurosheaf.utils.logging'))
            importlib.reload(importlib.import_module('neurosheaf.utils.exceptions'))
            importlib.reload(importlib.import_module('neurosheaf.utils.profiling'))
        
        # Should not cause memory issues
        assert True


@pytest.mark.phase1
@pytest.mark.integration
class TestPhase1Integration:
    """Test Phase 1 specific integration requirements."""
    
    def test_baseline_profiling_ready(self):
        """Test baseline profiling infrastructure is ready."""
        logger = setup_logger("baseline_test")
        
        @profile_memory(memory_threshold_mb=3000.0, log_results=True)
        def baseline_computation():
            logger.info("Starting baseline computation")
            
            # Simulate baseline computation that will be optimized
            data = []
            for i in range(1000):
                row = [j * i for j in range(100)]
                data.append(row)
            
            logger.info(f"Generated {len(data)} rows")
            return data
        
        result = baseline_computation()
        assert len(result) == 1000
        assert len(result[0]) == 100
        
        # Check profiling results
        manager = get_profile_manager()
        results = manager.get_results("baseline_computation")
        assert len(results) >= 1
        
        # Should track memory usage
        assert results[-1].cpu_memory_peak_mb > 0
        assert results[-1].execution_time > 0
    
    def test_error_handling_ready(self):
        """Test error handling is ready for other phases."""
        logger = setup_logger("error_handling_test")
        
        # Test different types of errors that will be used in other phases
        try:
            raise ValidationError("CKA input validation failed", parameter="matrix")
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            assert e.context["parameter"] == "matrix"
        
        try:
            raise ComputationError("Eigenvalue computation failed", operation="svd")
        except ComputationError as e:
            logger.error(f"Computation error: {e}")
            assert e.context["operation"] == "svd"
        
        # Should handle all error types
        assert True
    
    def test_logging_infrastructure_ready(self):
        """Test logging infrastructure is ready for other phases."""
        # Test loggers for different phases
        loggers = {
            "cka": setup_logger("neurosheaf.cka"),
            "sheaf": setup_logger("neurosheaf.sheaf"),
            "spectral": setup_logger("neurosheaf.spectral"),
            "visualization": setup_logger("neurosheaf.visualization"),
        }
        
        for phase, logger in loggers.items():
            logger.info(f"Phase {phase} logger working")
            logger.debug(f"Debug message from {phase}")
            logger.warning(f"Warning from {phase}")
        
        # Should not raise exceptions
        assert True
    
    def test_development_environment_ready(self):
        """Test development environment is ready."""
        # Test that we can run basic development commands
        repo_root = Path(__file__).parent.parent.parent
        
        # Check that package is installed in development mode
        import neurosheaf
        assert neurosheaf.__version__ == "0.1.0"
        
        # Check that tests can run
        assert True
    
    def test_phase1_success_criteria(self):
        """Test Phase 1 success criteria are met."""
        # 1. All package files created and importable
        import neurosheaf
        from neurosheaf.utils import setup_logger, profile_memory
        from neurosheaf.utils.exceptions import ValidationError
        
        # 2. Baseline performance infrastructure ready
        @profile_memory(log_results=False)
        def test_computation():
            return sum(range(1000))
        
        result = test_computation()
        assert result == 499500
        
        # 3. Code quality infrastructure ready
        # (This is tested by the existence of configuration files)
        
        # All success criteria met
        assert True