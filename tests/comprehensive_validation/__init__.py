"""Comprehensive validation framework for Week 7 Laplacian assembly and Phase 3.

This package provides extensive testing capabilities for validating the mathematical
correctness, performance, and robustness of the Week 7 implementation.

Modules:
- test_mathematical_correctness: Rigorous mathematical property validation
- test_performance_benchmarks: Performance and scalability testing
- test_robustness_edge_cases: Edge case and error handling testing
- run_comprehensive_validation: Orchestrated validation runner

Usage:
    # Run quick validation
    python run_comprehensive_validation.py --quick
    
    # Run full validation
    python run_comprehensive_validation.py --full
    
    # Run specific test module
    python test_mathematical_correctness.py
    python test_performance_benchmarks.py  
    python test_robustness_edge_cases.py
"""

__version__ = "1.0.0"
__author__ = "Neurosheaf Development Team"

# Import main validation components
try:
    from .test_mathematical_correctness import MathematicalCorrectnessValidator
    from .test_performance_benchmarks import PerformanceBenchmarkSuite
    from .test_robustness_edge_cases import RobustnessTestSuite
    from .run_comprehensive_validation import ComprehensiveValidationRunner
    
    __all__ = [
        "MathematicalCorrectnessValidator",
        "PerformanceBenchmarkSuite", 
        "RobustnessTestSuite",
        "ComprehensiveValidationRunner"
    ]
    
except ImportError as e:
    # Handle import errors gracefully for CI/CD environments
    import warnings
    warnings.warn(f"Some comprehensive validation components not available: {e}")
    __all__ = []