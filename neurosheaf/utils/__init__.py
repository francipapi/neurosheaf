"""Utilities for the Neurosheaf package.

This module contains common utilities used throughout the Neurosheaf framework:
- Logging infrastructure
- Custom exception hierarchy
- Performance profiling tools
- Memory management utilities
- Validation helpers
- Model loading utilities
- Persistence diagram distance metrics
"""

from .logging import setup_logger
from .exceptions import (
    NeurosheafError,
    ValidationError,
    ComputationError,
    MemoryError,
    ArchitectureError,
)
from .profiling import profile_memory
from .simple_model_loader import (
    load_model,
    save_model,
    list_model_info,
    validate_model_compatibility,
    get_model_summary
)
from .persistence_distances import (
    wasserstein_distance,
    bottleneck_distance,
    sliced_wasserstein_distance,
    compute_pairwise_distances,
    preprocess_diagram,
    add_diagonal_points,
    persistence_fisher_distance,
    extract_persistence_diagram_array,
)
from .dtw_similarity import (
    FiltrationDTW,
    create_filtration_dtw_comparator,
    quick_dtw_comparison,
)

__all__ = [
    "setup_logger",
    "profile_memory",
    "NeurosheafError",
    "ValidationError",
    "ComputationError",
    "MemoryError",
    "ArchitectureError",
    "load_model",
    "save_model",
    "list_model_info",
    "validate_model_compatibility",
    "get_model_summary",
    "wasserstein_distance",
    "bottleneck_distance",
    "sliced_wasserstein_distance",
    "compute_pairwise_distances",
    "preprocess_diagram",
    "add_diagonal_points",
    "persistence_fisher_distance",
    "extract_persistence_diagram_array",
    "FiltrationDTW",
    "create_filtration_dtw_comparator",
    "quick_dtw_comparison",
]