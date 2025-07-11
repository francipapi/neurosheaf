"""CKA (Centered Kernel Alignment) implementation module.

This module contains the debiased CKA implementation with NO double-centering,
baseline implementation for profiling, memory-efficient sampling strategies,
and pairwise computation utilities.

Phase 1 & Phase 2 Week 3-4 implementation complete.
"""

# Core CKA implementations
from .debiased import DebiasedCKA
from .baseline import BaselineCKA

# Phase 2 Week 3 implementations
from .sampling import AdaptiveSampler
from .pairwise import PairwiseCKA

# Phase 2 Week 4 implementations
from .nystrom import NystromCKA

__all__ = [
    "DebiasedCKA",
    "BaselineCKA",
    "AdaptiveSampler", 
    "PairwiseCKA",
    "NystromCKA",
]