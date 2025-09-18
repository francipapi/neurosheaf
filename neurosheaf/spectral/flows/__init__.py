"""Flow-based spectral analysis methods.

This module provides flow-based spectral analysis methods for comparing different
GW sheaves using distributional summaries rather than threshold-dependent persistence.

Available flows:
- α-flow: Non-trivial order-free analysis using baseline/residual decomposition
- t-flow: Multi-scale diffusion analysis using heat kernel summaries
"""

from .alpha_flow import (
    AlphaGroupingPolicy,
    AlphaFlowBuild,
    AlphaFlowBuilder
)

from .diffusion_flow import (
    DiffusionSpec,
    DiffusionSummaries,
    DiffusionFlowAnalyzer
)

__all__ = [
    'AlphaGroupingPolicy',
    'AlphaFlowBuild', 
    'AlphaFlowBuilder',
    'DiffusionSpec',
    'DiffusionSummaries',
    'DiffusionFlowAnalyzer'
]