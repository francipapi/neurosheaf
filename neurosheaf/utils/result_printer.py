"""Pretty printing utilities for neurosheaf analysis results.

This module provides formatted output for various analysis results,
making them human-readable for debugging and reporting.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from ..spectral.persistent import AlphaFlowResult, AlphaFlowPoint


def print_alpha_flow_results(
    result: AlphaFlowResult,
    show_eigenvalues: bool = True,
    show_moments: bool = True,
    show_metadata: bool = True,
    eigenvalue_limit: int = 15,
    moment_precision: int = 3,
    eigenvalue_precision: int = 6
) -> str:
    """Print formatted α-flow analysis results.
    
    Args:
        result: AlphaFlowResult from analyze_alpha_flow
        show_eigenvalues: Whether to display eigenvalue tables
        show_moments: Whether to display Hutchinson moments
        show_metadata: Whether to display metadata sections
        eigenvalue_limit: Maximum eigenvalues to display per α
        moment_precision: Decimal places for moment values
        eigenvalue_precision: Decimal places for eigenvalues
        
    Returns:
        Formatted string representation of results
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("α-FLOW ANALYSIS RESULTS")
    lines.append("=" * 80)
    
    # Summary section
    lines.append("\n📊 ANALYSIS SUMMARY")
    lines.append("-" * 40)
    lines.append(f"• α values analyzed: {len(result.points)}")
    lines.append(f"• Matrix size: {result.build_meta.get('matrix_size', 'N/A')}")
    lines.append(f"• Analysis time: {result.analysis_time:.2f}s")
    lines.append(f"• Eigenvalue method: {'CSR' if result.build_meta.get('eigen_use_csr', False) else 'LinearOperator'}")
    lines.append(f"• Mass mode: {result.build_meta.get('mass_mode', 'N/A')}")
    lines.append(f"• Precision: {result.build_meta.get('precision', 'N/A')}")
    
    if show_metadata:
        # Edge grouping metadata
        lines.append(f"\n🔗 EDGE GROUPING")
        lines.append("-" * 40)
        lines.append(f"• Base edges: {result.grouping_meta.get('n_base_edges', 'N/A')}")
        lines.append(f"• Residual edges: {result.grouping_meta.get('n_resid_edges', 'N/A')}")
        lines.append(f"• Grouping policy: {result.spec.grouping.kind} (param={result.spec.grouping.param})")
    
    # Eigenvalues section
    if show_eigenvalues and result.points:
        lines.append(f"\n⚡ EIGENVALUE EVOLUTION")
        lines.append("-" * 40)
        
        # Create eigenvalue table
        eigenvalue_table = _format_eigenvalue_table(
            result.points, eigenvalue_limit, eigenvalue_precision
        )
        lines.extend(eigenvalue_table)
    
    # Hutchinson moments section
    if show_moments and result.points:
        lines.append(f"\n🎯 HUTCHINSON MOMENTS")
        lines.append("-" * 40)
        
        # Get available moments from first point
        if result.points[0].moments:
            moment_table = _format_moments_table(
                result.points, moment_precision
            )
            lines.extend(moment_table)
        else:
            lines.append("No moment data available")
    
    # Normalized metrics section
    if result.points:
        lines.append(f"\n📈 NORMALIZED METRICS")
        lines.append("-" * 40)
        
        metrics_table = _format_metrics_table(result.points, moment_precision)
        lines.extend(metrics_table)
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def _format_eigenvalue_table(
    points: List[AlphaFlowPoint], 
    limit: int, 
    precision: int
) -> List[str]:
    """Format eigenvalue evolution table."""
    lines = []
    
    if not points or not points[0].eigenvalues.size:
        lines.append("No eigenvalue data available")
        return lines
    
    # Header
    alpha_header = "α".rjust(8)
    eig_headers = [f"λ_{i+1}".rjust(precision + 4) for i in range(min(limit, len(points[0].eigenvalues)))]
    header = alpha_header + " │ " + " ".join(eig_headers)
    lines.append(header)
    lines.append("─" * len(header))
    
    # Data rows
    for point in points:
        alpha_str = f"{point.alpha:8.3f}"
        
        if point.eigenvalues.size == 0:
            eig_strs = ["--"] * min(limit, 1)
        else:
            n_show = min(limit, len(point.eigenvalues))
            eig_strs = [
                _format_scientific(point.eigenvalues[i], precision + 4) 
                for i in range(n_show)
            ]
        
        row = alpha_str + " │ " + " ".join(eig_strs)
        lines.append(row)
    
    return lines


def _format_moments_table(
    points: List[AlphaFlowPoint], 
    precision: int
) -> List[str]:
    """Format Hutchinson moments table."""
    lines = []
    
    if not points or not points[0].moments:
        lines.append("No moment data available")
        return lines
    
    # Get moment orders
    moment_orders = sorted(points[0].moments.keys())
    
    # Header
    alpha_header = "α".rjust(8)
    moment_headers = [f"Tr(L^{k})".rjust(precision + 6) for k in moment_orders]
    header = alpha_header + " │ " + " ".join(moment_headers)
    lines.append(header)
    lines.append("─" * len(header))
    
    # Data rows
    for point in points:
        alpha_str = f"{point.alpha:8.3f}"
        
        moment_strs = []
        for k in moment_orders:
            if k in point.moments and not np.isnan(point.moments[k]):
                moment_strs.append(_format_scientific(point.moments[k], precision + 6))
            else:
                moment_strs.append("--".rjust(precision + 6))
        
        row = alpha_str + " │ " + " ".join(moment_strs)
        lines.append(row)
    
    return lines


def _format_metrics_table(
    points: List[AlphaFlowPoint], 
    precision: int
) -> List[str]:
    """Format normalized metrics table."""
    lines = []
    
    # Header
    alpha_header = "α".rjust(8)
    trace_header = "Tr/n".rjust(precision + 4)
    frob_header = "||L||²/n²".rjust(precision + 4)
    header = alpha_header + " │ " + trace_header + " " + frob_header
    lines.append(header)
    lines.append("─" * len(header))
    
    # Data rows
    for point in points:
        alpha_str = f"{point.alpha:8.3f}"
        
        if not np.isnan(point.trace_normalized):
            trace_str = _format_scientific(point.trace_normalized, precision + 4)
        else:
            trace_str = "--".rjust(precision + 4)
            
        if not np.isnan(point.frobenius_normalized):
            frob_str = _format_scientific(point.frobenius_normalized, precision + 4)
        else:
            frob_str = "--".rjust(precision + 4)
        
        row = alpha_str + " │ " + trace_str + " " + frob_str
        lines.append(row)
    
    return lines


def _format_scientific(value: float, width: int) -> str:
    """Format number in scientific notation with fixed width."""
    if np.isnan(value) or np.isinf(value):
        return "--".rjust(width)
    
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        # Use scientific notation
        formatted = f"{value:.2e}"
    else:
        # Use fixed-point notation
        if width >= 8:
            formatted = f"{value:.{max(0, width-6)}f}"
        else:
            formatted = f"{value:.2f}"
    
    return formatted.rjust(width)


# Add __str__ method integration
def _add_str_method_to_result():
    """Add __str__ method to AlphaFlowResult class."""
    def __str__(self):
        return print_alpha_flow_results(self)
    
    AlphaFlowResult.__str__ = __str__


# Auto-install the __str__ method when module is imported
_add_str_method_to_result()