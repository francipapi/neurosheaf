"""Intelligent edge rendering with restriction map visualization.

This module provides advanced edge visualization that encodes restriction
map properties, dimensional changes, and information flow.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from ...utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class EdgeVisualProperties:
    """Visual properties for an edge."""
    width: float  # Line width based on information flow
    color: str  # Color based on dimensional change
    opacity: float
    dash_pattern: Optional[str]  # solid, dash, dot, dashdot
    arrow_style: str  # normal, expanding, contracting, bidirectional
    curvature: float  # 0 for straight, positive for curved
    markers: List[Dict[str, Any]]  # Intermediate markers
    gradient_colors: Optional[List[str]]  # For gradient effects
    animation_speed: Optional[float]  # For animated edges


class IntelligentEdgeRenderer:
    """Enhanced edge rendering with restriction map information."""
    
    def __init__(self):
        """Initialize the intelligent edge renderer."""
        self._init_color_schemes()
        self._init_dash_patterns()
        self._init_arrow_styles()
        
    def _init_color_schemes(self):
        """Initialize color schemes for different edge properties."""
        self.dimensional_change_colors = {
            'preserve': '#2196F3',  # Blue - dimension preserved
            'expand': '#4CAF50',    # Green - dimension increased
            'contract': '#F44336',  # Red - dimension reduced
            'strong_expand': '#00C853',  # Bright green - large expansion
            'strong_contract': '#D50000',  # Bright red - large contraction
        }
        
        self.flow_strength_gradient = [
            '#E3F2FD',  # Very light blue - minimal flow
            '#90CAF9',  # Light blue
            '#42A5F5',  # Medium blue
            '#1E88E5',  # Strong blue
            '#1565C0',  # Very strong blue
            '#0D47A1',  # Maximum flow
        ]
        
        self.restriction_type_colors = {
            'identity': '#9E9E9E',      # Gray - identity map
            'projection': '#FF5722',    # Deep orange - projection
            'embedding': '#4CAF50',     # Green - embedding
            'orthogonal': '#3F51B5',    # Indigo - orthogonal transform
            'general': '#2196F3',       # Blue - general transform
        }
        
    def _init_dash_patterns(self):
        """Initialize dash patterns for different edge types."""
        self.dash_patterns = {
            'solid': None,
            'residual': 'dash',  # For residual connections
            'attention': 'dot',  # For attention connections
            'skip': 'dashdot',  # For skip connections
            'weak': '2,5',      # For weak connections
            'strong': None,     # Solid for strong connections
        }
        
    def _init_arrow_styles(self):
        """Initialize arrow styles for different transformations."""
        self.arrow_templates = {
            'normal': {
                'arrowhead': 2,
                'arrowsize': 1,
                'arrowwidth': 2,
                'arrowcolor': None,  # Inherit from line
            },
            'expanding': {
                'arrowhead': 2,
                'arrowsize': 1.5,
                'arrowwidth': 3,
                'arrowcolor': None,
            },
            'contracting': {
                'arrowhead': 4,  # Vee shape
                'arrowsize': 0.8,
                'arrowwidth': 2,
                'arrowcolor': None,
            },
            'bidirectional': {
                'arrowhead': 2,
                'arrowsize': 1,
                'arrowwidth': 2,
                'arrowcolor': None,
                'startarrowhead': 2,
            },
        }
        
    def analyze_restriction_map(self, restriction_map: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze a restriction map to determine its properties.
        
        Args:
            restriction_map: The restriction map tensor
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {}
        
        # Dimensional analysis
        target_dim, source_dim = restriction_map.shape
        analysis['source_dim'] = source_dim
        analysis['target_dim'] = target_dim
        analysis['dimensional_ratio'] = target_dim / source_dim if source_dim > 0 else 0
        
        # Dimensional change type
        if target_dim == source_dim:
            analysis['dimensional_change'] = 'preserve'
        elif target_dim > source_dim:
            ratio = target_dim / source_dim
            analysis['dimensional_change'] = 'strong_expand' if ratio > 2 else 'expand'
        else:
            ratio = source_dim / target_dim
            analysis['dimensional_change'] = 'strong_contract' if ratio > 2 else 'contract'
            
        # Information flow strength (Frobenius norm)
        analysis['frobenius_norm'] = torch.norm(restriction_map, 'fro').item()
        
        # Singular value decomposition for deeper analysis
        try:
            U, S, Vt = torch.linalg.svd(restriction_map, full_matrices=False)
            analysis['singular_values'] = S.cpu().numpy()
            analysis['condition_number'] = (S[0] / S[-1]).item() if S[-1] > 1e-10 else float('inf')
            analysis['effective_rank'] = torch.sum(S > 1e-6).item()
            
            # Determine restriction type based on singular values
            if torch.allclose(S, torch.ones_like(S)):
                analysis['restriction_type'] = 'orthogonal'
            elif torch.allclose(restriction_map, torch.eye(min(target_dim, source_dim))[:target_dim, :source_dim]):
                analysis['restriction_type'] = 'identity'
            elif analysis['effective_rank'] < min(target_dim, source_dim) * 0.8:
                analysis['restriction_type'] = 'projection'
            elif target_dim > source_dim:
                analysis['restriction_type'] = 'embedding'
            else:
                analysis['restriction_type'] = 'general'
                
        except Exception as e:
            logger.warning(f"SVD analysis failed: {e}")
            analysis['restriction_type'] = 'general'
            analysis['condition_number'] = None
            analysis['effective_rank'] = None
            
        # Sparsity analysis
        analysis['sparsity'] = (torch.sum(torch.abs(restriction_map) < 1e-6).item() / 
                               (target_dim * source_dim))
        
        # Energy distribution
        if 'singular_values' in analysis:
            total_energy = np.sum(analysis['singular_values'] ** 2)
            if total_energy > 0:
                cumsum = np.cumsum(analysis['singular_values'] ** 2)
                analysis['energy_90_percent'] = np.argmax(cumsum >= 0.9 * total_energy) + 1
            else:
                analysis['energy_90_percent'] = 0
                
        return analysis
        
    def create_edge_visual_properties(self,
                                    edge: Tuple[str, str],
                                    restriction_map: torch.Tensor,
                                    edge_attrs: Optional[Dict[str, Any]] = None) -> EdgeVisualProperties:
        """
        Create comprehensive visual properties for an edge.
        
        Args:
            edge: Tuple of (source, target) node names
            restriction_map: The restriction map tensor
            edge_attrs: Optional additional edge attributes
            
        Returns:
            EdgeVisualProperties with all visual settings
        """
        # Analyze the restriction map
        analysis = self.analyze_restriction_map(restriction_map)
        
        # Determine edge width based on information flow
        norm = analysis['frobenius_norm']
        if norm > 10:
            width = 6
        elif norm > 5:
            width = 4
        elif norm > 1:
            width = 2 + (norm - 1) * 0.5
        else:
            width = 1 + norm
            
        # Determine color based on dimensional change
        color = self.dimensional_change_colors[analysis['dimensional_change']]
        
        # Determine opacity based on condition number
        if analysis.get('condition_number'):
            if analysis['condition_number'] > 100:
                opacity = 0.5  # Poor conditioning
            elif analysis['condition_number'] > 10:
                opacity = 0.7  # Moderate conditioning
            else:
                opacity = 0.9  # Good conditioning
        else:
            opacity = 0.8
            
        # Determine dash pattern based on edge type
        if edge_attrs and 'edge_type' in edge_attrs:
            dash_pattern = self.dash_patterns.get(edge_attrs['edge_type'], None)
        else:
            # Use sparsity to determine pattern
            if analysis['sparsity'] > 0.8:
                dash_pattern = 'dot'
            elif analysis['sparsity'] > 0.5:
                dash_pattern = 'dash'
            else:
                dash_pattern = None
                
        # Determine arrow style
        arrow_style = 'normal'
        if analysis['dimensional_change'] in ['expand', 'strong_expand']:
            arrow_style = 'expanding'
        elif analysis['dimensional_change'] in ['contract', 'strong_contract']:
            arrow_style = 'contracting'
            
        # Create gradient colors for flow strength
        if norm > 5:
            gradient_colors = [self.flow_strength_gradient[-2], self.flow_strength_gradient[-1]]
        else:
            gradient_idx = int(norm / 10 * len(self.flow_strength_gradient))
            gradient_colors = None  # Use solid color for moderate flows
            
        # Create intermediate markers for complex transformations
        markers = []
        if analysis.get('effective_rank') and analysis['effective_rank'] < min(analysis['source_dim'], analysis['target_dim']) * 0.5:
            # Add a marker indicating dimension reduction
            markers.append({
                'position': 0.5,  # Middle of edge
                'symbol': 'diamond',
                'size': 8,
                'color': '#FF9800',  # Orange
                'label': f"rank={analysis['effective_rank']}"
            })
            
        return EdgeVisualProperties(
            width=width,
            color=color,
            opacity=opacity,
            dash_pattern=dash_pattern,
            arrow_style=arrow_style,
            curvature=0.1 if len(markers) > 0 else 0,  # Curve if markers present
            markers=markers,
            gradient_colors=gradient_colors,
            animation_speed=None
        )
        
    def create_edge_trace(self,
                         edge: Tuple[str, str],
                         pos: Dict[str, Tuple[float, float]],
                         visual_props: EdgeVisualProperties,
                         label: Optional[str] = None) -> go.Scatter:
        """
        Create a Plotly trace for an edge with all visual properties.
        
        Args:
            edge: Tuple of (source, target) node names
            pos: Dictionary of node positions
            visual_props: Visual properties for the edge
            label: Optional edge label
            
        Returns:
            Plotly Scatter trace for the edge
        """
        source, target = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Apply curvature if specified
        if visual_props.curvature > 0:
            # Create curved path using quadratic Bezier curve
            t = np.linspace(0, 1, 20)
            # Control point for curve
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2 + visual_props.curvature * abs(x1 - x0)
            
            x_curve = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
            y_curve = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
            
            x_coords = list(x_curve) + [None]
            y_coords = list(y_curve) + [None]
        else:
            x_coords = [x0, x1, None]
            y_coords = [y0, y1, None]
            
        # Create line trace
        line_dict = {
            'width': visual_props.width,
            'color': visual_props.color,
        }
        
        if visual_props.dash_pattern:
            line_dict['dash'] = visual_props.dash_pattern
            
        # Add gradient if specified
        if visual_props.gradient_colors:
            line_dict['color'] = visual_props.gradient_colors[0]  # Plotly limitation
            
        trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=line_dict,
            opacity=visual_props.opacity,
            hoverinfo='skip',  # Hover handled separately
            showlegend=False,
            name=f"edge_{source}_{target}"
        )
        
        return trace
        
    def create_arrow_annotations(self,
                               edge: Tuple[str, str],
                               pos: Dict[str, Tuple[float, float]],
                               visual_props: EdgeVisualProperties) -> List[Dict[str, Any]]:
        """
        Create arrow annotations for an edge.
        
        Args:
            edge: Tuple of (source, target) node names
            pos: Dictionary of node positions
            visual_props: Visual properties for the edge
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        source, target = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Get arrow template
        arrow_template = self.arrow_templates[visual_props.arrow_style]
        
        # Create main arrow annotation
        arrow_dict = {
            'x': x1,
            'y': y1,
            'ax': x0,
            'ay': y0,
            'xref': 'x',
            'yref': 'y',
            'axref': 'x',
            'ayref': 'y',
            'showarrow': True,
            'arrowhead': arrow_template['arrowhead'],
            'arrowsize': arrow_template['arrowsize'],
            'arrowwidth': arrow_template['arrowwidth'],
            'arrowcolor': arrow_template.get('arrowcolor') or visual_props.color,
            'opacity': visual_props.opacity,
        }
        
        annotations.append(arrow_dict)
        
        # Add reverse arrow for bidirectional
        if 'startarrowhead' in arrow_template:
            reverse_arrow = arrow_dict.copy()
            reverse_arrow['x'] = x0
            reverse_arrow['y'] = y0
            reverse_arrow['ax'] = x1
            reverse_arrow['ay'] = y1
            annotations.append(reverse_arrow)
            
        return annotations
        
    def create_dimensional_flow_indicators(self,
                                         edge: Tuple[str, str],
                                         source_dim: int,
                                         target_dim: int,
                                         pos: Dict[str, Tuple[float, float]]) -> List[go.Scatter]:
        """
        Create visual indicators for dimensional changes.
        
        Args:
            edge: Tuple of (source, target) node names
            source_dim: Source dimension
            target_dim: Target dimension
            pos: Dictionary of node positions
            
        Returns:
            List of Plotly traces for dimensional indicators
        """
        traces = []
        source, target = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Calculate midpoint
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        if target_dim > source_dim:
            # Expanding - create diverging lines
            ratio = target_dim / source_dim
            symbol = '▶' if ratio <= 2 else '▶▶'
            
            trace = go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='text',
                text=symbol,
                textfont=dict(size=12, color='#4CAF50'),
                hoverinfo='skip',
                showlegend=False
            )
            traces.append(trace)
            
        elif target_dim < source_dim:
            # Contracting - create converging lines
            ratio = source_dim / target_dim
            symbol = '◀' if ratio <= 2 else '◀◀'
            
            trace = go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='text',
                text=symbol,
                textfont=dict(size=12, color='#F44336'),
                hoverinfo='skip',
                showlegend=False
            )
            traces.append(trace)
            
        # Add dimension labels
        dim_label = f"{source_dim}→{target_dim}"
        label_trace = go.Scatter(
            x=[mid_x],
            y=[mid_y - 0.1],  # Slightly below midpoint
            mode='text',
            text=dim_label,
            textfont=dict(size=9, color='gray'),
            hoverinfo='skip',
            showlegend=False
        )
        traces.append(label_trace)
        
        return traces
        
    def create_edge_hover_info(self,
                             edge: Tuple[str, str],
                             analysis: Dict[str, Any]) -> str:
        """
        Create comprehensive hover information for an edge.
        
        Args:
            edge: Tuple of (source, target) node names
            analysis: Restriction map analysis results
            
        Returns:
            HTML-formatted hover text
        """
        source, target = edge
        
        hover_text = f"<b>Edge: {source} → {target}</b><br>"
        hover_text += f"<b>Dimensional Change:</b> {analysis['source_dim']} → {analysis['target_dim']}<br>"
        hover_text += f"<b>Type:</b> {analysis['dimensional_change'].replace('_', ' ').title()}<br>"
        hover_text += f"<b>Restriction Type:</b> {analysis['restriction_type'].title()}<br>"
        hover_text += f"<b>Information Flow:</b> {analysis['frobenius_norm']:.3f}<br>"
        
        if analysis.get('condition_number') and analysis['condition_number'] < float('inf'):
            hover_text += f"<b>Condition Number:</b> {analysis['condition_number']:.2f}<br>"
            
        if analysis.get('effective_rank'):
            hover_text += f"<b>Effective Rank:</b> {analysis['effective_rank']}<br>"
            
        hover_text += f"<b>Sparsity:</b> {analysis['sparsity']*100:.1f}%<br>"
        
        if analysis.get('energy_90_percent'):
            hover_text += f"<b>90% Energy:</b> {analysis['energy_90_percent']} components<br>"
            
        return hover_text