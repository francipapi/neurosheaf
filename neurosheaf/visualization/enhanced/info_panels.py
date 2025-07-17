"""Multi-modal information panels for rich data display.

This module provides comprehensive information panels that display
multiple types of data in an organized and visually appealing way.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
from .design_system import DesignSystem
from ...utils.logging import setup_logger

logger = setup_logger(__name__)


class InfoPanelManager:
    """Manages rich information panels with multiple data types."""
    
    def __init__(self, design_system: Optional[DesignSystem] = None):
        """Initialize the info panel manager."""
        self.design_system = design_system or DesignSystem()
        
    def create_node_info_panel(self,
                              node_name: str,
                              node_data: Dict[str, Any],
                              stalk_data: Optional[torch.Tensor] = None) -> go.Figure:
        """
        Create a comprehensive information panel for a node.
        
        Args:
            node_name: Name of the node
            node_data: Dictionary containing node information
            stalk_data: Optional stalk tensor for the node
            
        Returns:
            Plotly figure with node information
        """
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Node Properties",
                "Parameters",
                "Stalk Information",
                "Connections"
            ],
            specs=[
                [{"type": "table"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Apply design system layout
        layout_config = self.design_system.get_layout_config()
        fig.update_layout(**layout_config)
        
        # 1. Node Properties Table
        self._add_node_properties_table(fig, node_name, node_data, row=1, col=1)
        
        # 2. Parameter Distribution
        self._add_parameter_info(fig, node_data, row=1, col=2)
        
        # 3. Stalk Information
        if stalk_data is not None:
            self._add_stalk_analysis(fig, stalk_data, row=2, col=1)
        else:
            self._add_placeholder(fig, "No stalk data available", row=2, col=1)
            
        # 4. Connection Information
        self._add_connection_info(fig, node_data, row=2, col=2)
        
        # Set title
        title_style = self.design_system.get_title_style(1)
        title_style['text'] = f"Node Analysis: {node_name}"
        fig.update_layout(title=title_style)
        
        return fig
        
    def _add_node_properties_table(self,
                                  fig: go.Figure,
                                  node_name: str,
                                  node_data: Dict[str, Any],
                                  row: int,
                                  col: int):
        """Add node properties table."""
        properties = [
            ["Property", "Value"],
            ["Name", node_name],
            ["Type", node_data.get('type', 'Unknown')],
            ["Operation", node_data.get('op', 'N/A')],
            ["Target", node_data.get('target', 'N/A')],
            ["Parameters", f"{node_data.get('param_count', 0):,}"],
            ["FLOPs", f"{node_data.get('flops', 0):,}"],
            ["Memory", f"{node_data.get('memory_usage', 0):.2f} MB"]
        ]
        
        # Extract header and data
        header = properties[0]
        data = list(zip(*properties[1:]))
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=header,
                    fill_color=self.design_system.current_theme.secondary_color,
                    align="left",
                    font=dict(color="white", size=12)
                ),
                cells=dict(
                    values=data,
                    fill_color="lavender",
                    align="left",
                    font=dict(color=self.design_system.current_theme.text_color, size=11)
                )
            ),
            row=row, col=col
        )
        
    def _add_parameter_info(self,
                           fig: go.Figure,
                           node_data: Dict[str, Any],
                           row: int,
                           col: int):
        """Add parameter information chart."""
        param_count = node_data.get('param_count', 0)
        flops = node_data.get('flops', 0)
        memory = node_data.get('memory_usage', 0)
        
        if param_count > 0:
            # Create parameter breakdown
            categories = ['Parameters', 'FLOPs (M)', 'Memory (MB)']
            values = [param_count, flops / 1e6, memory]
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=self.design_system.current_theme.primary_color,
                    text=[f"{v:.1f}" for v in values],
                    textposition='auto'
                ),
                row=row, col=col
            )
        else:
            self._add_placeholder(fig, "No parameter information", row, col)
            
    def _add_stalk_analysis(self,
                           fig: go.Figure,
                           stalk_data: torch.Tensor,
                           row: int,
                           col: int):
        """Add stalk analysis visualization."""
        # Compute eigenvalues for analysis
        try:
            eigenvals = torch.linalg.eigvals(stalk_data).real
            eigenvals = eigenvals[eigenvals > 1e-10]  # Filter numerical zeros
            
            if len(eigenvals) > 0:
                # Plot eigenvalue distribution
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(eigenvals))),
                        y=eigenvals.cpu().numpy(),
                        mode='lines+markers',
                        name='Eigenvalues',
                        line=dict(color=self.design_system.current_theme.spectral_colors['eigenvalue'])
                    ),
                    row=row, col=col
                )
                
                # Add annotations
                fig.add_annotation(
                    x=0.5, y=0.95,
                    text=f"Rank: {len(eigenvals)}, Condition: {eigenvals.max()/eigenvals.min():.2f}",
                    xref=f"x{'' if col == 1 else col} domain",
                    yref=f"y{'' if row == 1 else row} domain",
                    showarrow=False,
                    font=dict(size=10)
                )
            else:
                self._add_placeholder(fig, "No valid eigenvalues", row, col)
                
        except Exception as e:
            logger.warning(f"Stalk analysis failed: {e}")
            self._add_placeholder(fig, "Stalk analysis failed", row, col)
            
    def _add_connection_info(self,
                            fig: go.Figure,
                            node_data: Dict[str, Any],
                            row: int,
                            col: int):
        """Add connection information."""
        # Mock connection data (would be populated from actual graph)
        in_degree = node_data.get('in_degree', 0)
        out_degree = node_data.get('out_degree', 0)
        
        categories = ['Incoming', 'Outgoing']
        values = [in_degree, out_degree]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=self.design_system.current_theme.accent_color,
                text=[str(v) for v in values],
                textposition='auto'
            ),
            row=row, col=col
        )
        
    def _add_placeholder(self,
                        fig: go.Figure,
                        message: str,
                        row: int,
                        col: int):
        """Add a placeholder for missing data."""
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            xref=f"x{'' if col == 1 else col} domain",
            yref=f"y{'' if row == 1 else row} domain",
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
    def create_edge_info_panel(self,
                              edge: Tuple[str, str],
                              restriction_analysis: Dict[str, Any],
                              restriction_map: torch.Tensor) -> go.Figure:
        """
        Create a comprehensive information panel for an edge.
        
        Args:
            edge: Tuple of (source, target) node names
            restriction_analysis: Analysis results from edge renderer
            restriction_map: The restriction map tensor
            
        Returns:
            Plotly figure with edge information
        """
        source, target = edge
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Restriction Properties",
                "Singular Values",
                "Restriction Map",
                "Dimensional Analysis"
            ],
            specs=[
                [{"type": "table"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Apply design system layout
        layout_config = self.design_system.get_layout_config()
        fig.update_layout(**layout_config)
        
        # 1. Restriction Properties Table
        self._add_restriction_properties_table(fig, edge, restriction_analysis, row=1, col=1)
        
        # 2. Singular Values
        self._add_singular_values_plot(fig, restriction_analysis, row=1, col=2)
        
        # 3. Restriction Map Heatmap
        self._add_restriction_map_heatmap(fig, restriction_map, row=2, col=1)
        
        # 4. Dimensional Analysis
        self._add_dimensional_analysis(fig, restriction_analysis, row=2, col=2)
        
        # Set title
        title_style = self.design_system.get_title_style(1)
        title_style['text'] = f"Edge Analysis: {source} → {target}"
        fig.update_layout(title=title_style)
        
        return fig
        
    def _add_restriction_properties_table(self,
                                        fig: go.Figure,
                                        edge: Tuple[str, str],
                                        analysis: Dict[str, Any],
                                        row: int,
                                        col: int):
        """Add restriction properties table."""
        source, target = edge
        
        properties = [
            ["Property", "Value"],
            ["Source", source],
            ["Target", target],
            ["Source Dim", str(analysis['source_dim'])],
            ["Target Dim", str(analysis['target_dim'])],
            ["Type", analysis['restriction_type'].title()],
            ["Frobenius Norm", f"{analysis['frobenius_norm']:.4f}"],
            ["Effective Rank", str(analysis.get('effective_rank', 'N/A'))],
            ["Condition Number", f"{analysis.get('condition_number', 'N/A'):.2f}"],
            ["Sparsity", f"{analysis['sparsity']*100:.1f}%"]
        ]
        
        # Extract header and data
        header = properties[0]
        data = list(zip(*properties[1:]))
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=header,
                    fill_color=self.design_system.current_theme.secondary_color,
                    align="left",
                    font=dict(color="white", size=12)
                ),
                cells=dict(
                    values=data,
                    fill_color="lavender",
                    align="left",
                    font=dict(color=self.design_system.current_theme.text_color, size=11)
                )
            ),
            row=row, col=col
        )
        
    def _add_singular_values_plot(self,
                                 fig: go.Figure,
                                 analysis: Dict[str, Any],
                                 row: int,
                                 col: int):
        """Add singular values plot."""
        if 'singular_values' in analysis:
            singular_values = analysis['singular_values']
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(singular_values))),
                    y=singular_values,
                    mode='lines+markers',
                    name='Singular Values',
                    line=dict(color=self.design_system.current_theme.spectral_colors['eigenvalue'])
                ),
                row=row, col=col
            )
            
            # Add energy threshold line
            if 'energy_90_percent' in analysis:
                fig.add_vline(
                    x=analysis['energy_90_percent'],
                    line_dash="dash",
                    annotation_text="90% Energy",
                    row=row, col=col
                )
        else:
            self._add_placeholder(fig, "No singular value data", row, col)
            
    def _add_restriction_map_heatmap(self,
                                    fig: go.Figure,
                                    restriction_map: torch.Tensor,
                                    row: int,
                                    col: int):
        """Add restriction map heatmap."""
        if restriction_map.numel() < 10000:  # Only for reasonable sizes
            fig.add_trace(
                go.Heatmap(
                    z=restriction_map.cpu().numpy(),
                    colorscale='RdBu',
                    zmid=0,
                    showscale=False
                ),
                row=row, col=col
            )
        else:
            self._add_placeholder(fig, f"Map too large ({restriction_map.shape})", row, col)
            
    def _add_dimensional_analysis(self,
                                 fig: go.Figure,
                                 analysis: Dict[str, Any],
                                 row: int,
                                 col: int):
        """Add dimensional analysis chart."""
        source_dim = analysis['source_dim']
        target_dim = analysis['target_dim']
        effective_rank = analysis.get('effective_rank', min(source_dim, target_dim))
        
        categories = ['Source Dim', 'Target Dim', 'Effective Rank']
        values = [source_dim, target_dim, effective_rank]
        
        colors = [
            self.design_system.current_theme.edge_colors['preserve'],
            self.design_system.current_theme.edge_colors['expand'] if target_dim > source_dim else self.design_system.current_theme.edge_colors['contract'],
            self.design_system.current_theme.accent_color
        ]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[str(v) for v in values],
                textposition='auto'
            ),
            row=row, col=col
        )