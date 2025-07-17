"""Enhanced poset visualizer integrating all advanced components.

This module provides the main enhanced poset visualizer that combines
node classification, edge rendering, layout algorithms, and information
panels into a comprehensive visualization system.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from .enhanced import (
    EnhancedNodeClassifier, 
    IntelligentEdgeRenderer,
    ArchitectureAwareLayout,
    InfoPanelManager,
    DesignSystem,
    NodeType
)
from ..sheaf.data_structures import Sheaf
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError

logger = setup_logger(__name__)


class EnhancedPosetVisualizer:
    """Advanced poset visualization with comprehensive features."""
    
    def __init__(self,
                 theme: str = 'neurosheaf_default',
                 layout_type: str = 'auto'):
        """
        Initialize the enhanced poset visualizer.
        
        Args:
            theme: Design theme to use
            layout_type: Default layout algorithm
        """
        self.design_system = DesignSystem(theme)
        self.node_classifier = EnhancedNodeClassifier()
        self.edge_renderer = IntelligentEdgeRenderer()
        self.layout_engine = ArchitectureAwareLayout(self.node_classifier)
        self.info_manager = InfoPanelManager(self.design_system)
        self.default_layout_type = layout_type
        
        logger.info("EnhancedPosetVisualizer initialized with advanced features")
        
    def create_visualization(self,
                           sheaf: Sheaf,
                           title: str = "Enhanced Neural Network Architecture",
                           width: int = 1200,
                           height: int = 800,
                           layout_type: Optional[str] = None,
                           show_info_panels: bool = True,
                           interactive_mode: bool = True) -> go.Figure:
        """
        Create a comprehensive enhanced visualization.
        
        Args:
            sheaf: Sheaf object to visualize
            title: Plot title
            width: Plot width
            height: Plot height
            layout_type: Layout algorithm to use
            show_info_panels: Whether to show information panels
            interactive_mode: Enable interactive features
            
        Returns:
            Enhanced Plotly figure
        """
        if not sheaf.poset.nodes():
            raise ValidationError("Sheaf poset is empty")
            
        layout_type = layout_type or self.default_layout_type
        
        logger.info(f"Creating enhanced visualization: {len(sheaf.poset.nodes())} nodes, "
                   f"{len(sheaf.poset.edges())} edges, layout={layout_type}")
        
        # Analyze architecture and create layout
        pos = self.layout_engine.create_layout(sheaf.poset, sheaf.metadata, layout_type)
        
        # Create main figure
        fig = go.Figure()
        
        # Apply design system layout
        layout_config = self.design_system.get_layout_config()
        layout_config.update({
            'width': width,
            'height': height,
            'title': {
                **self.design_system.get_title_style(1),
                'text': title
            },
            'xaxis': {
                **self.design_system.get_axis_style(),
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'title': "Network Flow →"
            },
            'yaxis': {
                **self.design_system.get_axis_style(),
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'scaleanchor': 'x',
                'scaleratio': 1
            }
        })
        
        fig.update_layout(**layout_config)
        
        # Add edges with enhanced rendering
        self._add_enhanced_edges(fig, sheaf, pos)
        
        # Add nodes with intelligent classification
        self._add_enhanced_nodes(fig, sheaf, pos)
        
        # Add architectural annotations
        self._add_architectural_annotations(fig, sheaf, pos)
        
        # Add legends and guides
        self._add_legends_and_guides(fig)
        
        logger.info("Enhanced visualization created successfully")
        return fig
        
    def _add_enhanced_edges(self,
                           fig: go.Figure,
                           sheaf: Sheaf,
                           pos: Dict[str, Tuple[float, float]]):
        """Add edges with enhanced rendering and information encoding."""
        edge_traces = []
        hover_info = []
        
        for edge in sheaf.poset.edges():
            if edge not in sheaf.restrictions:
                continue
                
            restriction_map = sheaf.restrictions[edge]
            
            # Analyze restriction map
            analysis = self.edge_renderer.analyze_restriction_map(restriction_map)
            
            # Create visual properties
            visual_props = self.edge_renderer.create_edge_visual_properties(
                edge, restriction_map
            )
            
            # Create edge trace
            edge_trace = self.edge_renderer.create_edge_trace(
                edge, pos, visual_props
            )
            
            # Create hover information
            hover_text = self.edge_renderer.create_edge_hover_info(edge, analysis)
            
            # Update trace with hover info
            edge_trace.update(
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            )
            
            fig.add_trace(edge_trace)
            
            # Add dimensional flow indicators
            flow_traces = self.edge_renderer.create_dimensional_flow_indicators(
                edge, analysis['source_dim'], analysis['target_dim'], pos
            )
            
            for flow_trace in flow_traces:
                fig.add_trace(flow_trace)
                
    def _add_enhanced_nodes(self,
                           fig: go.Figure,
                           sheaf: Sheaf,
                           pos: Dict[str, Tuple[float, float]]):
        """Add nodes with intelligent classification and styling."""
        # Group nodes by type for better organization
        node_groups = {}
        
        for node in sheaf.poset.nodes():
            node_attrs = sheaf.poset.nodes[node]
            
            # Classify node with model context from sheaf metadata
            model_context = {
                'traced_model': sheaf.metadata.get('traced_model'),
                'module_types': sheaf.metadata.get('module_types', {})
            }
            node_type = self.node_classifier.classify_node(node, node_attrs, model_context)
            
            # Get visual properties
            visual_props = self.node_classifier.get_node_visual_properties(node_type, node_attrs)
            
            # Get node statistics
            stats = self.node_classifier.get_node_statistics(node_attrs, sheaf.stalks)
            
            # Create node group
            if node_type not in node_groups:
                node_groups[node_type] = {
                    'nodes': [],
                    'x': [],
                    'y': [],
                    'text': [],
                    'hover': [],
                    'visual_props': visual_props
                }
                
            group = node_groups[node_type]
            group['nodes'].append(node)
            
            x, y = pos[node]
            group['x'].append(x)
            group['y'].append(y)
            
            # Create label
            label = self.node_classifier.generate_node_label(node, node_type, node_attrs)
            group['text'].append(label)
            
            # Create comprehensive hover info
            hover_text = self._create_node_hover_info(node, node_type, node_attrs, stats)
            group['hover'].append(hover_text)
            
        # Add trace for each node type
        for node_type, group in node_groups.items():
            if not group['nodes']:
                continue
                
            visual_props = group['visual_props']
            
            fig.add_trace(go.Scatter(
                x=group['x'],
                y=group['y'],
                mode='markers+text',
                text=group['text'],
                textposition='middle center',
                hoverinfo='text',
                hovertext=group['hover'],
                marker=dict(
                    size=[visual_props.size_factor * 40] * len(group['nodes']),
                    color=visual_props.primary_color,
                    line=dict(
                        width=visual_props.border_width,
                        color=visual_props.secondary_color
                    ),
                    opacity=visual_props.opacity,
                    symbol=self._get_plotly_symbol(visual_props.shape)
                ),
                textfont=dict(
                    size=10,
                    color='white' if visual_props.opacity > 0.7 else 'black'
                ),
                name=f"{node_type.value.title()} Layers",
                legendgroup=node_type.value,
                showlegend=True
            ))
            
    def _get_plotly_symbol(self, shape: str) -> str:
        """Convert shape name to Plotly symbol."""
        symbol_map = {
            'circle': 'circle',
            'square': 'square',
            'diamond': 'diamond',
            'hexagon': 'hexagon',
            'rectangle': 'square',
            'ellipse': 'circle',
            'star': 'star'
        }
        return symbol_map.get(shape, 'circle')
        
    def _create_node_hover_info(self,
                               node: str,
                               node_type: NodeType,
                               node_attrs: Dict[str, Any],
                               stats: Dict[str, Any]) -> str:
        """Create comprehensive hover information for a node."""
        hover_text = f"<b>{node}</b><br>"
        hover_text += f"<b>Type:</b> {node_type.value.title()}<br>"
        hover_text += f"<b>Operation:</b> {node_attrs.get('op', 'N/A')}<br>"
        
        # Add statistics
        if stats.get('param_count', 0) > 0:
            hover_text += f"<b>Parameters:</b> {stats['param_count']:,}<br>"
        if stats.get('flops', 0) > 0:
            hover_text += f"<b>FLOPs:</b> {stats['flops']:,}<br>"
        if stats.get('memory_usage', 0) > 0:
            hover_text += f"<b>Memory:</b> {stats['memory_usage']:.2f} MB<br>"
            
        # Add stalk information
        if 'stalk_dimension' in stats:
            hover_text += f"<b>Stalk Dimension:</b> {stats['stalk_dimension']}<br>"
        if 'stalk_rank' in stats:
            hover_text += f"<b>Stalk Rank:</b> {stats['stalk_rank']}<br>"
        if 'stalk_condition' in stats:
            hover_text += f"<b>Condition Number:</b> {stats['stalk_condition']:.2f}<br>"
            
        return hover_text
        
    def _add_architectural_annotations(self,
                                     fig: go.Figure,
                                     sheaf: Sheaf,
                                     pos: Dict[str, Tuple[float, float]]):
        """Add architectural annotations and groupings."""
        # Add flow direction annotation
        fig.add_annotation(
            x=0.02, y=0.98,
            text="Data Flow Direction →",
            xref="paper", yref="paper",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=self.design_system.current_theme.accent_color,
            ax=0.15, ay=0.98,
            font=dict(
                size=12,
                color=self.design_system.current_theme.text_color
            )
        )
        
        # Add architectural summary
        summary_text = self._create_architecture_summary(sheaf)
        fig.add_annotation(
            x=0.02, y=0.02,
            text=summary_text,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            font=dict(
                size=10,
                color=self.design_system.current_theme.text_color
            ),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
    def _create_architecture_summary(self, sheaf: Sheaf) -> str:
        """Create a summary of the architecture."""
        # Count node types
        type_counts = {}
        model_context = {
            'traced_model': sheaf.metadata.get('traced_model'),
            'module_types': sheaf.metadata.get('module_types', {})
        }
        for node in sheaf.poset.nodes():
            node_attrs = sheaf.poset.nodes[node]
            node_type = self.node_classifier.classify_node(node, node_attrs, model_context)
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            
        # Create summary
        summary = "<b>Architecture Summary:</b><br>"
        for node_type, count in type_counts.items():
            summary += f"• {node_type.value.title()}: {count}<br>"
            
        # Add total parameters if available
        total_params = sum(
            sheaf.poset.nodes[node].get('param_count', 0) 
            for node in sheaf.poset.nodes()
        )
        if total_params > 0:
            summary += f"<b>Total Parameters:</b> {total_params:,}<br>"
            
        return summary
        
    def _add_legends_and_guides(self, fig: go.Figure):
        """Add legends and visual guides."""
        # Add color legend for edge types
        edge_legend_text = (
            "<b>Edge Colors:</b><br>"
            "🔵 Dimension Preserved<br>"
            "🟢 Dimension Expanded<br>"
            "🔴 Dimension Reduced<br>"
            "📊 Information Flow (width)"
        )
        
        fig.add_annotation(
            x=0.98, y=0.02,
            text=edge_legend_text,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            font=dict(
                size=10,
                color=self.design_system.current_theme.text_color
            ),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            xanchor="right"
        )
        
    def create_node_detail_view(self,
                               node_name: str,
                               sheaf: Sheaf) -> go.Figure:
        """Create a detailed view for a specific node."""
        if node_name not in sheaf.poset.nodes():
            raise ValidationError(f"Node '{node_name}' not found in sheaf")
            
        node_attrs = sheaf.poset.nodes[node_name]
        stalk_data = sheaf.stalks.get(node_name)
        
        # Create detailed info panel
        return self.info_manager.create_node_info_panel(
            node_name, node_attrs, stalk_data
        )
        
    def create_edge_detail_view(self,
                               edge: Tuple[str, str],
                               sheaf: Sheaf) -> go.Figure:
        """Create a detailed view for a specific edge."""
        if edge not in sheaf.restrictions:
            raise ValidationError(f"Edge '{edge}' not found in sheaf restrictions")
            
        restriction_map = sheaf.restrictions[edge]
        analysis = self.edge_renderer.analyze_restriction_map(restriction_map)
        
        # Create detailed info panel
        return self.info_manager.create_edge_info_panel(
            edge, analysis, restriction_map
        )
        
    def plot_summary_stats(self, sheaf: Sheaf) -> go.Figure:
        """Create summary statistics plot for the sheaf."""
        # Analyze architecture
        type_counts = {}
        total_params = 0
        
        model_context = {
            'traced_model': sheaf.metadata.get('traced_model'),
            'module_types': sheaf.metadata.get('module_types', {})
        }
        
        for node in sheaf.poset.nodes():
            node_attrs = sheaf.poset.nodes[node]
            node_type = self.node_classifier.classify_node(node, node_attrs, model_context)
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            total_params += node_attrs.get('param_count', 0)
            
        # Create summary plot
        fig = go.Figure()
        
        # Node type distribution
        node_types = list(type_counts.keys())
        counts = [type_counts[nt] for nt in node_types]
        
        fig.add_trace(go.Bar(
            x=[nt.value.title() for nt in node_types],
            y=counts,
            name="Node Types",
            marker=dict(color="steelblue"),
            text=counts,
            textposition="outside"
        ))
        
        fig.update_layout(
            title="Sheaf Architecture Summary",
            xaxis_title="Node Type",
            yaxis_title="Count",
            annotations=[
                dict(
                    text=f"Total Nodes: {len(sheaf.poset.nodes())}<br>"
                         f"Total Edges: {len(sheaf.poset.edges())}<br>"
                         f"Total Parameters: {total_params:,}",
                    x=0.98, y=0.98,
                    xref="paper", yref="paper",
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            ]
        )
        
        return fig