"""Interactive poset visualization with data-flow layout.

This module provides visualization of neural network computational graphs
using topological sort-based layouts that clearly show data flow through
the network architecture.

Key Features:
- Topological sort-based data-flow layout from left to right
- Interactive hover information for nodes and edges
- Color-coded nodes by operation type
- Node size scaling by stalk dimension
- Edge weight visualization via restriction map norms
"""

import networkx as nx
import plotly.graph_objects as go
import torch
from typing import Dict, Optional, Tuple, Any
from ..sheaf.data_structures import Sheaf
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError

logger = setup_logger(__name__)


class PosetVisualizer:
    """Interactive visualization of neural network posets with data-flow layout.
    
    This class creates interactive plots of the computational graph underlying
    a neural sheaf, arranged to show the natural flow of data through the
    network from input to output layers.
    
    The visualization uses a topological sort to arrange nodes from left to right
    according to their position in the computational flow, providing an intuitive
    representation of the network's structure.
    
    Attributes:
        default_node_size: Base size for nodes
        size_scaling_factor: Factor for scaling node size by stalk dimension
        edge_color: Default color for edges
        node_colors: Mapping of operation types to colors
    """
    
    def __init__(self,
                 default_node_size: int = 15,
                 size_scaling_factor: float = 0.5,
                 edge_color: str = '#888',
                 node_colors: Optional[Dict[str, str]] = None):
        """Initialize PosetVisualizer.
        
        Args:
            default_node_size: Base size for nodes
            size_scaling_factor: Factor for scaling node size by stalk dimension
            edge_color: Default color for edges
            node_colors: Custom mapping of operation types to colors
        """
        self.default_node_size = default_node_size
        self.size_scaling_factor = size_scaling_factor
        self.edge_color = edge_color
        
        # Default color scheme for different operation types
        self.node_colors = node_colors or {
            'call_module': '#1f77b4',     # Blue for module calls
            'call_function': '#2ca02c',   # Green for function calls
            'call_method': '#ff7f0e',     # Orange for method calls
            'get_attr': '#d62728',        # Red for attribute access
            'placeholder': '#9467bd',     # Purple for inputs
            'output': '#8c564b',          # Brown for outputs
            'default': '#7f7f7f'          # Gray for unknown
        }
        
        logger.info("PosetVisualizer initialized with data-flow layout")
    
    def _data_flow_layout(self, poset: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Creates a layout based on topological sort to show data flow.
        
        This method arranges nodes from left to right according to their
        topological order in the computational graph, creating a natural
        visualization of data flow through the network.
        
        Args:
            poset: NetworkX directed graph representing the network structure
            
        Returns:
            Dictionary mapping node names to (x, y) positions
            
        Raises:
            ValidationError: If the graph contains cycles
        """
        if not nx.is_directed_acyclic_graph(poset):
            raise ValidationError("Poset must be a directed acyclic graph")
        
        pos = {}
        
        try:
            # Group nodes by their topological generation (level in data flow)
            generations = list(nx.topological_generations(poset))
            logger.debug(f"Found {len(generations)} topological generations")
            
            for i, generation in enumerate(generations):
                generation_list = sorted(list(generation))
                n_nodes = len(generation_list)
                
                # Spread nodes vertically within each generation
                if n_nodes == 1:
                    y_positions = [0.0]
                else:
                    y_positions = [j - (n_nodes - 1) / 2 for j in range(n_nodes)]
                
                # Assign positions
                for node, y_pos in zip(generation_list, y_positions):
                    pos[node] = (float(i), float(y_pos))
                
                logger.debug(f"Generation {i}: {n_nodes} nodes")
            
        except Exception as e:
            logger.error(f"Failed to compute topological layout: {e}")
            # Fallback to spring layout
            pos = nx.spring_layout(poset)
            logger.warning("Using fallback spring layout")
        
        return pos
    
    def _get_node_color(self, node_attrs: Dict[str, Any]) -> str:
        """Get color for a node based on its operation type.
        
        Args:
            node_attrs: Node attributes from the poset
            
        Returns:
            Color string for the node
        """
        op_type = node_attrs.get('op', 'default')
        return self.node_colors.get(op_type, self.node_colors['default'])
    
    def _get_node_size(self, stalk_dim: int) -> float:
        """Calculate node size based on stalk dimension.
        
        Args:
            stalk_dim: Dimension of the stalk at this node
            
        Returns:
            Scaled node size
        """
        return self.default_node_size + stalk_dim * self.size_scaling_factor
    
    def _create_edge_traces(self, poset: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                           sheaf: Sheaf) -> Tuple[go.Scatter, list]:
        """Create edge traces for the visualization.
        
        Args:
            poset: Network graph
            pos: Node positions
            sheaf: Sheaf object containing restriction maps
            
        Returns:
            Tuple of (edge_trace, edge_hover_texts)
        """
        edge_x, edge_y = [], []
        edge_hover_texts = []
        
        for edge in poset.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Add edge coordinates
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Create hover text for restriction map details
            restriction = sheaf.restrictions.get(edge)
            if restriction is not None:
                source_dim = restriction.shape[1]
                target_dim = restriction.shape[0]
                weight = torch.norm(restriction, 'fro').item()
                hover_text = (f"Edge: {edge[0]} → {edge[1]}<br>"
                            f"Restriction: {source_dim} → {target_dim}<br>"
                            f"Frobenius norm: {weight:.4f}")
            else:
                hover_text = f"Edge: {edge[0]} → {edge[1]}<br>No restriction map"
            
            edge_hover_texts.append(hover_text)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color=self.edge_color),
            hoverinfo='text',
            text=edge_hover_texts,
            mode='lines',
            name='Edges',
            showlegend=False
        )
        
        return edge_trace, edge_hover_texts
    
    def _create_node_traces(self, poset: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                           sheaf: Sheaf) -> go.Scatter:
        """Create node traces for the visualization.
        
        Args:
            poset: Network graph
            pos: Node positions  
            sheaf: Sheaf object containing stalk information
            
        Returns:
            Plotly scatter trace for nodes
        """
        node_x, node_y = [], []
        node_text, node_hover_info = [], []
        node_colors, node_sizes = [], []
        
        for node in poset.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node label (shortened for display)
            node_label = node if len(node) <= 15 else node[:12] + "..."
            node_text.append(node_label)
            
            # Get node attributes
            node_attrs = poset.nodes[node]
            node_op = node_attrs.get('op', 'N/A')
            node_target = node_attrs.get('target', 'N/A')
            
            # Get stalk dimension from metadata
            stalk_ranks = sheaf.metadata.get('stalk_ranks', {})
            stalk_dim = stalk_ranks.get(node, 1)
            
            # Create detailed hover information
            hover_text = (f"<b>{node}</b><br>"
                         f"Operation: {node_op}<br>"
                         f"Target: {node_target}<br>"
                         f"Stalk Dimension: {stalk_dim}")
            
            # Add whitening information if available
            if node in sheaf.whitening_maps:
                whitening_shape = sheaf.whitening_maps[node].shape
                hover_text += f"<br>Whitening: {whitening_shape}"
            
            node_hover_info.append(hover_text)
            
            # Styling based on properties
            node_colors.append(self._get_node_color(node_attrs))
            node_sizes.append(self._get_node_size(stalk_dim))
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            hovertext=node_hover_info,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name='Nodes',
            showlegend=False
        )
        
        return node_trace
    
    def plot(self, sheaf: Sheaf, 
             title: str = "Interactive Network Poset (Data-Flow Layout)",
             width: int = 1000,
             height: int = 600) -> go.Figure:
        """Generate an interactive plot of the sheaf's poset.
        
        Args:
            sheaf: Sheaf object to visualize
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            
        Returns:
            Interactive Plotly figure
            
        Raises:
            ValidationError: If sheaf structure is invalid
        """
        if not sheaf.poset.nodes():
            raise ValidationError("Sheaf poset is empty")
        
        logger.info(f"Creating poset visualization for {len(sheaf.poset.nodes())} nodes, "
                   f"{len(sheaf.poset.edges())} edges")
        
        try:
            # Compute data-flow layout
            pos = self._data_flow_layout(sheaf.poset)
            
            # Create edge and node traces
            edge_trace, _ = self._create_edge_traces(sheaf.poset, pos, sheaf)
            node_trace = self._create_node_traces(sheaf.poset, pos, sheaf)
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=title,
                        x=0.5,
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Data flows from left to right. Hover for details.",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="gray", size=12)
                        )
                    ],
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        title="Computational Flow →"
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False
                    ),
                    width=width,
                    height=height,
                    plot_bgcolor='white'
                )
            )
            
            logger.info("Poset visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create poset visualization: {e}")
            raise ValidationError(f"Poset visualization failed: {e}")
    
    def plot_summary_stats(self, sheaf: Sheaf) -> go.Figure:
        """Create a summary statistics plot for the sheaf structure.
        
        Args:
            sheaf: Sheaf object to analyze
            
        Returns:
            Bar plot with sheaf statistics
        """
        # Collect statistics
        stats = {
            'Nodes': len(sheaf.poset.nodes()),
            'Edges': len(sheaf.poset.edges()),
            'Stalks': len(sheaf.stalks),
            'Restrictions': len(sheaf.restrictions),
            'Whitening Maps': len(sheaf.whitening_maps)
        }
        
        # Get stalk dimensions if available
        stalk_ranks = sheaf.metadata.get('stalk_ranks', {})
        if stalk_ranks:
            stats['Total Stalk Dim'] = sum(stalk_ranks.values())
            stats['Avg Stalk Dim'] = sum(stalk_ranks.values()) / len(stalk_ranks)
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=list(stats.keys()),
                y=list(stats.values()),
                marker_color='lightblue',
                text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in stats.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Sheaf Structure Summary",
            xaxis_title="Component",
            yaxis_title="Count/Dimension",
            showlegend=False
        )
        
        return fig