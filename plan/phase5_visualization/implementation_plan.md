# Phase 5: Visualization Implementation Plan (Weeks 11-13)

## Overview
Implement comprehensive visualization suite with log-scale support, automatic backend switching, interactive dashboard, and WebGL rendering for large-scale visualizations.

## Week 11: Core Visualization Components

### Day 1-2: CKA Matrix Visualization (Log-Scale)
**Reference**: docs/visualization-plan-v3.md - "Log-scale support for CKA matrices"
- [ ] Implement log-scale CKA heatmap visualization
- [ ] Create automatic scale detection and switching
- [ ] Add colormap optimization for log-scale data
- [ ] Implement zoom and pan functionality
- [ ] Create annotation and labeling system

### Day 3-4: Poset Structure Visualization
- [ ] Implement network graph visualization for posets
- [ ] Create hierarchical layout algorithms
- [ ] Add node and edge styling based on properties
- [ ] Implement interactive node selection
- [ ] Create layer information display

### Day 5: Persistence Diagram Visualization
- [ ] Implement persistence diagram plotting
- [ ] Create barcode visualization
- [ ] Add interactive point selection
- [ ] Implement multiple diagram comparison
- [ ] Create persistence statistics display

## Week 12: Backend Management and Optimization

### Day 6-7: Automatic Backend Switching
**Reference**: docs/visualization-plan-v3.md - "Automatic backend switching"
- [ ] Implement matplotlib backend for small datasets
- [ ] Create Plotly backend for interactive visualizations
- [ ] Add WebGL backend for large-scale rendering
- [ ] Implement automatic backend selection logic
- [ ] Create performance monitoring and switching

### Day 8-9: Interactive Dashboard Framework
- [ ] Build Dash-based interactive dashboard
- [ ] Create real-time data updates
- [ ] Implement user session management
- [ ] Add customization controls
- [ ] Create export functionality

### Day 10: Performance Optimization
- [ ] Implement data downsampling for large visualizations
- [ ] Add progressive loading for large datasets
- [ ] Create caching system for rendered plots
- [ ] Implement lazy loading for dashboard components
- [ ] Add memory usage monitoring

## Week 13: Advanced Features and Integration

### Day 11-12: Advanced Visualization Features
- [ ] Implement multi-parameter persistence visualization
- [ ] Create animation support for filtration sequences
- [ ] Add 3D visualization capabilities
- [ ] Implement custom color schemes and themes
- [ ] Create publication-ready export formats

### Day 13-14: Integration and Testing
- [ ] Integrate with spectral analysis module
- [ ] Create comprehensive test suite
- [ ] Add error handling and edge cases
- [ ] Implement user input validation
- [ ] Create performance benchmarks

### Day 15: Documentation and Examples
- [ ] Create visualization gallery
- [ ] Write comprehensive documentation
- [ ] Add interactive examples
- [ ] Create tutorial notebooks
- [ ] Prepare for deployment integration

## Implementation Details

### Log-Scale CKA Visualization
```python
# neurosheaf/visualization/stalks.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError

logger = setup_logger(__name__)

class CKAMatrixVisualizer:
    """Visualize CKA matrices with log-scale support."""
    
    def __init__(self, 
                 backend: str = 'auto',
                 log_scale_threshold: float = 1e-6,
                 colormap: str = 'viridis'):
        self.backend = backend
        self.log_scale_threshold = log_scale_threshold
        self.colormap = colormap
        
    def plot_cka_matrix(self,
                       cka_matrix: torch.Tensor,
                       layer_names: Optional[List[str]] = None,
                       title: str = "CKA Similarity Matrix",
                       use_log_scale: Optional[bool] = None) -> Union[plt.Figure, go.Figure]:
        """Plot CKA matrix with automatic log-scale detection.
        
        Args:
            cka_matrix: CKA similarity matrix [n_layers, n_layers]
            layer_names: Names of layers (optional)
            title: Plot title
            use_log_scale: Force log scale (auto-detect if None)
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        # Validate input
        if cka_matrix.dim() != 2 or cka_matrix.shape[0] != cka_matrix.shape[1]:
            raise ValidationError("CKA matrix must be square 2D tensor")
        
        n_layers = cka_matrix.shape[0]
        
        # Generate layer names if not provided
        if layer_names is None:
            layer_names = [f"Layer_{i}" for i in range(n_layers)]
        
        # Auto-detect log scale need
        if use_log_scale is None:
            use_log_scale = self._should_use_log_scale(cka_matrix)
        
        # Choose backend
        backend = self._choose_backend(n_layers)
        
        if backend == 'matplotlib':
            return self._plot_matplotlib(cka_matrix, layer_names, title, use_log_scale)
        elif backend == 'plotly':
            return self._plot_plotly(cka_matrix, layer_names, title, use_log_scale)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _should_use_log_scale(self, cka_matrix: torch.Tensor) -> bool:
        """Determine if log scale should be used based on data distribution."""
        # Convert to numpy for analysis
        data = cka_matrix.detach().cpu().numpy()
        
        # Remove diagonal (always 1.0)
        off_diagonal = data[~np.eye(data.shape[0], dtype=bool)]
        
        # Check for small values
        min_val = np.min(off_diagonal)
        max_val = np.max(off_diagonal)
        
        # Use log scale if dynamic range is large or has very small values
        dynamic_range = max_val / max(min_val, 1e-10)
        
        should_use_log = (
            min_val < self.log_scale_threshold or
            dynamic_range > 100 or
            np.any(off_diagonal < 0.01)
        )
        
        if should_use_log:
            logger.info(f"Using log scale: min={min_val:.2e}, max={max_val:.2e}, range={dynamic_range:.1f}")
        
        return should_use_log
    
    def _choose_backend(self, n_layers: int) -> str:
        """Choose appropriate backend based on data size and user preference."""
        if self.backend != 'auto':
            return self.backend
        
        # Auto-select based on size
        if n_layers < 20:
            return 'matplotlib'  # Fast for small matrices
        elif n_layers < 100:
            return 'plotly'      # Interactive for medium matrices
        else:
            return 'plotly'      # Handle large matrices with downsampling
    
    def _plot_matplotlib(self,
                        cka_matrix: torch.Tensor,
                        layer_names: List[str],
                        title: str,
                        use_log_scale: bool) -> plt.Figure:
        """Create matplotlib heatmap."""
        data = cka_matrix.detach().cpu().numpy()
        
        # Apply log scale if needed
        if use_log_scale:
            # Avoid log(0) by adding small epsilon
            data = np.log10(np.clip(data, self.log_scale_threshold, None))
            title += " (Log Scale)"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(data, cmap=self.colormap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if use_log_scale:
            cbar.set_label('log₁₀(CKA Similarity)')
        else:
            cbar.set_label('CKA Similarity')
        
        # Set labels
        ax.set_xticks(range(len(layer_names)))
        ax.set_yticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_yticklabels(layer_names)
        
        # Add title
        ax.set_title(title)
        
        # Add value annotations for small matrices
        if len(layer_names) <= 10:
            for i in range(len(layer_names)):
                for j in range(len(layer_names)):
                    if use_log_scale:
                        text = f"{10**data[i,j]:.3f}"
                    else:
                        text = f"{data[i,j]:.3f}"
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white' if data[i,j] < data.mean() else 'black')
        
        plt.tight_layout()
        return fig
    
    def _plot_plotly(self,
                    cka_matrix: torch.Tensor,
                    layer_names: List[str],
                    title: str,
                    use_log_scale: bool) -> go.Figure:
        """Create interactive Plotly heatmap."""
        data = cka_matrix.detach().cpu().numpy()
        
        # Apply log scale if needed
        if use_log_scale:
            # Keep original for hover text
            original_data = data.copy()
            data = np.log10(np.clip(data, self.log_scale_threshold, None))
            title += " (Log Scale)"
            
            # Create custom hover text
            hover_text = []
            for i in range(len(layer_names)):
                hover_row = []
                for j in range(len(layer_names)):
                    hover_row.append(
                        f"Layer {i} vs Layer {j}<br>"
                        f"CKA: {original_data[i,j]:.4f}<br>"
                        f"Log₁₀: {data[i,j]:.2f}"
                    )
                hover_text.append(hover_row)
        else:
            # Standard hover text
            hover_text = []
            for i in range(len(layer_names)):
                hover_row = []
                for j in range(len(layer_names)):
                    hover_row.append(
                        f"Layer {i} vs Layer {j}<br>"
                        f"CKA: {data[i,j]:.4f}"
                    )
                hover_text.append(hover_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=layer_names,
            y=layer_names,
            colorscale=self.colormap,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            colorbar=dict(
                title="log₁₀(CKA)" if use_log_scale else "CKA Similarity"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Layer",
            yaxis_title="Layer",
            width=800,
            height=600
        )
        
        return fig
```

### Poset Visualization
```python
# neurosheaf/visualization/poset.py
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PosetVisualizer:
    """Visualize poset structure of neural networks."""
    
    def __init__(self, layout_algorithm: str = 'hierarchical'):
        self.layout_algorithm = layout_algorithm
        
    def plot_poset(self,
                  poset: nx.DiGraph,
                  node_properties: Optional[Dict] = None,
                  edge_properties: Optional[Dict] = None,
                  title: str = "Neural Network Poset") -> go.Figure:
        """Plot poset structure as interactive network graph.
        
        Args:
            poset: NetworkX directed graph representing poset
            node_properties: Dict with node properties for styling
            edge_properties: Dict with edge properties for styling
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Generate layout
        pos = self._generate_layout(poset)
        
        # Create edge traces
        edge_traces = self._create_edge_traces(poset, pos, edge_properties)
        
        # Create node trace
        node_trace = self._create_node_trace(poset, pos, node_properties)
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        fig.update_layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Hover over nodes for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888888", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=600
        )
        
        return fig
    
    def _generate_layout(self, poset: nx.DiGraph) -> Dict:
        """Generate node positions for visualization."""
        if self.layout_algorithm == 'hierarchical':
            return self._hierarchical_layout(poset)
        elif self.layout_algorithm == 'force':
            return nx.spring_layout(poset)
        elif self.layout_algorithm == 'circular':
            return nx.circular_layout(poset)
        else:
            return nx.spring_layout(poset)
    
    def _hierarchical_layout(self, poset: nx.DiGraph) -> Dict:
        """Create hierarchical layout based on topological ordering."""
        # Compute levels
        levels = {}
        for node in nx.topological_sort(poset):
            level = 0
            for pred in poset.predecessors(node):
                level = max(level, levels[pred] + 1)
            levels[node] = level
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Position nodes
        pos = {}
        max_level = max(levels.values()) if levels else 0
        
        for level, nodes in level_groups.items():
            y = 1.0 - (level / max_level)  # Top to bottom
            
            # Spread nodes horizontally
            if len(nodes) == 1:
                x_positions = [0.5]
            else:
                x_positions = np.linspace(0.1, 0.9, len(nodes))
            
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y)
        
        return pos
    
    def _create_edge_traces(self,
                           poset: nx.DiGraph,
                           pos: Dict,
                           edge_properties: Optional[Dict]) -> List[go.Scatter]:
        """Create edge traces for visualization."""
        edge_traces = []
        
        # Default edge properties
        default_props = {
            'width': 2,
            'color': '#888888',
            'dash': 'solid'
        }
        
        for edge in poset.edges():
            source, target = edge
            
            # Get edge properties
            if edge_properties and edge in edge_properties:
                props = {**default_props, **edge_properties[edge]}
            else:
                props = default_props
            
            # Create edge trace
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=props['width'],
                    color=props['color'],
                    dash=props['dash']
                ),
                hoverinfo='none',
                showlegend=False
            )
            
            edge_traces.append(edge_trace)
        
        return edge_traces
    
    def _create_node_trace(self,
                          poset: nx.DiGraph,
                          pos: Dict,
                          node_properties: Optional[Dict]) -> go.Scatter:
        """Create node trace for visualization."""
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        # Default node properties
        default_size = 20
        default_color = '#1f77b4'
        
        for node in poset.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text (label)
            node_text.append(str(node))
            
            # Node info (hover)
            adjacencies = list(poset.neighbors(node))
            node_info.append(
                f"Node: {node}<br>"
                f"Degree: {poset.degree(node)}<br>"
                f"Neighbors: {', '.join(map(str, adjacencies))}"
            )
            
            # Node styling
            if node_properties and node in node_properties:
                props = node_properties[node]
                node_colors.append(props.get('color', default_color))
                node_sizes.append(props.get('size', default_size))
            else:
                node_colors.append(default_color)
                node_sizes.append(default_size)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        return node_trace
```

### Persistence Diagram Visualization
```python
# neurosheaf/visualization/persistence.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PersistenceVisualizer:
    """Visualize persistence diagrams and barcodes."""
    
    def __init__(self, backend: str = 'plotly'):
        self.backend = backend
        
    def plot_persistence_diagram(self,
                               birth_death_pairs: List[Dict],
                               infinite_bars: Optional[List[Dict]] = None,
                               title: str = "Persistence Diagram") -> go.Figure:
        """Plot persistence diagram.
        
        Args:
            birth_death_pairs: List of {'birth': float, 'death': float} dicts
            infinite_bars: List of infinite persistence bars
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if birth_death_pairs:
            births = [pair['birth'] for pair in birth_death_pairs]
            deaths = [pair['death'] for pair in birth_death_pairs]
            lifetimes = [pair.get('lifetime', d - b) for pair, b, d in zip(birth_death_pairs, births, deaths)]
            
            # Create scatter plot
            fig.add_trace(go.Scatter(
                x=births,
                y=deaths,
                mode='markers',
                marker=dict(
                    size=8,
                    color=lifetimes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Lifetime")
                ),
                name="Finite Bars",
                hovertemplate=(
                    "Birth: %{x:.3f}<br>"
                    "Death: %{y:.3f}<br>"
                    "Lifetime: %{marker.color:.3f}<br>"
                    "<extra></extra>"
                )
            ))
        
        # Add infinite bars if present
        if infinite_bars:
            infinite_births = [bar['birth'] for bar in infinite_bars]
            max_death = max([pair['death'] for pair in birth_death_pairs]) if birth_death_pairs else 1.0
            
            fig.add_trace(go.Scatter(
                x=infinite_births,
                y=[max_death * 1.1] * len(infinite_births),
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='triangle-up'
                ),
                name="Infinite Bars",
                hovertemplate=(
                    "Birth: %{x:.3f}<br>"
                    "Death: ∞<br>"
                    "<extra></extra>"
                )
            ))
        
        # Add diagonal line
        if birth_death_pairs:
            min_val = min(min(births), min(deaths))
            max_val = max(max(births), max(deaths))
        else:
            min_val, max_val = 0, 1
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name="y = x",
            hoverinfo='none',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Birth",
            yaxis_title="Death",
            width=600,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_barcode(self,
                    birth_death_pairs: List[Dict],
                    infinite_bars: Optional[List[Dict]] = None,
                    title: str = "Persistence Barcode") -> go.Figure:
        """Plot persistence barcode.
        
        Args:
            birth_death_pairs: List of {'birth': float, 'death': float} dicts
            infinite_bars: List of infinite persistence bars
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        y_positions = []
        colors = []
        
        # Plot finite bars
        for i, pair in enumerate(birth_death_pairs):
            birth = pair['birth']
            death = pair['death']
            lifetime = pair.get('lifetime', death - birth)
            
            fig.add_trace(go.Scatter(
                x=[birth, death],
                y=[i, i],
                mode='lines',
                line=dict(
                    width=5,
                    color=lifetime,
                    colorscale='Viridis'
                ),
                name=f"Bar {i}",
                hovertemplate=(
                    f"Bar {i}<br>"
                    f"Birth: {birth:.3f}<br>"
                    f"Death: {death:.3f}<br>"
                    f"Lifetime: {lifetime:.3f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))
            
            y_positions.append(i)
        
        # Plot infinite bars
        if infinite_bars:
            max_x = max([pair['death'] for pair in birth_death_pairs]) if birth_death_pairs else 1.0
            
            for i, bar in enumerate(infinite_bars):
                bar_idx = len(birth_death_pairs) + i
                birth = bar['birth']
                
                fig.add_trace(go.Scatter(
                    x=[birth, max_x * 1.1],
                    y=[bar_idx, bar_idx],
                    mode='lines',
                    line=dict(
                        width=5,
                        color='red'
                    ),
                    name=f"Infinite Bar {i}",
                    hovertemplate=(
                        f"Infinite Bar {i}<br>"
                        f"Birth: {birth:.3f}<br>"
                        f"Death: ∞<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))
                
                y_positions.append(bar_idx)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Filtration Parameter",
            yaxis_title="Bar Index",
            width=800,
            height=max(400, len(y_positions) * 30),
            showlegend=False
        )
        
        return fig
```

### Interactive Dashboard
```python
# neurosheaf/visualization/dashboard.py
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional
import torch
from ..spectral.persistent import PersistentSpectralAnalyzer
from ..sheaf.construction import SheafBuilder
from .stalks import CKAMatrixVisualizer
from .poset import PosetVisualizer
from .persistence import PersistenceVisualizer

class NeurosheafDashboard:
    """Interactive dashboard for neural sheaf analysis."""
    
    def __init__(self, port: int = 8050):
        self.port = port
        self.app = dash.Dash(__name__)
        self.cka_visualizer = CKAMatrixVisualizer()
        self.poset_visualizer = PosetVisualizer()
        self.persistence_visualizer = PersistenceVisualizer()
        
        # Initialize layout
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Neurosheaf Analysis Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.H3("Controls"),
                
                # File upload
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                
                # Analysis parameters
                html.Div([
                    html.Label("Filtration Steps:"),
                    dcc.Slider(
                        id='n-steps-slider',
                        min=5,
                        max=100,
                        value=20,
                        marks={i: str(i) for i in range(5, 101, 15)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginTop': 20}),
                
                html.Div([
                    html.Label("Filtration Type:"),
                    dcc.Dropdown(
                        id='filtration-type',
                        options=[
                            {'label': 'Threshold', 'value': 'threshold'},
                            {'label': 'CKA-based', 'value': 'cka_based'}
                        ],
                        value='threshold'
                    )
                ], style={'marginTop': 20}),
                
                # Analysis button
                html.Button(
                    'Run Analysis',
                    id='run-analysis-btn',
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'height': '40px',
                        'marginTop': 20,
                        'backgroundColor': '#007bff',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px'
                    }
                ),
                
                # Progress indicator
                html.Div(id='progress-indicator', style={'marginTop': 20})
                
            ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': 20}),
            
            # Main visualization area
            html.Div([
                
                # Tabs for different visualizations
                dcc.Tabs(id='viz-tabs', value='cka-tab', children=[
                    dcc.Tab(label='CKA Matrix', value='cka-tab'),
                    dcc.Tab(label='Poset Structure', value='poset-tab'),
                    dcc.Tab(label='Persistence Diagram', value='persistence-tab'),
                    dcc.Tab(label='Features', value='features-tab')
                ]),
                
                # Tab content
                html.Div(id='tab-content', style={'marginTop': 20}),
                
            ], style={'width': '78%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': 20})
        ])
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('viz-tabs', 'value'),
             Input('run-analysis-btn', 'n_clicks')],
            [State('upload-data', 'contents'),
             State('n-steps-slider', 'value'),
             State('filtration-type', 'value')]
        )
        def update_tab_content(active_tab, n_clicks, file_contents, n_steps, filtration_type):
            """Update tab content based on selection."""
            if n_clicks == 0 or file_contents is None:
                return html.Div("Please upload data and run analysis")
            
            # This would be implemented with actual data loading and analysis
            # For now, return placeholder content
            
            if active_tab == 'cka-tab':
                return html.Div([
                    html.H4("CKA Similarity Matrix"),
                    dcc.Graph(id='cka-graph', figure=self._create_sample_cka_plot())
                ])
            
            elif active_tab == 'poset-tab':
                return html.Div([
                    html.H4("Network Poset Structure"),
                    dcc.Graph(id='poset-graph', figure=self._create_sample_poset_plot())
                ])
            
            elif active_tab == 'persistence-tab':
                return html.Div([
                    html.H4("Persistence Diagram"),
                    dcc.Graph(id='persistence-graph', figure=self._create_sample_persistence_plot())
                ])
            
            elif active_tab == 'features-tab':
                return html.Div([
                    html.H4("Persistence Features"),
                    dcc.Graph(id='features-graph', figure=self._create_sample_features_plot())
                ])
        
        @self.app.callback(
            Output('progress-indicator', 'children'),
            [Input('run-analysis-btn', 'n_clicks')]
        )
        def update_progress(n_clicks):
            """Update progress indicator."""
            if n_clicks > 0:
                return html.Div([
                    html.P("Analysis in progress..."),
                    dcc.Interval(id='progress-interval', interval=1000, n_intervals=0)
                ])
            return ""
    
    def _create_sample_cka_plot(self):
        """Create sample CKA plot for demonstration."""
        # This would be replaced with actual CKA matrix visualization
        sample_matrix = torch.randn(5, 5)
        sample_matrix = torch.abs(sample_matrix)
        sample_matrix = sample_matrix / sample_matrix.max()
        
        return self.cka_visualizer.plot_cka_matrix(
            sample_matrix,
            layer_names=[f"Layer_{i}" for i in range(5)],
            title="Sample CKA Matrix"
        )
    
    def _create_sample_poset_plot(self):
        """Create sample poset plot for demonstration."""
        # This would be replaced with actual poset visualization
        import networkx as nx
        
        sample_poset = nx.DiGraph()
        sample_poset.add_edges_from([
            ('Input', 'Conv1'),
            ('Conv1', 'Conv2'),
            ('Conv2', 'FC1'),
            ('FC1', 'Output')
        ])
        
        return self.poset_visualizer.plot_poset(
            sample_poset,
            title="Sample Network Poset"
        )
    
    def _create_sample_persistence_plot(self):
        """Create sample persistence plot for demonstration."""
        # This would be replaced with actual persistence diagram
        sample_pairs = [
            {'birth': 0.1, 'death': 0.5, 'lifetime': 0.4},
            {'birth': 0.2, 'death': 0.8, 'lifetime': 0.6},
            {'birth': 0.3, 'death': 0.7, 'lifetime': 0.4}
        ]
        
        return self.persistence_visualizer.plot_persistence_diagram(
            sample_pairs,
            title="Sample Persistence Diagram"
        )
    
    def _create_sample_features_plot(self):
        """Create sample features plot for demonstration."""
        # This would be replaced with actual features visualization
        import plotly.express as px
        
        sample_data = pd.DataFrame({
            'Step': list(range(20)),
            'Spectral Gap': np.random.exponential(0.1, 20),
            'Effective Dimension': np.random.uniform(2, 8, 20)
        })
        
        fig = px.line(sample_data, x='Step', y=['Spectral Gap', 'Effective Dimension'],
                     title="Sample Persistence Features")
        
        return fig
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        print(f"Starting Neurosheaf Dashboard on port {self.port}")
        self.app.run_server(debug=debug, port=self.port)
```

## Testing Suite

### Test Structure
```
tests/phase5_visualization/
├── unit/
│   ├── test_cka_visualization.py
│   ├── test_poset_visualization.py
│   └── test_persistence_visualization.py
├── integration/
│   ├── test_dashboard.py
│   └── test_backend_switching.py
└── visual/
    ├── test_plot_outputs.py
    └── test_visual_regression.py
```

### Critical Test: Log-Scale Visualization
```python
# tests/phase5_visualization/unit/test_cka_visualization.py
import pytest
import torch
import numpy as np
from neurosheaf.visualization.stalks import CKAMatrixVisualizer

class TestCKAVisualization:
    """Test CKA matrix visualization with log-scale support."""
    
    def test_log_scale_detection(self):
        """Test automatic log-scale detection."""
        visualizer = CKAMatrixVisualizer()
        
        # Matrix with large dynamic range
        matrix = torch.tensor([
            [1.0, 0.001, 0.1],
            [0.001, 1.0, 0.01],
            [0.1, 0.01, 1.0]
        ])
        
        should_use_log = visualizer._should_use_log_scale(matrix)
        assert should_use_log  # Should detect need for log scale
        
        # Matrix with small dynamic range
        matrix_small = torch.tensor([
            [1.0, 0.8, 0.9],
            [0.8, 1.0, 0.7],
            [0.9, 0.7, 1.0]
        ])
        
        should_use_log_small = visualizer._should_use_log_scale(matrix_small)
        assert not should_use_log_small  # Should not need log scale
    
    def test_backend_selection(self):
        """Test automatic backend selection."""
        visualizer = CKAMatrixVisualizer(backend='auto')
        
        # Small matrix -> matplotlib
        backend_small = visualizer._choose_backend(10)
        assert backend_small == 'matplotlib'
        
        # Large matrix -> plotly
        backend_large = visualizer._choose_backend(50)
        assert backend_large == 'plotly'
    
    def test_plot_generation(self):
        """Test plot generation doesn't crash."""
        visualizer = CKAMatrixVisualizer()
        
        # Create test matrix
        matrix = torch.rand(5, 5)
        matrix = matrix @ matrix.T  # Make symmetric
        torch.fill_diagonal_(matrix, 1.0)  # Set diagonal to 1
        
        # Test matplotlib backend
        fig_mpl = visualizer.plot_cka_matrix(
            matrix,
            layer_names=[f"Layer_{i}" for i in range(5)],
            title="Test Matrix"
        )
        
        # Should return matplotlib figure
        assert hasattr(fig_mpl, 'axes')
        
        # Test plotly backend
        visualizer.backend = 'plotly'
        fig_plotly = visualizer.plot_cka_matrix(
            matrix,
            layer_names=[f"Layer_{i}" for i in range(5)],
            title="Test Matrix"
        )
        
        # Should return plotly figure
        assert hasattr(fig_plotly, 'data')
```

## Success Criteria

1. **Visualization Quality**
   - Log-scale CKA matrices display clearly
   - Poset structures are intuitive and interactive
   - Persistence diagrams follow mathematical conventions

2. **Performance**
   - Renders 100x100 CKA matrices smoothly
   - Dashboard responds within 2 seconds
   - Memory usage <500MB for large visualizations

3. **User Experience**
   - Intuitive dashboard interface
   - Clear error messages and validation
   - Export functionality works correctly

4. **Integration**
   - Seamless integration with analysis modules
   - Proper data validation and error handling
   - Consistent styling and theming

## Phase 5 Deliverables

1. **Core Visualization Components**
   - Log-scale CKA matrix visualization
   - Interactive poset structure plots
   - Persistence diagram and barcode plots

2. **Dashboard Framework**
   - Interactive Dash-based dashboard
   - Real-time analysis updates
   - Export and sharing capabilities

3. **Backend Management**
   - Automatic backend switching
   - Performance optimization
   - WebGL support for large datasets

4. **Documentation**
   - Visualization gallery
   - User guide and tutorials
   - API documentation

5. **Testing Suite**
   - Unit tests for all components
   - Visual regression tests
   - Performance benchmarks