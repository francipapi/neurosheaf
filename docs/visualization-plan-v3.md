# Visualization Implementation Plan v3

## Overview
This v3 plan implements a comprehensive visualization suite for persistent sheaf Laplacians with responsive layouts, log-scale options, and interactive dashboards based on project knowledge and v2 improvements.

## Key Improvements from v2
- **Log-scale option** for Gram matrix heatmaps
- **pygraphviz guard** with helpful user instructions
- **Responsive layout** switching to Plotly for large matrices (>1000×1000)
- **WebGL support** for large-scale visualizations
- **Interactive dashboard** with real-time updates

## Core Visualization Components

### 1. Stalk Visualization with Log-Scale Support

```python
# neursheaf/visualization/stalks.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from typing import Dict, Optional, Union
import warnings

class StalkVisualizer:
    """
    Visualize stalk data (CKA Gram matrices) with adaptive rendering.
    
    Features:
    - Log-scale normalization for better contrast
    - Automatic backend selection based on matrix size
    - Interactive hover information
    """
    
    def __init__(self, backend: str = 'auto'):
        self.backend = backend
        self.size_threshold = 1000  # Switch to Plotly for larger matrices
        
    def visualize_stalk(self,
                       stalk_data: np.ndarray,
                       node_name: str,
                       log_scale: bool = False,
                       show_values: bool = True,
                       cmap: str = 'viridis') -> Union[plt.Figure, go.Figure]:
        """
        Visualize a single stalk (CKA Gram matrix).
        
        Parameters
        ----------
        stalk_data : np.ndarray
            CKA Gram matrix for the stalk
        node_name : str
            Name of the node/layer
        log_scale : bool
            Use logarithmic color scale
        show_values : bool
            Show values in cells (only for small matrices)
        cmap : str
            Colormap name
            
        Returns
        -------
        fig : matplotlib.Figure or plotly.Figure
            Visualization figure
        """
        n = stalk_data.shape[0]
        
        # Choose backend
        if self.backend == 'auto':
            use_plotly = n > self.size_threshold
        else:
            use_plotly = self.backend == 'plotly'
            
        if use_plotly:
            return self._plotly_heatmap(stalk_data, node_name, log_scale, cmap)
        else:
            return self._matplotlib_heatmap(stalk_data, node_name, log_scale, show_values, cmap)
    
    def _matplotlib_heatmap(self, data, title, log_scale, show_values, cmap):
        """Create matplotlib/seaborn heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare normalization
        if log_scale:
            # Add small epsilon to avoid log(0)
            data_positive = data + 1e-10
            norm = LogNorm(vmin=data_positive.min(), vmax=data_positive.max())
        else:
            norm = None
            
        # Create heatmap
        if show_values and data.shape[0] <= 20:
            # Show values for small matrices
            sns.heatmap(data, 
                       annot=True, 
                       fmt='.3f',
                       norm=norm,
                       cmap=cmap,
                       ax=ax,
                       cbar_kws={'label': 'CKA Value'})
        else:
            sns.heatmap(data,
                       norm=norm,
                       cmap=cmap,
                       ax=ax,
                       cbar_kws={'label': 'CKA Value (log scale)' if log_scale else 'CKA Value'})
        
        ax.set_title(f'Stalk at {title}', fontsize=14)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sample Index')
        
        plt.tight_layout()
        return fig
    
    def _plotly_heatmap(self, data, title, log_scale, cmap):
        """Create interactive Plotly heatmap for large matrices"""
        # Prepare data
        if log_scale:
            z_data = np.log10(data + 1e-10)
            colorbar_title = 'log₁₀(CKA Value)'
        else:
            z_data = data
            colorbar_title = 'CKA Value'
            
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            colorscale=cmap,
            colorbar=dict(title=colorbar_title),
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Stalk at {title} (Interactive)',
            xaxis_title='Sample Index',
            yaxis_title='Sample Index',
            width=800,
            height=700
        )
        
        return fig
    
    def visualize_all_stalks(self,
                           sheaf,
                           log_scale: bool = False,
                           max_stalks: int = 9) -> plt.Figure:
        """
        Visualize multiple stalks in a grid layout.
        
        Parameters
        ----------
        sheaf : NeuralNetworkSheaf
            Sheaf containing stalks
        log_scale : bool
            Use log scale for all heatmaps
        max_stalks : int
            Maximum number of stalks to display
            
        Returns
        -------
        fig : matplotlib.Figure
            Grid of stalk visualizations
        """
        stalk_names = list(sheaf.stalks.keys())[:max_stalks]
        n_stalks = len(stalk_names)
        
        # Calculate grid dimensions
        n_cols = min(3, n_stalks)
        n_rows = (n_stalks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_stalks == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i, (ax, name) in enumerate(zip(axes, stalk_names)):
            stalk_data = sheaf.stalks[name]
            
            # Create heatmap
            if log_scale:
                norm = LogNorm(vmin=stalk_data.min()+1e-10, vmax=stalk_data.max())
            else:
                norm = None
                
            im = ax.imshow(stalk_data, cmap='viridis', norm=norm, aspect='auto')
            ax.set_title(f'{name}', fontsize=10)
            ax.set_xlabel('Sample')
            ax.set_ylabel('Sample')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
            
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        return fig
```

### 2. Network Poset Graph Visualization

```python
# neursheaf/visualization/poset.py
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Optional, Tuple

class PosetVisualizer:
    """
    Visualize neural network poset structure with hierarchical layout.
    
    Handles pygraphviz availability gracefully.
    """
    
    def __init__(self):
        self.has_graphviz = self._check_graphviz()
        
    def _check_graphviz(self) -> bool:
        """Check if pygraphviz is available"""
        try:
            import pygraphviz
            return True
        except ImportError:
            return False
    
    def visualize_poset(self,
                       poset: nx.DiGraph,
                       node_colors: Optional[Dict] = None,
                       edge_colors: Optional[Dict] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize poset structure with hierarchical layout.
        
        Parameters
        ----------
        poset : nx.DiGraph
            Neural network poset
        node_colors : dict, optional
            Node name -> color mapping
        edge_colors : dict, optional
            Edge -> color mapping
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.Figure
            Poset visualization
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get layout
        pos = self._compute_layout(poset)
        
        # Default colors
        if node_colors is None:
            node_colors = self._compute_node_colors(poset)
            
        if edge_colors is None:
            edge_colors = 'gray'
            
        # Draw nodes
        nx.draw_networkx_nodes(poset, pos, 
                              node_color=[node_colors.get(n, 'lightblue') for n in poset.nodes()],
                              node_size=800,
                              ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(poset, pos,
                              edge_color=edge_colors,
                              arrows=True,
                              arrowsize=20,
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(poset, pos,
                               font_size=8,
                               font_weight='bold',
                               ax=ax)
        
        ax.set_title('Neural Network Poset Structure', fontsize=14)
        ax.axis('off')
        
        return fig
    
    def _compute_layout(self, poset: nx.DiGraph) -> Dict:
        """Compute hierarchical layout for poset"""
        if self.has_graphviz:
            try:
                # Use graphviz for better hierarchical layout
                pos = nx.nx_agraph.graphviz_layout(poset, prog='dot')
                return pos
            except Exception as e:
                warnings.warn(f"Graphviz layout failed: {e}. Falling back to spring layout.")
        else:
            warnings.warn(
                "For better hierarchical layouts, install pygraphviz:\n"
                "  Linux/Mac: brew install graphviz && pip install pygraphviz\n"
                "  Windows: conda install -c conda-forge pygraphviz"
            )
        
        # Fallback to spring layout
        return nx.spring_layout(poset, k=2, iterations=50)
    
    def _compute_node_colors(self, poset: nx.DiGraph) -> Dict:
        """Compute node colors based on layer depth"""
        # Compute depths
        try:
            depths = nx.shortest_path_length(poset, source='input')
        except nx.NetworkXError:
            # No 'input' node, use topological generations
            depths = {}
            for i, generation in enumerate(nx.topological_generations(poset)):
                for node in generation:
                    depths[node] = i
                    
        # Map depths to colors
        max_depth = max(depths.values()) if depths else 0
        cmap = plt.cm.viridis
        
        node_colors = {}
        for node, depth in depths.items():
            color = cmap(depth / (max_depth + 1))
            node_colors[node] = color
            
        return node_colors
    
    def visualize_restriction_maps(self,
                                 sheaf,
                                 show_scales: bool = True) -> plt.Figure:
        """
        Visualize restriction maps with edge attributes.
        
        Shows Procrustes scale factors on edges.
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        poset = sheaf.poset
        pos = self._compute_layout(poset)
        
        # Draw nodes
        nx.draw_networkx_nodes(poset, pos, node_size=1000, ax=ax)
        
        # Draw edges with scale information
        for edge in poset.edges():
            if edge in sheaf.restrictions:
                scale = sheaf.restrictions[edge]['scale']
                
                # Color by scale
                color = plt.cm.coolwarm(scale / 2.0)  # Normalize to [0,1]
                
                nx.draw_networkx_edges(poset, pos,
                                     edgelist=[edge],
                                     edge_color=[color],
                                     width=2,
                                     arrows=True,
                                     arrowsize=20,
                                     ax=ax)
                
                if show_scales:
                    # Add scale label
                    x1, y1 = pos[edge[0]]
                    x2, y2 = pos[edge[1]]
                    label_x = (x1 + x2) / 2
                    label_y = (y1 + y2) / 2
                    
                    ax.text(label_x, label_y, f'{scale:.2f}',
                           fontsize=8,
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   alpha=0.8))
        
        # Draw node labels
        nx.draw_networkx_labels(poset, pos, font_size=10, ax=ax)
        
        ax.set_title('Restriction Maps with Scale Factors', fontsize=14)
        ax.axis('off')
        
        # Add colorbar for scale
        sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                                   norm=plt.Normalize(vmin=0, vmax=2))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Procrustes Scale Factor')
        
        return fig
```

### 3. Persistence Diagram Visualization

```python
# neursheaf/visualization/persistence.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Tuple, Optional

class PersistenceVisualizer:
    """
    Visualize persistence diagrams and barcodes for spectral features.
    """
    
    def plot_persistence_diagram(self,
                                persistence_pairs: List[Tuple[float, float]],
                                title: str = "Persistence Diagram",
                                interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot persistence diagram showing birth-death pairs.
        
        Parameters
        ----------
        persistence_pairs : list of tuples
            (birth, death) pairs
        title : str
            Plot title
        interactive : bool
            Use Plotly for interactive visualization
            
        Returns
        -------
        fig : Figure
            Persistence diagram
        """
        if not persistence_pairs:
            warnings.warn("No persistence pairs to plot")
            return self._empty_diagram(title)
            
        births = [b for b, d in persistence_pairs]
        deaths = [d for b, d in persistence_pairs]
        
        if interactive:
            return self._plotly_persistence_diagram(births, deaths, title)
        else:
            return self._matplotlib_persistence_diagram(births, deaths, title)
    
    def _matplotlib_persistence_diagram(self, births, deaths, title):
        """Create static persistence diagram"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot points
        ax.scatter(births, deaths, s=50, alpha=0.7, c='blue', edgecolors='black')
        
        # Plot diagonal
        max_val = max(max(births), max(deaths))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        
        # Add persistence lines
        for b, d in zip(births, deaths):
            ax.plot([b, b], [b, d], 'gray', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plotly_persistence_diagram(self, births, deaths, title):
        """Create interactive persistence diagram"""
        fig = go.Figure()
        
        # Add points
        persistence = [d - b for b, d in zip(births, deaths)]
        
        fig.add_trace(go.Scatter(
            x=births,
            y=deaths,
            mode='markers',
            marker=dict(
                size=8,
                color=persistence,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Persistence")
            ),
            text=[f'Birth: {b:.3f}<br>Death: {d:.3f}<br>Persistence: {d-b:.3f}' 
                  for b, d in zip(births, deaths)],
            hoverinfo='text',
            name='Features'
        ))
        
        # Add diagonal
        max_val = max(max(births), max(deaths))
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Birth',
            yaxis_title='Death',
            width=700,
            height=700,
            hovermode='closest'
        )
        
        return fig
    
    def plot_persistence_barcode(self,
                               persistence_pairs: List[Tuple[float, float]],
                               title: str = "Persistence Barcode") -> plt.Figure:
        """
        Plot persistence barcode showing feature lifespans.
        
        Parameters
        ----------
        persistence_pairs : list of tuples
            (birth, death) pairs
        title : str
            Plot title
            
        Returns
        -------
        fig : matplotlib.Figure
            Barcode plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by birth time
        sorted_pairs = sorted(persistence_pairs, key=lambda x: x[0])
        
        # Plot bars
        for i, (birth, death) in enumerate(sorted_pairs):
            ax.plot([birth, death], [i, i], 'b-', linewidth=2, alpha=0.7)
            ax.scatter([birth], [i], color='green', s=30, zorder=3)
            ax.scatter([death], [i], color='red', s=30, zorder=3)
        
        ax.set_xlabel('Filtration Value', fontsize=12)
        ax.set_ylabel('Feature Index', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g', label='Birth'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', label='Death'),
            Line2D([0], [0], color='b', linewidth=2, label='Lifetime')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        return fig
    
    def plot_spectral_evolution(self,
                              spectral_features: Dict,
                              n_eigenvalues: int = 10) -> plt.Figure:
        """
        Plot evolution of eigenvalues across filtration.
        
        Parameters
        ----------
        spectral_features : dict
            Output from PersistentSpectralAnalyzer
        n_eigenvalues : int
            Number of eigenvalues to plot
            
        Returns
        -------
        fig : matplotlib.Figure
            Spectral evolution plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        thresholds = spectral_features['thresholds']
        
        # Plot harmonic spectra (Betti numbers)
        betti_numbers = spectral_features['betti_numbers']
        ax1.plot(thresholds, betti_numbers, 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Filtration Value')
        ax1.set_ylabel('Betti Number')
        ax1.set_title('Harmonic Spectrum Evolution (Topological)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot non-harmonic spectra
        non_harmonic = spectral_features['non_harmonic_spectra']
        
        # Extract first n eigenvalues at each threshold
        eigenvalue_tracks = []
        for i in range(min(n_eigenvalues, min(len(s) for s in non_harmonic))):
            track = [s[i].item() if i < len(s) else np.nan 
                    for s in non_harmonic]
            eigenvalue_tracks.append(track)
        
        # Plot each eigenvalue track
        for i, track in enumerate(eigenvalue_tracks):
            ax2.plot(thresholds, track, '-', alpha=0.7, label=f'λ_{i+1}')
        
        ax2.set_xlabel('Filtration Value')
        ax2.set_ylabel('Eigenvalue')
        ax2.set_title('Non-Harmonic Spectrum Evolution (Functional)', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
```

### 4. Interactive Dashboard

```python
# neursheaf/visualization/dashboard.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import base64
import io

class NeuralSheafDashboard:
    """
    Interactive dashboard for exploring neural sheaf analysis results.
    
    Features:
    - Real-time parameter updates
    - WebGL rendering for large networks
    - Comparative visualizations
    - Export functionality
    """
    
    def __init__(self, results: Dict):
        self.results = results
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Neural Sheaf Analysis Dashboard", 
                   style={'textAlign': 'center'}),
            
            # Control panel
            html.Div([
                html.H3("Controls"),
                
                # Layer selector
                html.Label("Select Layer:"),
                dcc.Dropdown(
                    id='layer-selector',
                    options=[{'label': name, 'value': name} 
                            for name in self.results['sheaf'].stalks.keys()],
                    value=list(self.results['sheaf'].stalks.keys())[0]
                ),
                
                # Visualization options
                html.Label("Visualization Type:"),
                dcc.RadioItems(
                    id='viz-type',
                    options=[
                        {'label': 'Stalk Heatmap', 'value': 'stalk'},
                        {'label': 'Persistence Diagram', 'value': 'persistence'},
                        {'label': 'Spectral Evolution', 'value': 'spectral'},
                        {'label': 'Network Structure', 'value': 'network'}
                    ],
                    value='stalk'
                ),
                
                # Log scale toggle
                html.Label("Log Scale:"),
                dcc.Checklist(
                    id='log-scale',
                    options=[{'label': 'Enable', 'value': 'log'}],
                    value=[]
                ),
                
                # Export button
                html.Button('Export Current View', id='export-btn', n_clicks=0),
                dcc.Download(id="download-image")
                
            ], style={'width': '25%', 'float': 'left', 'padding': '20px'}),
            
            # Main visualization area
            html.Div([
                dcc.Graph(id='main-viz', style={'height': '600px'}),
                
                # Summary statistics
                html.Div(id='summary-stats', style={'marginTop': '20px'})
                
            ], style={'width': '70%', 'float': 'right', 'padding': '20px'}),
            
            # Comparison section
            html.Div([
                html.H3("Model Comparison"),
                dcc.Graph(id='comparison-viz', style={'height': '400px'})
            ], style={'clear': 'both', 'padding': '20px'})
        ])
    
    def _setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            Output('main-viz', 'figure'),
            [Input('layer-selector', 'value'),
             Input('viz-type', 'value'),
             Input('log-scale', 'value')]
        )
        def update_main_viz(layer_name, viz_type, log_scale):
            """Update main visualization based on controls"""
            use_log = 'log' in log_scale
            
            if viz_type == 'stalk':
                return self._create_stalk_viz(layer_name, use_log)
            elif viz_type == 'persistence':
                return self._create_persistence_viz()
            elif viz_type == 'spectral':
                return self._create_spectral_viz()
            elif viz_type == 'network':
                return self._create_network_viz()
                
        @self.app.callback(
            Output('summary-stats', 'children'),
            [Input('layer-selector', 'value')]
        )
        def update_summary(layer_name):
            """Update summary statistics"""
            stalk = self.results['sheaf'].stalks[layer_name]
            
            stats = [
                html.P(f"Stalk dimension: {stalk.shape[0]}"),
                html.P(f"Mean CKA: {np.mean(stalk):.3f}"),
                html.P(f"Std CKA: {np.std(stalk):.3f}"),
                html.P(f"Condition number: {np.linalg.cond(stalk):.2f}")
            ]
            
            return html.Div(stats)
        
        @self.app.callback(
            Output("download-image", "data"),
            Input("export-btn", "n_clicks"),
            State("main-viz", "figure"),
            prevent_initial_call=True
        )
        def export_figure(n_clicks, figure):
            """Export current visualization"""
            if figure is None:
                return None
                
            # Convert to static image
            img_bytes = go.Figure(figure).to_image(format="png", width=1200, height=800)
            
            return dcc.send_bytes(img_bytes, "neural_sheaf_viz.png")
    
    def _create_stalk_viz(self, layer_name: str, use_log: bool) -> go.Figure:
        """Create interactive stalk heatmap"""
        stalk = self.results['sheaf'].stalks[layer_name]
        
        # Use WebGL for large matrices
        use_webgl = stalk.shape[0] > 500
        
        if use_log:
            z_data = np.log10(stalk + 1e-10)
            colorbar_title = 'log₁₀(CKA)'
        else:
            z_data = stalk
            colorbar_title = 'CKA Value'
            
        trace_type = go.Heatmapgl if use_webgl else go.Heatmap
        
        fig = go.Figure(data=trace_type(
            z=z_data,
            colorscale='Viridis',
            colorbar=dict(title=colorbar_title)
        ))
        
        fig.update_layout(
            title=f'Stalk at {layer_name}' + (' (WebGL)' if use_webgl else ''),
            xaxis_title='Sample Index',
            yaxis_title='Sample Index'
        )
        
        return fig
    
    def _create_network_viz(self) -> go.Figure:
        """Create interactive network visualization"""
        poset = self.results['sheaf'].poset
        
        # Compute layout
        pos = nx.spring_layout(poset, k=2)
        
        # Extract node positions
        node_x = []
        node_y = []
        node_text = []
        
        for node in poset.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        # Extract edge positions
        edge_x = []
        edge_y = []
        
        for edge in poset.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=20,
                colorbar=dict(
                    thickness=15,
                    title='Node Degree',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Color by degree
        node_degrees = [poset.degree(node) for node in poset.nodes()]
        node_trace.marker.color = node_degrees
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='Neural Network Structure',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)
```

### 5. Performance Visualizations

```python
# neursheaf/visualization/performance.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

class PerformanceVisualizer:
    """
    Visualize performance metrics and benchmarks.
    """
    
    def plot_memory_comparison(self,
                             baseline_memory: float,
                             psl_memory: float,
                             target_memory: float = 3.0) -> plt.Figure:
        """
        Plot memory usage comparison.
        
        Parameters
        ----------
        baseline_memory : float
            Baseline PH memory usage (GB)
        psl_memory : float
            Our PSL memory usage (GB)
        target_memory : float
            Target memory limit (GB)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['Baseline PH', 'PSL (Ours)', 'Target']
        memory_values = [baseline_memory, psl_memory, target_memory]
        colors = ['red', 'green', 'blue']
        
        bars = ax.bar(methods, memory_values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, memory_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f} GB',
                   ha='center', va='bottom', fontsize=12)
        
        # Add reduction factor
        reduction = baseline_memory / psl_memory
        ax.text(0.5, 0.95, f'{reduction:.1f}× reduction',
               transform=ax.transAxes,
               fontsize=14,
               weight='bold',
               ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_ylabel('Memory Usage (GB)', fontsize=12)
        ax.set_title('Memory Usage Comparison', fontsize=14)
        ax.set_ylim(0, max(memory_values) * 1.2)
        
        # Add log scale option
        ax2 = ax.twinx()
        ax2.set_yscale('log')
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Memory Usage (log scale)', fontsize=12)
        
        return fig
    
    def plot_scaling_analysis(self,
                            network_sizes: List[int],
                            baseline_times: List[float],
                            psl_times: List[float]) -> plt.Figure:
        """Plot computational scaling analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute times
        ax1.plot(network_sizes, baseline_times, 'o-', label='Baseline PH', 
                markersize=8, linewidth=2)
        ax1.plot(network_sizes, psl_times, 's-', label='PSL (Ours)', 
                markersize=8, linewidth=2)
        
        ax1.set_xlabel('Network Size (# nodes)', fontsize=12)
        ax1.set_ylabel('Computation Time (s)', fontsize=12)
        ax1.set_title('Scaling Comparison', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Speedup factor
        speedups = [b/p for b, p in zip(baseline_times, psl_times)]
        
        ax2.plot(network_sizes, speedups, 'go-', markersize=8, linewidth=2)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Network Size (# nodes)', fontsize=12)
        ax2.set_ylabel('Speedup Factor', fontsize=12)
        ax2.set_title('PSL Speedup vs Baseline', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Add speedup annotations
        for i, (size, speedup) in enumerate(zip(network_sizes, speedups)):
            if i % 2 == 0:  # Annotate every other point
                ax2.annotate(f'{speedup:.1f}×',
                           (size, speedup),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center')
        
        plt.tight_layout()
        return fig
```

### 6. Integration with Main Pipeline

```python
# neursheaf/visualization/integration.py
from .stalks import StalkVisualizer
from .poset import PosetVisualizer
from .persistence import PersistenceVisualizer
from .performance import PerformanceVisualizer
from .dashboard import NeuralSheafDashboard

class VisualizationSuite:
    """
    Complete visualization suite for neural sheaf analysis.
    
    Provides unified interface to all visualization components.
    """
    
    def __init__(self):
        self.stalk_viz = StalkVisualizer()
        self.poset_viz = PosetVisualizer()
        self.persistence_viz = PersistenceVisualizer()
        self.performance_viz = PerformanceVisualizer()
        
    def create_analysis_report(self, 
                             results: Dict,
                             save_dir: str = './reports') -> None:
        """
        Create comprehensive analysis report with all visualizations.
        
        Parameters
        ----------
        results : dict
            Complete analysis results
        save_dir : str
            Directory to save report
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Sheaf structure
        fig_poset = self.poset_viz.visualize_poset(
            results['sheaf'].poset
        )
        fig_poset.savefig(f'{save_dir}/poset_structure.png', dpi=150)
        
        # 2. Sample stalks
        fig_stalks = self.stalk_viz.visualize_all_stalks(
            results['sheaf'], 
            log_scale=True
        )
        fig_stalks.savefig(f'{save_dir}/stalk_samples.png', dpi=150)
        
        # 3. Persistence diagrams
        fig_persistence = self.persistence_viz.plot_persistence_diagram(
            results['spectral_features']['persistence_diagrams']
        )
        fig_persistence.savefig(f'{save_dir}/persistence_diagram.png', dpi=150)
        
        # 4. Spectral evolution
        fig_spectral = self.persistence_viz.plot_spectral_evolution(
            results['spectral_features']
        )
        fig_spectral.savefig(f'{save_dir}/spectral_evolution.png', dpi=150)
        
        # 5. Performance metrics
        if 'performance' in results:
            fig_perf = self.performance_viz.plot_memory_comparison(
                results['performance']['baseline_memory'],
                results['performance']['psl_memory']
            )
            fig_perf.savefig(f'{save_dir}/memory_comparison.png', dpi=150)
        
        # Generate HTML report
        self._generate_html_report(results, save_dir)
        
    def launch_dashboard(self, results: Dict, port: int = 8050) -> None:
        """
        Launch interactive dashboard for exploring results.
        
        Parameters
        ----------
        results : dict
            Analysis results
        port : int
            Port for dashboard server
        """
        dashboard = NeuralSheafDashboard(results)
        
        print(f"Launching dashboard at http://localhost:{port}")
        print("Press Ctrl+C to stop")
        
        dashboard.run(debug=False, port=port)
```

## Testing

```python
# tests/test_visualization.py
import pytest
import numpy as np
from neursheaf.visualization import VisualizationSuite

class TestVisualization:
    
    def test_log_scale_heatmap(self):
        """Test log-scale option for heatmaps"""
        viz = StalkVisualizer()
        
        # Create test data with wide range
        data = np.random.lognormal(0, 2, (100, 100))
        data = (data @ data.T) / 100  # Make symmetric PSD
        
        # Test both scales
        fig_linear = viz.visualize_stalk(data, "test", log_scale=False)
        fig_log = viz.visualize_stalk(data, "test", log_scale=True)
        
        assert fig_linear is not None
        assert fig_log is not None
        
    def test_large_matrix_handling(self):
        """Test automatic backend switching for large matrices"""
        viz = StalkVisualizer(backend='auto')
        
        # Small matrix - should use matplotlib
        small_data = np.random.rand(100, 100)
        fig_small = viz.visualize_stalk(small_data, "small")
        assert 'matplotlib' in str(type(fig_small))
        
        # Large matrix - should use plotly
        large_data = np.random.rand(1500, 1500)
        fig_large = viz.visualize_stalk(large_data, "large")
        assert 'plotly' in str(type(fig_large))
        
    def test_pygraphviz_fallback(self):
        """Test graceful handling of missing pygraphviz"""
        viz = PosetVisualizer()
        
        # Should work even without pygraphviz
        import networkx as nx
        G = nx.DiGraph([('a', 'b'), ('b', 'c')])
        
        fig = viz.visualize_poset(G)
        assert fig is not None
```

## Documentation

```python
"""
Visualization Suite for Neural Sheaf Analysis

This module provides comprehensive visualization capabilities for
persistent sheaf Laplacian analysis, including:

Components
----------
1. **Stalk Visualization**: CKA Gram matrices with log-scale support
2. **Poset Visualization**: Neural network structure with hierarchical layout  
3. **Persistence Visualization**: Diagrams, barcodes, and spectral evolution
4. **Performance Visualization**: Memory usage and scaling analysis
5. **Interactive Dashboard**: Real-time exploration with Dash/Plotly

Features
--------
- Automatic backend selection based on data size
- WebGL support for large-scale visualizations (>10k edges)
- Log-scale normalization for wide-range data
- Export functionality for all visualizations
- Responsive layouts adapting to data characteristics

Examples
--------
Basic usage:

>>> from neursheaf.visualization import VisualizationSuite
>>> viz = VisualizationSuite()
>>> 
>>> # Create analysis report
>>> viz.create_analysis_report(results, save_dir='./my_report')
>>> 
>>> # Launch interactive dashboard
>>> viz.launch_dashboard(results, port=8050)

Individual visualizations:

>>> # Visualize stalk with log scale
>>> fig = viz.stalk_viz.visualize_stalk(
...     sheaf.stalks['conv1'], 
...     'conv1',
...     log_scale=True
... )
>>> 
>>> # Plot persistence diagram
>>> fig = viz.persistence_viz.plot_persistence_diagram(
...     results['persistence_diagrams'],
...     interactive=True
... )

References
----------
- Project knowledge: Visualization techniques from persistent homology
- Integration with all analysis components
- Built on matplotlib, plotly, dash, networkx
"""
```

## Summary

This v3 Visualization Implementation Plan provides:

1. **Log-scale support** for better visualization of wide-range CKA values
2. **Automatic backend switching** (matplotlib → Plotly for >1000×1000 matrices)
3. **pygraphviz graceful handling** with helpful installation instructions
4. **WebGL rendering** for large network visualizations
5. **Interactive dashboard** with real-time parameter updates
6. **Comprehensive visualization suite** covering all aspects of the analysis

All components properly reference:
- Project knowledge for persistence visualization theory
- Integration with sheaf construction and spectral analysis
- External tools: matplotlib, plotly, dash, networkx for infrastructure