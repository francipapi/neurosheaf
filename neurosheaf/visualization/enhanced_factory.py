"""Enhanced visualization factory integrating all advanced components.

This module provides a comprehensive factory that creates enhanced
visualizations with all advanced features integrated.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from .enhanced_poset import EnhancedPosetVisualizer
from .enhanced_spectral import EnhancedSpectralVisualizer
from .enhanced.design_system import DesignSystem
from .persistence import PersistenceVisualizer
from ..sheaf.data_structures import Sheaf
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError

logger = setup_logger(__name__)


class EnhancedVisualizationFactory:
    """Comprehensive visualization factory with all advanced features."""
    
    def __init__(self, theme: str = 'neurosheaf_default'):
        """Initialize the enhanced visualization factory."""
        self.design_system = DesignSystem(theme)
        self.poset_visualizer = EnhancedPosetVisualizer(theme)
        self.spectral_visualizer = EnhancedSpectralVisualizer(self.design_system)
        self.persistence_visualizer = PersistenceVisualizer(
            default_colorscale='Hot',
            infinite_marker_symbol='triangle-up',
            infinite_marker_color='red'
        )
        
        logger.info("EnhancedVisualizationFactory initialized")
        
    def create_comprehensive_analysis_dashboard(self,
                                              sheaf: Sheaf,
                                              analysis_result: Dict[str, Any],
                                              title: str = "Neural Sheaf Analysis Dashboard") -> go.Figure:
        """
        Create a comprehensive analysis dashboard with all enhanced features.
        
        Args:
            sheaf: Sheaf object
            analysis_result: Complete analysis result from PersistentSpectralAnalyzer
            title: Dashboard title
            
        Returns:
            Comprehensive enhanced dashboard
        """
        logger.info("Creating comprehensive enhanced dashboard")
        
        # Validate inputs
        self._validate_inputs(sheaf, analysis_result)
        
        # Create main dashboard layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Network Architecture (Enhanced)",
                "Eigenvalue Evolution (Multi-Scale)",
                "Persistence Diagram (Contextual)",
                "Spectral Gap Analysis",
                "Restriction Map Analysis",
                "Persistence Barcode",
                "Architectural Statistics",
                "Spectral Density",
                "Analysis Summary"
            ],
            specs=[
                [{"colspan": 2}, None, {"rowspan": 2}],
                [{"rowspan": 2}, {"rowspan": 2}, None],
                [None, None, {"type": "table"}],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            column_widths=[0.4, 0.35, 0.25],
            row_heights=[0.4, 0.35, 0.25]
        )
        
        # Apply design system
        layout_config = self.design_system.get_layout_config()
        layout_config.update({
            'width': 1800,
            'height': 1200,
            'title': {
                **self.design_system.get_title_style(1),
                'text': title,
                'y': 0.98
            },
            'showlegend': True,
            'legend': dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        })
        
        fig.update_layout(**layout_config)
        
        # Extract data
        eigenvalue_sequences = analysis_result['persistence_result']['eigenvalue_sequences']
        filtration_params = analysis_result['filtration_params']
        diagrams = analysis_result['diagrams']
        
        # 1. Enhanced Network Architecture (spans 2 columns)
        self._add_enhanced_network_view(fig, sheaf, row=1, col=1)
        
        # 2. Persistence Diagram (spans 2 rows)
        self._add_enhanced_persistence_diagram(fig, diagrams, row=1, col=3)
        
        # 3. Spectral Gap Analysis (spans 2 rows)
        self._add_spectral_gap_analysis(fig, eigenvalue_sequences, filtration_params, row=2, col=1)
        
        # 4. Enhanced Eigenvalue Evolution (spans 2 rows)
        self._add_enhanced_eigenvalue_evolution(fig, eigenvalue_sequences, filtration_params, row=2, col=2)
        
        # 5. Analysis Summary Table
        self._add_analysis_summary_table(fig, sheaf, analysis_result, row=3, col=3)
        
        # Add interactive annotations
        self._add_interactive_annotations(fig, analysis_result)
        
        logger.info("Enhanced dashboard created successfully")
        return fig
        
    def _validate_inputs(self, sheaf: Sheaf, analysis_result: Dict[str, Any]):
        """Validate dashboard inputs."""
        if not sheaf.poset.nodes():
            raise ValidationError("Sheaf poset is empty")
            
        required_keys = ['persistence_result', 'diagrams', 'filtration_params']
        for key in required_keys:
            if key not in analysis_result:
                raise ValidationError(f"Missing required key '{key}' in analysis_result")
                
    def _add_enhanced_network_view(self, fig: go.Figure, sheaf: Sheaf, row: int, col: int):
        """Add enhanced network architecture view."""
        # Create standalone network visualization
        network_fig = self.poset_visualizer.create_visualization(
            sheaf,
            title="",
            width=800,
            height=500,
            show_info_panels=False
        )
        
        # Extract traces and add to main figure
        for trace in network_fig.data:
            trace.showlegend = False  # Disable individual legends
            fig.add_trace(trace, row=row, col=col)
            
        # Update axes
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title_text="Network Data Flow →",
            row=row, col=col
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor=f"x{col}",
            scaleratio=1,
            row=row, col=col
        )
        
    def _add_enhanced_persistence_diagram(self, fig: go.Figure, diagrams: Dict[str, Any], row: int, col: int):
        """Add enhanced persistence diagram."""
        # Create standalone persistence diagram
        pers_fig = self.persistence_visualizer.plot_diagram(
            diagrams,
            title="",
            width=400,
            height=600
        )
        
        # Extract traces and add to main figure
        for trace in pers_fig.data:
            trace.showlegend = False
            fig.add_trace(trace, row=row, col=col)
            
        # Update axes
        fig.update_xaxes(title_text="Birth", row=row, col=col)
        fig.update_yaxes(title_text="Death", row=row, col=col)
        
        # Add diagonal line
        if diagrams.get('birth_death_pairs'):
            births = [p['birth'] for p in diagrams['birth_death_pairs']]
            deaths = [p['death'] for p in diagrams['birth_death_pairs']]
            if births and deaths:
                min_val = min(births)
                max_val = max(deaths)
                fig.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="gray", width=1, dash="dash"),
                    row=row, col=col
                )
                
    def _add_spectral_gap_analysis(self, fig: go.Figure, eigenvalue_sequences: List, 
                                  filtration_params: List[float], row: int, col: int):
        """Add enhanced spectral gap analysis."""
        # Compute spectral gaps
        gaps = []
        for eigenvals in eigenvalue_sequences:
            if len(eigenvals) >= 2:
                sorted_vals = torch.sort(eigenvals)[0]
                gap = (sorted_vals[1] - sorted_vals[0]).item()
                gaps.append(gap)
            else:
                gaps.append(0.0)
                
        # Create spectral gap trace
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=gaps,
                mode='lines+markers',
                name='Spectral Gap',
                line=dict(
                    color=self.design_system.current_theme.spectral_colors['gap'],
                    width=3
                ),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor="rgba(0,150,136,0.3)",
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add critical points
        if len(gaps) > 2:
            gaps_array = np.array(gaps)
            # Find local minima
            minima = []
            for i in range(1, len(gaps_array) - 1):
                if gaps_array[i] < gaps_array[i-1] and gaps_array[i] < gaps_array[i+1]:
                    minima.append(i)
                    
            if minima:
                fig.add_trace(
                    go.Scatter(
                        x=[filtration_params[i] for i in minima],
                        y=[gaps_array[i] for i in minima],
                        mode='markers',
                        name='Critical Points',
                        marker=dict(
                            color=self.design_system.current_theme.warning_color,
                            size=10,
                            symbol='x'
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Spectral Gap", row=row, col=col)
        
    def _add_enhanced_eigenvalue_evolution(self, fig: go.Figure, eigenvalue_sequences: List,
                                         filtration_params: List[float], row: int, col: int):
        """Add enhanced eigenvalue evolution plot."""
        # Prepare eigenvalue matrix
        max_eigenvals = max(len(seq) for seq in eigenvalue_sequences if len(seq) > 0)
        eigenval_matrix = np.full((max_eigenvals, len(filtration_params)), np.nan)
        
        for i, seq in enumerate(eigenvalue_sequences):
            if len(seq) > 0:
                eigenvals = seq.detach().cpu().numpy()
                eigenvals = np.maximum(eigenvals, 1e-12)
                eigenval_matrix[:len(eigenvals), i] = eigenvals
                
        # Plot ALL eigenvalue tracks
        n_plot = max_eigenvals
        colors = self.design_system.get_colorscale('spectral')
        
        for i in range(n_plot):
            track = eigenval_matrix[i, :]
            valid_mask = ~np.isnan(track)
            
            if np.any(valid_mask):
                color_idx = int(i / n_plot * (len(colors) - 1))
                color = colors[color_idx][1]
                
                fig.add_trace(
                    go.Scatter(
                        x=np.array(filtration_params)[valid_mask],
                        y=track[valid_mask],
                        mode='lines',
                        name=f'λ_{i}',
                        line=dict(color=color, width=2),
                        showlegend=i < 5,
                        legendgroup='eigenvalues'
                    ),
                    row=row, col=col
                )
                
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Eigenvalue (log scale)", type="log", row=row, col=col)
        
    def _add_analysis_summary_table(self, fig: go.Figure, sheaf: Sheaf, 
                                   analysis_result: Dict[str, Any], row: int, col: int):
        """Add comprehensive analysis summary table."""
        # Gather summary statistics
        metadata = analysis_result.get('analysis_metadata', {})
        diagrams = analysis_result.get('diagrams', {})
        
        # Node type analysis
        node_types = {}
        model_context = {
            'traced_model': sheaf.metadata.get('traced_model'),
            'module_types': sheaf.metadata.get('module_types', {})
        }
        for node in sheaf.poset.nodes():
            node_attrs = sheaf.poset.nodes[node]
            node_type = self.poset_visualizer.node_classifier.classify_node(node, node_attrs, model_context)
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        # Create table data
        table_data = [
            ["Metric", "Value"],
            ["Network Nodes", str(len(sheaf.poset.nodes()))],
            ["Network Edges", str(len(sheaf.poset.edges()))],
            ["Filtration Steps", str(len(analysis_result.get('filtration_params', [])))],
            ["Finite Features", str(len(diagrams.get('birth_death_pairs', [])))],
            ["Infinite Features", str(len(diagrams.get('infinite_bars', [])))],
            ["Analysis Time", f"{metadata.get('analysis_time', 0):.2f}s"],
            ["Mean Lifetime", f"{diagrams.get('statistics', {}).get('mean_lifetime', 0):.4f}"],
        ]
        
        # Add most common node types
        if node_types:
            most_common = max(node_types.items(), key=lambda x: x[1])
            table_data.append(["Main Node Type", f"{most_common[0].value} ({most_common[1]})"])
            
        # Extract header and data
        header = table_data[0]
        data = list(zip(*table_data[1:]))
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=header,
                    fill_color=self.design_system.current_theme.primary_color,
                    align="left",
                    font=dict(color="white", size=12)
                ),
                cells=dict(
                    values=data,
                    fill_color="rgba(240,240,240,0.8)",
                    align="left",
                    font=dict(color=self.design_system.current_theme.text_color, size=11)
                )
            ),
            row=row, col=col
        )
        
    def _add_interactive_annotations(self, fig: go.Figure, analysis_result: Dict[str, Any]):
        """Add interactive annotations and guides."""
        # Add main dashboard annotation
        fig.add_annotation(
            x=0.5, y=0.02,
            text="🎯 <b>Enhanced Neural Sheaf Analysis Dashboard</b> | Interactive features: hover for details, click legends to toggle",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(
                size=12,
                color=self.design_system.current_theme.text_color
            ),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            xanchor="center"
        )
        
        # Add feature explanation
        feature_text = (
            "<b>🔍 Enhanced Features:</b><br>"
            "• Smart node classification & styling<br>"
            "• Intelligent edge weight visualization<br>"
            "• Architecture-aware layout<br>"
            "• Multi-scale spectral analysis<br>"
            "• Contextual persistence features"
        )
        
        fig.add_annotation(
            x=0.98, y=0.98,
            text=feature_text,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            xanchor="right"
        )
        
    def create_node_focused_analysis(self, node_name: str, sheaf: Sheaf) -> go.Figure:
        """Create detailed analysis focused on a specific node."""
        return self.poset_visualizer.create_node_detail_view(node_name, sheaf)
        
    def create_edge_focused_analysis(self, edge: Tuple[str, str], sheaf: Sheaf) -> go.Figure:
        """Create detailed analysis focused on a specific edge."""
        return self.poset_visualizer.create_edge_detail_view(edge, sheaf)
        
    def create_spectral_deep_dive(self, eigenvalue_sequences: List, 
                                filtration_params: List[float]) -> go.Figure:
        """Create comprehensive spectral analysis deep dive."""
        return self.spectral_visualizer.create_comprehensive_spectral_view(
            eigenvalue_sequences, filtration_params
        )
        
    def create_persistence_diagram(self, diagrams: Dict[str, Any], 
                                 title: str = "Persistence Diagram",
                                 **kwargs) -> go.Figure:
        """Create persistence diagram."""
        return self.persistence_visualizer.plot_diagram(diagrams, title=title, **kwargs)
        
    def create_persistence_barcode(self, diagrams: Dict[str, Any], 
                                 title: str = "Persistence Barcode",
                                 **kwargs) -> go.Figure:
        """Create persistence barcode."""
        return self.persistence_visualizer.plot_barcode(diagrams, title=title, **kwargs)
        
    def create_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create analysis summary collection."""
        summary_plots = {}
        
        # Create individual summary plots
        eigenval_seqs = analysis_result['persistence_result']['eigenvalue_sequences']
        filtration_params = analysis_result['filtration_params']
        diagrams = analysis_result['diagrams']
        
        # Eigenvalue evolution
        if eigenval_seqs:
            summary_plots['eigenvalue_evolution'] = self.spectral_visualizer.plot_eigenvalue_evolution(
                eigenval_seqs, filtration_params, title="Eigenvalue Evolution Summary"
            )
        
        # Spectral gap evolution
        if eigenval_seqs:
            summary_plots['spectral_gap'] = self.spectral_visualizer.plot_spectral_gap_evolution(
                eigenval_seqs, filtration_params, title="Spectral Gap Evolution Summary"
            )
        
        # Persistence diagram
        if diagrams:
            summary_plots['persistence_diagram'] = self.persistence_visualizer.plot_diagram(
                diagrams, title="Persistence Diagram Summary"
            )
        
        # Persistence barcode
        if diagrams:
            summary_plots['persistence_barcode'] = self.persistence_visualizer.plot_barcode(
                diagrams, title="Persistence Barcode Summary"
            )
        
        return summary_plots
        
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'theme': getattr(self.design_system, 'theme', 'neurosheaf_default'),
            'default_config': self.design_system.current_theme.__dict__,
            'visualizers': {
                'poset': 'EnhancedPosetVisualizer',
                'spectral': 'EnhancedSpectralVisualizer', 
                'persistence': 'PersistenceVisualizer'
            }
        }