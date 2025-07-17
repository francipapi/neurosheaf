"""Unified visualization factory for consistent interfaces.

This module provides a centralized factory for creating and managing
different types of visualizations in the neurosheaf package, ensuring
consistent interfaces and configuration management.

Key Features:
- Unified interface for all visualization types
- Consistent configuration management
- Automatic plot type detection and routing
- Integration with existing analysis pipeline
- Comprehensive plotting workflows
"""

from typing import Dict, Any, Optional, Union, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .poset import PosetVisualizer
from .persistence import PersistenceVisualizer
from .spectral import SpectralVisualizer
from ..sheaf.data_structures import Sheaf
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError

logger = setup_logger(__name__)


class VisualizationFactory:
    """Unified factory for creating neurosheaf visualizations.
    
    This class provides a consistent interface for creating all types of
    visualizations in the neurosheaf package, handling configuration
    management and ensuring proper integration with the analysis pipeline.
    
    The factory supports:
    - Individual visualization creation
    - Combined multi-panel dashboards
    - Consistent styling and configuration
    - Automatic data validation and error handling
    
    Attributes:
        poset_visualizer: PosetVisualizer instance
        persistence_visualizer: PersistenceVisualizer instance
        spectral_visualizer: SpectralVisualizer instance
        default_config: Default configuration for all visualizations
    """
    
    def __init__(self,
                 poset_config: Optional[Dict[str, Any]] = None,
                 persistence_config: Optional[Dict[str, Any]] = None,
                 spectral_config: Optional[Dict[str, Any]] = None,
                 default_config: Optional[Dict[str, Any]] = None):
        """Initialize VisualizationFactory.
        
        Args:
            poset_config: Configuration for PosetVisualizer
            persistence_config: Configuration for PersistenceVisualizer
            spectral_config: Configuration for SpectralVisualizer
            default_config: Default configuration for all visualizers
        """
        # Set up default configuration
        self.default_config = default_config or {
            'width': 800,
            'height': 600,
            'font_size': 12,
            'color_palette': 'plotly',
            'show_legend': True,
            'interactive': True
        }
        
        # Initialize visualizers with configurations
        poset_kwargs = {**(poset_config or {})}
        
        # Set default persistence visualization styling
        persistence_defaults = {
            'default_colorscale': 'Hot',
            'infinite_marker_symbol': 'triangle-up',
            'infinite_marker_color': 'red'
        }
        persistence_kwargs = {**persistence_defaults, **(persistence_config or {})}
        
        spectral_kwargs = {**(spectral_config or {})}
        
        self.poset_visualizer = PosetVisualizer(**poset_kwargs)
        self.persistence_visualizer = PersistenceVisualizer(**persistence_kwargs)
        self.spectral_visualizer = SpectralVisualizer(**spectral_kwargs)
        
        logger.info("VisualizationFactory initialized with all visualizers")
    
    def create_poset_plot(self, 
                         sheaf: Sheaf,
                         title: Optional[str] = None,
                         **kwargs) -> go.Figure:
        """Create a poset visualization plot.
        
        Args:
            sheaf: Sheaf object to visualize
            title: Plot title (auto-generated if None)
            **kwargs: Additional arguments passed to PosetVisualizer.plot()
            
        Returns:
            Interactive Plotly figure
        """
        title = title or "Neural Network Computational Graph"
        
        # Filter config to only include valid parameters for PosetVisualizer.plot()
        valid_params = {'width', 'height'}
        plot_kwargs = {k: v for k, v in {**self.default_config, **kwargs}.items() 
                      if k in valid_params}
        
        logger.info("Creating poset visualization")
        return self.poset_visualizer.plot(sheaf, title=title, **plot_kwargs)
    
    def create_persistence_diagram(self,
                                 diagrams: Dict[str, Any],
                                 title: Optional[str] = None,
                                 **kwargs) -> go.Figure:
        """Create a persistence diagram.
        
        Args:
            diagrams: Persistence diagram data from PersistentSpectralAnalyzer
            title: Plot title (auto-generated if None)
            **kwargs: Additional arguments passed to PersistenceVisualizer.plot_diagram()
            
        Returns:
            Interactive Plotly figure
        """
        title = title or "Persistence Diagram"
        
        # Filter config to only include valid parameters for PersistenceVisualizer.plot_diagram()
        valid_params = {'width', 'height', 'show_diagonal', 'colorscale'}
        plot_kwargs = {k: v for k, v in {**self.default_config, **kwargs}.items() 
                      if k in valid_params}
        
        logger.info("Creating persistence diagram")
        return self.persistence_visualizer.plot_diagram(diagrams, title=title, **plot_kwargs)
    
    def create_persistence_barcode(self,
                                 diagrams: Dict[str, Any],
                                 title: Optional[str] = None,
                                 **kwargs) -> go.Figure:
        """Create a persistence barcode.
        
        Args:
            diagrams: Persistence diagram data from PersistentSpectralAnalyzer
            title: Plot title (auto-generated if None)
            **kwargs: Additional arguments passed to PersistenceVisualizer.plot_barcode()
            
        Returns:
            Interactive Plotly figure
        """
        title = title or "Persistence Barcode"
        
        # Filter config to only include valid parameters for PersistenceVisualizer.plot_barcode()
        valid_params = {'width', 'height', 'sort_by', 'max_bars'}
        plot_kwargs = {k: v for k, v in {**self.default_config, **kwargs}.items() 
                      if k in valid_params}
        
        logger.info("Creating persistence barcode")
        return self.persistence_visualizer.plot_barcode(diagrams, title=title, **plot_kwargs)
    
    def create_eigenvalue_evolution(self,
                                  eigenvalue_sequences: List,
                                  filtration_params: List[float],
                                  title: Optional[str] = None,
                                  **kwargs) -> go.Figure:
        """Create an eigenvalue evolution plot.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: Filtration parameter values
            title: Plot title (auto-generated if None)
            **kwargs: Additional arguments passed to SpectralVisualizer.plot_eigenvalue_evolution()
            
        Returns:
            Interactive Plotly figure
        """
        title = title or "Eigenvalue Evolution"
        
        # Filter config to only include valid parameters for SpectralVisualizer.plot_eigenvalue_evolution()
        valid_params = {'width', 'height', 'log_scale', 'max_eigenvalues', 'show_legend'}
        plot_kwargs = {k: v for k, v in {**self.default_config, **kwargs}.items() 
                      if k in valid_params}
        
        logger.info("Creating eigenvalue evolution plot")
        return self.spectral_visualizer.plot_eigenvalue_evolution(
            eigenvalue_sequences, filtration_params, title=title, **plot_kwargs
        )
    
    def create_comprehensive_dashboard(self,
                                     sheaf: Sheaf,
                                     analysis_result: Dict[str, Any],
                                     title: str = "Neurosheaf Analysis Dashboard") -> go.Figure:
        """Create a comprehensive multi-panel dashboard.
        
        This method creates a complete dashboard combining all visualization
        types into a single interactive figure with multiple subplots.
        
        Args:
            sheaf: Sheaf object
            analysis_result: Complete analysis result from PersistentSpectralAnalyzer
            title: Dashboard title
            
        Returns:
            Multi-panel Plotly figure
            
        Raises:
            ValidationError: If analysis_result is incomplete
        """
        logger.info("Creating comprehensive analysis dashboard")
        
        # Validate analysis result structure
        required_keys = ['persistence_result', 'diagrams', 'filtration_params']
        for key in required_keys:
            if key not in analysis_result:
                raise ValidationError(f"Missing required key '{key}' in analysis_result")
        
        # Extract data from analysis result
        persistence_result = analysis_result['persistence_result']
        diagrams = analysis_result['diagrams']
        filtration_params = analysis_result['filtration_params']
        eigenvalue_sequences = persistence_result.get('eigenvalue_sequences', [])
        
        # Create subplot layout (2x2 grid)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Network Structure (Data Flow)",
                "Persistence Diagram", 
                "Eigenvalue Evolution",
                "Persistence Barcode"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # 1. Poset visualization (top-left)
        try:
            poset_fig = self.create_poset_plot(sheaf, width=400, height=300)
            for trace in poset_fig.data:
                trace.showlegend = False  # Disable legends for subplots
                fig.add_trace(trace, row=1, col=1)
        except Exception as e:
            logger.warning(f"Failed to create poset subplot: {e}")
        
        # 2. Persistence diagram (top-right)
        try:
            pers_fig = self.create_persistence_diagram(diagrams, width=400, height=300)
            for trace in pers_fig.data:
                trace.showlegend = False
                fig.add_trace(trace, row=1, col=2)
        except Exception as e:
            logger.warning(f"Failed to create persistence diagram subplot: {e}")
        
        # 3. Eigenvalue evolution (bottom-left)
        try:
            if eigenvalue_sequences:
                eigen_fig = self.create_eigenvalue_evolution(
                    eigenvalue_sequences, filtration_params, 
                    width=400, height=300, show_legend=False
                )
                # Only show first few eigenvalue traces to avoid clutter
                for i, trace in enumerate(eigen_fig.data[:10]):  # Limit to 10 traces
                    trace.showlegend = False
                    fig.add_trace(trace, row=2, col=1)
        except Exception as e:
            logger.warning(f"Failed to create eigenvalue evolution subplot: {e}")
        
        # 4. Persistence barcode (bottom-right) 
        try:
            barcode_fig = self.create_persistence_barcode(diagrams, width=400, height=300)
            for trace in barcode_fig.data:
                trace.showlegend = False
                fig.add_trace(trace, row=2, col=2)
        except Exception as e:
            logger.warning(f"Failed to create persistence barcode subplot: {e}")
        
        # Update layout for dashboard
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18)
            ),
            width=1200,
            height=800,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        # Update axis titles for subplots
        fig.update_xaxes(title_text="Computational Flow", row=1, col=1)
        fig.update_yaxes(title_text="", row=1, col=1)
        
        fig.update_xaxes(title_text="Birth", row=1, col=2)
        fig.update_yaxes(title_text="Death", row=1, col=2)
        
        fig.update_xaxes(title_text="Filtration Parameter", row=2, col=1)
        fig.update_yaxes(title_text="Eigenvalue (log)", row=2, col=1)
        
        fig.update_xaxes(title_text="Filtration Parameter", row=2, col=2)
        fig.update_yaxes(title_text="Feature Index", row=2, col=2)
        
        # Add summary statistics as annotation
        try:
            stats = analysis_result.get('analysis_metadata', {})
            summary_text = (f"Analysis Summary:<br>"
                          f"• {stats.get('sheaf_nodes', 0)} nodes<br>"
                          f"• {stats.get('sheaf_edges', 0)} edges<br>"
                          f"• {stats.get('n_filtration_steps', 0)} filtration steps<br>"
                          f"• {len(diagrams.get('birth_death_pairs', []))} finite features<br>"
                          f"• {len(diagrams.get('infinite_bars', []))} infinite features")
            
            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1
            )
        except Exception as e:
            logger.warning(f"Failed to add summary annotation: {e}")
        
        logger.info("Comprehensive dashboard created successfully")
        return fig
    
    def create_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create a collection of summary plots for analysis results.
        
        Args:
            analysis_result: Complete analysis result from PersistentSpectralAnalyzer
            
        Returns:
            Dictionary mapping plot names to Plotly figures
        """
        logger.info("Creating analysis summary plots")
        
        plots = {}
        
        try:
            # Persistence statistics
            diagrams = analysis_result.get('diagrams', {})
            if diagrams:
                plots['persistence_diagram'] = self.create_persistence_diagram(diagrams)
                plots['persistence_barcode'] = self.create_persistence_barcode(diagrams)
                plots['lifetime_distribution'] = self.persistence_visualizer.plot_lifetime_distribution(diagrams)
        except Exception as e:
            logger.warning(f"Failed to create persistence plots: {e}")
        
        try:
            # Spectral statistics
            persistence_result = analysis_result.get('persistence_result', {})
            eigenvalue_sequences = persistence_result.get('eigenvalue_sequences', [])
            filtration_params = analysis_result.get('filtration_params', [])
            
            if eigenvalue_sequences and filtration_params:
                plots['eigenvalue_evolution'] = self.create_eigenvalue_evolution(
                    eigenvalue_sequences, filtration_params
                )
                plots['spectral_gap'] = self.spectral_visualizer.plot_spectral_gap_evolution(
                    eigenvalue_sequences, filtration_params
                )
                plots['eigenvalue_statistics'] = self.spectral_visualizer.plot_eigenvalue_statistics(
                    eigenvalue_sequences, filtration_params
                )
        except Exception as e:
            logger.warning(f"Failed to create spectral plots: {e}")
        
        logger.info(f"Created {len(plots)} summary plots")
        return plots
    
    def save_plots(self, 
                   plots: Dict[str, go.Figure],
                   output_dir: str = ".",
                   format: str = "html",
                   **kwargs) -> None:
        """Save multiple plots to files.
        
        Args:
            plots: Dictionary mapping plot names to figures
            output_dir: Output directory
            format: Output format ('html', 'png', 'svg', 'pdf')
            **kwargs: Additional arguments for plot saving
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in plots.items():
            filename = f"{name}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            try:
                if format == "html":
                    fig.write_html(filepath, **kwargs)
                elif format in ["png", "svg", "pdf"]:
                    fig.write_image(filepath, **kwargs)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                logger.info(f"Saved plot '{name}' to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot '{name}': {e}")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration of all visualizers.
        
        Returns:
            Dictionary with configuration details
        """
        return {
            'default_config': self.default_config,
            'poset_visualizer': {
                'default_node_size': self.poset_visualizer.default_node_size,
                'size_scaling_factor': self.poset_visualizer.size_scaling_factor,
                'edge_color': self.poset_visualizer.edge_color,
                'node_colors': self.poset_visualizer.node_colors
            },
            'persistence_visualizer': {
                'default_colorscale': self.persistence_visualizer.default_colorscale,
                'infinite_marker_symbol': self.persistence_visualizer.infinite_marker_symbol,
                'infinite_marker_color': self.persistence_visualizer.infinite_marker_color
            },
            'spectral_visualizer': {
                'default_log_scale': self.spectral_visualizer.default_log_scale,
                'min_eigenvalue_threshold': self.spectral_visualizer.min_eigenvalue_threshold,
                'color_palette': self.spectral_visualizer.color_palette
            }
        }