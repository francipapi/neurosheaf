"""Interactive spectral visualization with multi-scale support.

This module provides visualization of eigenvalue evolution through
filtration processes, with special attention to multi-scale data
requiring logarithmic visualization.

Key Features:
- Logarithmic y-axis for multi-scale eigenvalue visualization
- Individual traces for eigenvalue evolution tracking
- Interactive highlighting and path tracking
- Integration with SubspaceTracker for eigenspace continuity
- Comprehensive spectral statistics visualization
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError

logger = setup_logger(__name__)


class SpectralVisualizer:
    """Interactive visualization of spectral evolution with multi-scale support.
    
    This class creates detailed visualizations of eigenvalue evolution through
    filtration processes, with particular emphasis on handling values that
    span multiple orders of magnitude through logarithmic scaling.
    
    Key capabilities:
    - Logarithmic y-axis scaling for multi-scale eigenvalues
    - Individual eigenvalue path tracking
    - Interactive hover information with eigenspace details
    - Integration with continuous path tracking from SubspaceTracker
    - Comprehensive spectral statistics and summaries
    
    Attributes:
        default_log_scale: Whether to use logarithmic y-axis by default
        min_eigenvalue_threshold: Minimum eigenvalue for log scale (avoid log(0))
        color_palette: Color palette for eigenvalue traces
    """
    
    def __init__(self,
                 default_log_scale: bool = True,
                 min_eigenvalue_threshold: float = 1e-12,
                 color_palette: Optional[List[str]] = None):
        """Initialize SpectralVisualizer.
        
        Args:
            default_log_scale: Whether to use logarithmic y-axis by default
            min_eigenvalue_threshold: Minimum eigenvalue for log scale
            color_palette: Custom color palette for eigenvalue traces
        """
        self.default_log_scale = default_log_scale
        self.min_eigenvalue_threshold = min_eigenvalue_threshold
        
        # Default color palette for eigenvalue traces
        self.color_palette = color_palette or [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        logger.info(f"SpectralVisualizer initialized with "
                   f"log_scale={default_log_scale}, "
                   f"min_threshold={min_eigenvalue_threshold}")
    
    def _validate_eigenvalue_sequences(self, eigenvalue_sequences: List[torch.Tensor],
                                     filtration_params: List[float]) -> None:
        """Validate eigenvalue sequences and filtration parameters.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: List of filtration parameter values
            
        Raises:
            ValidationError: If input data is invalid
        """
        if not eigenvalue_sequences:
            raise ValidationError("Empty eigenvalue sequences")
        
        if len(eigenvalue_sequences) != len(filtration_params):
            raise ValidationError(
                f"Length mismatch: {len(eigenvalue_sequences)} eigenvalue sequences "
                f"vs {len(filtration_params)} filtration parameters"
            )
        
        # Check for valid eigenvalue tensors
        for i, seq in enumerate(eigenvalue_sequences):
            if not isinstance(seq, torch.Tensor):
                raise ValidationError(f"Eigenvalue sequence {i} is not a tensor")
            
            if seq.numel() > 0 and torch.any(seq < 0):
                logger.warning(f"Negative eigenvalues found in sequence {i}")
    
    def _prepare_eigenvalue_paths(self, eigenvalue_sequences: List[torch.Tensor],
                                filtration_params: List[float]) -> np.ndarray:
        """Prepare eigenvalue paths for visualization.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: List of filtration parameter values
            
        Returns:
            2D numpy array with eigenvalue paths (eigenvalue_index, filtration_step)
        """
        # Determine maximum number of eigenvalues across all steps
        max_k = max(len(seq) for seq in eigenvalue_sequences if len(seq) > 0)
        
        if max_k == 0:
            raise ValidationError("No eigenvalues found in any sequence")
        
        # Create array to hold all paths, padded with NaN for missing values
        eigen_paths = np.full((max_k, len(filtration_params)), np.nan)
        
        for i, seq in enumerate(eigenvalue_sequences):
            if len(seq) > 0:
                # Convert to numpy and handle potential device issues
                eigenvals = seq.detach().cpu().numpy()
                
                # Apply minimum threshold for log scale
                if self.default_log_scale:
                    eigenvals = np.maximum(eigenvals, self.min_eigenvalue_threshold)
                
                # Store eigenvalues
                eigen_paths[:len(eigenvals), i] = eigenvals
        
        logger.debug(f"Prepared eigenvalue paths: {max_k} eigenvalues, "
                    f"{len(filtration_params)} steps")
        
        return eigen_paths
    
    def plot_eigenvalue_evolution(self,
                                eigenvalue_sequences: List[torch.Tensor],
                                filtration_params: List[float],
                                title: str = "Eigenvalue Evolution",
                                width: int = 1000,
                                height: int = 600,
                                log_scale: Optional[bool] = None,
                                max_eigenvalues: Optional[int] = None,
                                show_legend: bool = True,
                                enable_scale_toggle: bool = True) -> go.Figure:
        """Plot the evolution of eigenvalues with logarithmic scale support.
        
        This is the core visualization method that creates interactive plots
        of eigenvalue evolution through the filtration process.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors for each filtration step
            filtration_params: Filtration parameter values
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            log_scale: Whether to use logarithmic y-axis (None = use default)
            max_eigenvalues: Maximum number of eigenvalues to plot (None = all)
            show_legend: Whether to show the legend
            enable_scale_toggle: Whether to add interactive scale toggle buttons
            
        Returns:
            Interactive Plotly figure
            
        Raises:
            ValidationError: If input data is invalid
        """
        self._validate_eigenvalue_sequences(eigenvalue_sequences, filtration_params)
        
        use_log_scale = log_scale if log_scale is not None else self.default_log_scale
        
        logger.info(f"Creating eigenvalue evolution plot: {len(eigenvalue_sequences)} steps, "
                   f"log_scale={use_log_scale}")
        
        # Prepare eigenvalue paths
        eigen_paths = self._prepare_eigenvalue_paths(eigenvalue_sequences, filtration_params)
        max_k = eigen_paths.shape[0]
        
        # Limit number of eigenvalues if specified (but default to showing all)
        if max_eigenvalues is not None and max_k > max_eigenvalues:
            eigen_paths = eigen_paths[:max_eigenvalues, :]
            max_k = max_eigenvalues
            logger.info(f"Limiting display to {max_eigenvalues} eigenvalues")
        else:
            logger.info(f"Displaying all {max_k} eigenvalues")
        
        # Prepare both linear and log versions of eigenvalue paths
        eigen_paths_linear = eigen_paths.copy()
        eigen_paths_log = np.maximum(eigen_paths, self.min_eigenvalue_threshold)
        
        fig = go.Figure()
        
        # Create trace for each eigenvalue path (both linear and log versions)
        for i in range(max_k):
            eigenvalue_path_linear = eigen_paths_linear[i, :]
            eigenvalue_path_log = eigen_paths_log[i, :]
            
            # Skip entirely NaN paths
            if np.all(np.isnan(eigenvalue_path_linear)):
                continue
            
            # Choose color from palette (cycling if needed)
            color = self.color_palette[i % len(self.color_palette)]
            
            # Create hover text with detailed information
            hover_texts = []
            for j, (param, eigenval) in enumerate(zip(filtration_params, eigenvalue_path_linear)):
                if not np.isnan(eigenval):
                    hover_text = (f"Eigenvalue λ<sub>{i}</sub><br>"
                                f"Parameter: {param:.4f}<br>"
                                f"Value: {eigenval:.6e}<br>"
                                f"Step: {j}")
                    hover_texts.append(hover_text)
                else:
                    hover_texts.append("")
            
            # Add linear scale trace
            fig.add_trace(go.Scatter(
                x=filtration_params,
                y=eigenvalue_path_linear,
                mode='lines+markers',
                name=f'λ<sub>{i}</sub>',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                connectgaps=False,
                hoverinfo='text',
                hovertext=hover_texts,
                visible=True if not use_log_scale else False,
                legendgroup=f'eigenvalue_{i}',
                showlegend=True if i < 20 else False  # Only show first 20 in legend to avoid clutter
            ))
            
            # Add log scale trace (with same styling)
            fig.add_trace(go.Scatter(
                x=filtration_params,
                y=eigenvalue_path_log,
                mode='lines+markers',
                name=f'λ<sub>{i}</sub>',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                connectgaps=False,
                hoverinfo='text',
                hovertext=hover_texts,
                visible=True if use_log_scale else False,
                legendgroup=f'eigenvalue_{i}',
                showlegend=False  # Don't show in legend (duplicate)
            ))
        
        # Configure y-axis for initial display
        initial_y_axis_config = {
            'title': "Eigenvalue" + (" (log scale)" if use_log_scale else ""),
            'showgrid': True,
            'gridcolor': 'lightgray'
        }
        
        if use_log_scale:
            initial_y_axis_config['type'] = 'log'
            # Set range to avoid issues with very small values
            non_nan_values = eigen_paths[~np.isnan(eigen_paths)]
            if len(non_nan_values) > 0:
                min_val = max(np.min(non_nan_values), self.min_eigenvalue_threshold)
                max_val = np.max(non_nan_values)
                initial_y_axis_config['range'] = [np.log10(min_val * 0.1), np.log10(max_val * 10)]
        
        # Prepare update menus for scale toggle
        updatemenus = []
        if enable_scale_toggle:
            # Calculate optimal ranges for both scales
            non_nan_values = eigen_paths[~np.isnan(eigen_paths)]
            linear_range = None
            log_range = None
            
            if len(non_nan_values) > 0:
                min_val = np.min(non_nan_values)
                max_val = np.max(non_nan_values)
                
                # Linear scale range
                padding = (max_val - min_val) * 0.1
                linear_range = [min_val - padding, max_val + padding]
                
                # Log scale range
                min_val_log = max(min_val, self.min_eigenvalue_threshold)
                log_range = [np.log10(min_val_log * 0.1), np.log10(max_val * 10)]
            
            # Create visibility arrays for switching between linear and log traces
            n_traces = max_k * 2  # Each eigenvalue has 2 traces (linear + log)
            linear_visibility = []
            log_visibility = []
            
            for i in range(max_k):
                # Linear visibility: show ALL linear traces, hide ALL log traces
                linear_visibility.extend([True, False])
                # Log visibility: hide ALL linear traces, show ALL log traces  
                log_visibility.extend([False, True])
            
            updatemenus = [
                {
                    "buttons": [
                        {
                            "label": "Linear Scale",
                            "method": "update",
                            "args": [
                                {"visible": linear_visibility},
                                {
                                    "yaxis": {
                                        "title": "Eigenvalue",
                                        "type": "linear",
                                        "showgrid": True,
                                        "gridcolor": "lightgray",
                                        "range": linear_range
                                    }
                                }
                            ]
                        },
                        {
                            "label": "Log Scale", 
                            "method": "update",
                            "args": [
                                {"visible": log_visibility},
                                {
                                    "yaxis": {
                                        "title": "Eigenvalue (log scale)",
                                        "type": "log",
                                        "showgrid": True,
                                        "gridcolor": "lightgray",
                                        "range": log_range
                                    }
                                }
                            ]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.01,
                    "xanchor": "left",
                    "y": 1.02,
                    "yanchor": "top",
                    "bgcolor": "#f0f0f0",
                    "bordercolor": "#cccccc",
                    "borderwidth": 1,
                    "font": {"size": 12}
                }
            ]
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Filtration Parameter",
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=initial_y_axis_config,
            width=width,
            height=height,
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=1.01,
                font=dict(size=10)
            ),
            hovermode='closest',
            plot_bgcolor='white',
            updatemenus=updatemenus
        )
        
        # Add annotation about log scale if enabled
        if use_log_scale:
            fig.add_annotation(
                text="⚠️ Logarithmic y-axis scale for multi-scale visualization",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                font=dict(size=10, color="orange"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="orange",
                borderwidth=1
            )
        
        logger.info(f"Created eigenvalue evolution plot with {max_k} traces")
        
        return fig
    
    def plot_spectral_gap_evolution(self,
                                   eigenvalue_sequences: List[torch.Tensor],
                                   filtration_params: List[float],
                                   title: str = "Spectral Gap Evolution") -> go.Figure:
        """Plot evolution of the spectral gap (difference between first two eigenvalues).
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: Filtration parameter values
            title: Plot title
            
        Returns:
            Interactive Plotly figure
        """
        self._validate_eigenvalue_sequences(eigenvalue_sequences, filtration_params)
        
        # Compute spectral gaps
        spectral_gaps = []
        valid_params = []
        
        for param, eigenvals in zip(filtration_params, eigenvalue_sequences):
            if len(eigenvals) >= 2:
                # Sort eigenvalues to ensure proper gap calculation
                sorted_eigenvals = torch.sort(eigenvals)[0]
                gap = (sorted_eigenvals[1] - sorted_eigenvals[0]).item()
                spectral_gaps.append(gap)
                valid_params.append(param)
            else:
                # No gap if less than 2 eigenvalues
                spectral_gaps.append(np.nan)
                valid_params.append(param)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=valid_params,
            y=spectral_gaps,
            mode='lines+markers',
            name='Spectral Gap',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            connectgaps=False
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Filtration Parameter",
            yaxis_title="Spectral Gap (λ₁ - λ₀)",
            showlegend=False
        )
        
        return fig
    
    def plot_eigenvalue_statistics(self,
                                 eigenvalue_sequences: List[torch.Tensor],
                                 filtration_params: List[float],
                                 title: str = "Eigenvalue Statistics Evolution") -> go.Figure:
        """Plot statistical summaries of eigenvalue evolution.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: Filtration parameter values
            title: Plot title
            
        Returns:
            Multi-trace figure with statistical summaries
        """
        self._validate_eigenvalue_sequences(eigenvalue_sequences, filtration_params)
        
        # Compute statistics for each step
        means, stds, maxs, mins, counts = [], [], [], [], []
        
        for eigenvals in eigenvalue_sequences:
            if len(eigenvals) > 0:
                eigenvals_np = eigenvals.detach().cpu().numpy()
                means.append(np.mean(eigenvals_np))
                stds.append(np.std(eigenvals_np))
                maxs.append(np.max(eigenvals_np))
                mins.append(np.min(eigenvals_np))
                counts.append(len(eigenvals_np))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                maxs.append(np.nan)
                mins.append(np.nan)
                counts.append(0)
        
        fig = go.Figure()
        
        # Add traces for different statistics
        fig.add_trace(go.Scatter(
            x=filtration_params, y=means,
            mode='lines+markers',
            name='Mean',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=filtration_params, y=maxs,
            mode='lines+markers',
            name='Maximum',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=filtration_params, y=mins,
            mode='lines+markers',
            name='Minimum',
            line=dict(color='green')
        ))
        
        # Add secondary y-axis for count
        fig.add_trace(go.Scatter(
            x=filtration_params, y=counts,
            mode='lines+markers',
            name='Count',
            line=dict(color='orange'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Filtration Parameter",
            yaxis=dict(
                title="Eigenvalue",
                side='left'
            ),
            yaxis2=dict(
                title="Count",
                side='right',
                overlaying='y'
            ),
            legend=dict(x=1.1, y=1)
        )
        
        return fig
    
    def plot_eigenvalue_heatmap(self,
                              eigenvalue_sequences: List[torch.Tensor],
                              filtration_params: List[float],
                              title: str = "Eigenvalue Evolution Heatmap",
                              max_eigenvalues: int = 50) -> go.Figure:
        """Create a heatmap visualization of eigenvalue evolution.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: Filtration parameter values
            title: Plot title
            max_eigenvalues: Maximum number of eigenvalues to include
            
        Returns:
            Heatmap figure
        """
        self._validate_eigenvalue_sequences(eigenvalue_sequences, filtration_params)
        
        # Prepare eigenvalue matrix
        eigen_paths = self._prepare_eigenvalue_paths(eigenvalue_sequences, filtration_params)
        
        # Limit eigenvalues if needed
        if eigen_paths.shape[0] > max_eigenvalues:
            eigen_paths = eigen_paths[:max_eigenvalues, :]
        
        # Apply log transformation if using log scale
        if self.default_log_scale:
            eigen_paths = np.log10(eigen_paths + self.min_eigenvalue_threshold)
        
        fig = go.Figure(data=go.Heatmap(
            z=eigen_paths,
            x=filtration_params,
            y=[f'λ{i}' for i in range(eigen_paths.shape[0])],
            colorscale='Viridis',
            colorbar=dict(
                title="Log₁₀(Eigenvalue)" if self.default_log_scale else "Eigenvalue"
            )
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Filtration Parameter",
            yaxis_title="Eigenvalue Index"
        )
        
        return fig
    
    def create_spectral_summary(self,
                              eigenvalue_sequences: List[torch.Tensor],
                              filtration_params: List[float]) -> Dict[str, Any]:
        """Create comprehensive summary statistics for spectral evolution.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: Filtration parameter values
            
        Returns:
            Dictionary with summary statistics
        """
        self._validate_eigenvalue_sequences(eigenvalue_sequences, filtration_params)
        
        # Overall statistics
        all_eigenvals = []
        step_counts = []
        
        for eigenvals in eigenvalue_sequences:
            if len(eigenvals) > 0:
                all_eigenvals.extend(eigenvals.detach().cpu().numpy())
                step_counts.append(len(eigenvals))
            else:
                step_counts.append(0)
        
        if not all_eigenvals:
            return {
                'total_eigenvalues': 0,
                'n_steps': len(eigenvalue_sequences)
            }
        
        all_eigenvals = np.array(all_eigenvals)
        
        summary = {
            'n_steps': len(eigenvalue_sequences),
            'total_eigenvalues': len(all_eigenvals),
            'mean_eigenvalues_per_step': np.mean(step_counts),
            'max_eigenvalues_per_step': max(step_counts),
            'min_eigenvalues_per_step': min(step_counts),
            'eigenvalue_statistics': {
                'mean': np.mean(all_eigenvals),
                'std': np.std(all_eigenvals),
                'min': np.min(all_eigenvals),
                'max': np.max(all_eigenvals),
                'median': np.median(all_eigenvals)
            },
            'filtration_range': (min(filtration_params), max(filtration_params)),
            'zero_eigenvalues': np.sum(all_eigenvals < 1e-10),
            'small_eigenvalues': np.sum(all_eigenvals < 1e-6)
        }
        
        # Spectral gap statistics (where available)
        spectral_gaps = []
        for eigenvals in eigenvalue_sequences:
            if len(eigenvals) >= 2:
                sorted_eigenvals = torch.sort(eigenvals)[0]
                gap = (sorted_eigenvals[1] - sorted_eigenvals[0]).item()
                spectral_gaps.append(gap)
        
        if spectral_gaps:
            summary['spectral_gap_statistics'] = {
                'mean': np.mean(spectral_gaps),
                'std': np.std(spectral_gaps),
                'min': min(spectral_gaps),
                'max': max(spectral_gaps)
            }
        
        return summary
    
    def plot_dtw_alignment(self,
                          alignment_data: Dict[str, Any],
                          title: str = "DTW Alignment Visualization",
                          width: int = 1000,
                          height: int = 700) -> go.Figure:
        """Plot DTW alignment between two eigenvalue evolution sequences.
        
        This method creates a comprehensive visualization of DTW alignment
        showing how eigenvalue evolution patterns are matched between two
        different neural networks or conditions.
        
        Args:
            alignment_data: DTW alignment data from FiltrationDTW
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            
        Returns:
            Interactive Plotly figure showing DTW alignment
        """
        # Extract data from alignment
        sequence1 = alignment_data['sequence1']
        sequence2 = alignment_data['sequence2']
        filtration_params1 = alignment_data['filtration_params1']
        filtration_params2 = alignment_data['filtration_params2']
        alignment = alignment_data['alignment']
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Network 1: Eigenvalue Evolution",
                "Network 2: Eigenvalue Evolution",
                "DTW Alignment Path",
                "Warping Function",
                "Distance Matrix (Heatmap)",
                "Alignment Quality Metrics"
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"rowspan": 2}, {"secondary_y": False}],
                [None, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Network 1 eigenvalue evolution
        if isinstance(sequence1[0], list):
            # Multivariate sequence
            for i, eigenval_series in enumerate(zip(*sequence1)):
                if i < 5:  # Limit to first 5 eigenvalues
                    fig.add_trace(
                        go.Scatter(
                            x=filtration_params1,
                            y=eigenval_series,
                            mode='lines+markers',
                            name=f'Network 1: λ_{i}',
                            line=dict(color=self.color_palette[i % len(self.color_palette)]),
                            showlegend=i < 3
                        ),
                        row=1, col=1
                    )
        else:
            # Univariate sequence
            fig.add_trace(
                go.Scatter(
                    x=filtration_params1,
                    y=sequence1,
                    mode='lines+markers',
                    name='Network 1',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
        
        # Plot 2: Network 2 eigenvalue evolution
        if isinstance(sequence2[0], list):
            # Multivariate sequence
            for i, eigenval_series in enumerate(zip(*sequence2)):
                if i < 5:  # Limit to first 5 eigenvalues
                    fig.add_trace(
                        go.Scatter(
                            x=filtration_params2,
                            y=eigenval_series,
                            mode='lines+markers',
                            name=f'Network 2: λ_{i}',
                            line=dict(color=self.color_palette[i % len(self.color_palette)], dash='dash'),
                            showlegend=i < 3
                        ),
                        row=1, col=2
                    )
        else:
            # Univariate sequence
            fig.add_trace(
                go.Scatter(
                    x=filtration_params2,
                    y=sequence2,
                    mode='lines+markers',
                    name='Network 2',
                    line=dict(color='red', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
        
        # Plot 3: DTW alignment path
        if alignment:
            alignment_x = [filtration_params1[i] for i, j in alignment]
            alignment_y = [filtration_params2[j] for i, j in alignment]
            
            fig.add_trace(
                go.Scatter(
                    x=alignment_x,
                    y=alignment_y,
                    mode='lines+markers',
                    name='DTW Alignment Path',
                    line=dict(color='green', width=2),
                    marker=dict(size=4, color='green')
                ),
                row=2, col=1
            )
            
            # Add diagonal reference line
            min_param = min(min(filtration_params1), min(filtration_params2))
            max_param = max(max(filtration_params1), max(filtration_params2))
            fig.add_trace(
                go.Scatter(
                    x=[min_param, max_param],
                    y=[min_param, max_param],
                    mode='lines',
                    name='Perfect Alignment',
                    line=dict(color='gray', dash='dash', width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 4: Warping function
        if alignment:
            warping_indices = [i for i, j in alignment]
            warping_targets = [j for i, j in alignment]
            
            fig.add_trace(
                go.Scatter(
                    x=warping_indices,
                    y=warping_targets,
                    mode='lines+markers',
                    name='Warping Function',
                    line=dict(color='purple', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=2
            )
            
            # Add diagonal reference
            max_idx = max(max(warping_indices), max(warping_targets))
            fig.add_trace(
                go.Scatter(
                    x=[0, max_idx],
                    y=[0, max_idx],
                    mode='lines',
                    name='No Warping',
                    line=dict(color='gray', dash='dash', width=1),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Plot 5: Distance matrix heatmap (if available)
        if len(sequence1) < 50 and len(sequence2) < 50:  # Only for small sequences
            distance_matrix = self._compute_distance_matrix(sequence1, sequence2)
            
            fig.add_trace(
                go.Heatmap(
                    z=distance_matrix,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Distance", x=0.48, len=0.4)
                ),
                row=3, col=1
            )
        
        # Plot 6: Alignment quality metrics
        alignment_quality = alignment_data.get('alignment_quality', 0.0)
        quality_metrics = [
            alignment_quality,
            1.0 - alignment_quality,  # Complement
            len(alignment) / max(len(sequence1), len(sequence2)) if alignment else 0.0
        ]
        
        fig.add_trace(
            go.Bar(
                x=['Alignment Quality', 'Misalignment', 'Path Coverage'],
                y=quality_metrics,
                name='Quality Metrics',
                marker=dict(color=['green', 'red', 'blue'])
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Filtration Parameter", row=1, col=1)
        fig.update_xaxes(title_text="Filtration Parameter", row=1, col=2)
        fig.update_xaxes(title_text="Network 1 Filtration", row=2, col=1)
        fig.update_xaxes(title_text="Network 1 Index", row=2, col=2)
        fig.update_xaxes(title_text="Network 2 Index", row=3, col=1)
        fig.update_xaxes(title_text="Metrics", row=3, col=2)
        
        fig.update_yaxes(title_text="Eigenvalue", row=1, col=1)
        fig.update_yaxes(title_text="Eigenvalue", row=1, col=2)
        fig.update_yaxes(title_text="Network 2 Filtration", row=2, col=1)
        fig.update_yaxes(title_text="Network 2 Index", row=2, col=2)
        fig.update_yaxes(title_text="Network 1 Index", row=3, col=1)
        fig.update_yaxes(title_text="Quality Score", row=3, col=2)
        
        # Add annotations
        fig.add_annotation(
            text=f"DTW Distance: {alignment_data.get('distance', 0):.4f}<br>"
                 f"Alignment Quality: {alignment_quality:.3f}<br>"
                 f"Path Length: {len(alignment) if alignment else 0}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1
        )
        
        logger.info(f"Created DTW alignment visualization with {len(alignment) if alignment else 0} alignment points")
        
        return fig
    
    def _compute_distance_matrix(self, sequence1: List[float], sequence2: List[float]) -> np.ndarray:
        """Compute distance matrix between two sequences for visualization."""
        n1, n2 = len(sequence1), len(sequence2)
        distance_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                if isinstance(sequence1[i], list) and isinstance(sequence2[j], list):
                    # Multivariate case
                    distance_matrix[i, j] = np.linalg.norm(
                        np.array(sequence1[i]) - np.array(sequence2[j])
                    )
                else:
                    # Univariate case
                    distance_matrix[i, j] = abs(sequence1[i] - sequence2[j])
        
        return distance_matrix
    
    def plot_dtw_comparison_summary(self,
                                   comparison_results: List[Dict[str, Any]],
                                   network_names: List[str],
                                   title: str = "DTW Comparison Summary") -> go.Figure:
        """Plot summary of multiple DTW comparisons.
        
        Args:
            comparison_results: List of DTW comparison results
            network_names: Names of the networks being compared
            title: Plot title
            
        Returns:
            Summary visualization of DTW comparisons
        """
        # Create distance matrix from comparison results
        n_networks = len(network_names)
        distance_matrix = np.zeros((n_networks, n_networks))
        
        # Fill distance matrix
        for i, result in enumerate(comparison_results):
            if 'distance_matrix' in result:
                distance_matrix = result['distance_matrix']
                break
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=distance_matrix,
            x=network_names,
            y=network_names,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="DTW Distance"),
            text=np.round(distance_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Network",
            yaxis_title="Network",
            width=600,
            height=600
        )
        
        return fig