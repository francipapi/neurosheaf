"""Enhanced spectral visualization with advanced analysis features.

This module provides advanced spectral analysis visualization including
intelligent eigenvalue tracking, multi-scale display, and contextual
analysis integrated with the design system.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from .enhanced.design_system import DesignSystem
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class EnhancedSpectralVisualizer:
    """Advanced spectral visualization with comprehensive features."""
    
    def __init__(self, design_system: Optional[DesignSystem] = None):
        """Initialize enhanced spectral visualizer."""
        self.design_system = design_system or DesignSystem()
        self._eigenvalue_cache = {}
        
    def create_comprehensive_spectral_view(self,
                                         eigenvalue_sequences: List[torch.Tensor],
                                         filtration_params: List[float],
                                         title: str = "Spectral Evolution Analysis",
                                         width: int = 1400,
                                         height: int = 900) -> go.Figure:
        """
        Create a comprehensive spectral analysis view.
        
        Args:
            eigenvalue_sequences: List of eigenvalue tensors
            filtration_params: Filtration parameter values
            title: Plot title
            width: Plot width
            height: Plot height
            
        Returns:
            Comprehensive spectral analysis figure
        """
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Eigenvalue Evolution (Log Scale)",
                "Spectral Gap Analysis",
                "Eigenvalue Statistics",
                "Spectral Density Heatmap",
                "Cumulative Spectral Energy",
                "Stability Analysis"
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
                [{"type": "heatmap"}, {"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Apply design system
        layout_config = self.design_system.get_layout_config()
        layout_config.update({
            'width': width,
            'height': height,
            'title': {
                **self.design_system.get_title_style(1),
                'text': title
            }
        })
        fig.update_layout(**layout_config)
        
        # Analyze eigenvalue sequences
        analysis = self._analyze_eigenvalue_sequences(eigenvalue_sequences, filtration_params)
        
        # 1. Eigenvalue Evolution (Log Scale)
        self._add_eigenvalue_evolution_plot(fig, analysis, row=1, col=1)
        
        # 2. Spectral Gap Analysis
        self._add_spectral_gap_analysis(fig, analysis, row=1, col=2)
        
        # 3. Eigenvalue Statistics
        self._add_eigenvalue_statistics(fig, analysis, row=1, col=3)
        
        # 4. Spectral Density Heatmap
        self._add_spectral_density_heatmap(fig, analysis, row=2, col=1)
        
        # 5. Cumulative Spectral Energy
        self._add_cumulative_energy_plot(fig, analysis, row=2, col=2)
        
        # 6. Stability Analysis
        self._add_stability_analysis(fig, analysis, row=2, col=3)
        
        # Add comprehensive annotations
        self._add_spectral_annotations(fig, analysis)
        
        return fig
        
    def _analyze_eigenvalue_sequences(self,
                                    eigenvalue_sequences: List[torch.Tensor],
                                    filtration_params: List[float]) -> Dict[str, Any]:
        """Comprehensive analysis of eigenvalue sequences."""
        analysis = {
            'filtration_params': filtration_params,
            'eigenvalue_sequences': eigenvalue_sequences,
            'n_steps': len(eigenvalue_sequences),
            'max_eigenvals': max(len(seq) for seq in eigenvalue_sequences if len(seq) > 0),
        }
        
        # Prepare eigenvalue matrix
        eigenval_matrix = np.full((analysis['max_eigenvals'], analysis['n_steps']), np.nan)
        
        for i, seq in enumerate(eigenvalue_sequences):
            if len(seq) > 0:
                eigenvals = seq.detach().cpu().numpy()
                eigenvals = np.maximum(eigenvals, 1e-12)  # Prevent log(0)
                eigenval_matrix[:len(eigenvals), i] = eigenvals
                
        analysis['eigenval_matrix'] = eigenval_matrix
        
        # Compute spectral gaps
        analysis['spectral_gaps'] = self._compute_spectral_gaps(eigenvalue_sequences)
        
        # Compute eigenvalue statistics
        analysis['statistics'] = self._compute_eigenvalue_statistics(eigenvalue_sequences)
        
        # Compute stability metrics
        analysis['stability'] = self._compute_stability_metrics(eigenval_matrix)
        
        # Compute spectral energy
        analysis['spectral_energy'] = self._compute_spectral_energy(eigenvalue_sequences)
        
        return analysis
        
    def _add_eigenvalue_evolution_plot(self,
                                     fig: go.Figure,
                                     analysis: Dict[str, Any],
                                     row: int,
                                     col: int):
        """Add eigenvalue evolution plot with intelligent tracking."""
        eigenval_matrix = analysis['eigenval_matrix']
        filtration_params = analysis['filtration_params']
        
        # Color scheme for eigenvalue tracks
        colors = self.design_system.get_colorscale('spectral')
        n_colors = len(colors)
        
        # Plot eigenvalue tracks
        n_plot = eigenval_matrix.shape[0]  # Plot ALL eigenvalue tracks
        
        for i in range(n_plot):
            eigenval_track = eigenval_matrix[i, :]
            
            # Filter valid values
            valid_mask = ~np.isnan(eigenval_track)
            if not np.any(valid_mask):
                continue
                
            valid_params = np.array(filtration_params)[valid_mask]
            valid_eigenvals = eigenval_track[valid_mask]
            
            # Choose color
            color_idx = int(i / n_plot * (n_colors - 1))
            color = colors[color_idx][1]
            
            # Create trace
            fig.add_trace(
                go.Scatter(
                    x=valid_params,
                    y=valid_eigenvals,
                    mode='lines+markers',
                    name=f'λ_{i}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=i < 5,  # Only show first 5 in legend
                    legendgroup='eigenvalues'
                ),
                row=row, col=col
            )
            
        # Set log scale
        fig.update_yaxes(type="log", row=row, col=col)
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Eigenvalue (log scale)", row=row, col=col)
        
    def _add_spectral_gap_analysis(self,
                                  fig: go.Figure,
                                  analysis: Dict[str, Any],
                                  row: int,
                                  col: int):
        """Add spectral gap analysis."""
        spectral_gaps = analysis['spectral_gaps']
        filtration_params = analysis['filtration_params']
        
        # Main spectral gap
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=spectral_gaps,
                mode='lines+markers',
                name='Spectral Gap',
                line=dict(color=self.design_system.current_theme.spectral_colors['gap'], width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor="rgba(0,150,136,0.3)"
            ),
            row=row, col=col
        )
        
        # Add critical points
        gap_array = np.array(spectral_gaps)
        if len(gap_array) > 2:
            # Find local minima (potential phase transitions)
            minima_indices = []
            for i in range(1, len(gap_array) - 1):
                if gap_array[i] < gap_array[i-1] and gap_array[i] < gap_array[i+1]:
                    minima_indices.append(i)
                    
            if minima_indices:
                fig.add_trace(
                    go.Scatter(
                        x=[filtration_params[i] for i in minima_indices],
                        y=[gap_array[i] for i in minima_indices],
                        mode='markers',
                        name='Critical Points',
                        marker=dict(
                            color=self.design_system.current_theme.warning_color,
                            size=10,
                            symbol='x'
                        )
                    ),
                    row=row, col=col
                )
                
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Spectral Gap", row=row, col=col)
        
    def _add_eigenvalue_statistics(self,
                                  fig: go.Figure,
                                  analysis: Dict[str, Any],
                                  row: int,
                                  col: int):
        """Add eigenvalue statistics plot."""
        stats = analysis['statistics']
        filtration_params = analysis['filtration_params']
        
        # Mean eigenvalue
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=stats['mean_eigenvals'],
                mode='lines',
                name='Mean',
                line=dict(color=self.design_system.current_theme.primary_color, width=2),
                yaxis='y'
            ),
            row=row, col=col
        )
        
        # Eigenvalue count (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=stats['eigenval_counts'],
                mode='lines',
                name='Count',
                line=dict(color=self.design_system.current_theme.secondary_color, width=2, dash='dash'),
                yaxis='y2'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Mean Eigenvalue", row=row, col=col)
        
        # Update secondary y-axis
        fig.update_layout(**{
            f'yaxis{4 if row == 2 else col + 1}': dict(
                title="Eigenvalue Count",
                side="right",
                overlaying=f'y{4 if row == 2 else col + 1}',
                color=self.design_system.current_theme.secondary_color
            )
        })
        
    def _add_spectral_density_heatmap(self,
                                     fig: go.Figure,
                                     analysis: Dict[str, Any],
                                     row: int,
                                     col: int):
        """Add spectral density heatmap."""
        eigenval_matrix = analysis['eigenval_matrix']
        filtration_params = analysis['filtration_params']
        
        # Take log for better visualization
        log_eigenvals = np.log10(eigenval_matrix + 1e-12)
        
        # Use all eigenvalues for comprehensive view
        max_eigenvals = eigenval_matrix.shape[0]
        
        fig.add_trace(
            go.Heatmap(
                z=log_eigenvals[:max_eigenvals, :],
                x=filtration_params,
                y=[f'λ_{i}' for i in range(max_eigenvals)],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="log₁₀(Eigenvalue)",
                    titleside="right"
                ),
                hoverongaps=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Eigenvalue Index", row=row, col=col)
        
    def _add_cumulative_energy_plot(self,
                                   fig: go.Figure,
                                   analysis: Dict[str, Any],
                                   row: int,
                                   col: int):
        """Add cumulative spectral energy plot."""
        spectral_energy = analysis['spectral_energy']
        filtration_params = analysis['filtration_params']
        
        # Total spectral energy
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=spectral_energy['total_energy'],
                mode='lines+markers',
                name='Total Energy',
                line=dict(color=self.design_system.current_theme.accent_color, width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor="rgba(255,193,7,0.3)"
            ),
            row=row, col=col
        )
        
        # 90% energy cutoff
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=spectral_energy['energy_90_percent'],
                mode='lines',
                name='90% Energy Level',
                line=dict(color=self.design_system.current_theme.info_color, width=2, dash='dash')
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Spectral Energy", row=row, col=col)
        
    def _add_stability_analysis(self,
                               fig: go.Figure,
                               analysis: Dict[str, Any],
                               row: int,
                               col: int):
        """Add stability analysis."""
        stability = analysis['stability']
        filtration_params = analysis['filtration_params']
        
        # Eigenvalue variance (measure of instability)
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=stability['eigenval_variance'],
                mode='lines+markers',
                name='Eigenvalue Variance',
                line=dict(color=self.design_system.current_theme.warning_color, width=2),
                marker=dict(size=4)
            ),
            row=row, col=col
        )
        
        # Condition number
        fig.add_trace(
            go.Scatter(
                x=filtration_params,
                y=stability['condition_numbers'],
                mode='lines',
                name='Condition Number',
                line=dict(color=self.design_system.current_theme.error_color, width=2),
                yaxis='y2'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Eigenvalue Variance", row=row, col=col)
        
    def _add_spectral_annotations(self,
                                 fig: go.Figure,
                                 analysis: Dict[str, Any]):
        """Add comprehensive spectral annotations."""
        # Summary statistics
        n_steps = analysis['n_steps']
        max_eigenvals = analysis['max_eigenvals']
        
        summary_text = (
            f"<b>Spectral Analysis Summary:</b><br>"
            f"• Filtration Steps: {n_steps}<br>"
            f"• Max Eigenvalues: {max_eigenvals}<br>"
            f"• Parameter Range: [{analysis['filtration_params'][0]:.3f}, {analysis['filtration_params'][-1]:.3f}]<br>"
            f"• Mean Spectral Gap: {np.mean(analysis['spectral_gaps']):.4f}"
        )
        
        fig.add_annotation(
            x=0.02, y=0.98,
            text=summary_text,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
        
    def _compute_spectral_gaps(self, eigenvalue_sequences: List[torch.Tensor]) -> List[float]:
        """Compute spectral gaps for each filtration step."""
        gaps = []
        
        for eigenvals in eigenvalue_sequences:
            if len(eigenvals) >= 2:
                sorted_vals = torch.sort(eigenvals)[0]
                gap = (sorted_vals[1] - sorted_vals[0]).item()
                gaps.append(gap)
            else:
                gaps.append(0.0)
                
        return gaps
        
    def _compute_eigenvalue_statistics(self, eigenvalue_sequences: List[torch.Tensor]) -> Dict[str, List[float]]:
        """Compute comprehensive eigenvalue statistics."""
        stats = {
            'mean_eigenvals': [],
            'eigenval_counts': [],
            'max_eigenvals': [],
            'min_eigenvals': [],
            'eigenval_std': []
        }
        
        for eigenvals in eigenvalue_sequences:
            if len(eigenvals) > 0:
                stats['mean_eigenvals'].append(torch.mean(eigenvals).item())
                stats['eigenval_counts'].append(len(eigenvals))
                stats['max_eigenvals'].append(torch.max(eigenvals).item())
                stats['min_eigenvals'].append(torch.min(eigenvals).item())
                stats['eigenval_std'].append(torch.std(eigenvals).item())
            else:
                stats['mean_eigenvals'].append(0.0)
                stats['eigenval_counts'].append(0)
                stats['max_eigenvals'].append(0.0)
                stats['min_eigenvals'].append(0.0)
                stats['eigenval_std'].append(0.0)
                
        return stats
        
    def _compute_stability_metrics(self, eigenval_matrix: np.ndarray) -> Dict[str, List[float]]:
        """Compute stability metrics."""
        stability = {
            'eigenval_variance': [],
            'condition_numbers': []
        }
        
        for i in range(eigenval_matrix.shape[1]):
            eigenvals = eigenval_matrix[:, i]
            valid_eigenvals = eigenvals[~np.isnan(eigenvals)]
            
            if len(valid_eigenvals) > 1:
                stability['eigenval_variance'].append(np.var(valid_eigenvals))
                # Compute condition number safely
                positive_eigenvals = valid_eigenvals[valid_eigenvals > 1e-10]
                if len(positive_eigenvals) > 0:
                    condition_num = np.max(positive_eigenvals) / np.min(positive_eigenvals)
                    stability['condition_numbers'].append(condition_num)
                else:
                    stability['condition_numbers'].append(1.0)
            else:
                stability['eigenval_variance'].append(0.0)
                stability['condition_numbers'].append(1.0)
                
        return stability
        
    def _compute_spectral_energy(self, eigenvalue_sequences: List[torch.Tensor]) -> Dict[str, List[float]]:
        """Compute spectral energy measures."""
        energy = {
            'total_energy': [],
            'energy_90_percent': []
        }
        
        for eigenvals in eigenvalue_sequences:
            if len(eigenvals) > 0:
                total_energy = torch.sum(eigenvals ** 2).item()
                energy['total_energy'].append(total_energy)
                
                # Find 90% energy level
                sorted_vals = torch.sort(eigenvals, descending=True)[0]
                cumsum = torch.cumsum(sorted_vals ** 2, dim=0)
                energy_90_idx = torch.argmax((cumsum >= 0.9 * total_energy).float())
                energy['energy_90_percent'].append(sorted_vals[energy_90_idx].item())
            else:
                energy['total_energy'].append(0.0)
                energy['energy_90_percent'].append(0.0)
                
        return energy
        
    def plot_eigenvalue_evolution(self, eigenvalue_sequences: List[torch.Tensor], 
                                filtration_params: List[float],
                                title: str = "Eigenvalue Evolution",
                                enable_scale_toggle: bool = True) -> go.Figure:
        """Plot eigenvalue evolution with interactive scale toggle."""
        fig = go.Figure()
        
        # Prepare eigenvalue matrix
        max_eigenvals = max(len(seq) for seq in eigenvalue_sequences if len(seq) > 0)
        eigenval_matrix = np.full((max_eigenvals, len(filtration_params)), np.nan)
        eigenval_matrix_log = np.full((max_eigenvals, len(filtration_params)), np.nan)
        
        for i, seq in enumerate(eigenvalue_sequences):
            if len(seq) > 0:
                eigenvals = seq.detach().cpu().numpy()
                eigenval_matrix[:len(eigenvals), i] = eigenvals
                eigenval_matrix_log[:len(eigenvals), i] = np.maximum(eigenvals, 1e-12)
                
        # Plot ALL eigenvalue tracks (no limit)
        n_plot = max_eigenvals  # Plot all eigenvalues, no limit
        colors = self.design_system.get_colorscale('spectral')
        
        for i in range(n_plot):
            track_linear = eigenval_matrix[i, :]
            track_log = eigenval_matrix_log[i, :]
            valid_mask = ~np.isnan(track_linear)
            
            if np.any(valid_mask):
                color_idx = int(i / n_plot * (len(colors) - 1))
                color = colors[color_idx][1]
                
                # Linear trace
                fig.add_trace(go.Scatter(
                    x=np.array(filtration_params)[valid_mask],
                    y=track_linear[valid_mask],
                    mode='lines+markers',
                    name=f'λ_{i}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    visible=False,  # Start with log scale (hide all linear)
                    legendgroup=f'eigenvalue_{i}',
                    showlegend=True if i < 20 else False  # Only show first 20 in legend to avoid clutter
                ))
                
                # Log trace
                fig.add_trace(go.Scatter(
                    x=np.array(filtration_params)[valid_mask],
                    y=track_log[valid_mask],
                    mode='lines+markers',
                    name=f'λ_{i}',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    visible=True,  # Start with log scale
                    legendgroup=f'eigenvalue_{i}',
                    showlegend=False  # Don't duplicate in legend
                ))
        
        # Prepare update menus for scale toggle
        updatemenus = []
        if enable_scale_toggle:
            # Calculate optimal ranges
            non_nan_linear = eigenval_matrix[~np.isnan(eigenval_matrix)]
            non_nan_log = eigenval_matrix_log[~np.isnan(eigenval_matrix_log)]
            
            linear_range = None
            log_range = None
            
            if len(non_nan_linear) > 0:
                min_val = np.min(non_nan_linear)
                max_val = np.max(non_nan_linear)
                padding = (max_val - min_val) * 0.1
                linear_range = [min_val - padding, max_val + padding]
                
                min_val_log = np.min(non_nan_log)
                max_val_log = np.max(non_nan_log)
                log_range = [np.log10(min_val_log * 0.1), np.log10(max_val_log * 10)]
            
            # Create visibility arrays
            n_traces = n_plot * 2
            linear_visibility = []
            log_visibility = []
            
            for i in range(n_plot):
                linear_visibility.extend([True, False])  # Show ALL linear, hide ALL log
                log_visibility.extend([False, True])     # Hide ALL linear, show ALL log
            
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
        
        # Apply layout configuration
        layout_config = self.design_system.get_layout_config()
        layout_config.update({
            'title': title,
            'xaxis_title': "Filtration Parameter",
            'yaxis_title': "Eigenvalue (log scale)",
            'yaxis_type': "log",
            'updatemenus': updatemenus
        })
        
        fig.update_layout(**layout_config)
        
        return fig
        
    def plot_spectral_gap_evolution(self, eigenvalue_sequences: List[torch.Tensor],
                                   filtration_params: List[float],
                                   title: str = "Spectral Gap Evolution") -> go.Figure:
        """Plot spectral gap evolution."""
        fig = go.Figure()
        
        # Compute spectral gaps
        gaps = self._compute_spectral_gaps(eigenvalue_sequences)
        
        fig.add_trace(go.Scatter(
            x=filtration_params,
            y=gaps,
            mode='lines+markers',
            name='Spectral Gap',
            line=dict(color=self.design_system.current_theme.spectral_colors['gap'], width=3),
            marker=dict(size=6),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Filtration Parameter",
            yaxis_title="Spectral Gap",
            **self.design_system.get_layout_config()
        )
        
        return fig
        
    def plot_eigenvalue_statistics(self, eigenvalue_sequences: List[torch.Tensor],
                                 filtration_params: List[float],
                                 title: str = "Eigenvalue Statistics") -> go.Figure:
        """Plot eigenvalue statistics."""
        fig = go.Figure()
        
        stats = self._compute_eigenvalue_statistics(eigenvalue_sequences)
        
        # Mean eigenvalue
        fig.add_trace(go.Scatter(
            x=filtration_params,
            y=stats['mean_eigenvals'],
            mode='lines+markers',
            name='Mean Eigenvalue',
            line=dict(color=self.design_system.current_theme.primary_color, width=2)
        ))
        
        # Standard deviation
        fig.add_trace(go.Scatter(
            x=filtration_params,
            y=stats['eigenval_std'],
            mode='lines+markers',
            name='Standard Deviation',
            line=dict(color=self.design_system.current_theme.secondary_color, width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Filtration Parameter",
            yaxis_title="Eigenvalue Statistics",
            **self.design_system.get_layout_config()
        )
        
        return fig
        
    def plot_eigenvalue_heatmap(self, eigenvalue_sequences: List[torch.Tensor],
                              filtration_params: List[float],
                              title: str = "Eigenvalue Heatmap",
                              max_eigenvalues: int = None) -> go.Figure:
        """Plot eigenvalue heatmap."""
        fig = go.Figure()
        
        # Prepare eigenvalue matrix
        actual_max_eigenvals = max(len(seq) for seq in eigenvalue_sequences if len(seq) > 0)
        max_eigenvals = max_eigenvalues if max_eigenvalues is not None else actual_max_eigenvals
        eigenval_matrix = np.full((max_eigenvals, len(filtration_params)), np.nan)
        
        for i, seq in enumerate(eigenvalue_sequences):
            if len(seq) > 0:
                eigenvals = seq.detach().cpu().numpy()
                eigenvals = np.maximum(eigenvals, 1e-12)
                eigenval_matrix[:min(len(eigenvals), max_eigenvals), i] = eigenvals[:max_eigenvals]
                
        # Take log for better visualization
        log_eigenvals = np.log10(eigenval_matrix)
        
        fig.add_trace(go.Heatmap(
            z=log_eigenvals,
            x=filtration_params,
            y=[f'λ_{i}' for i in range(max_eigenvals)],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="log₁₀(Eigenvalue)")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Filtration Parameter",
            yaxis_title="Eigenvalue Index",
            **self.design_system.get_layout_config()
        )
        
        return fig