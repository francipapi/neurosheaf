"""Interactive persistence diagram and barcode visualization.

This module provides detailed visualization of persistence diagrams and 
barcodes computed from persistent spectral analysis of neural sheaves.

Key Features:
- Persistence diagrams with lifetime-based color coding
- Interactive barcodes sorted by birth time
- Proper handling of infinite persistence bars
- Statistical summaries and hover information
- Integration with PersistentSpectralAnalyzer output
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError

logger = setup_logger(__name__)


class PersistenceVisualizer:
    """Interactive visualization of persistence diagrams and barcodes.
    
    This class creates detailed, interactive visualizations of topological
    persistence features extracted from neural sheaf spectral analysis.
    
    The visualizations include:
    - Persistence diagrams showing (birth, death) pairs
    - Persistence barcodes showing feature lifetimes  
    - Statistical summaries and interactive hover information
    - Proper handling of infinite persistence features
    
    Attributes:
        default_colorscale: Default colorscale for lifetime visualization
        infinite_marker_symbol: Symbol for infinite persistence points
        infinite_marker_color: Color for infinite persistence markers
    """
    
    def __init__(self,
                 default_colorscale: str = 'Hot',
                 infinite_marker_symbol: str = 'triangle-up',
                 infinite_marker_color: str = 'red'):
        """Initialize PersistenceVisualizer.
        
        Args:
            default_colorscale: Plotly colorscale for lifetime visualization
            infinite_marker_symbol: Symbol for infinite persistence points
            infinite_marker_color: Color for infinite persistence markers
        """
        self.default_colorscale = default_colorscale
        self.infinite_marker_symbol = infinite_marker_symbol
        self.infinite_marker_color = infinite_marker_color
        
        logger.info("PersistenceVisualizer initialized")
    
    def _validate_diagrams(self, diagrams: Dict) -> None:
        """Validate persistence diagrams structure.
        
        Args:
            diagrams: Dictionary containing persistence diagram data
            
        Raises:
            ValidationError: If diagrams structure is invalid
        """
        required_keys = ['birth_death_pairs', 'infinite_bars']
        for key in required_keys:
            if key not in diagrams:
                raise ValidationError(f"Missing required key '{key}' in diagrams")
        
        # Validate finite pairs structure
        for pair in diagrams['birth_death_pairs']:
            if not all(k in pair for k in ['birth', 'death', 'lifetime']):
                raise ValidationError("Invalid birth_death_pairs structure")
        
        # Validate infinite bars structure  
        for bar in diagrams['infinite_bars']:
            if 'birth' not in bar:
                raise ValidationError("Invalid infinite_bars structure")
    
    def plot_diagram(self, 
                    diagrams: Dict,
                    title: str = "Persistence Diagram",
                    width: int = 600,
                    height: int = 600,
                    show_diagonal: bool = True,
                    colorscale: Optional[str] = None) -> go.Figure:
        """Plot a detailed and interactive persistence diagram.
        
        Args:
            diagrams: Dictionary containing persistence diagram data
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            show_diagonal: Whether to show the diagonal line (birth = death)
            colorscale: Colorscale for lifetime visualization
            
        Returns:
            Interactive Plotly figure
            
        Raises:
            ValidationError: If diagrams structure is invalid
        """
        self._validate_diagrams(diagrams)
        
        logger.info("Creating persistence diagram visualization")
        
        colorscale = colorscale or self.default_colorscale
        
        # Extract finite pairs
        pairs = diagrams.get('birth_death_pairs', [])
        infinite_bars = diagrams.get('infinite_bars', [])
        
        fig = go.Figure()
        
        # Plot finite persistence pairs
        if pairs:
            births = [p['birth'] for p in pairs]
            deaths = [p['death'] for p in pairs]
            lifetimes = [p['lifetime'] for p in pairs]
            
            # Create hover text with detailed information
            hover_texts = []
            for i, pair in enumerate(pairs):
                hover_text = (f"Birth: {pair['birth']:.4f}<br>"
                            f"Death: {pair['death']:.4f}<br>"
                            f"Lifetime: {pair['lifetime']:.4f}")
                
                # Add additional information if available
                if 'birth_step' in pair:
                    hover_text += f"<br>Birth Step: {pair['birth_step']}"
                if 'death_step' in pair:
                    hover_text += f"<br>Death Step: {pair['death_step']}"
                if 'path_id' in pair:
                    hover_text += f"<br>Path ID: {pair['path_id']}"
                
                hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=births, y=deaths,
                mode='markers',
                marker=dict(
                    color=lifetimes,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(
                        title='Lifetime',
                        titleside='right'
                    ),
                    size=10,
                    line=dict(width=2, color='black'),
                    symbol='circle'
                ),
                text=hover_texts,
                hoverinfo='text',
                name='Finite Features'
            ))
        
        # Plot infinite persistence bars
        if infinite_bars:
            inf_births = [p['birth'] for p in infinite_bars]
            
            # Position infinite bars above the finite points
            if pairs:
                max_death = max([p['death'] for p in pairs])
                inf_y_pos = max_death * 1.1
            else:
                inf_y_pos = 1.0
            
            # Create hover text for infinite bars
            inf_hover_texts = []
            for bar in infinite_bars:
                hover_text = f"Birth: {bar['birth']:.4f}<br>Death: ∞"
                if 'birth_step' in bar:
                    hover_text += f"<br>Birth Step: {bar['birth_step']}"
                if 'path_id' in bar:
                    hover_text += f"<br>Path ID: {bar['path_id']}"
                inf_hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=inf_births,
                y=[inf_y_pos] * len(inf_births),
                mode='markers',
                marker=dict(
                    symbol=self.infinite_marker_symbol,
                    color=self.infinite_marker_color,
                    size=14,
                    line=dict(width=2, color='black')
                ),
                text=inf_hover_texts,
                hoverinfo='text',
                name='Infinite Features'
            ))
        
        # Add diagonal line (birth = death)
        if show_diagonal and pairs:
            min_val = min([p['birth'] for p in pairs])
            max_val = max([p['death'] for p in pairs])
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="gray", width=2, dash="dash"),
                name="y = x"
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, family="Arial, sans-serif")
            ),
            xaxis_title="Birth",
            yaxis_title="Death",
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        
        # Add statistics annotation if available
        if 'statistics' in diagrams:
            stats = diagrams['statistics']
            stats_text = (f"Finite pairs: {stats.get('n_finite_pairs', 0)}<br>"
                         f"Infinite bars: {stats.get('n_infinite_bars', 0)}")
            if 'mean_lifetime' in stats:
                stats_text += f"<br>Mean lifetime: {stats['mean_lifetime']:.4f}"
            
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        logger.info(f"Created persistence diagram with {len(pairs)} finite pairs, "
                   f"{len(infinite_bars)} infinite bars")
        
        return fig
    
    def plot_barcode(self,
                    diagrams: Dict,
                    title: str = "Persistence Barcode",
                    width: int = 800,
                    height: int = 400,
                    sort_by: str = 'birth',
                    max_bars: Optional[int] = None) -> go.Figure:
        """Plot a persistence barcode sorted by birth time.
        
        Args:
            diagrams: Dictionary containing persistence diagram data
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
            sort_by: Sort bars by 'birth', 'death', or 'lifetime'
            max_bars: Maximum number of bars to display (None for all)
            
        Returns:
            Interactive Plotly figure
            
        Raises:
            ValidationError: If diagrams structure is invalid
        """
        self._validate_diagrams(diagrams)
        
        logger.info("Creating persistence barcode visualization")
        
        # Combine finite and infinite pairs
        all_pairs = []
        
        # Add finite pairs
        for pair in diagrams.get('birth_death_pairs', []):
            all_pairs.append({
                'birth': pair['birth'],
                'death': pair['death'],
                'lifetime': pair['lifetime'],
                'type': 'finite',
                'path_id': pair.get('path_id', -1)
            })
        
        # Add infinite bars
        for bar in diagrams.get('infinite_bars', []):
            all_pairs.append({
                'birth': bar['birth'],
                'death': float('inf'),
                'lifetime': float('inf'),
                'type': 'infinite',
                'path_id': bar.get('path_id', -1)
            })
        
        if not all_pairs:
            # Create empty plot
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis_title="Filtration Parameter",
                yaxis_title="Feature Index",
                annotations=[dict(
                    text="No persistence features to display",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )]
            )
            return fig
        
        # Sort pairs
        if sort_by == 'birth':
            all_pairs.sort(key=lambda p: p['birth'])
        elif sort_by == 'death':
            all_pairs.sort(key=lambda p: p['death'] if p['death'] != float('inf') else float('inf'))
        elif sort_by == 'lifetime':
            all_pairs.sort(key=lambda p: p['lifetime'] if p['lifetime'] != float('inf') else float('inf'))
        
        # Limit number of bars if specified
        if max_bars is not None and len(all_pairs) > max_bars:
            all_pairs = all_pairs[:max_bars]
            logger.info(f"Limiting display to {max_bars} bars")
        
        fig = go.Figure()
        
        # Determine extent for infinite bars
        finite_deaths = [p['death'] for p in all_pairs if p['death'] != float('inf')]
        max_finite_death = max(finite_deaths) if finite_deaths else 1.0
        
        # Create bars
        for i, pair in enumerate(all_pairs):
            birth = pair['birth']
            death = pair['death']
            
            # Handle infinite bars
            if death == float('inf'):
                death_display = max_finite_death * 1.2
                line_color = self.infinite_marker_color
                line_width = 4
            else:
                death_display = death
                line_color = 'steelblue'
                line_width = 3
            
            # Create hover text
            death_str = '∞' if pair['death'] == float('inf') else f"{pair['death']:.4f}"
            lifetime_str = '∞' if pair['lifetime'] == float('inf') else f"{pair['lifetime']:.4f}"
            hover_text = (f"Birth: {pair['birth']:.4f}<br>"
                         f"Death: {death_str}<br>"
                         f"Lifetime: {lifetime_str}")
            
            if pair['path_id'] != -1:
                hover_text += f"<br>Path ID: {pair['path_id']}"
            
            fig.add_trace(go.Scatter(
                x=[birth, death_display],
                y=[i, i],
                mode='lines',
                line=dict(width=line_width, color=line_color),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False,
                name=f"Feature {i}"
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, family="Arial, sans-serif")
            ),
            xaxis_title="Filtration Parameter",
            yaxis_title="Feature Index",
            yaxis=dict(
                showticklabels=False,  # Hide y-axis labels as they are just indices
                range=[-0.5, len(all_pairs) - 0.5]
            ),
            width=width,
            height=height,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor='black'
        )
        
        # Add annotation for infinite bars if present
        infinite_count = sum(1 for p in all_pairs if p['type'] == 'infinite')
        if infinite_count > 0:
            fig.add_annotation(
                text=f"Red bars indicate infinite persistence ({infinite_count} features)",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                font=dict(size=10, color=self.infinite_marker_color),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        logger.info(f"Created barcode with {len(all_pairs)} features "
                   f"({len(all_pairs) - infinite_count} finite, {infinite_count} infinite)")
        
        return fig
    
    def plot_lifetime_distribution(self,
                                  diagrams: Dict,
                                  title: str = "Persistence Lifetime Distribution",
                                  bins: int = 20) -> go.Figure:
        """Plot distribution of persistence lifetimes.
        
        Args:
            diagrams: Dictionary containing persistence diagram data
            title: Plot title
            bins: Number of histogram bins
            
        Returns:
            Histogram figure
        """
        self._validate_diagrams(diagrams)
        
        # Extract finite lifetimes only
        pairs = diagrams.get('birth_death_pairs', [])
        lifetimes = [p['lifetime'] for p in pairs if p['lifetime'] != float('inf')]
        
        if not lifetimes:
            # Create empty plot
            fig = go.Figure()
            fig.update_layout(
                title=title,
                annotations=[dict(
                    text="No finite lifetimes to display",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )]
            )
            return fig
        
        fig = go.Figure(data=[
            go.Histogram(
                x=lifetimes,
                nbinsx=bins,
                marker_color='lightblue',
                opacity=0.7,
                name='Lifetimes'
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Lifetime",
            yaxis_title="Count",
            showlegend=False
        )
        
        # Add statistics
        mean_lifetime = np.mean(lifetimes)
        median_lifetime = np.median(lifetimes)
        
        fig.add_vline(
            x=mean_lifetime,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_lifetime:.4f}"
        )
        
        fig.add_vline(
            x=median_lifetime,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_lifetime:.4f}"
        )
        
        return fig
    
    def create_summary_stats(self, diagrams: Dict) -> Dict[str, Any]:
        """Create comprehensive summary statistics for persistence diagrams.
        
        Args:
            diagrams: Dictionary containing persistence diagram data
            
        Returns:
            Dictionary with summary statistics
        """
        self._validate_diagrams(diagrams)
        
        pairs = diagrams.get('birth_death_pairs', [])
        infinite_bars = diagrams.get('infinite_bars', [])
        
        if not pairs and not infinite_bars:
            return {
                'n_finite_pairs': 0,
                'n_infinite_bars': 0,
                'total_features': 0
            }
        
        stats = {
            'n_finite_pairs': len(pairs),
            'n_infinite_bars': len(infinite_bars),
            'total_features': len(pairs) + len(infinite_bars)
        }
        
        # Statistics for finite pairs
        if pairs:
            lifetimes = [p['lifetime'] for p in pairs]
            births = [p['birth'] for p in pairs]
            deaths = [p['death'] for p in pairs]
            
            stats.update({
                'mean_lifetime': np.mean(lifetimes),
                'median_lifetime': np.median(lifetimes),
                'std_lifetime': np.std(lifetimes),
                'max_lifetime': max(lifetimes),
                'min_lifetime': min(lifetimes),
                'total_persistence': sum(lifetimes),
                'birth_range': (min(births), max(births)),
                'death_range': (min(deaths), max(deaths))
            })
        
        # Add infinite bars statistics
        if infinite_bars:
            inf_births = [b['birth'] for b in infinite_bars]
            stats['infinite_birth_range'] = (min(inf_births), max(inf_births))
        
        return stats
        
    def plot_lifetime_distribution(self, diagrams: Dict, 
                                  title: str = "Persistence Lifetime Distribution",
                                  bins: int = 20,
                                  **kwargs) -> go.Figure:
        """Plot distribution of persistence lifetimes."""
        self._validate_diagrams(diagrams)
        
        pairs = diagrams.get('birth_death_pairs', [])
        if not pairs:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis_title="Lifetime",
                yaxis_title="Count",
                annotations=[
                    dict(text="No finite persistence pairs found", 
                         x=0.5, y=0.5, showarrow=False, 
                         xref="paper", yref="paper")
                ]
            )
            return fig
        
        lifetimes = [p['lifetime'] for p in pairs]
        
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=lifetimes,
            nbinsx=bins,
            name="Lifetime Distribution",
            marker=dict(color="steelblue", opacity=0.7),
            hovertemplate="Lifetime: %{x}<br>Count: %{y}<extra></extra>"
        ))
        
        # Add mean line
        mean_lifetime = np.mean(lifetimes)
        fig.add_vline(
            x=mean_lifetime,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text=f"Mean: {mean_lifetime:.4f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Lifetime",
            yaxis_title="Count",
            **kwargs
        )
        
        return fig