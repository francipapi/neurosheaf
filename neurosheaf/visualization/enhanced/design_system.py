"""Unified design system for neurosheaf visualizations.

This module provides consistent styling, color schemes, and visual
elements across all visualization components.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class DesignTheme:
    """A complete design theme for visualizations."""
    name: str
    primary_color: str
    secondary_color: str
    accent_color: str
    background_color: str
    text_color: str
    font_family: str
    font_size: int
    
    # Semantic colors
    success_color: str
    warning_color: str
    error_color: str
    info_color: str
    
    # Visualization specific
    node_colors: Dict[str, str]
    edge_colors: Dict[str, str]
    persistence_colors: Dict[str, str]
    spectral_colors: Dict[str, str]


class DesignSystem:
    """Unified design system for consistent styling."""
    
    def __init__(self, theme_name: str = 'neurosheaf_default'):
        """Initialize with a specific theme."""
        self.themes = self._init_themes()
        self.current_theme = self.themes[theme_name]
        
    def _init_themes(self) -> Dict[str, DesignTheme]:
        """Initialize available themes."""
        return {
            'neurosheaf_default': DesignTheme(
                name='Neurosheaf Default',
                primary_color='#2E7D32',
                secondary_color='#1976D2',
                accent_color='#FF6F00',
                background_color='#FAFAFA',
                text_color='#212121',
                font_family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                font_size=12,
                success_color='#2E7D32',
                warning_color='#F57C00',
                error_color='#C62828',
                info_color='#0277BD',
                node_colors={
                    'linear': '#1E88E5',
                    'conv': '#43A047',
                    'activation': '#FF8F00',
                    'normalization': '#8E24AA',
                    'dropout': '#E53935',
                    'attention': '#FFB300',
                    'input': '#4CAF50',
                    'output': '#F44336',
                    'unknown': '#757575'
                },
                edge_colors={
                    'preserve': '#2196F3',
                    'expand': '#4CAF50',
                    'contract': '#F44336',
                    'flow': '#2196F3'
                },
                persistence_colors={
                    'finite': '#1976D2',
                    'infinite': '#D32F2F',
                    'lifetime': '#E65100'
                },
                spectral_colors={
                    'eigenvalue': '#3F51B5',
                    'gap': '#009688',
                    'evolution': '#673AB7'
                }
            ),
            'dark': DesignTheme(
                name='Dark Theme',
                primary_color='#66BB6A',
                secondary_color='#42A5F5',
                accent_color='#FFB74D',
                background_color='#121212',
                text_color='#E0E0E0',
                font_family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                font_size=12,
                success_color='#66BB6A',
                warning_color='#FFB74D',
                error_color='#EF5350',
                info_color='#42A5F5',
                node_colors={
                    'linear': '#42A5F5',
                    'conv': '#66BB6A',
                    'activation': '#FFB74D',
                    'normalization': '#AB47BC',
                    'dropout': '#EF5350',
                    'attention': '#FFCA28',
                    'input': '#66BB6A',
                    'output': '#EF5350',
                    'unknown': '#9E9E9E'
                },
                edge_colors={
                    'preserve': '#42A5F5',
                    'expand': '#66BB6A',
                    'contract': '#EF5350',
                    'flow': '#42A5F5'
                },
                persistence_colors={
                    'finite': '#42A5F5',
                    'infinite': '#EF5350',
                    'lifetime': '#FF7043'
                },
                spectral_colors={
                    'eigenvalue': '#5C6BC0',
                    'gap': '#26A69A',
                    'evolution': '#9575CD'
                }
            )
        }
        
    def get_layout_config(self) -> Dict[str, Any]:
        """Get standard layout configuration."""
        return {
            'plot_bgcolor': self.current_theme.background_color,
            'paper_bgcolor': self.current_theme.background_color,
            'font': {
                'family': self.current_theme.font_family,
                'size': self.current_theme.font_size,
                'color': self.current_theme.text_color
            },
            'showlegend': True,
            'legend': {
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': '#CCCCCC',
                'borderwidth': 1,
                'font': {
                    'family': self.current_theme.font_family,
                    'size': self.current_theme.font_size - 1
                }
            },
            'margin': dict(l=40, r=40, t=60, b=40),
            'hovermode': 'closest'
        }
        
    def get_node_style(self, node_type: str) -> Dict[str, Any]:
        """Get styling for a specific node type."""
        color = self.current_theme.node_colors.get(node_type, self.current_theme.node_colors['unknown'])
        return {
            'color': color,
            'line': {
                'width': 2,
                'color': self.current_theme.text_color
            },
            'opacity': 0.8
        }
        
    def get_edge_style(self, edge_type: str) -> Dict[str, Any]:
        """Get styling for a specific edge type."""
        color = self.current_theme.edge_colors.get(edge_type, self.current_theme.edge_colors['flow'])
        return {
            'color': color,
            'width': 2,
            'opacity': 0.7
        }
        
    def get_colorscale(self, scale_type: str) -> List[List]:
        """Get color scale for different visualization types."""
        scales = {
            'persistence': [
                [0, '#E3F2FD'],
                [0.25, '#90CAF9'],
                [0.5, '#42A5F5'],
                [0.75, '#1E88E5'],
                [1, '#1565C0']
            ],
            'spectral': [
                [0, '#F3E5F5'],
                [0.25, '#CE93D8'],
                [0.5, '#AB47BC'],
                [0.75, '#8E24AA'],
                [1, '#6A1B9A']
            ],
            'flow': [
                [0, '#E8F5E8'],
                [0.25, '#A5D6A7'],
                [0.5, '#66BB6A'],
                [0.75, '#4CAF50'],
                [1, '#388E3C']
            ]
        }
        return scales.get(scale_type, scales['persistence'])
        
    def get_title_style(self, level: int = 1) -> Dict[str, Any]:
        """Get title styling for different hierarchy levels."""
        sizes = {1: 18, 2: 16, 3: 14, 4: 12}
        return {
            'text': '',
            'font': {
                'family': self.current_theme.font_family,
                'size': sizes.get(level, 12),
                'color': self.current_theme.text_color
            },
            'x': 0.5,
            'xanchor': 'center'
        }
        
    def get_axis_style(self) -> Dict[str, Any]:
        """Get axis styling."""
        return {
            'color': self.current_theme.text_color,
            'gridcolor': '#E0E0E0',
            'zerolinecolor': '#CCCCCC',
            'titlefont': {
                'family': self.current_theme.font_family,
                'size': self.current_theme.font_size,
                'color': self.current_theme.text_color
            },
            'tickfont': {
                'family': self.current_theme.font_family,
                'size': self.current_theme.font_size - 1,
                'color': self.current_theme.text_color
            }
        }
        
    def create_hover_template(self, fields: List[str]) -> str:
        """Create consistent hover template."""
        template = "<b>%{text}</b><br>"
        for field in fields:
            template += f"<b>{field}:</b> %{{{field}}}<br>"
        template += "<extra></extra>"
        return template