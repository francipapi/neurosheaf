#!/usr/bin/env python3
"""
Computational Graph Visualization Script
Creates a professional diagram of the computational flow: Input → Linear → ReLU → Linear → ReLU
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse
import matplotlib.patches as patches

def create_computational_graph(save_path=None, show_dimensions=True, layout='horizontal', dpi=300):
    """
    Create a professional computational graph visualization.
    
    Args:
        save_path (str, optional): Path to save the figure
        show_dimensions (bool): Whether to show tensor dimensions
        layout (str): 'horizontal' or 'vertical' layout
        dpi (int): DPI for saved figure
    """
    
    # Set up the figure with professional styling
    plt.style.use('default')
    if layout == 'horizontal':
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 14))  # Wider to accommodate dimension labels
    
    # Define colors (matching MLP diagram style)
    colors = {
        'input': '#4A90E2',      # Blue
        'linear': '#7ED321',     # Green
        'relu': '#BD10E0',       # Purple
        'arrow': '#2C3E50',      # Dark blue-gray
        'text': '#2C3E50',       # Dark blue-gray
        'dimension': '#666666'    # Gray
    }
    
    # Define the computational nodes
    nodes = [
        {'name': 'Input', 'type': 'input', 'label': 'Input\n(x)', 'dim': '[2]'},
        {'name': 'Linear1', 'type': 'linear', 'label': 'Linear\n(W₁x + b₁)', 'dim': '[3]'},
        {'name': 'ReLU1', 'type': 'relu', 'label': 'ReLU\nmax(0, z)', 'dim': '[3]'},
        {'name': 'Linear2', 'type': 'linear', 'label': 'Linear\n(W₂a + b₂)', 'dim': '[1]'},
        {'name': 'ReLU2', 'type': 'relu', 'label': 'ReLU\nmax(0, z)', 'dim': '[1]'}
    ]
    
    # Layout parameters
    if layout == 'horizontal':
        spacing = 2.5
        positions = [(i * spacing, 0) for i in range(len(nodes))]
        arrow_direction = 'horizontal'
        node_width = 1.8
        node_height = 1.2
    else:
        spacing = 2.8  # Increased spacing to prevent overlap
        positions = [(0, -i * spacing) for i in range(len(nodes))]
        arrow_direction = 'vertical'
        node_width = 2.2  # Wider nodes for better readability
        node_height = 1.4  # Taller nodes for better text spacing
    
    # Draw arrows between nodes
    for i in range(len(nodes) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        
        if layout == 'horizontal':
            # Horizontal arrows
            arrow_start_x = x1 + node_width/2
            arrow_end_x = x2 - node_width/2
            arrow_y = y1
            
            ax.annotate('', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
        else:
            # Vertical arrows
            arrow_start_y = y1 - node_height/2
            arrow_end_y = y2 + node_height/2
            arrow_x = x1
            
            ax.annotate('', xy=(arrow_x, arrow_end_y), xytext=(arrow_x, arrow_start_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
    
    # Draw nodes
    for i, node in enumerate(nodes):
        x, y = positions[i]
        node_type = node['type']
        
        # Choose shape and color based on node type
        if node_type == 'input':
            # Ellipse for input
            shape = Ellipse((x, y), node_width, node_height, 
                          facecolor=colors['input'], edgecolor='white', 
                          linewidth=2, alpha=0.9)
        elif node_type == 'linear':
            # Rectangle for linear transformations
            shape = Rectangle((x - node_width/2, y - node_height/2), 
                            node_width, node_height,
                            facecolor=colors['linear'], edgecolor='white', 
                            linewidth=2, alpha=0.9)
        else:  # relu
            # Rounded rectangle for activation functions
            shape = FancyBboxPatch((x - node_width/2, y - node_height/2), 
                                 node_width, node_height,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['relu'], edgecolor='white', 
                                 linewidth=2, alpha=0.9)
        
        ax.add_patch(shape)
        
        # Add node label
        ax.text(x, y, node['label'], ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
        
        # Add dimension annotation if enabled
        if show_dimensions:
            if layout == 'horizontal':
                dim_y = y - node_height/2 - 0.3
                dim_x = x
            else:
                dim_y = y
                dim_x = x + node_width/2 + 0.5  # Increased spacing from node edge
                
            ax.text(dim_x, dim_y, node['dim'], ha='left' if layout == 'vertical' else 'center', 
                   va='center', fontsize=10, style='italic', color=colors['dimension'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor='lightgray', linewidth=1))
    
    # Add title
    if layout == 'horizontal':
        title_x = (len(nodes) - 1) * spacing / 2
        title_y = 2.0
    else:
        title_x = 0
        title_y = 1.5
        
    ax.text(title_x, title_y, 'Computational Graph', 
           ha='center', va='center', fontsize=18, fontweight='bold',
           color=colors['text'])
    
    subtitle = 'Input → Linear → ReLU → Linear → ReLU'
    ax.text(title_x, title_y - 0.4, subtitle, 
           ha='center', va='center', fontsize=12, style='italic',
           color=colors['text'])
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['input'], 
                   markersize=12, label='Input Data'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['linear'], 
                   markersize=12, label='Linear Transform'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['relu'], 
                   markersize=12, label='ReLU Activation')
    ]
    
    if layout == 'horizontal':
        legend_loc = 'upper right'
        bbox_to_anchor = (1, 0.95)
    else:
        legend_loc = 'lower right'
        bbox_to_anchor = (0.98, 0.02)  # Position at bottom right to avoid title
    
    ax.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
             frameon=True, fancybox=True, shadow=True)
    
    # Set axis properties
    if layout == 'horizontal':
        ax.set_xlim(-1, len(nodes) * spacing)
        ax.set_ylim(-2.5, 3)
    else:
        ax.set_xlim(-3.5, 4)  # Extended to accommodate dimension labels
        ax.set_ylim(-(len(nodes) - 1) * spacing - 2, 2.5)  # Adjusted for new spacing
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        # Also save as PDF for publication quality
        pdf_path = save_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Computational graph saved as {save_path} and {pdf_path}")
    
    return fig, ax

def create_detailed_graph(save_path=None, dpi=300):
    """
    Create a more detailed computational graph with mathematical expressions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Colors
    colors = {
        'input': '#4A90E2',
        'linear': '#7ED321', 
        'relu': '#BD10E0',
        'arrow': '#2C3E50',
        'text': '#2C3E50'
    }
    
    # Detailed nodes with mathematical expressions
    detailed_nodes = [
        {'label': 'Input\nx ∈ ℝ²', 'type': 'input', 'math': '', 'output': 'x'},
        {'label': 'Linear Layer 1\nz₁ = W₁x + b₁', 'type': 'linear', 'math': 'W₁ ∈ ℝ³ˣ², b₁ ∈ ℝ³', 'output': 'z₁ ∈ ℝ³'},
        {'label': 'ReLU Activation\na₁ = max(0, z₁)', 'type': 'relu', 'math': 'Element-wise', 'output': 'a₁ ∈ ℝ³'},
        {'label': 'Linear Layer 2\nz₂ = W₂a₁ + b₂', 'type': 'linear', 'math': 'W₂ ∈ ℝ¹ˣ³, b₂ ∈ ℝ¹', 'output': 'z₂ ∈ ℝ¹'},
        {'label': 'ReLU Activation\nŷ = max(0, z₂)', 'type': 'relu', 'math': 'Element-wise', 'output': 'ŷ ∈ ℝ¹'}
    ]
    
    # Layout
    spacing = 3.0
    positions = [(i * spacing, 0) for i in range(len(detailed_nodes))]
    node_width = 2.2
    node_height = 1.5
    
    # Draw connections and nodes
    for i, node in enumerate(detailed_nodes):
        x, y = positions[i]
        
        # Draw arrow to next node
        if i < len(detailed_nodes) - 1:
            next_x, next_y = positions[i + 1]
            ax.annotate('', xy=(next_x - node_width/2, next_y), 
                       xytext=(x + node_width/2, y),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['arrow']))
        
        # Choose color
        if node['type'] == 'input':
            color = colors['input']
            shape = FancyBboxPatch((x - node_width/2, y - node_height/2), 
                                 node_width, node_height,
                                 boxstyle="round,pad=0.15", facecolor=color, 
                                 edgecolor='white', linewidth=2, alpha=0.9)
        elif node['type'] == 'linear':
            color = colors['linear']
            shape = Rectangle((x - node_width/2, y - node_height/2), 
                            node_width, node_height, facecolor=color, 
                            edgecolor='white', linewidth=2, alpha=0.9)
        else:  # relu
            color = colors['relu']
            shape = FancyBboxPatch((x - node_width/2, y - node_height/2), 
                                 node_width, node_height,
                                 boxstyle="round,pad=0.1", facecolor=color, 
                                 edgecolor='white', linewidth=2, alpha=0.9)
        
        ax.add_patch(shape)
        
        # Main label
        ax.text(x, y + 0.15, node['label'], ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Mathematical details
        if node['math']:
            ax.text(x, y - 0.25, node['math'], ha='center', va='center', 
                   fontsize=8, style='italic', color='white', alpha=0.8)
        
        # Output annotation
        ax.text(x, y - node_height/2 - 0.4, node['output'], ha='center', va='center',
               fontsize=9, style='italic', color=colors['text'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                        alpha=0.9, edgecolor='gray'))
    
    # Title
    title_x = (len(detailed_nodes) - 1) * spacing / 2
    ax.text(title_x, 2.5, 'Detailed Computational Graph', 
           ha='center', va='center', fontsize=18, fontweight='bold',
           color=colors['text'])
    
    ax.text(title_x, 2.0, 'Forward Pass Through 2-3-1 MLP with Mathematical Expressions', 
           ha='center', va='center', fontsize=12, style='italic',
           color=colors['text'])
    
    # Set axis properties
    ax.set_xlim(-1.5, len(detailed_nodes) * spacing)
    ax.set_ylim(-3, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        pdf_path = save_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Detailed computational graph saved as {save_path} and {pdf_path}")
    
    return fig, ax

def main():
    """Main function to create computational graphs."""
    
    print("Creating computational graph visualizations...")
    
    # Create standard horizontal computational graph
    fig1, ax1 = create_computational_graph(save_path='comp_graph_horizontal.png', 
                                          show_dimensions=True, layout='horizontal')
    
    # Create vertical layout version
    fig2, ax2 = create_computational_graph(save_path='comp_graph_vertical.png',
                                          show_dimensions=True, layout='vertical')
    
    # Create detailed version with mathematical expressions
    fig3, ax3 = create_detailed_graph(save_path='comp_graph_detailed.png')
    
    print("\nCreated three computational graph versions:")
    print("1. Horizontal layout (comp_graph_horizontal.png)")
    print("2. Vertical layout (comp_graph_vertical.png)")  
    print("3. Detailed with math expressions (comp_graph_detailed.png)")
    print("All saved as PNG and PDF formats for publication quality")
    
    # Display the horizontal version
    plt.figure(fig1.number)
    plt.show()

if __name__ == "__main__":
    main()