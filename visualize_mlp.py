#!/usr/bin/env python3
"""
Professional MLP Visualization Script
Creates a clean, publication-quality diagram of a 2-3-1 MLP with ReLU activations.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as patches

def create_mlp_diagram(save_path=None, show_weights=False, dpi=300):
    """
    Create a professional MLP diagram for 2-3-1 architecture with ReLU activations.
    
    Args:
        save_path (str, optional): Path to save the figure
        show_weights (bool): Whether to show connection weights
        dpi (int): DPI for saved figure
    """
    
    # Set up the figure with professional styling
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors
    colors = {
        'input': '#4A90E2',      # Blue
        'hidden': '#7ED321',     # Green  
        'output': '#F5A623',     # Orange
        'relu': '#BD10E0',       # Purple
        'connection': '#9B9B9B',  # Gray
        'text': '#2C3E50'        # Dark blue-gray
    }
    
    # Network architecture
    layers = [2, 3, 1]  # 2-3-1 architecture
    layer_names = ['Input Layer', 'Hidden Layer', 'Output Layer']
    
    # Layout parameters
    layer_spacing = 3.0
    max_neurons = max(layers)
    neuron_radius = 0.3
    
    # Calculate positions for each neuron
    positions = {}
    layer_positions = {}
    
    for layer_idx, num_neurons in enumerate(layers):
        x = layer_idx * layer_spacing
        layer_positions[layer_idx] = x
        
        # Center neurons vertically
        if num_neurons == 1:
            y_positions = [0]
        else:
            y_start = -(num_neurons - 1) * 0.8 / 2
            y_positions = [y_start + i * 0.8 for i in range(num_neurons)]
        
        positions[layer_idx] = [(x, y) for y in y_positions]
    
    # Draw connections between layers
    np.random.seed(42)  # For consistent weight visualization
    for layer_idx in range(len(layers) - 1):
        for i, (x1, y1) in enumerate(positions[layer_idx]):
            for j, (x2, y2) in enumerate(positions[layer_idx + 1]):
                # Generate random weight for visualization
                weight = np.random.normal(0, 0.5)
                
                # Line thickness based on weight magnitude
                linewidth = max(0.5, min(3.0, abs(weight) * 2))
                alpha = max(0.3, min(1.0, abs(weight)))
                
                # Draw connection
                ax.plot([x1 + neuron_radius, x2 - neuron_radius], 
                       [y1, y2], 
                       color=colors['connection'], 
                       linewidth=linewidth, 
                       alpha=alpha,
                       zorder=1)
                
                # Optionally show weight values
                if show_weights:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f'{weight:.2f}', 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.8))
    
    # Draw neurons
    for layer_idx, layer_positions_list in positions.items():
        if layer_idx == 0:
            color = colors['input']
            layer_name = 'Input'
        elif layer_idx == len(layers) - 1:
            color = colors['output']  
            layer_name = 'Output'
        else:
            color = colors['hidden']
            layer_name = 'Hidden'
            
        for neuron_idx, (x, y) in enumerate(layer_positions_list):
            # Draw neuron circle
            circle = Circle((x, y), neuron_radius, 
                          facecolor=color, edgecolor='white', 
                          linewidth=2, zorder=3)
            ax.add_patch(circle)
            
            # Add neuron labels
            if layer_idx == 0:
                label = f'x₁' if neuron_idx == 0 else f'x₂'
            elif layer_idx == len(layers) - 1:
                label = 'ŷ'
            else:
                label = f'z₁' if neuron_idx == 0 else f'z₂' if neuron_idx == 1 else f'z₃'
                
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white', zorder=4)
    
    # Add ReLU activation indicators between layers
    for layer_idx in range(len(layers) - 1):
        x_pos = (layer_positions[layer_idx] + layer_positions[layer_idx + 1]) / 2
        
        # Only show ReLU for hidden layers (not output)
        if layer_idx < len(layers) - 2:  # Between input-hidden and hidden-hidden
            # ReLU activation box
            rect = FancyBboxPatch((x_pos - 0.3, -2.2), 0.6, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['relu'], 
                                 edgecolor='white',
                                 linewidth=1.5,
                                 alpha=0.9,
                                 zorder=2)
            ax.add_patch(rect)
            
            ax.text(x_pos, -2.0, 'ReLU', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white', zorder=4)
    
    # Add layer labels
    for layer_idx, layer_name in enumerate(layer_names):
        x_pos = layer_positions[layer_idx]
        ax.text(x_pos, -3.0, layer_name, ha='center', va='center',
               fontsize=14, fontweight='bold', color=colors['text'])
    
    # Add title
    ax.text(layer_spacing, 2.5, 'Multi-Layer Perceptron (2-3-1)', 
           ha='center', va='center', fontsize=18, fontweight='bold',
           color=colors['text'])
    
    ax.text(layer_spacing, 2.1, 'ReLU Activation Functions', 
           ha='center', va='center', fontsize=12, style='italic',
           color=colors['text'])
    
    # Set axis properties
    ax.set_xlim(-0.8, layer_spacing * 2 + 0.8)
    ax.set_ylim(-3.5, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['input'], 
                   markersize=12, label='Input Layer'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['hidden'], 
                   markersize=12, label='Hidden Layer'),  
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['output'], 
                   markersize=12, label='Output Layer'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['relu'], 
                   markersize=8, label='ReLU Activation')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
             frameon=True, fancybox=True, shadow=True)
    
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
        print(f"Diagram saved as {save_path} and {pdf_path}")
    
    return fig, ax

def main():
    """Main function to create and display the MLP diagram."""
    
    # Create the diagram
    fig, ax = create_mlp_diagram(save_path='mlp_2_3_1_diagram.png', 
                                show_weights=False, dpi=300)
    
    # Display the diagram
    plt.show()
    
    # Also create version with weights shown
    fig2, ax2 = create_mlp_diagram(save_path='mlp_2_3_1_with_weights.png',
                                  show_weights=True, dpi=300)
    
    print("\nCreated two versions:")
    print("1. Clean diagram (mlp_2_3_1_diagram.png)")
    print("2. Diagram with sample weights (mlp_2_3_1_with_weights.png)")
    print("Both saved as PNG and PDF formats for publication quality")

if __name__ == "__main__":
    main()