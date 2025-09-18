#!/usr/bin/env python3
"""
Plot eigenvalue evolution for 16 models in a 4x4 grid.
Creates a professional visualization showing eigenvalue evolution across different model types.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# Professional color scheme by model type - Enhanced but conservative
COLOR_SCHEMES = {
    'custom_random': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],    # Deep blue palette
    'custom_trained': ['#2D5016', '#61A12E', '#8FBC8F', '#228B22'],   # Forest green palette  
    'mlp_random': ['#D2691E', '#FF8C00', '#FFA500', '#FFB84D'],       # Warm orange palette
    'mlp_trained': ['#663399', '#8A2BE2', '#9370DB', '#B19CD9']       # Royal purple palette
}

def load_eigenvalue_data(filepath: str) -> Optional[Dict]:
    """Load eigenvalue data from npz file."""
    try:
        data = np.load(filepath, allow_pickle=True)
        return {
            'eigenvalue_matrix': data['eigenvalue_matrix'],
            'time_vector': data['time_vector'],
            'metadata': data['metadata'].item() if data['metadata'].size > 0 else {}
        }
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return None

def classify_model_type(filename: str) -> str:
    """Classify model type based on filename."""
    if 'custom_random' in filename:
        return 'custom_random'
    elif 'custom_trained' in filename:
        return 'custom_trained'
    elif 'mlp_random' in filename:
        return 'mlp_random'
    elif 'mlp_trained' in filename:
        return 'mlp_trained'
    else:
        return 'unknown'

def get_model_files(data_dir: str) -> Dict[str, List[str]]:
    """Get model files organized by type, selecting 4 of each type."""
    data_path = Path(data_dir)
    all_files = list(data_path.glob("*.npz"))
    
    model_files = {
        'custom_random': [],
        'custom_trained': [],
        'mlp_random': [],
        'mlp_trained': []
    }
    
    skipped_files = []
    
    for file in all_files:
        model_type = classify_model_type(file.name)
        if model_type in model_files:
            model_files[model_type].append(str(file))
        elif model_type == 'unknown':
            skipped_files.append(file.name)
    
    # Report skipped files
    if skipped_files:
        print(f"Skipped {len(skipped_files)} unknown model files: {', '.join(skipped_files)}")
    
    # Select first 4 of each type
    for model_type in model_files:
        model_files[model_type] = sorted(model_files[model_type])[:4]
        
    return model_files

def extract_model_info(filename: str) -> Dict[str, str]:
    """Extract model information from filename for display."""
    basename = Path(filename).stem
    
    # Extract version number
    version_match = re.search(r'v(\d+)', basename)
    version = version_match.group(1) if version_match else "00"
    
    # Extract accuracy if present
    acc_match = re.search(r'acc([0-9.]+)', basename)
    accuracy = f" (acc={acc_match.group(1)})" if acc_match else ""
    
    # Extract epoch if present
    epoch_match = re.search(r'ep(\d+)', basename)
    epoch = f" ep{epoch_match.group(1)}" if epoch_match else ""
    
    model_type = classify_model_type(basename)
    type_display = model_type.replace('_', ' ').title()
    
    return {
        'type': type_display,
        'version': version,
        'accuracy': accuracy,
        'epoch': epoch,
        'title': f"{type_display} v{version}{accuracy}{epoch}"
    }

def create_eigenvalue_grid_plot():
    """Create the main eigenvalue grid visualization."""
    
    # Setup data directory
    data_dir = "eigenvalueData"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Get model files
    model_files = get_model_files(data_dir)
    
    # Verify we have enough models
    total_models = sum(len(files) for files in model_files.values())
    print(f"Found {total_models} models total:")
    for model_type, files in model_files.items():
        print(f"  {model_type}: {len(files)} models")
    
    if total_models < 16:
        print(f"Warning: Only {total_models} models found, expected 16")
    
    # Create subplot titles
    subplot_titles = []
    all_files = []
    model_types = []
    
    for model_type in ['custom_random', 'custom_trained', 'mlp_random', 'mlp_trained']:
        for file in model_files[model_type]:
            info = extract_model_info(file)
            subplot_titles.append(info['title'])
            all_files.append(file)
            model_types.append(model_type)
    
    # Create 4x4 subplot grid
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Process each model
    for idx, (file, model_type) in enumerate(zip(all_files, model_types)):
        row = (idx // 4) + 1
        col = (idx % 4) + 1
        
        print(f"Processing {file} -> row {row}, col {col}")
        
        # Load data
        data = load_eigenvalue_data(file)
        if data is None:
            continue
            
        eigenvalue_matrix = data['eigenvalue_matrix']
        time_vector = data['time_vector']
        
        # Get colors for this model type
        colors = COLOR_SCHEMES[model_type]
        
        # Plot eigenvalue tracks
        # eigenvalue_matrix is (n_time_steps, n_eigenvalues)
        # time_vector has n_time_steps values
        n_time_steps, n_eigenvals = eigenvalue_matrix.shape
        
        print(f"  Model in row {row}, col {col}: plotting ALL {n_eigenvals} eigenvalues")
        
        # Plot ALL eigenvalues
        for eigenval_idx in range(n_eigenvals):
            # Extract eigenvalue track for eigenvalue eigenval_idx across all time steps
            eigenvals = eigenvalue_matrix[:, eigenval_idx]  # Column eigenval_idx contains eigenvalue eigenval_idx across time
            
            # Filter out invalid values
            valid_mask = (eigenvals > 0) & np.isfinite(eigenvals)
            if not np.any(valid_mask):
                continue
            
            # Use time_vector as filtration parameters for x-axis
            valid_filtration = time_vector[valid_mask]
            valid_eigenvals = eigenvals[valid_mask]
            
            # Ensure valid eigenvalues for linear scale
            valid_eigenvals = np.maximum(valid_eigenvals, 0)
            
            # Choose color (cycle through available colors)
            color = colors[eigenval_idx % len(colors)]
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=valid_filtration,
                    y=valid_eigenvals,
                    mode='lines',
                    name=f'λ_{eigenval_idx}',
                    line=dict(color=color, width=1.5),
                    showlegend=False,  # Too many traces for legend
                    hovertemplate=f'λ_{eigenval_idx}<br>Filtration: %{{x:.4f}}<br>Value: %{{y:.2e}}<extra></extra>'
                ),
                row=row, col=col
            )
    
    # Compute global ranges across all models for consistent axes
    all_filtration_values = []
    all_eigenvalue_values = []
    
    for file, model_type in zip(all_files, model_types):
        data = load_eigenvalue_data(file)
        if data is not None:
            all_filtration_values.extend(data['time_vector'])
            # Get all valid eigenvalues from this model
            eigenvalue_matrix = data['eigenvalue_matrix']
            valid_eigenvals = eigenvalue_matrix[(eigenvalue_matrix > 0) & np.isfinite(eigenvalue_matrix)]
            all_eigenvalue_values.extend(valid_eigenvals.flatten())
    
    # Calculate optimal ranges using same logic as EnhancedSpectralVisualizer
    if all_filtration_values:
        global_filtration_min = min(all_filtration_values)
        global_filtration_max = max(all_filtration_values)
        print(f"Global filtration range: [{global_filtration_min:.6f}, {global_filtration_max:.6f}]")
    else:
        global_filtration_min, global_filtration_max = 0, 1
        print("Warning: No filtration values found, using default range [0, 1]")
    
    # Calculate eigenvalue range with padding (same as EnhancedSpectralVisualizer)
    if all_eigenvalue_values:
        eigenval_min = min(all_eigenvalue_values)
        eigenval_max = max(all_eigenvalue_values)
        eigenval_padding = (eigenval_max - eigenval_min) * 0.1
        eigenval_range = [eigenval_min - eigenval_padding, eigenval_max + eigenval_padding]
        print(f"Global eigenvalue range: [{eigenval_range[0]:.6e}, {eigenval_range[1]:.6e}]")
    else:
        eigenval_range = None
        print("Warning: No valid eigenvalues found")
    
    # Update layout for all subplots
    for i in range(1, 17):  # 16 subplots
        row = ((i-1) // 4) + 1
        col = ((i-1) % 4) + 1
        
        # Set linear scale for y-axis with computed range
        fig.update_yaxes(type="linear", range=eigenval_range, row=row, col=col)
        
        # Set x-axis range to start from minimum filtration value
        fig.update_xaxes(range=[global_filtration_min, global_filtration_max], row=row, col=col)
        
        # Only show x-axis labels on bottom row
        if row == 4:
            fig.update_xaxes(title_text="Filtration Parameter", row=row, col=col)
        else:
            fig.update_xaxes(showticklabels=False, row=row, col=col)
            
        # Only show y-axis labels on left column  
        if col == 1:
            fig.update_yaxes(title_text="Eigenvalue", row=row, col=col)
        else:
            fig.update_yaxes(showticklabels=False, row=row, col=col)
    
    # Update overall layout
    fig.update_layout(
        title={
            'text': "Eigenvalue Evolution Analysis: 16 Model Comparison Grid",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        width=1600,
        height=1200,
        showlegend=False,
        paper_bgcolor='#fdfdfd',
        plot_bgcolor='rgba(250,250,250,0.9)',
        font=dict(size=10)
    )
    
    # Add color-coded annotation for model types
    annotations_text = (
        "<b>Model Types:</b> "
        "<span style='color:#2E86AB'>Custom Random</span> | "
        "<span style='color:#61A12E'>Custom Trained</span> | " 
        "<span style='color:#FF8C00'>MLP Random</span> | "
        "<span style='color:#8A2BE2'>MLP Trained</span>"
    )
    
    fig.add_annotation(
        x=0.5, y=0.02,
        text=annotations_text,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1,
        xanchor="center"
    )
    
    return fig

def main():
    """Main function to create and save the plot."""
    print("Creating eigenvalue evolution grid plot...")
    
    try:
        fig = create_eigenvalue_grid_plot()
        
        # Save as HTML
        output_file = "eigenvalue_evolution_grid.html"
        fig.write_html(output_file)
        print(f"Plot saved as {output_file}")
        
        # Also save as PNG if kaleido is available
        try:
            png_file = "eigenvalue_evolution_grid.png"
            fig.write_image(png_file, width=1600, height=1200, scale=2)
            print(f"Plot also saved as {png_file}")
        except Exception as e:
            print(f"Could not save PNG (install kaleido for PNG export): {e}")
        
        # Show plot
        fig.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        raise

if __name__ == "__main__":
    main()