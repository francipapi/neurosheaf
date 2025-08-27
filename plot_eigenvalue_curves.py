#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Mean Eigenvalue Evolution Curves

Creates visualization of eigenvalue evolution curves from all models in eigenvalueData/,
color-coded by model type (trained/random, custom/MLP).

Usage:
    python plot_eigenvalue_curves.py                    # Basic plot
    python plot_eigenvalue_curves.py --interactive      # Interactive HTML plot
    python plot_eigenvalue_curves.py --subplots         # Separate subplots by category
    python plot_eigenvalue_curves.py --log-scale        # Use log scale for eigenvalues
    python plot_eigenvalue_curves.py --statistics       # Save curve statistics
"""

import argparse
import glob
import os
import pathlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import seaborn as sns

# Optional interactive plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_eigenvalue_data(data_dir: str, include_unknown: bool = False) -> Dict[str, Dict]:
    """
    Load all eigenvalue evolution data from the specified directory.
    
    Returns:
        Dictionary with model names as keys and data dictionaries as values.
        Each data dict contains: 'eigenvalues', 'time', 'type', 'architecture', 'status'
    """
    print(f"[INFO] Loading eigenvalue data from {data_dir}")
    
    # Find all data files
    patterns = ["*.npz", "*.npy"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern)))
    
    # Filter out non-model files
    model_files = []
    for f in files:
        basename = os.path.basename(f)
        # Skip processing logs and summary files
        if not any(skip in basename.lower() for skip in ['processing', 'log', 'summary']):
            model_files.append(f)
    
    print(f"[INFO] Found {len(model_files)} model data files")
    
    models_data = {}
    failed_loads = []
    
    for file_path in model_files:
        try:
            model_name = pathlib.Path(file_path).stem
            data = load_single_file(file_path, model_name, include_unknown)
            if data:
                models_data[model_name] = data
        except Exception as e:
            failed_loads.append((file_path, str(e)))
            print(f"[WARN] Failed to load {file_path}: {e}")
    
    print(f"[INFO] Successfully loaded {len(models_data)} models")
    if failed_loads:
        print(f"[WARN] Failed to load {len(failed_loads)} files")
    
    return models_data

def load_single_file(file_path: str, model_name: str, include_unknown: bool = False) -> Optional[Dict]:
    """Load a single eigenvalue evolution file."""
    try:
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=False)
            if 'eigenvalue_matrix' in data and 'time_vector' in data:
                eigenvalues = np.asarray(data['eigenvalue_matrix'], dtype=float)
                time = np.asarray(data['time_vector'], dtype=float)
            else:
                print(f"[WARN] {file_path}: Missing required keys")
                return None
                
        elif file_path.endswith('.npy'):
            arr = np.load(file_path, allow_pickle=True)
            if isinstance(arr, dict):
                eigenvalues = np.asarray(arr['eigenvalue_matrix'], dtype=float)
                time = np.asarray(arr['time_vector'], dtype=float)
            else:
                # Raw array
                eigenvalues = np.asarray(arr, dtype=float)
                if eigenvalues.ndim != 2:
                    print(f"[WARN] {file_path}: Expected 2D array")
                    return None
                time = np.linspace(0, 1, eigenvalues.shape[0])
        else:
            print(f"[WARN] Unsupported file format: {file_path}")
            return None
        
        # Ensure proper orientation (time should be first dimension)
        if eigenvalues.shape[0] != len(time) and eigenvalues.shape[1] == len(time):
            eigenvalues = eigenvalues.T
        
        # Sort by time
        time_order = np.argsort(time)
        time = time[time_order]
        eigenvalues = eigenvalues[time_order, :]
        
        # Classify model
        classification = classify_model(model_name)
        
        # Skip unknown/other models unless explicitly requested
        if not include_unknown and (classification['architecture'] == 'Other' or classification['status'] == 'Unknown'):
            print(f"[INFO] Skipping unknown model: {model_name} (category: {classification['category']})")
            return None
        
        return {
            'eigenvalues': eigenvalues,
            'time': time,
            'architecture': classification['architecture'],
            'status': classification['status'],
            'category': classification['category'],
            'file_path': file_path
        }
        
    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return None

def classify_model(model_name: str) -> Dict[str, str]:
    """
    Classify model based on its name.
    
    Returns:
        Dict with 'architecture', 'status', and 'category'
    """
    name_lower = model_name.lower()
    
    # Handle MNIST models first (more specific patterns)
    if name_lower.startswith('mlp4layer_mnist'):
        # Pattern: mlp4layer_mnist_seedXX
        architecture = 'MNIST-MLP4'
        status = 'Trained'  # These models are trained on MNIST
    elif name_lower.startswith('tinycnn_mnist'):
        # Pattern: tinycnn_mnist.pth
        architecture = 'MNIST-CNN'
        status = 'Trained'  # These models are trained on MNIST
    elif name_lower.startswith('tinycnn_random'):
        # Pattern: tinycnn_random_XXX
        architecture = 'MNIST-CNN'
        status = 'Random'
    elif name_lower.startswith('tinycnn'):
        # General TinyCNN pattern
        architecture = 'MNIST-CNN'
        # Determine status from context
        if any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        elif any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        else:
            status = 'Trained'  # Default for TinyCNN
    elif name_lower.startswith('mnist_mlp_random'):
        # Pattern: mnist_mlp_random_XXX
        architecture = 'MNIST-MLP'
        status = 'Random'
    elif name_lower.startswith('mnist_mlp'):
        # General MNIST MLP pattern
        architecture = 'MNIST-MLP'
        # Determine status from context
        if any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        elif any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        else:
            status = 'Unknown'
    # Handle regular (non-MNIST) models
    elif any(x in name_lower for x in ['custom', 'conv']):
        architecture = 'Custom'
        # Determine training status
        if any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        elif any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        else:
            status = 'Unknown'
    elif any(x in name_lower for x in ['mlp']):
        architecture = 'MLP'
        # Determine training status
        if any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        elif any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        else:
            status = 'Unknown'
    else:
        architecture = 'Other'
        # Determine training status
        if any(x in name_lower for x in ['trained', 'acc']):
            status = 'Trained'
        elif any(x in name_lower for x in ['random', 'rand']):
            status = 'Random'
        else:
            status = 'Unknown'
    
    # Combined category
    category = f"{architecture}-{status}"
    
    return {
        'architecture': architecture,
        'status': status,
        'category': category
    }


def compute_mean_curve(eigenvalues: np.ndarray, use_log: bool = False) -> np.ndarray:
    """Compute mean eigenvalue curve across all eigenvalues at each time step."""
    # Handle negative values
    if use_log:
        # Use log1p for stability with small values
        eigenvalues = np.where(eigenvalues > 0, eigenvalues, 1e-10)
        eigenvalues = np.log1p(eigenvalues)
    
    # Compute mean across eigenvalues (axis=1)
    return np.nanmean(eigenvalues, axis=1)

def get_color_scheme() -> Dict[str, Dict]:
    """Define color scheme for different model categories."""
    return {
        # Original model categories
        'Custom-Trained': {
            'color': '#1f77b4',  # Blue
            'linestyle': '-',     # Solid
            'alpha': 0.7
        },
        'Custom-Random': {
            'color': '#1f77b4',  # Blue
            'linestyle': '--',    # Dashed
            'alpha': 0.6
        },
        'MLP-Trained': {
            'color': '#d62728',  # Red
            'linestyle': '-',     # Solid
            'alpha': 0.7
        },
        'MLP-Random': {
            'color': '#d62728',  # Red
            'linestyle': '--',    # Dashed
            'alpha': 0.6
        },
        'Other-Trained': {
            'color': '#2ca02c',  # Green
            'linestyle': '-',     # Solid
            'alpha': 0.7
        },
        'Other-Random': {
            'color': '#2ca02c',  # Green
            'linestyle': '--',    # Dashed
            'alpha': 0.6
        },
        'Other-Unknown': {
            'color': '#808080',  # Gray
            'linestyle': ':',     # Dotted
            'alpha': 0.5
        },
        # MNIST model categories
        'MNIST-MLP4-Trained': {
            'color': '#ff7f0e',  # Orange
            'linestyle': '-',     # Solid
            'alpha': 0.8
        },
        'MNIST-MLP-Random': {
            'color': '#9467bd',  # Purple
            'linestyle': '--',    # Dashed
            'alpha': 0.7
        },
        'MNIST-MLP-Trained': {
            'color': '#9467bd',  # Purple
            'linestyle': '-',     # Solid
            'alpha': 0.8
        },
        'MNIST-MLP-Unknown': {
            'color': '#9467bd',  # Purple
            'linestyle': ':',     # Dotted
            'alpha': 0.6
        },
        # TinyCNN categories - using different colors for trained vs random
        'MNIST-CNN-Trained': {
            'color': '#2ca02c',  # Green
            'linestyle': '-',     # Solid
            'alpha': 0.8
        },
        'MNIST-CNN-Random': {
            'color': '#17becf',  # Cyan
            'linestyle': '--',    # Dashed
            'alpha': 0.7
        },
        'MNIST-CNN-Unknown': {
            'color': '#808080',  # Gray
            'linestyle': ':',     # Dotted
            'alpha': 0.6
        }
    }

def detect_and_remove_outliers(eigenvalues: np.ndarray, time: np.ndarray, 
                               method: str = 'none', model_name: str = '', 
                               category: str = '') -> Tuple[np.ndarray, np.ndarray]:
    """
    Outlier detection and removal, including specific handling for TinyCNN models.
    
    Args:
        eigenvalues: (n_time, n_eigenvalues) array
        time: (n_time,) array
        method: Detection method ('none', 'iqr', 'zscore', 'trajectory_end')
        model_name: Name of the model for identification
        category: Category of the model (e.g., 'MNIST-CNN-Trained')
        
    Returns:
        Tuple of (cleaned_eigenvalues, cleaned_time)
    """
    
    if method == 'trajectory_end':
        # Remove last point from trained TinyCNN models
        # Identify TinyCNN models by name or category
        is_tinycnn_trained = (
            ('tinycnn' in model_name.lower() and 
             ('trained' in category.lower() or 'mnist-cnn-trained' in category.lower())) or
            (category.lower() == 'mnist-cnn-trained')
        )
        
        if is_tinycnn_trained and len(time) > 1:
            print(f"[INFO] {model_name}: Removing last datapoint (TinyCNN trained model)")
            # Remove last point from both eigenvalues and time
            return eigenvalues[:-1, :], time[:-1]
        else:
            # Not a TinyCNN trained model, return unchanged
            return eigenvalues, time
    
    elif method == 'iqr':
        # TODO: Implement IQR-based outlier detection
        print(f"[INFO] IQR outlier detection not yet implemented for {model_name}")
        return eigenvalues, time
    
    elif method == 'zscore':
        # TODO: Implement z-score-based outlier detection
        print(f"[INFO] Z-score outlier detection not yet implemented for {model_name}")
        return eigenvalues, time
    
    # Default: return unchanged
    return eigenvalues, time

def resample_to_common_grid(models_data: Dict, n_points: int = 200, 
                           outlier_method: str = 'none') -> Dict:
    """
    Resample all curves to a common [0, 1] time grid with outlier detection.
    
    Args:
        models_data: Dictionary of model data
        n_points: Number of interpolation points
        outlier_method: Method for outlier detection ('none', 'iqr', 'zscore', 'trajectory_end')
        
    Returns:
        Dictionary with resampled data normalized to [0, 1] time range
    """
    
    # Step 1: Apply outlier detection (placeholder for now)
    print(f"[INFO] Applying outlier detection method: {outlier_method}")
    cleaned_data = {}
    outlier_stats = {'models_processed': 0, 'outliers_found': 0}
    
    for name, data in models_data.items():
        eigenvalues = data['eigenvalues']
        time = data['time']
        
        # Apply outlier detection
        cleaned_eigenvalues, cleaned_time = detect_and_remove_outliers(
            eigenvalues, time, method=outlier_method,
            model_name=name, category=data.get('category', '')
        )
        
        # Track if any outliers were removed
        if len(cleaned_time) != len(time):
            outlier_stats['outliers_found'] += 1
            print(f"[INFO] {name}: Removed {len(time) - len(cleaned_time)} outlier points")
        
        outlier_stats['models_processed'] += 1
        
        cleaned_data[name] = {
            **data,
            'eigenvalues': cleaned_eigenvalues,
            'time': cleaned_time,
            'original_shape': eigenvalues.shape,
            'outliers_removed': len(time) - len(cleaned_time)
        }
    
    print(f"[INFO] Outlier detection complete: {outlier_stats['outliers_found']}/{outlier_stats['models_processed']} models had outliers")
    
    # Step 2: Normalize time to [0, 1] for all models
    print("[INFO] Normalizing all time coordinates to [0, 1] range")
    normalized_data = {}
    original_ranges = {}
    
    for name, data in cleaned_data.items():
        time = data['time']
        t_min, t_max = float(time.min()), float(time.max())
        original_ranges[name] = (t_min, t_max)
        
        # Normalize to [0, 1]
        if t_max > t_min:
            normalized_time = (time - t_min) / (t_max - t_min)
        else:
            # Handle edge case where all time points are the same
            normalized_time = np.zeros_like(time)
            print(f"[WARN] {name}: All time points identical, using zeros")
        
        normalized_data[name] = {
            **data,
            'time': normalized_time,
            'original_time_range': (t_min, t_max)
        }
    
    # Step 3: Create common grid in [0, 1]
    common_time = np.linspace(0, 1, n_points)
    print(f"[INFO] Created common time grid: [0, 1] with {n_points} points")
    
    # Step 4: Resample all models to common [0, 1] grid
    print("[INFO] Resampling all models to common [0, 1] grid")
    resampled_data = {}
    
    for name, data in normalized_data.items():
        try:
            eigenvalues = data['eigenvalues']
            time = data['time']
            
            # Resample each eigenvalue series to common grid
            resampled_eigenvalues = np.zeros((n_points, eigenvalues.shape[1]))
            for i in range(eigenvalues.shape[1]):
                resampled_eigenvalues[:, i] = np.interp(common_time, time, eigenvalues[:, i])
            
            resampled_data[name] = {
                **data,
                'eigenvalues': resampled_eigenvalues,
                'time': common_time,  # Now [0, 1] range
                'normalized': True
            }
        except Exception as e:
            print(f"[WARN] Failed to resample {name}: {e}")
    
    # Report summary
    if original_ranges:
        all_mins = [r[0] for r in original_ranges.values()]
        all_maxs = [r[1] for r in original_ranges.values()]
        print(f"[INFO] Original time ranges spanned: [{min(all_mins):.6f}, {max(all_maxs):.6f}]")
    
    print(f"[INFO] Successfully resampled {len(resampled_data)} models to normalized [0, 1] grid")
    return resampled_data


def plot_curves(models_data: Dict, use_log: bool = False, show_stats: bool = True, 
                output_file: str = "eigenvalue_curves.png") -> None:
    """Create the main plot with all eigenvalue curves."""
    
    color_scheme = get_color_scheme()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Group models by category
    categories = {}
    for name, data in models_data.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((name, data))
    
    print(f"[INFO] Found categories: {list(categories.keys())}")
    
    # Plot individual curves
    legend_handles = []
    category_curves = {}
    
    for category, models in categories.items():
        print(f"[INFO] Plotting {len(models)} models in category '{category}'")
        
        if category not in color_scheme:
            print(f"[WARN] No color scheme for category '{category}', using default")
            style = {'color': '#808080', 'linestyle': '-', 'alpha': 0.5}
        else:
            style = color_scheme[category]
        
        curves_for_stats = []
        
        for name, data in models:
            try:
                mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
                time = data['time']
                
                # Plot individual curve
                ax.plot(time, mean_curve, 
                       color=style['color'], 
                       linestyle=style['linestyle'],
                       alpha=style['alpha'], 
                       linewidth=0.8)
                
                curves_for_stats.append(mean_curve)
                
            except Exception as e:
                print(f"[WARN] Failed to plot {name}: {e}")
        
        # Store curves for statistics
        if curves_for_stats:
            category_curves[category] = np.array(curves_for_stats)
            
            # Create legend handle
            handle = mpatches.Patch(color=style['color'], 
                                  label=f"{category} (n={len(curves_for_stats)})")
            legend_handles.append(handle)
    
    # Plot category statistics if requested
    if show_stats and category_curves:
        for category, curves in category_curves.items():
            if category in color_scheme:
                style = color_scheme[category]
                
                # Compute mean and std across models
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                time = list(categories[category])[0][1]['time']  # Use time from first model
                
                # Plot mean with thicker line
                ax.plot(time, mean_curve, 
                       color=style['color'], 
                       linestyle=style['linestyle'],
                       alpha=1.0, 
                       linewidth=2.5,
                       label=f"{category} Mean")
                
                # Add standard deviation band
                ax.fill_between(time, 
                               mean_curve - std_curve,
                               mean_curve + std_curve,
                               color=style['color'], 
                               alpha=0.15)
    
    # Formatting
    ax.set_xlabel('Filtration Parameter', fontsize=12)
    if use_log:
        ax.set_ylabel('Mean Log(Eigenvalue + 1)', fontsize=12)
        ax.set_title('Mean Eigenvalue Evolution (Log Scale)', fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel('Mean Eigenvalue', fontsize=12)
        ax.set_title('Mean Eigenvalue Evolution', fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(handles=legend_handles, loc='best', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot saved to {output_file}")
    
    # Show plot
    plt.show()

def create_subplots(models_data: Dict, use_log: bool = False, 
                   output_file: str = "eigenvalue_curves_subplots.png") -> None:
    """Create subplots for each model category."""
    
    color_scheme = get_color_scheme()
    
    # Group models by category
    categories = {}
    for name, data in models_data.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((name, data))
    
    # Create subplots
    n_categories = len(categories)
    n_cols = min(2, n_categories)
    n_rows = (n_categories + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_categories == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (category, models) in enumerate(categories.items()):
        ax = axes[idx]
        
        if category in color_scheme:
            style = color_scheme[category]
        else:
            style = {'color': '#808080', 'linestyle': '-', 'alpha': 0.7}
        
        curves = []
        for name, data in models:
            try:
                mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
                time = data['time']
                
                ax.plot(time, mean_curve, 
                       color=style['color'], 
                       linestyle=style['linestyle'],
                       alpha=style['alpha'], 
                       linewidth=1.0)
                
                curves.append(mean_curve)
                
            except Exception as e:
                print(f"[WARN] Failed to plot {name} in subplot: {e}")
        
        # Plot mean
        if curves:
            mean_curve = np.mean(curves, axis=0)
            ax.plot(time, mean_curve, 
                   color=style['color'], 
                   linewidth=2.5, 
                   alpha=1.0)
        
        ax.set_title(f"{category} (n={len(models)})", fontweight='bold')
        ax.set_xlabel('Filtration Parameter')
        if use_log:
            ax.set_ylabel('Mean Log(Eigenvalue + 1)')
        else:
            ax.set_ylabel('Mean Eigenvalue')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(categories), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Subplots saved to {output_file}")
    plt.show()

def create_interactive_plot(models_data: Dict, use_log: bool = False,
                           output_file: str = "eigenvalue_curves.html") -> None:
    """Create interactive plot using Plotly."""
    if not PLOTLY_AVAILABLE:
        print("[WARN] Plotly not available. Install plotly for interactive plots.")
        return
    
    color_scheme = get_color_scheme()
    fig = go.Figure()
    
    # Group models by category
    categories = {}
    for name, data in models_data.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((name, data))
    
    for category, models in categories.items():
        if category in color_scheme:
            style = color_scheme[category]
        else:
            style = {'color': '#808080', 'linestyle': '-', 'alpha': 0.7}
        
        for name, data in models:
            try:
                mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
                time = data['time']
                
                # Convert linestyle to plotly dash
                dash = 'solid' if style['linestyle'] == '-' else 'dash'
                if style['linestyle'] == ':':
                    dash = 'dot'
                
                fig.add_trace(go.Scatter(
                    x=time,
                    y=mean_curve,
                    mode='lines',
                    name=name,
                    legendgroup=category,
                    legendgrouptitle_text=category,
                    line=dict(color=style['color'], 
                             dash=dash,
                             width=1),
                    opacity=style['alpha'],
                    hovertemplate=f"<b>{name}</b><br>Time: %{{x:.4f}}<br>Mean Eigenvalue: %{{y:.4f}}<extra></extra>"
                ))
                
            except Exception as e:
                print(f"[WARN] Failed to add {name} to interactive plot: {e}")
    
    # Update layout
    title = 'Interactive Mean Eigenvalue Evolution'
    if use_log:
        title += ' (Log Scale)'
        yaxis_title = 'Mean Log(Eigenvalue + 1)'
    else:
        yaxis_title = 'Mean Eigenvalue'
    
    fig.update_layout(
        title=title,
        xaxis_title='Filtration Parameter',
        yaxis_title=yaxis_title,
        hovermode='closest',
        showlegend=True,
        width=1200,
        height=700
    )
    
    # Save and show
    fig.write_html(output_file)
    print(f"[INFO] Interactive plot saved to {output_file}")
    fig.show()

def save_statistics(models_data: Dict, use_log: bool = False,
                   output_file: str = "curve_statistics.csv") -> None:
    """Save curve statistics to CSV file."""
    import pandas as pd
    
    stats_data = []
    
    for name, data in models_data.items():
        try:
            mean_curve = compute_mean_curve(data['eigenvalues'], use_log)
            
            stats_data.append({
                'model_name': name,
                'category': data['category'],
                'architecture': data['architecture'],
                'status': data['status'],
                'n_timepoints': len(mean_curve),
                'n_eigenvalues': data['eigenvalues'].shape[1],
                'mean_eigenvalue_overall': np.mean(mean_curve),
                'std_eigenvalue_overall': np.std(mean_curve),
                'min_eigenvalue': np.min(mean_curve),
                'max_eigenvalue': np.max(mean_curve),
                'final_eigenvalue': mean_curve[-1],
                'initial_eigenvalue': mean_curve[0]
            })
            
        except Exception as e:
            print(f"[WARN] Failed to compute statistics for {name}: {e}")
    
    # Save to CSV
    df = pd.DataFrame(stats_data)
    df.to_csv(output_file, index=False)
    print(f"[INFO] Statistics saved to {output_file}")
    
    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    summary = df.groupby('category').agg({
        'model_name': 'count',
        'mean_eigenvalue_overall': ['mean', 'std'],
        'n_eigenvalues': 'mean'
    })
    print(summary)

def main():
    parser = argparse.ArgumentParser(
        description="Plot eigenvalue evolution curves from all models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, default='eigenvalueData',
                       help='Directory containing eigenvalue data files')
    parser.add_argument('--interactive', action='store_true',
                       help='Create interactive HTML plot')
    parser.add_argument('--subplots', action='store_true',
                       help='Create separate subplots by category')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use log scale for eigenvalues')
    parser.add_argument('--statistics', action='store_true',
                       help='Save curve statistics to CSV')
    parser.add_argument('--no-stats-overlay', action='store_true',
                       help='Disable statistical overlays (mean curves and bands)')
    parser.add_argument('--output-prefix', type=str, default='eigenvalue_curves',
                       help='Output file prefix')
    parser.add_argument('--include-unknown', action='store_true',
                       help='Include unknown/unclassified models in the plots')
    parser.add_argument('--outlier-method', type=str, default='none',
                       choices=['none', 'iqr', 'zscore', 'trajectory_end'],
                       help='Method for outlier detection (placeholder for future implementation)')
    
    args = parser.parse_args()
    
    # Load data
    models_data = load_eigenvalue_data(args.data_dir, include_unknown=args.include_unknown)
    if not models_data:
        print("[ERROR] No models loaded!")
        return
    
    # Resample to common [0, 1] grid with outlier detection
    models_data = resample_to_common_grid(models_data, outlier_method=args.outlier_method)
    
    # Create main plot
    output_file = f"{args.output_prefix}.png"
    plot_curves(models_data, 
               use_log=args.log_scale, 
               show_stats=not args.no_stats_overlay,
               output_file=output_file)
    
    # Create subplots if requested
    if args.subplots:
        subplot_file = f"{args.output_prefix}_subplots.png"
        create_subplots(models_data, 
                       use_log=args.log_scale, 
                       output_file=subplot_file)
    
    # Create interactive plot if requested
    if args.interactive:
        interactive_file = f"{args.output_prefix}.html"
        create_interactive_plot(models_data, 
                               use_log=args.log_scale,
                               output_file=interactive_file)
    
    # Save statistics if requested
    if args.statistics:
        stats_file = f"{args.output_prefix}_statistics.csv"
        save_statistics(models_data, 
                       use_log=args.log_scale,
                       output_file=stats_file)
    
    print(f"\n[INFO] Visualization complete! Processed {len(models_data)} models.")

if __name__ == "__main__":
    main()