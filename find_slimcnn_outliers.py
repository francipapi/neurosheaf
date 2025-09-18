#!/usr/bin/env python3
"""
Find Outliers in Trained SlimCNN Models

This script identifies outlier models in the trained SlimCNN eigenvalue evolution data
by analyzing final eigenvalue values and detecting models with significantly different behavior.

Usage:
    python find_slimcnn_outliers.py
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

def load_eigenvalue_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load eigenvalue data from a single file.
    
    Returns:
        Tuple of (eigenvalues, time) arrays, or (None, None) if loading fails
    """
    try:
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=False)
            if 'eigenvalue_matrix' in data and 'time_vector' in data:
                eigenvalues = np.asarray(data['eigenvalue_matrix'], dtype=float)
                time = np.asarray(data['time_vector'], dtype=float)
                return eigenvalues, time
            else:
                print(f"Missing required keys in {file_path}")
                return None, None
        else:
            print(f"Unsupported file format: {file_path}")
            return None, None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_model_info(file_path: str) -> Dict[str, str]:
    """Extract model information from filename."""
    filename = Path(file_path).stem
    
    # Remove _eigenvalues suffix if present
    if filename.endswith('_eigenvalues'):
        filename = filename[:-12]
    
    # Parse SlimCNN trained models: slimcnn_digits_seed##
    parts = filename.split('_')
    
    return {
        'filename': filename,
        'architecture': 'SlimCNN',
        'dataset': 'digits',
        'type': 'trained' if 'seed' in filename else 'random',
        'seed': parts[-1] if 'seed' in filename else 'unknown',
        'file_path': file_path
    }

def compute_final_eigenvalue_stats(eigenvalues: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for the final eigenvalues.
    
    Args:
        eigenvalues: 2D array of shape (n_timepoints, n_eigenvalues)
        
    Returns:
        Dictionary with final eigenvalue statistics
    """
    if eigenvalues.size == 0:
        return {}
    
    # Get final timepoint eigenvalues
    final_eigs = eigenvalues[-1, :]
    final_eigs = final_eigs[np.isfinite(final_eigs)]  # Remove NaN/Inf
    
    if len(final_eigs) == 0:
        return {}
    
    return {
        'mean_final_eig': np.mean(final_eigs),
        'min_final_eig': np.min(final_eigs),
        'max_final_eig': np.max(final_eigs),
        'std_final_eig': np.std(final_eigs),
        'median_final_eig': np.median(final_eigs),
        'n_eigenvalues': len(final_eigs)
    }

def detect_outliers_zscore(values: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Detect outliers using Z-score method."""
    if len(values) < 3:
        return np.array([], dtype=bool)
    
    z_scores = np.abs((values - np.mean(values)) / np.std(values))
    return z_scores > threshold

def detect_outliers_iqr(values: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """Detect outliers using IQR method."""
    if len(values) < 3:
        return np.array([], dtype=bool)
    
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return (values < lower_bound) | (values > upper_bound)

def analyze_slimcnn_models() -> pd.DataFrame:
    """
    Analyze all SlimCNN trained models and detect outliers.
    
    Returns:
        DataFrame with analysis results
    """
    data_dir = "eigenvalueData"
    
    # Find all SlimCNN trained model files
    pattern = os.path.join(data_dir, "slimcnn_digits_seed*_eigenvalues.npz")
    files = glob.glob(pattern)
    
    if not files:
        print("No SlimCNN trained model files found!")
        return pd.DataFrame()
    
    print(f"Found {len(files)} SlimCNN trained model files")
    
    results = []
    
    for file_path in sorted(files):
        # Extract model info
        model_info = extract_model_info(file_path)
        
        # Load eigenvalue data
        eigenvalues, time = load_eigenvalue_file(file_path)
        
        if eigenvalues is None:
            continue
        
        # Compute final eigenvalue statistics
        final_stats = compute_final_eigenvalue_stats(eigenvalues)
        
        if not final_stats:
            continue
        
        # Combine all information
        result = {**model_info, **final_stats}
        results.append(result)
        
        print(f"Loaded {model_info['filename']}: mean_final={final_stats['mean_final_eig']:.6f}")
    
    return pd.DataFrame(results)

def identify_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify outliers in the dataset using multiple methods.
    
    Args:
        df: DataFrame with model analysis results
        
    Returns:
        DataFrame with outlier flags added
    """
    if df.empty:
        return df
    
    # Use mean final eigenvalue as the main metric for outlier detection
    values = df['mean_final_eig'].values
    
    # Apply different outlier detection methods
    df['outlier_zscore'] = detect_outliers_zscore(values, threshold=2.0)
    df['outlier_iqr'] = detect_outliers_iqr(values, multiplier=1.5)
    
    # Combined outlier flag (outlier by any method)
    df['is_outlier'] = df['outlier_zscore'] | df['outlier_iqr']
    
    # Additional analysis for low-value outliers specifically
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Flag models with mean final eigenvalue significantly below average
    df['low_final_eig'] = values < (mean_val - 1.5 * std_val)
    
    return df

def create_visualization(df: pd.DataFrame) -> None:
    """Create visualizations showing the outliers."""
    if df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SlimCNN Trained Models: Final Eigenvalue Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean final eigenvalue distribution
    ax1 = axes[0, 0]
    normal_models = df[~df['is_outlier']]['mean_final_eig']
    outlier_models = df[df['is_outlier']]['mean_final_eig']
    
    ax1.hist(normal_models, bins=10, alpha=0.7, label='Normal models', color='blue', edgecolor='black')
    if len(outlier_models) > 0:
        ax1.hist(outlier_models, bins=5, alpha=0.9, label='Outliers', color='red', edgecolor='black')
    
    ax1.set_xlabel('Mean Final Eigenvalue')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Mean Final Eigenvalues')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    ax2 = axes[0, 1]
    ax2.boxplot([normal_models, outlier_models] if len(outlier_models) > 0 else [normal_models], 
                labels=['Normal', 'Outliers'] if len(outlier_models) > 0 else ['All models'])
    ax2.set_ylabel('Mean Final Eigenvalue')
    ax2.set_title('Box Plot of Final Eigenvalues')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model comparison
    ax3 = axes[1, 0]
    x_pos = range(len(df))
    colors = ['red' if outlier else 'blue' for outlier in df['is_outlier']]
    
    bars = ax3.bar(x_pos, df['mean_final_eig'], color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Model Index')
    ax3.set_ylabel('Mean Final Eigenvalue')
    ax3.set_title('Mean Final Eigenvalue by Model')
    ax3.grid(True, alpha=0.3)
    
    # Add model names for outliers
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['is_outlier']:
            ax3.annotate(row['seed'], (i, row['mean_final_eig']), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, rotation=45)
    
    # Plot 4: Min vs Max final eigenvalue scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['min_final_eig'], df['max_final_eig'], 
                         c=['red' if outlier else 'blue' for outlier in df['is_outlier']], 
                         alpha=0.7, s=50, edgecolors='black')
    ax4.set_xlabel('Min Final Eigenvalue')
    ax4.set_ylabel('Max Final Eigenvalue')
    ax4.set_title('Min vs Max Final Eigenvalue')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for outliers
    for idx, row in df.iterrows():
        if row['is_outlier']:
            ax4.annotate(row['seed'], (row['min_final_eig'], row['max_final_eig']), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('slimcnn_outlier_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'slimcnn_outlier_analysis.png'")
    plt.show()

def generate_report(df: pd.DataFrame) -> None:
    """Generate a detailed report of the analysis."""
    print("\n" + "="*80)
    print("SLIMCNN TRAINED MODELS - OUTLIER ANALYSIS REPORT")
    print("="*80)
    
    if df.empty:
        print("No data to analyze!")
        return
    
    print(f"Total models analyzed: {len(df)}")
    print(f"Outliers detected: {df['is_outlier'].sum()}")
    print(f"Models with low final eigenvalues: {df['low_final_eig'].sum()}")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Mean final eigenvalue: {df['mean_final_eig'].mean():.6f} ± {df['mean_final_eig'].std():.6f}")
    print(f"Range: [{df['mean_final_eig'].min():.6f}, {df['mean_final_eig'].max():.6f}]")
    print(f"Median: {df['mean_final_eig'].median():.6f}")
    
    # Outliers details
    outliers = df[df['is_outlier']]
    if not outliers.empty:
        print(f"\n🚨 IDENTIFIED OUTLIERS:")
        print("-" * 50)
        for idx, row in outliers.iterrows():
            print(f"Model: {row['filename']}")
            print(f"  Seed: {row['seed']}")
            print(f"  Mean final eigenvalue: {row['mean_final_eig']:.6f}")
            print(f"  Min final eigenvalue: {row['min_final_eig']:.6f}")
            print(f"  Max final eigenvalue: {row['max_final_eig']:.6f}")
            print(f"  Standard deviation: {row['std_final_eig']:.6f}")
            print(f"  Z-score outlier: {row['outlier_zscore']}")
            print(f"  IQR outlier: {row['outlier_iqr']}")
            print(f"  Low final eigenvalue: {row['low_final_eig']}")
            print()
    
    # Low eigenvalue models (specifically what you're looking for)
    low_models = df[df['low_final_eig']]
    if not low_models.empty:
        print(f"🔍 MODELS WITH UNUSUALLY LOW FINAL EIGENVALUES:")
        print("-" * 50)
        for idx, row in low_models.iterrows():
            deviation = (row['mean_final_eig'] - df['mean_final_eig'].mean()) / df['mean_final_eig'].std()
            print(f"❌ {row['filename']} (seed {row['seed']}): {row['mean_final_eig']:.6f} ({deviation:.2f}σ below mean)")
    
    # Save detailed results to CSV
    output_file = 'slimcnn_outlier_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

def main():
    """Main analysis function."""
    print("Starting SlimCNN Outlier Analysis...")
    
    # Load and analyze all models
    df = analyze_slimcnn_models()
    
    if df.empty:
        print("No models to analyze!")
        return
    
    # Identify outliers
    df = identify_outliers(df)
    
    # Generate report
    generate_report(df)
    
    # Create visualization
    create_visualization(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()