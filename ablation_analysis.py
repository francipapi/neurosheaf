#!/usr/bin/env python3
"""
Comprehensive analysis script for ablation study results.

This script analyzes the output from ablation_test.py and generates:
- Summary statistics and rankings
- Parameter importance analysis  
- Publication-ready visualizations
- LaTeX tables for papers
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from scipy.stats import spearmanr, pearsonr

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load and combine all summary.csv files from ablation study."""
    all_results = []
    
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            summary_file = subdir / "summary.csv"
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                df['result_source'] = subdir.name
                all_results.append(df)
    
    if not all_results:
        raise FileNotFoundError(f"No summary.csv files found in {results_dir}")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df

def clean_and_enhance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and add derived metrics."""
    # Calculate Inter/Intra ratio
    df['InterIntraRatio'] = df['BetweenMean'] / df['WithinMean']
    
    # Create a composite score (higher is better)
    df['CompositeScore'] = (
        0.3 * df['ARI'] + 
        0.2 * df['V_measure'] + 
        0.2 * df['Silhouette'] + 
        0.15 * df['InterIntraRatio'] / 5.0 +  # Normalize to similar scale
        0.15 * (1 - df['DaviesBouldin_on_MDS2'] / df['DaviesBouldin_on_MDS2'].max())  # Invert DB (lower is better)
    )
    
    # Add simplified config names for easier reading
    df['SimplifiedConfig'] = df.apply(lambda row: f"{row['distance']}_amp={row['amp_norm']}_smooth={row['smooth']}", axis=1)
    
    return df

def parameter_importance_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze importance of each parameter via correlation with performance metrics."""
    # Define performance metrics (higher = better)
    performance_cols = ['ARI', 'V_measure', 'Silhouette', 'InterIntraRatio', 'CompositeScore']
    
    # Encode categorical variables
    categorical_params = ['distance', 'amp_norm', 'smooth', 'srvf_norm']
    for param in categorical_params:
        if param in df.columns:
            df[f'{param}_encoded'] = pd.Categorical(df[param]).codes
    
    # Analyze correlations
    param_importance = {}
    numeric_params = ['resample', 'smooth_win', 'lambda_warp', 'window_frac', 'dtw_window_frac']
    encoded_params = [f'{p}_encoded' for p in categorical_params if f'{p}_encoded' in df.columns]
    boolean_params = ['time_scaling', 'use_arc_length', 'pad_with_last']
    
    all_params = numeric_params + encoded_params + boolean_params
    
    for param in all_params:
        if param in df.columns and df[param].notna().sum() > 1:
            correlations = []
            for perf_metric in performance_cols:
                if perf_metric in df.columns:
                    # Use Spearman correlation (robust to non-linear relationships)
                    corr, _ = spearmanr(df[param].fillna(df[param].median()), 
                                      df[perf_metric].fillna(0))
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                param_importance[param] = np.mean(correlations)
    
    return param_importance

def create_performance_ranking_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Create a ranking table of top performing configurations."""
    # Sort by composite score (descending)
    top_configs = df.nlargest(top_n, 'CompositeScore')
    
    # Select relevant columns for the table
    table_cols = [
        'distance', 'amp_norm', 'smooth', 'smooth_win', 'time_scaling', 
        'ARI', 'V_measure', 'Silhouette', 'InterIntraRatio', 'CompositeScore'
    ]
    
    # Include elastic-specific parameters if present
    if 'srvf_norm' in top_configs.columns:
        table_cols.insert(-5, 'srvf_norm')
    if 'lambda_warp' in top_configs.columns:
        table_cols.insert(-5, 'lambda_warp')
    if 'window_frac' in top_configs.columns:
        table_cols.insert(-5, 'window_frac')
    
    return top_configs[table_cols].round(3)

def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table code for publication."""
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].round(3)
    
    latex_str = df.to_latex(index=False, escape=False)
    
    # Add caption and label
    latex_str = latex_str.replace(
        '\\end{tabular}',
        f'\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}'
    )
    
    return latex_str

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create publication-ready visualizations."""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Distance method comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    distance_comparison = df.groupby('distance').agg({
        'ARI': ['mean', 'std'],
        'Silhouette': ['mean', 'std'],
        'InterIntraRatio': ['mean', 'std']
    }).round(3)
    
    distance_comparison.columns = ['_'.join(col) for col in distance_comparison.columns]
    distance_means = distance_comparison[[col for col in distance_comparison.columns if 'mean' in col]]
    distance_stds = distance_comparison[[col for col in distance_comparison.columns if 'std' in col]]
    
    x = np.arange(len(distance_means.index))
    width = 0.25
    
    metrics = ['ARI', 'Silhouette', 'InterIntraRatio']
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        means = distance_means[f'{metric}_mean']
        stds = distance_stds[f'{metric}_std']
        # Normalize InterIntraRatio to [0,1] scale for comparison
        if metric == 'InterIntraRatio':
            means = means / 5.0  # Approximate normalization
            stds = stds / 5.0
        
        ax.bar(x + i*width, means, width, yerr=stds, label=metric, color=color, alpha=0.7)
    
    ax.set_xlabel('Distance Method')
    ax.set_ylabel('Performance Score')
    ax.set_title('Distance Method Comparison (Mean ± Std)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(distance_means.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter heatmap (if elastic parameters exist)
    if 'lambda_warp' in df.columns and 'window_frac' in df.columns:
        elastic_df = df[df['distance'] == 'elastic']
        if len(elastic_df) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create pivot table for heatmap
            heatmap_data = elastic_df.pivot_table(
                values='ARI', 
                index='lambda_warp', 
                columns='window_frac', 
                aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title('Elastic Distance: λ_warp vs window_frac (ARI Performance)')
            ax.set_xlabel('Window Fraction')
            ax.set_ylabel('Lambda Warp')
            plt.tight_layout()
            plt.savefig(output_dir / 'elastic_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Performance distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    performance_metrics = ['ARI', 'Silhouette', 'InterIntraRatio', 'CompositeScore']
    for i, metric in enumerate(performance_metrics):
        ax = axes[i//2, i%2]
        df.boxplot(column=metric, by='distance', ax=ax)
        ax.set_title(f'{metric} by Distance Method')
        ax.set_xlabel('Distance Method')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
    plt.suptitle('Performance Metrics Distribution by Distance Method')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis pipeline."""
    # Configuration
    results_dir = Path("ablation_full")
    output_dir = Path("ablation_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading ablation study results...")
    try:
        df = load_all_results(results_dir)
        print(f"Loaded {len(df)} configurations from {len(df['result_source'].unique())} result sources")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print("Cleaning and enhancing data...")
    df = clean_and_enhance_data(df)
    
    print("Analyzing parameter importance...")
    param_importance = parameter_importance_analysis(df)
    
    # Sort by importance
    sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nParameter Importance Ranking:")
    print("=" * 50)
    for i, (param, importance) in enumerate(sorted_params, 1):
        print(f"{i:2d}. {param:20s}: {importance:.4f}")
    
    print(f"\nCreating performance ranking table...")
    top_configs = create_performance_ranking_table(df, top_n=20)
    
    print(f"\nTop 10 Configurations:")
    print("=" * 80)
    print(top_configs.head(10).to_string(index=False))
    
    print(f"\nGenerating visualizations...")
    create_visualizations(df, output_dir / "figures")
    
    # Save results
    print(f"\nSaving results to {output_dir}/...")
    
    # Save full results
    df.to_csv(output_dir / "combined_results.csv", index=False)
    
    # Save top configurations
    top_configs.to_csv(output_dir / "top_configurations.csv", index=False)
    
    # Save parameter importance
    param_df = pd.DataFrame(sorted_params, columns=['Parameter', 'Importance'])
    param_df.to_csv(output_dir / "parameter_importance.csv", index=False)
    
    # Generate LaTeX tables
    with open(output_dir / "latex_tables.tex", "w") as f:
        f.write("% Top Configurations Table\n")
        f.write(generate_latex_table(
            top_configs.head(10), 
            "Top 10 performing configurations in ablation study",
            "tab:top_configs"
        ))
        f.write("\n\n% Parameter Importance Table\n")
        f.write(generate_latex_table(
            param_df.head(10),
            "Parameter importance ranking based on correlation with performance metrics",
            "tab:param_importance"
        ))
    
    print("\nAnalysis Summary:")
    print("=" * 50)
    print(f"Best configuration (ARI={df['ARI'].max():.3f}):")
    best_idx = df['ARI'].idxmax()
    best_config = df.loc[best_idx]
    for col in ['distance', 'amp_norm', 'smooth', 'smooth_win', 'time_scaling']:
        if col in best_config:
            print(f"  {col}: {best_config[col]}")
    
    print(f"\nFiles saved:")
    print(f"  - Combined results: {output_dir}/combined_results.csv")
    print(f"  - Top configurations: {output_dir}/top_configurations.csv")
    print(f"  - Parameter importance: {output_dir}/parameter_importance.csv")
    print(f"  - LaTeX tables: {output_dir}/latex_tables.tex")
    print(f"  - Figures: {output_dir}/figures/")
    print("\nDone!")

if __name__ == "__main__":
    main()