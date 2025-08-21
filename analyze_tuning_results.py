#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Script for EDR Parameter Tuning Results

Analyzes the results from parameter tuning to identify the best configurations
and understand the impact of different parameters on functional similarity capture.

Usage:
    python analyze_tuning_results.py                          # Analyze default results
    python analyze_tuning_results.py --results-dir custom/    # Analyze custom directory
    python analyze_tuning_results.py --plot                   # Generate plots
"""

import argparse
import json
import pathlib
from typing import Dict, List, Any
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARN] Matplotlib/Seaborn not available. Plotting disabled.")

def load_results(results_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Load all tuning results from the specified directory."""
    all_results = []
    
    # Try to load the main results file
    main_file = results_dir / "all_results.json"
    if main_file.exists():
        with open(main_file, 'r') as f:
            all_results = json.load(f)
        print(f"[INFO] Loaded {len(all_results)} results from {main_file}")
        return all_results
    
    # Otherwise, load batch files
    batch_files = list(results_dir.glob("results_batch_*.json"))
    if not batch_files:
        raise ValueError(f"No results found in {results_dir}")
    
    for batch_file in sorted(batch_files):
        with open(batch_file, 'r') as f:
            batch_results = json.load(f)
            all_results.extend(batch_results)
    
    print(f"[INFO] Loaded {len(all_results)} results from {len(batch_files)} batch files")
    return all_results

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform comprehensive analysis of tuning results."""
    # Filter valid results
    valid_results = [r for r in results if 'metrics' in r and 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Total configurations: {len(results)}")
    print(f"Successful configurations: {len(valid_results)}")
    print(f"Failed configurations: {len(failed_results)}")
    
    if not valid_results:
        print("[ERROR] No valid results to analyze!")
        return {}
    
    # Extract metrics
    df_data = []
    for result in valid_results:
        config = result['config']
        metrics = result['metrics']
        
        row = config.copy()
        row.update(metrics)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Basic statistics
    print(f"\n=== SEPARATION METRICS ===")
    print(f"Separation Ratio:")
    print(f"  Best: {df['separation_ratio'].max():.4f}")
    print(f"  Mean: {df['separation_ratio'].mean():.4f}")
    print(f"  Std:  {df['separation_ratio'].std():.4f}")
    print(f"  Min:  {df['separation_ratio'].min():.4f}")
    
    print(f"\nMargin:")
    print(f"  Best: {df['margin'].max():.4f}")
    print(f"  Mean: {df['margin'].mean():.4f}")
    print(f"  Std:  {df['margin'].std():.4f}")
    print(f"  Min:  {df['margin'].min():.4f}")
    
    # Find best configurations
    top_n = 5
    best_by_ratio = df.nlargest(top_n, 'separation_ratio')
    best_by_margin = df.nlargest(top_n, 'margin')
    
    print(f"\n=== TOP {top_n} CONFIGURATIONS BY SEPARATION RATIO ===")
    for i, (idx, row) in enumerate(best_by_ratio.iterrows(), 1):
        print(f"\n#{i} - Ratio: {row['separation_ratio']:.4f}, Margin: {row['margin']:.4f}")
        for param in ['K', 'USE_LOG1P', 'NORMALIZE_MODE', 'TOP_K_EIGS', 'REL_EPS_MULT', 'EDR_GAP_COST']:
            if param in row:
                print(f"  {param}: {row[param]}")
    
    print(f"\n=== TOP {top_n} CONFIGURATIONS BY MARGIN ===")
    for i, (idx, row) in enumerate(best_by_margin.iterrows(), 1):
        print(f"\n#{i} - Ratio: {row['separation_ratio']:.4f}, Margin: {row['margin']:.4f}")
        for param in ['K', 'USE_LOG1P', 'NORMALIZE_MODE', 'TOP_K_EIGS', 'REL_EPS_MULT', 'EDR_GAP_COST']:
            if param in row:
                print(f"  {param}: {row[param]}")
    
    # Parameter impact analysis
    print(f"\n=== PARAMETER IMPACT ANALYSIS ===")
    categorical_params = ['USE_LOG1P', 'NORMALIZE_MODE', 'TOP_K_EIGS']
    numerical_params = ['K', 'REL_EPS_MULT', 'EDR_GAP_COST']
    
    for param in categorical_params:
        if param in df.columns:
            print(f"\n{param}:")
            grouped = df.groupby(param)['separation_ratio'].agg(['mean', 'std', 'count'])
            for value, row in grouped.iterrows():
                print(f"  {value}: mean={row['mean']:.4f}, std={row['std']:.4f}, count={row['count']}")
    
    # Correlation analysis for numerical parameters
    if len(numerical_params) > 1:
        print(f"\n=== CORRELATION ANALYSIS ===")
        corr_params = [p for p in numerical_params if p in df.columns] + ['separation_ratio', 'margin']
        if len(corr_params) > 2:
            corr_matrix = df[corr_params].corr()
            print("Correlation with separation_ratio:")
            for param in corr_params:
                if param != 'separation_ratio':
                    corr = corr_matrix.loc[param, 'separation_ratio']
                    print(f"  {param}: {corr:.4f}")
    
    # Identify optimal parameter ranges
    print(f"\n=== OPTIMAL PARAMETER RANGES ===")
    high_performance = df[df['separation_ratio'] >= df['separation_ratio'].quantile(0.8)]
    
    for param in df.columns:
        if param in ['K', 'REL_EPS_MULT', 'EDR_GAP_COST', 'USE_LOG1P', 'NORMALIZE_MODE', 'TOP_K_EIGS']:
            if df[param].dtype in ['object', 'bool'] or df[param].nunique() <= 10:
                # Categorical analysis
                best_values = high_performance[param].value_counts()
                print(f"\n{param} (in top 20% configs):")
                for value, count in best_values.items():
                    pct = count / len(high_performance) * 100
                    print(f"  {value}: {count}/{len(high_performance)} ({pct:.1f}%)")
            else:
                # Numerical analysis
                print(f"\n{param} (in top 20% configs):")
                print(f"  Range: {high_performance[param].min():.3f} - {high_performance[param].max():.3f}")
                print(f"  Mean: {high_performance[param].mean():.3f}")
                print(f"  Median: {high_performance[param].median():.3f}")
    
    return {
        'dataframe': df,
        'valid_results': len(valid_results),
        'failed_results': len(failed_results),
        'best_config': valid_results[df['separation_ratio'].idxmax()],
        'summary_stats': {
            'separation_ratio': {
                'best': df['separation_ratio'].max(),
                'mean': df['separation_ratio'].mean(),
                'std': df['separation_ratio'].std()
            },
            'margin': {
                'best': df['margin'].max(),
                'mean': df['margin'].mean(),
                'std': df['margin'].std()
            }
        }
    }

def create_plots(df: pd.DataFrame, output_dir: pathlib.Path):
    """Create visualization plots for parameter tuning results."""
    if not PLOTTING_AVAILABLE:
        print("[WARN] Plotting not available. Install matplotlib and seaborn.")
        return
    
    plt.style.use('default')
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Distribution of separation ratios
    plt.figure(figsize=(10, 6))
    plt.hist(df['separation_ratio'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Separation Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Separation Ratios')
    plt.axvline(df['separation_ratio'].mean(), color='red', linestyle='--', label=f'Mean: {df["separation_ratio"].mean():.3f}')
    plt.axvline(df['separation_ratio'].quantile(0.9), color='green', linestyle='--', label=f'90th percentile: {df["separation_ratio"].quantile(0.9):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "separation_ratio_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Separation ratio vs margin
    plt.figure(figsize=(10, 8))
    plt.scatter(df['separation_ratio'], df['margin'], alpha=0.6)
    plt.xlabel('Separation Ratio')
    plt.ylabel('Margin')
    plt.title('Separation Ratio vs Margin')
    # Highlight best points
    top_configs = df.nlargest(5, 'separation_ratio')
    plt.scatter(top_configs['separation_ratio'], top_configs['margin'], 
                color='red', s=100, alpha=0.8, label='Top 5 configs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "ratio_vs_margin.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Parameter impact heatmaps
    categorical_params = ['USE_LOG1P', 'NORMALIZE_MODE', 'TOP_K_EIGS']
    
    for param in categorical_params:
        if param in df.columns and df[param].nunique() > 1:
            plt.figure(figsize=(12, 6))
            
            # Box plot
            plt.subplot(1, 2, 1)
            df.boxplot(column='separation_ratio', by=param, ax=plt.gca())
            plt.title(f'Separation Ratio by {param}')
            plt.suptitle('')  # Remove default title
            
            # Bar plot with means
            plt.subplot(1, 2, 2)
            means = df.groupby(param)['separation_ratio'].mean()
            means.plot(kind='bar')
            plt.title(f'Mean Separation Ratio by {param}')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(fig_dir / f"impact_{param.lower()}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Correlation heatmap
    numerical_cols = ['K', 'REL_EPS_MULT', 'EDR_GAP_COST', 'separation_ratio', 'margin']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(numerical_cols) > 2:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.3f')
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[INFO] Plots saved to {fig_dir}")

def save_report(analysis: Dict[str, Any], output_dir: pathlib.Path):
    """Save a comprehensive report of the analysis."""
    report_file = output_dir / "tuning_report.json"
    
    # Make the report JSON-serializable
    report = {
        'summary': {
            'total_configs': analysis['valid_results'] + analysis['failed_results'],
            'successful_configs': analysis['valid_results'],
            'failed_configs': analysis['failed_results']
        },
        'best_config': analysis['best_config'],
        'summary_stats': analysis['summary_stats']
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[INFO] Report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter tuning results for EDR/LCSS metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results-dir', type=str, default='tuning_results', 
                       help='Directory containing tuning results')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    parser.add_argument('--save-report', action='store_true', help='Save analysis report')
    
    args = parser.parse_args()
    
    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    # Load and analyze results
    results = load_results(results_dir)
    analysis = analyze_results(results)
    
    if not analysis:
        return
    
    # Generate plots if requested
    if args.plot and 'dataframe' in analysis:
        create_plots(analysis['dataframe'], results_dir)
    
    # Save report if requested
    if args.save_report:
        save_report(analysis, results_dir)
    
    print(f"\n[INFO] Analysis complete! Check {results_dir} for outputs.")

if __name__ == "__main__":
    main()