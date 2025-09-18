#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outlier Analysis Script

Analyzes the best EDR configuration to identify specific model pairs causing 
negative margins and preventing positive functional similarity detection.

Usage:
    python outlier_analysis.py                    # Analyze default best config
    python outlier_analysis.py --results-dir custom/  # Custom results directory
    python outlier_analysis.py --plot             # Generate visualization plots
"""

import argparse
import json
import pathlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARN] Matplotlib/Seaborn not available. Plotting disabled.")

def load_best_configuration(results_dir: pathlib.Path) -> Dict[str, Any]:
    """Load the best configuration from tuning results."""
    # First try the tuning report
    report_file = results_dir / "tuning_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
            return report['best_config']
    
    # Otherwise, look for best_config.json
    best_file = results_dir / "best_config.json"
    if best_file.exists():
        with open(best_file, 'r') as f:
            return json.load(f)
    
    # Last resort: find best from all_results.json
    all_results_file = results_dir / "all_results.json"
    if all_results_file.exists():
        with open(all_results_file, 'r') as f:
            results = json.load(f)
            valid_results = [r for r in results if 'metrics' in r]
            if valid_results:
                best_idx = np.argmax([r['metrics']['separation_ratio'] for r in valid_results])
                return valid_results[best_idx]
    
    raise ValueError(f"No valid configuration found in {results_dir}")

def analyze_distance_matrix(distance_matrix: List[List[float]], 
                          model_names: List[str]) -> Dict[str, Any]:
    """Analyze the distance matrix to identify outliers."""
    D = np.array(distance_matrix)
    N = len(model_names)
    
    # Classify models
    trained_indices = []
    random_indices = []
    
    for i, name in enumerate(model_names):
        name_lower = name.lower()
        if 'trained' in name_lower or 'acc' in name_lower:
            trained_indices.append(i)
        elif 'random' in name_lower:
            random_indices.append(i)
    
    print(f"[INFO] Found {len(trained_indices)} trained models, {len(random_indices)} random models")
    
    # Compute distance categories
    within_trained_pairs = []
    cross_pairs = []
    within_random_pairs = []
    
    for i in trained_indices:
        for j in trained_indices:
            if i < j:  # Only upper triangle
                within_trained_pairs.append({
                    'i': i, 'j': j,
                    'model_i': model_names[i],
                    'model_j': model_names[j],
                    'distance': D[i, j],
                    'category': 'within_trained'
                })
    
    for i in trained_indices:
        for j in random_indices:
            cross_pairs.append({
                'i': i, 'j': j,
                'model_i': model_names[i],
                'model_j': model_names[j],
                'distance': D[i, j],
                'category': 'cross'
            })
    
    for i in random_indices:
        for j in random_indices:
            if i < j:  # Only upper triangle
                within_random_pairs.append({
                    'i': i, 'j': j,
                    'model_i': model_names[i],
                    'model_j': model_names[j],
                    'distance': D[i, j],
                    'category': 'within_random'
                })
    
    return {
        'within_trained_pairs': within_trained_pairs,
        'cross_pairs': cross_pairs,
        'within_random_pairs': within_random_pairs,
        'trained_indices': trained_indices,
        'random_indices': random_indices
    }

def identify_outliers(pairs_data: Dict[str, List[Dict]], 
                     top_n: int = 10) -> Dict[str, List[Dict]]:
    """Identify the most problematic pairs for margin calculation."""
    
    within_trained = pairs_data['within_trained_pairs']
    cross_pairs = pairs_data['cross_pairs']
    
    # Sort within-trained pairs by distance (highest first - these hurt margin)
    worst_within_trained = sorted(within_trained, 
                                 key=lambda x: x['distance'], 
                                 reverse=True)[:top_n]
    
    # Sort cross pairs by distance (lowest first - these also hurt margin)
    best_cross = sorted(cross_pairs, 
                       key=lambda x: x['distance'])[:top_n]
    
    return {
        'worst_within_trained': worst_within_trained,
        'best_cross': best_cross
    }

def classify_model_architecture(model_name: str) -> Tuple[str, str]:
    """Classify model by architecture and training status."""
    name_lower = model_name.lower()
    
    # Architecture
    if any(x in name_lower for x in ['custom', 'conv']):
        architecture = 'Custom'
    elif 'mlp' in name_lower:
        architecture = 'MLP'
    else:
        architecture = 'Other'
    
    # Training status
    if any(x in name_lower for x in ['trained', 'acc']):
        status = 'Trained'
    elif 'random' in name_lower:
        status = 'Random'
    else:
        status = 'Unknown'
    
    return architecture, status

def analyze_architecture_effects(pairs_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze how different architectures affect distances."""
    within_trained = pairs_data['within_trained_pairs']
    
    # Classify pairs by architecture combination
    same_arch_pairs = []
    diff_arch_pairs = []
    
    for pair in within_trained:
        arch_i, _ = classify_model_architecture(pair['model_i'])
        arch_j, _ = classify_model_architecture(pair['model_j'])
        
        pair_info = pair.copy()
        pair_info['arch_i'] = arch_i
        pair_info['arch_j'] = arch_j
        pair_info['same_architecture'] = (arch_i == arch_j)
        
        if arch_i == arch_j:
            same_arch_pairs.append(pair_info)
        else:
            diff_arch_pairs.append(pair_info)
    
    # Statistics
    same_arch_distances = [p['distance'] for p in same_arch_pairs]
    diff_arch_distances = [p['distance'] for p in diff_arch_pairs]
    
    return {
        'same_architecture_pairs': same_arch_pairs,
        'different_architecture_pairs': diff_arch_pairs,
        'same_arch_stats': {
            'count': len(same_arch_distances),
            'mean': np.mean(same_arch_distances) if same_arch_distances else 0,
            'std': np.std(same_arch_distances) if same_arch_distances else 0,
            'max': np.max(same_arch_distances) if same_arch_distances else 0,
            'min': np.min(same_arch_distances) if same_arch_distances else 0
        },
        'diff_arch_stats': {
            'count': len(diff_arch_distances),
            'mean': np.mean(diff_arch_distances) if diff_arch_distances else 0,
            'std': np.std(diff_arch_distances) if diff_arch_distances else 0,
            'max': np.max(diff_arch_distances) if diff_arch_distances else 0,
            'min': np.min(diff_arch_distances) if diff_arch_distances else 0
        }
    }

def calculate_margin_impact(outliers: Dict[str, List[Dict]], 
                           pairs_data: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Calculate the impact of removing outliers on margin."""
    
    within_trained = pairs_data['within_trained_pairs']
    cross_pairs = pairs_data['cross_pairs']
    
    # Original metrics
    within_distances = [p['distance'] for p in within_trained]
    cross_distances = [p['distance'] for p in cross_pairs]
    
    original_max_within = np.max(within_distances)
    original_min_cross = np.min(cross_distances)
    original_margin = original_min_cross - original_max_within
    
    # Impact of removing worst within-trained pairs
    worst_models = set()
    for pair in outliers['worst_within_trained'][:5]:  # Top 5 worst
        worst_models.add(pair['model_i'])
        worst_models.add(pair['model_j'])
    
    # Filter out pairs involving worst models
    filtered_within = [p for p in within_trained 
                      if p['model_i'] not in worst_models and p['model_j'] not in worst_models]
    filtered_cross = [p for p in cross_pairs 
                     if p['model_i'] not in worst_models]
    
    if filtered_within and filtered_cross:
        filtered_within_distances = [p['distance'] for p in filtered_within]
        filtered_cross_distances = [p['distance'] for p in filtered_cross]
        
        filtered_max_within = np.max(filtered_within_distances)
        filtered_min_cross = np.min(filtered_cross_distances)
        filtered_margin = filtered_min_cross - filtered_max_within
    else:
        filtered_margin = float('-inf')
    
    return {
        'original_margin': original_margin,
        'original_max_within': original_max_within,
        'original_min_cross': original_min_cross,
        'filtered_margin': filtered_margin,
        'margin_improvement': filtered_margin - original_margin,
        'models_to_remove': list(worst_models),
        'n_models_removed': len(worst_models)
    }

def print_analysis_results(config: Dict[str, Any], 
                          outliers: Dict[str, List[Dict]],
                          arch_analysis: Dict[str, Any],
                          margin_impact: Dict[str, float]) -> None:
    """Print comprehensive analysis results."""
    
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS RESULTS")
    print("="*60)
    
    # Configuration info
    print(f"\nBest Configuration:")
    print(f"  Separation Ratio: {config['metrics']['separation_ratio']:.4f}")
    print(f"  Margin: {config['metrics']['margin']:.4f}")
    print(f"  Parameters: {config['config']}")
    
    # Worst within-trained pairs
    print(f"\n--- TOP 10 WORST WITHIN-TRAINED PAIRS ---")
    for i, pair in enumerate(outliers['worst_within_trained'], 1):
        arch_i, _ = classify_model_architecture(pair['model_i'])
        arch_j, _ = classify_model_architecture(pair['model_j'])
        print(f"{i:2d}. {pair['distance']:.4f} | {arch_i:6s} {pair['model_i'][:30]:30s} ↔ {arch_j:6s} {pair['model_j'][:30]:30s}")
    
    # Best cross pairs
    print(f"\n--- TOP 10 BEST (LOWEST) CROSS PAIRS ---")
    for i, pair in enumerate(outliers['best_cross'], 1):
        arch_i, _ = classify_model_architecture(pair['model_i'])
        arch_j, _ = classify_model_architecture(pair['model_j'])
        print(f"{i:2d}. {pair['distance']:.4f} | {arch_i:6s} {pair['model_i'][:30]:30s} ↔ {arch_j:6s} {pair['model_j'][:30]:30s}")
    
    # Architecture analysis
    print(f"\n--- ARCHITECTURE ANALYSIS ---")
    print(f"Same Architecture Pairs:")
    print(f"  Count: {arch_analysis['same_arch_stats']['count']}")
    print(f"  Mean Distance: {arch_analysis['same_arch_stats']['mean']:.4f}")
    print(f"  Max Distance: {arch_analysis['same_arch_stats']['max']:.4f}")
    
    print(f"Different Architecture Pairs:")
    print(f"  Count: {arch_analysis['diff_arch_stats']['count']}")
    print(f"  Mean Distance: {arch_analysis['diff_arch_stats']['mean']:.4f}")
    print(f"  Max Distance: {arch_analysis['diff_arch_stats']['max']:.4f}")
    
    # Margin impact analysis
    print(f"\n--- MARGIN IMPACT ANALYSIS ---")
    print(f"Original Margin: {margin_impact['original_margin']:.4f}")
    print(f"  Max Within-Trained: {margin_impact['original_max_within']:.4f}")
    print(f"  Min Cross: {margin_impact['original_min_cross']:.4f}")
    
    print(f"After Removing {margin_impact['n_models_removed']} Worst Models:")
    print(f"  New Margin: {margin_impact['filtered_margin']:.4f}")
    print(f"  Improvement: {margin_impact['margin_improvement']:.4f}")
    
    print(f"\nModels to Consider Removing:")
    for model in sorted(margin_impact['models_to_remove']):
        arch, status = classify_model_architecture(model)
        print(f"  {arch:6s} {status:7s} {model}")
    
    # Recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    if margin_impact['filtered_margin'] > 0:
        print(f"✅ POSITIVE MARGIN achievable by removing {margin_impact['n_models_removed']} outlier models!")
    else:
        print(f"❌ Outlier removal alone insufficient. Need additional strategies:")
        
    if arch_analysis['diff_arch_stats']['mean'] > arch_analysis['same_arch_stats']['mean'] * 1.5:
        print(f"  • Consider architecture-specific analysis")
        
    print(f"  • Try alternative distance metrics")
    print(f"  • Implement adaptive epsilon tuning")
    print(f"  • Use ensemble approach with multiple metrics")

def create_visualizations(outliers: Dict[str, List[Dict]],
                         arch_analysis: Dict[str, Any],
                         output_dir: pathlib.Path) -> None:
    """Create visualization plots for outlier analysis."""
    if not PLOTTING_AVAILABLE:
        print("[WARN] Plotting not available.")
        return
        
    fig_dir = output_dir / "outlier_figures"
    fig_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    
    # 1. Distance distribution by pair type
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Within-trained distances
    within_distances = [p['distance'] for p in outliers['worst_within_trained']]
    cross_distances = [p['distance'] for p in outliers['best_cross']]
    
    axes[0].hist(within_distances, bins=20, alpha=0.7, color='red', label='Worst Within-Trained')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Frequency') 
    axes[0].set_title('Distribution of Worst Within-Trained Distances')
    axes[0].axvline(np.mean(within_distances), color='darkred', linestyle='--', 
                   label=f'Mean: {np.mean(within_distances):.3f}')
    axes[0].legend()
    
    axes[1].hist(cross_distances, bins=20, alpha=0.7, color='blue', label='Best Cross Distances')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Best (Lowest) Cross Distances')
    axes[1].axvline(np.mean(cross_distances), color='darkblue', linestyle='--',
                   label=f'Mean: {np.mean(cross_distances):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / "outlier_distances.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Architecture comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    same_arch_distances = [p['distance'] for p in arch_analysis['same_architecture_pairs']]
    diff_arch_distances = [p['distance'] for p in arch_analysis['different_architecture_pairs']]
    
    data_to_plot = []
    if same_arch_distances:
        data_to_plot.extend([(d, 'Same Architecture') for d in same_arch_distances])
    if diff_arch_distances:
        data_to_plot.extend([(d, 'Different Architecture') for d in diff_arch_distances])
    
    if data_to_plot:
        df_arch = pd.DataFrame(data_to_plot, columns=['Distance', 'Architecture_Type'])
        
        import seaborn as sns
        sns.boxplot(data=df_arch, x='Architecture_Type', y='Distance', ax=ax)
        ax.set_title('Distance Distribution by Architecture Pairing')
        ax.set_ylabel('Distance')
        ax.set_xlabel('Architecture Pairing Type')
        
        # Add statistical annotations
        if same_arch_distances and diff_arch_distances:
            same_mean = np.mean(same_arch_distances)
            diff_mean = np.mean(diff_arch_distances)
            ax.text(0, ax.get_ylim()[1] * 0.9, f'Mean: {same_mean:.3f}', 
                   ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat'))
            ax.text(1, ax.get_ylim()[1] * 0.9, f'Mean: {diff_mean:.3f}', 
                   ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(fig_dir / "architecture_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Visualizations saved to {fig_dir}")

def save_detailed_results(outliers: Dict[str, List[Dict]],
                         arch_analysis: Dict[str, Any],
                         margin_impact: Dict[str, float],
                         output_dir: pathlib.Path) -> None:
    """Save detailed analysis results to files."""
    
    # Save outlier pairs
    outlier_file = output_dir / "outlier_pairs.json"
    with open(outlier_file, 'w') as f:
        json.dump(outliers, f, indent=2)
    
    # Save architecture analysis
    arch_file = output_dir / "architecture_analysis.json"
    with open(arch_file, 'w') as f:
        json.dump(arch_analysis, f, indent=2)
    
    # Save margin impact analysis
    margin_file = output_dir / "margin_impact_analysis.json"
    with open(margin_file, 'w') as f:
        json.dump(margin_impact, f, indent=2)
    
    # Save CSV of worst pairs for easy inspection
    worst_pairs_csv = output_dir / "worst_within_trained_pairs.csv"
    worst_df = pd.DataFrame(outliers['worst_within_trained'])
    if not worst_df.empty:
        # Add architecture info
        worst_df['arch_i'] = worst_df['model_i'].apply(lambda x: classify_model_architecture(x)[0])
        worst_df['arch_j'] = worst_df['model_j'].apply(lambda x: classify_model_architecture(x)[0])
        worst_df['same_arch'] = worst_df['arch_i'] == worst_df['arch_j']
        worst_df.to_csv(worst_pairs_csv, index=False)
    
    # Save CSV of best cross pairs
    best_cross_csv = output_dir / "best_cross_pairs.csv"
    best_df = pd.DataFrame(outliers['best_cross'])
    if not best_df.empty:
        best_df['arch_trained'] = best_df['model_i'].apply(lambda x: classify_model_architecture(x)[0])
        best_df['arch_random'] = best_df['model_j'].apply(lambda x: classify_model_architecture(x)[0])
        best_df.to_csv(best_cross_csv, index=False)
    
    print(f"[INFO] Detailed results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze outliers preventing positive margins in functional similarity detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results-dir', type=str, default='tuning_results',
                       help='Directory containing tuning results')
    parser.add_argument('--output-dir', type=str, default='outlier_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top outliers to analyze')
    
    args = parser.parse_args()
    
    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] Loading best configuration from {results_dir}")
    
    # Load best configuration
    try:
        best_config = load_best_configuration(results_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        return
    
    # Extract data
    distance_matrix = best_config['distance_matrix']
    model_names = best_config['model_names']
    
    print(f"[INFO] Analyzing {len(model_names)} models")
    
    # Analyze distance matrix
    pairs_data = analyze_distance_matrix(distance_matrix, model_names)
    
    # Identify outliers
    outliers = identify_outliers(pairs_data, args.top_n)
    
    # Architecture analysis
    arch_analysis = analyze_architecture_effects(pairs_data)
    
    # Margin impact analysis
    margin_impact = calculate_margin_impact(outliers, pairs_data)
    
    # Print results
    print_analysis_results(best_config, outliers, arch_analysis, margin_impact)
    
    # Save detailed results
    save_detailed_results(outliers, arch_analysis, margin_impact, output_dir)
    
    # Create visualizations if requested
    if args.plot:
        create_visualizations(outliers, arch_analysis, output_dir)
    
    print(f"\n[INFO] Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()