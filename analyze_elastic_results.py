#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Elastic Distance Results

Analyzes the elastic distance matrix computed from eigenvalue evolution curves
to evaluate functional similarity detection between trained and random models.
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results() -> Tuple[np.ndarray, List[str]]:
    """Load the elastic distance results."""
    
    # Load distance matrix
    D = np.load("elastic_eigs_distance.npy")
    
    # Load model index
    with open("elastic_eigs_index.json", "r") as f:
        model_names = json.load(f)
    
    return D, model_names

def classify_models(model_names: List[str]) -> Dict[str, List[int]]:
    """Classify models by type and extract indices."""
    
    trained_indices = []
    random_indices = []
    other_indices = []
    
    for i, name in enumerate(model_names):
        name_lower = name.lower()
        
        if 'random' in name_lower:
            random_indices.append(i)
        elif any(x in name_lower for x in ['trained', 'acc']):
            trained_indices.append(i)
        elif 'mnist' in name_lower and 'random' not in name_lower:
            # MNIST models without explicit 'trained' or 'acc' are trained models
            # This handles patterns like tinycnn_mnist1, mlp4layer_mnist_seed42
            trained_indices.append(i)
        else:
            other_indices.append(i)
    
    return {
        'trained': trained_indices,
        'random': random_indices,
        'other': other_indices
    }

def extract_model_info(model_name: str) -> Dict[str, Any]:
    """Extract detailed information from model name."""
    import re
    
    name_lower = model_name.lower()
    
    info = {
        'name': model_name,
        'type': 'unknown',
        'accuracy': None,
        'epochs': None,
        'architecture': 'unknown',
        'dataset': None,
        'seed': None,
        'variant': None
    }
    
    # Extract accuracy
    acc_match = re.search(r'acc(\d+)', name_lower)
    if acc_match:
        info['accuracy'] = int(acc_match.group(1))
    
    # Extract epochs
    ep_match = re.search(r'ep(\d+)', name_lower)
    if ep_match:
        info['epochs'] = int(ep_match.group(1))
    
    # Extract seed
    seed_match = re.search(r'seed(\d+)', name_lower)
    if seed_match:
        info['seed'] = int(seed_match.group(1))
    
    # Extract variant number (for patterns like mnist1, mnist2, etc.)
    variant_match = re.search(r'mnist(\d+)', name_lower)
    if variant_match:
        info['variant'] = int(variant_match.group(1))
    
    # Determine dataset
    if 'mnist' in name_lower:
        info['dataset'] = 'mnist'
    elif 'cifar' in name_lower:
        info['dataset'] = 'cifar'
    
    # Determine type
    if 'random' in name_lower:
        info['type'] = 'random'
    elif any(x in name_lower for x in ['trained', 'acc']):
        info['type'] = 'trained'
    elif 'mnist' in name_lower and 'random' not in name_lower:
        # MNIST models without explicit 'trained' or 'acc' are trained models
        info['type'] = 'trained'
    
    # Architecture classification
    if 'tinycnn' in name_lower:
        info['architecture'] = 'tinycnn'
    elif 'mlp4layer' in name_lower:
        info['architecture'] = 'mlp4layer'
    elif 'custom' in name_lower:
        info['architecture'] = 'custom'
    elif 'mlp' in name_lower:
        info['architecture'] = 'mlp'
    elif any(x in name_lower for x in ['conv', 'cnn']):
        info['architecture'] = 'conv'
    elif 'resnet' in name_lower:
        info['architecture'] = 'resnet'
    elif 'vgg' in name_lower:
        info['architecture'] = 'vgg'
    
    return info

def calculate_separation_metrics(D: np.ndarray, model_indices: Dict[str, List[int]]) -> Dict[str, Any]:
    """Calculate separation metrics between trained and random models."""
    
    trained_idx = model_indices['trained']
    random_idx = model_indices['random']
    
    if len(trained_idx) < 2 or len(random_idx) < 2:
        return {'error': 'Insufficient trained or random models'}
    
    # Within-trained distances
    within_trained = []
    for i in range(len(trained_idx)):
        for j in range(i + 1, len(trained_idx)):
            within_trained.append(D[trained_idx[i], trained_idx[j]])
    
    # Cross distances (trained vs random)
    cross_distances = []
    for i in trained_idx:
        for j in random_idx:
            cross_distances.append(D[i, j])
    
    # Within-random distances
    within_random = []
    for i in range(len(random_idx)):
        for j in range(i + 1, len(random_idx)):
            within_random.append(D[random_idx[i], random_idx[j]])
    
    # Calculate metrics
    margin = np.min(cross_distances) - np.max(within_trained)
    separation_ratio = np.mean(cross_distances) / np.mean(within_trained) if np.mean(within_trained) > 0 else float('inf')
    
    # Statistical tests
    try:
        t_stat_cross_within, p_cross_within = stats.ttest_ind(cross_distances, within_trained)
        t_stat_cross_random, p_cross_random = stats.ttest_ind(cross_distances, within_random)
    except:
        t_stat_cross_within, p_cross_within = None, None
        t_stat_cross_random, p_cross_random = None, None
    
    return {
        'margin': margin,
        'separation_ratio': separation_ratio,
        'within_trained_mean': np.mean(within_trained),
        'within_trained_std': np.std(within_trained),
        'within_random_mean': np.mean(within_random),
        'within_random_std': np.std(within_random),
        'cross_mean': np.mean(cross_distances),
        'cross_std': np.std(cross_distances),
        'n_trained': len(trained_idx),
        'n_random': len(random_idx),
        'n_within_trained_pairs': len(within_trained),
        'n_within_random_pairs': len(within_random),
        'n_cross_pairs': len(cross_distances),
        't_stat_cross_within': t_stat_cross_within,
        'p_value_cross_within': p_cross_within,
        't_stat_cross_random': t_stat_cross_random,
        'p_value_cross_random': p_cross_random,
        'within_trained_distances': within_trained,
        'within_random_distances': within_random,
        'cross_distances': cross_distances
    }

def analyze_by_architecture(D: np.ndarray, model_names: List[str]) -> Dict[str, Any]:
    """Analyze separation by architecture."""
    
    model_infos = [extract_model_info(name) for name in model_names]
    
    # Group by architecture
    arch_groups = {}
    for i, info in enumerate(model_infos):
        if info['type'] == 'trained':
            arch = info['architecture']
            if arch not in arch_groups:
                arch_groups[arch] = []
            arch_groups[arch].append(i)
    
    # Calculate within-architecture distances
    arch_analysis = {}
    for arch, indices in arch_groups.items():
        if len(indices) >= 2:
            within_arch_distances = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    within_arch_distances.append(D[indices[i], indices[j]])
            
            arch_analysis[arch] = {
                'n_models': len(indices),
                'mean_distance': np.mean(within_arch_distances),
                'std_distance': np.std(within_arch_distances),
                'distances': within_arch_distances
            }
    
    return arch_analysis

def create_visualizations(D: np.ndarray, model_names: List[str], model_indices: Dict[str, List[int]], 
                         metrics: Dict[str, Any]) -> None:
    """Create visualization plots."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distance matrix heatmap
    ax1 = axes[0, 0]
    
    # Reorder matrix to group by type
    trained_idx = model_indices['trained']
    random_idx = model_indices['random']
    other_idx = model_indices['other']
    ordered_idx = trained_idx + random_idx + other_idx
    
    D_ordered = D[np.ix_(ordered_idx, ordered_idx)]
    
    im = ax1.imshow(D_ordered, cmap='viridis', aspect='auto')
    ax1.set_title('Elastic Distance Matrix\n(Trained | Random | Other)')
    ax1.set_xlabel('Model Index')
    ax1.set_ylabel('Model Index')
    
    # Add dividing lines
    n_trained = len(trained_idx)
    n_random = len(random_idx)
    ax1.axhline(y=n_trained - 0.5, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(x=n_trained - 0.5, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(y=n_trained + n_random - 0.5, color='orange', linestyle='--', alpha=0.7)
    ax1.axvline(x=n_trained + n_random - 0.5, color='orange', linestyle='--', alpha=0.7)
    
    plt.colorbar(im, ax=ax1)
    
    # 2. Distance distributions
    ax2 = axes[0, 1]
    
    within_trained = metrics['within_trained_distances']
    within_random = metrics['within_random_distances']
    cross_distances = metrics['cross_distances']
    
    ax2.hist(within_trained, bins=20, alpha=0.7, label='Within Trained', density=True)
    ax2.hist(within_random, bins=20, alpha=0.7, label='Within Random', density=True)
    ax2.hist(cross_distances, bins=20, alpha=0.7, label='Cross (Trained vs Random)', density=True)
    ax2.set_xlabel('Elastic Distance')
    ax2.set_ylabel('Density')
    ax2.set_title('Distance Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = axes[1, 0]
    
    box_data = [within_trained, within_random, cross_distances]
    box_labels = ['Within\nTrained', 'Within\nRandom', 'Cross\n(T vs R)']
    
    bp = ax3.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    bp['boxes'][2].set_facecolor('lightcoral')
    
    ax3.set_ylabel('Elastic Distance')
    ax3.set_title('Distance Distributions by Type')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    p_cross_within_str = f"{metrics['p_value_cross_within']:.6f}" if metrics['p_value_cross_within'] else 'N/A'
    p_cross_random_str = f"{metrics['p_value_cross_random']:.6f}" if metrics['p_value_cross_random'] else 'N/A'
    margin_status = 'YES' if metrics['margin'] > 0 else 'NO'
    
    stats_text = f"""ELASTIC DISTANCE ANALYSIS SUMMARY

📊 Dataset:
  Total models: {len(model_names)}
  Trained models: {metrics['n_trained']}
  Random models: {metrics['n_random']}

🎯 Separation Metrics:
  Margin: {metrics['margin']:+.6f}
  Separation Ratio: {metrics['separation_ratio']:.4f}

📈 Distance Statistics:
  Within Trained: {metrics['within_trained_mean']:.4f} ± {metrics['within_trained_std']:.4f}
  Within Random:  {metrics['within_random_mean']:.4f} ± {metrics['within_random_std']:.4f}
  Cross Distance: {metrics['cross_mean']:.4f} ± {metrics['cross_std']:.4f}

🔬 Statistical Significance:
  Cross vs Within Trained: p = {p_cross_within_str}
  Cross vs Within Random:  p = {p_cross_random_str}

✅ Positive Margin: {margin_status}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('elastic_distance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_accuracy_correlation(D: np.ndarray, model_names: List[str]) -> Dict[str, Any]:
    """Analyze correlation between model accuracy and distance patterns."""
    
    model_infos = [extract_model_info(name) for name in model_names]
    
    # Filter trained models with accuracy information
    trained_with_acc = []
    for i, info in enumerate(model_infos):
        if info['type'] == 'trained' and info['accuracy'] is not None:
            trained_with_acc.append((i, info['accuracy']))
    
    if len(trained_with_acc) < 3:
        return {'error': 'Insufficient trained models with accuracy information'}
    
    # Calculate average distance to all other trained models for each trained model
    trained_indices = [idx for idx, _ in trained_with_acc]
    accuracies = [acc for _, acc in trained_with_acc]
    
    avg_distances = []
    for i, idx in enumerate(trained_indices):
        distances_to_others = []
        for j, other_idx in enumerate(trained_indices):
            if i != j:
                distances_to_others.append(D[idx, other_idx])
        avg_distances.append(np.mean(distances_to_others))
    
    # Calculate correlation
    try:
        corr_coeff, p_value = stats.pearsonr(accuracies, avg_distances)
    except:
        corr_coeff, p_value = None, None
    
    return {
        'accuracies': accuracies,
        'avg_distances': avg_distances,
        'correlation': corr_coeff,
        'p_value': p_value,
        'n_models': len(trained_with_acc)
    }

def print_detailed_analysis(D: np.ndarray, model_names: List[str]) -> None:
    """Print comprehensive analysis results."""
    
    print("="*80)
    print("ELASTIC DISTANCE ANALYSIS RESULTS")
    print("="*80)
    
    # Basic info
    model_indices = classify_models(model_names)
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Total models: {len(model_names)}")
    print(f"  Trained models: {len(model_indices['trained'])}")
    print(f"  Random models: {len(model_indices['random'])}")
    print(f"  Other models: {len(model_indices['other'])}")
    
    # Distance matrix properties
    print(f"\n📏 DISTANCE MATRIX PROPERTIES:")
    print(f"  Shape: {D.shape}")
    print(f"  Min distance: {np.min(D[D > 0]):.6f}")
    print(f"  Max distance: {np.max(D):.6f}")
    print(f"  Mean distance: {np.mean(D[D > 0]):.6f}")
    print(f"  Std distance: {np.std(D[D > 0]):.6f}")
    
    # Separation analysis
    metrics = calculate_separation_metrics(D, model_indices)
    
    if 'error' not in metrics:
        print(f"\n🎯 SEPARATION ANALYSIS:")
        print(f"  Margin: {metrics['margin']:+.6f}")
        print(f"  Separation Ratio: {metrics['separation_ratio']:.4f}")
        
        print(f"\n  Within Trained Models:")
        print(f"    Mean distance: {metrics['within_trained_mean']:.6f}")
        print(f"    Std distance:  {metrics['within_trained_std']:.6f}")
        print(f"    Min distance:  {np.min(metrics['within_trained_distances']):.6f}")
        print(f"    Max distance:  {np.max(metrics['within_trained_distances']):.6f}")
        print(f"    Pairs: {metrics['n_within_trained_pairs']}")
        
        print(f"\n  Within Random Models:")
        print(f"    Mean distance: {metrics['within_random_mean']:.6f}")
        print(f"    Std distance:  {metrics['within_random_std']:.6f}")
        print(f"    Min distance:  {np.min(metrics['within_random_distances']):.6f}")
        print(f"    Max distance:  {np.max(metrics['within_random_distances']):.6f}")
        print(f"    Pairs: {metrics['n_within_random_pairs']}")
        
        print(f"\n  Cross Distances (Trained vs Random):")
        print(f"    Mean distance: {metrics['cross_mean']:.6f}")
        print(f"    Std distance:  {metrics['cross_std']:.6f}")
        print(f"    Min distance:  {np.min(metrics['cross_distances']):.6f}")
        print(f"    Max distance:  {np.max(metrics['cross_distances']):.6f}")
        print(f"    Pairs: {metrics['n_cross_pairs']}")
        
        print(f"\n🔬 STATISTICAL SIGNIFICANCE:")
        if metrics['p_value_cross_within'] is not None:
            significance_cw = "SIGNIFICANT" if metrics['p_value_cross_within'] < 0.05 else "NOT SIGNIFICANT"
            print(f"  Cross vs Within Trained: t = {metrics['t_stat_cross_within']:.4f}, p = {metrics['p_value_cross_within']:.6f} ({significance_cw})")
        
        if metrics['p_value_cross_random'] is not None:
            significance_cr = "SIGNIFICANT" if metrics['p_value_cross_random'] < 0.05 else "NOT SIGNIFICANT"
            print(f"  Cross vs Within Random:  t = {metrics['t_stat_cross_random']:.4f}, p = {metrics['p_value_cross_random']:.6f} ({significance_cr})")
        
        # Success evaluation
        print(f"\n🏆 FUNCTIONAL SIMILARITY DETECTION:")
        if metrics['margin'] > 0:
            print(f"  ✅ SUCCESS: Positive margin achieved ({metrics['margin']:+.6f})")
            print(f"  ✅ Elastic distance successfully separates trained from random models")
        else:
            print(f"  ❌ Negative margin ({metrics['margin']:+.6f})")
            print(f"  📊 Separation ratio: {metrics['separation_ratio']:.4f} (>1.0 indicates some separation)")
        
        # Architecture analysis
        arch_analysis = analyze_by_architecture(D, model_names)
        if arch_analysis:
            print(f"\n🏗️ ARCHITECTURE ANALYSIS:")
            for arch, data in arch_analysis.items():
                print(f"  {arch.upper()} ({data['n_models']} models):")
                print(f"    Mean within-arch distance: {data['mean_distance']:.6f} ± {data['std_distance']:.6f}")
        
        # Accuracy correlation
        acc_analysis = analyze_accuracy_correlation(D, model_names)
        if 'error' not in acc_analysis:
            print(f"\n🎯 ACCURACY CORRELATION:")
            print(f"  Models analyzed: {acc_analysis['n_models']}")
            if acc_analysis['correlation'] is not None:
                print(f"  Correlation (accuracy vs avg distance): r = {acc_analysis['correlation']:.4f}")
                print(f"  P-value: {acc_analysis['p_value']:.6f}")
                if abs(acc_analysis['correlation']) > 0.3:
                    direction = "positive" if acc_analysis['correlation'] > 0 else "negative"
                    print(f"  📊 {direction.upper()} correlation detected")
                else:
                    print(f"  📊 No strong correlation detected")
    
    else:
        print(f"\n❌ Error in separation analysis: {metrics['error']}")

def main():
    """Main analysis function."""
    
    print("Loading elastic distance results...")
    D, model_names = load_results()
    
    print(f"Loaded {len(model_names)} models with {D.shape[0]}x{D.shape[1]} distance matrix")
    
    # Run detailed analysis
    print_detailed_analysis(D, model_names)
    
    # Create visualizations
    model_indices = classify_models(model_names)
    metrics = calculate_separation_metrics(D, model_indices)
    
    if 'error' not in metrics:
        print(f"\n🎨 Creating visualizations...")
        create_visualizations(D, model_names, model_indices, metrics)
        print(f"📊 Saved analysis plot as 'elastic_distance_analysis.png'")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()