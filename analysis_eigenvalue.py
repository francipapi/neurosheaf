#!/usr/bin/env python3
"""
Eigenvalue Evolution Analysis using ISW Distance

This script computes pairwise Integrated Sliced Wasserstein (ISW) distances 
between all eigenvalue evolution files in the eigenvalueData folder.

The analysis uses the following configuration:
- p=1 (L1 norm)
- n_quantiles=199 
- tail_trim=0.02 (2% tail trimming)
- time_weight_alpha=3.8 (exponential time weighting)
- eps=1e-3 (numerical stability floor)
- log_scale=True (log1p transformation)
- normalize='p95' (95th percentile normalization)
- K=200 (common time grid points)

Output:
- CSV file with pairwise distance matrix
- Heatmap visualization
- Console summary with key findings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Tuple, Dict
import warnings

# Neurosheaf imports
from neurosheaf.utils import EigenvalueISW

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def find_eigenvalue_files(data_dir: str = "eigenvalueData") -> List[Path]:
    """Find all .npz files in the eigenvalue data directory.
    
    Args:
        data_dir: Directory containing eigenvalue evolution files
        
    Returns:
        List of Path objects for eigenvalue files
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")
    
    # Find all .npz files
    eigenvalue_files = list(data_path.glob("*.npz"))
    
    if not eigenvalue_files:
        raise FileNotFoundError(f"No .npz files found in '{data_dir}'")
    
    # Sort for consistent ordering
    eigenvalue_files.sort()
    
    logger.info(f"Found {len(eigenvalue_files)} eigenvalue evolution files:")
    for file in eigenvalue_files:
        logger.info(f"  - {file.name}")
    
    return eigenvalue_files


def extract_model_info(eigenvalue_files: List[Path]) -> List[Dict]:
    """Extract model information from eigenvalue files.
    
    Args:
        eigenvalue_files: List of eigenvalue evolution files
        
    Returns:
        List of dictionaries with model information
    """
    model_info = []
    
    for file_path in eigenvalue_files:
        try:
            # Load metadata
            data = np.load(file_path, allow_pickle=True)
            metadata = data.get('metadata', {})
            
            if isinstance(metadata, np.ndarray):
                metadata = metadata.item()
            
            # Extract basic info
            info = {
                'filename': file_path.name,
                'model_name': file_path.stem,
                'eigenvalue_shape': data['eigenvalue_matrix'].shape,
                'time_steps': len(data['time_vector']),
                'time_range': (float(data['time_vector'].min()), float(data['time_vector'].max())),
                'metadata': metadata
            }
            
            model_info.append(info)
            
        except Exception as e:
            logger.warning(f"Could not extract info from {file_path.name}: {e}")
            
            # Add basic info even if metadata fails
            info = {
                'filename': file_path.name,
                'model_name': file_path.stem,
                'eigenvalue_shape': 'unknown',
                'time_steps': 'unknown',
                'time_range': 'unknown',
                'metadata': {}
            }
            model_info.append(info)
    
    return model_info


def compute_isw_distances(eigenvalue_files: List[Path], 
                         isw_params: Dict) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise ISW distances between eigenvalue evolution files.
    
    Args:
        eigenvalue_files: List of eigenvalue evolution files
        isw_params: ISW distance computation parameters
        
    Returns:
        Tuple of (distance_matrix, model_names)
    """
    logger.info("Computing pairwise ISW distances...")
    logger.info(f"ISW parameters: {isw_params}")
    
    # Extract K parameter (it goes to pairwise_isw_matrix, not constructor)
    K = isw_params.pop('K', 200)
    
    # Create ISW instance with specified parameters
    isw = EigenvalueISW(**isw_params)
    
    # Convert Path objects to strings for the ISW function
    file_paths = [str(f) for f in eigenvalue_files]
    
    # Compute pairwise distance matrix
    distance_matrix, model_names = isw.pairwise_isw_matrix(file_paths, K=K)
    
    logger.info(f"Computed {len(model_names)}x{len(model_names)} distance matrix")
    
    return distance_matrix, model_names


def save_results(distance_matrix: np.ndarray, 
                model_names: List[str],
                model_info: List[Dict],
                timestamp: str) -> Dict[str, str]:
    """Save analysis results to files.
    
    Args:
        distance_matrix: Pairwise ISW distance matrix
        model_names: List of model names
        model_info: List of model information dictionaries
        timestamp: Timestamp string for file naming
        
    Returns:
        Dictionary with output file paths
    """
    output_files = {}
    
    # 1. Save distance matrix as CSV
    csv_file = f"isw_distance_matrix_{timestamp}.csv"
    df_distances = pd.DataFrame(distance_matrix, 
                               index=model_names, 
                               columns=model_names)
    df_distances.to_csv(csv_file)
    output_files['distance_csv'] = csv_file
    logger.info(f"Saved distance matrix to {csv_file}")
    
    # 2. Save model information as CSV
    info_file = f"model_info_{timestamp}.csv"
    df_info = pd.DataFrame(model_info)
    df_info.to_csv(info_file, index=False)
    output_files['info_csv'] = info_file
    logger.info(f"Saved model information to {info_file}")
    
    # 3. Save raw numpy array for further analysis
    npy_file = f"isw_distance_matrix_{timestamp}.npy"
    np.save(npy_file, distance_matrix)
    output_files['distance_npy'] = npy_file
    
    return output_files


def create_visualizations(distance_matrix: np.ndarray,
                         model_names: List[str],
                         timestamp: str) -> Dict[str, str]:
    """Create visualizations of the distance matrix.
    
    Args:
        distance_matrix: Pairwise ISW distance matrix
        model_names: List of model names
        timestamp: Timestamp string for file naming
        
    Returns:
        Dictionary with visualization file paths
    """
    viz_files = {}
    
    # Set up matplotlib for better plots
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # 1. Heatmap visualization
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)  # Mask upper triangle
    heatmap = sns.heatmap(distance_matrix, 
                         annot=True, 
                         fmt='.4f',
                         xticklabels=model_names,
                         yticklabels=model_names,
                         cmap='viridis_r',  # Reverse colormap (dark = small distance)
                         mask=mask,
                         square=True,
                         cbar_kws={'label': 'ISW Distance'})
    
    plt.title('Pairwise ISW Distances Between Eigenvalue Evolutions\n' +
              'p=1, α=3.8, log-scale, 95%-norm', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file = f"isw_heatmap_{timestamp}.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    viz_files['heatmap'] = heatmap_file
    logger.info(f"Saved heatmap to {heatmap_file}")
    
    # Also save as PDF for publications
    heatmap_pdf = f"isw_heatmap_{timestamp}.pdf"
    plt.savefig(heatmap_pdf, bbox_inches='tight')
    viz_files['heatmap_pdf'] = heatmap_pdf
    
    plt.close()
    
    # 2. Distance distribution histogram
    plt.figure(figsize=(10, 6))
    
    # Extract upper triangle distances (exclude diagonal)
    upper_tri = np.triu(distance_matrix, k=1)
    distances = upper_tri[upper_tri > 0]
    
    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('ISW Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pairwise ISW Distances')
    plt.grid(True, alpha=0.3)
    
    # 3. Sorted distance plot
    plt.subplot(1, 2, 2)
    sorted_distances = np.sort(distances)
    plt.plot(range(len(sorted_distances)), sorted_distances, 'o-', color='coral')
    plt.xlabel('Pair Index (sorted)')
    plt.ylabel('ISW Distance')
    plt.title('Sorted Pairwise Distances')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save distribution plots
    dist_file = f"isw_distribution_{timestamp}.png"
    plt.savefig(dist_file, dpi=300, bbox_inches='tight')
    viz_files['distribution'] = dist_file
    logger.info(f"Saved distribution plots to {dist_file}")
    
    plt.close()
    
    return viz_files


def analyze_results(distance_matrix: np.ndarray, 
                   model_names: List[str],
                   model_info: List[Dict]) -> Dict:
    """Analyze the computed distance matrix and extract insights.
    
    Args:
        distance_matrix: Pairwise ISW distance matrix
        model_names: List of model names
        model_info: List of model information dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    # Extract upper triangle distances (exclude diagonal)
    upper_tri = np.triu(distance_matrix, k=1)
    distances = upper_tri[upper_tri > 0]
    
    # Basic statistics
    stats = {
        'n_models': len(model_names),
        'n_comparisons': len(distances),
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'median_distance': float(np.median(distances))
    }
    
    # Find most similar and most different pairs
    # Create masked version for finding minimum (exclude diagonal)
    masked_matrix = distance_matrix + np.eye(len(model_names)) * np.inf
    min_idx = np.unravel_index(np.argmin(masked_matrix), masked_matrix.shape)
    max_idx = np.unravel_index(np.argmax(upper_tri), upper_tri.shape)
    
    most_similar = {
        'models': (model_names[min_idx[0]], model_names[min_idx[1]]),
        'distance': float(distance_matrix[min_idx])
    }
    
    most_different = {
        'models': (model_names[max_idx[0]], model_names[max_idx[1]]),
        'distance': float(distance_matrix[max_idx])
    }
    
    # Group analysis (trained vs random models)
    trained_models = [name for name in model_names if not name.startswith('r')]
    random_models = [name for name in model_names if name.startswith('r')]
    
    groups = {
        'trained_models': trained_models,
        'random_models': random_models,
        'n_trained': len(trained_models),
        'n_random': len(random_models)
    }
    
    # Cross-group vs within-group distances
    if trained_models and random_models:
        trained_indices = [i for i, name in enumerate(model_names) if name in trained_models]
        random_indices = [i for i, name in enumerate(model_names) if name in random_models]
        
        # Within-group distances
        if len(trained_indices) > 1:
            within_trained = []
            for i in range(len(trained_indices)):
                for j in range(i+1, len(trained_indices)):
                    within_trained.append(distance_matrix[trained_indices[i], trained_indices[j]])
            groups['within_trained_distances'] = within_trained
            groups['mean_within_trained'] = float(np.mean(within_trained)) if within_trained else None
        
        if len(random_indices) > 1:
            within_random = []
            for i in range(len(random_indices)):
                for j in range(i+1, len(random_indices)):
                    within_random.append(distance_matrix[random_indices[i], random_indices[j]])
            groups['within_random_distances'] = within_random
            groups['mean_within_random'] = float(np.mean(within_random)) if within_random else None
        
        # Cross-group distances
        cross_group = []
        for t_idx in trained_indices:
            for r_idx in random_indices:
                cross_group.append(distance_matrix[t_idx, r_idx])
        groups['cross_group_distances'] = cross_group
        groups['mean_cross_group'] = float(np.mean(cross_group)) if cross_group else None
    
    return {
        'statistics': stats,
        'most_similar': most_similar,
        'most_different': most_different,
        'groups': groups
    }


def print_summary(analysis: Dict, model_info: List[Dict]):
    """Print a comprehensive summary of the analysis results.
    
    Args:
        analysis: Analysis results dictionary
        model_info: List of model information dictionaries
    """
    print("\n" + "="*80)
    print("EIGENVALUE EVOLUTION ISW DISTANCE ANALYSIS SUMMARY")
    print("="*80)
    
    stats = analysis['statistics']
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Models analyzed: {stats['n_models']}")
    print(f"   Pairwise comparisons: {stats['n_comparisons']}")
    print(f"   Mean distance: {stats['mean_distance']:.6f}")
    print(f"   Std deviation: {stats['std_distance']:.6f}")
    print(f"   Distance range: [{stats['min_distance']:.6f}, {stats['max_distance']:.6f}]")
    print(f"   Median distance: {stats['median_distance']:.6f}")
    
    print(f"\n📁 MODEL INFORMATION:")
    for info in model_info:
        print(f"   {info['model_name']}: {info['eigenvalue_shape']} eigenvalues, "
              f"{info['time_steps']} time steps")
    
    print(f"\n🔍 MOST SIMILAR MODELS:")
    sim = analysis['most_similar']
    print(f"   {sim['models'][0]} ↔ {sim['models'][1]}")
    print(f"   Distance: {sim['distance']:.6f}")
    
    print(f"\n🔍 MOST DIFFERENT MODELS:")
    diff = analysis['most_different']
    print(f"   {diff['models'][0]} ↔ {diff['models'][1]}")
    print(f"   Distance: {diff['distance']:.6f}")
    
    groups = analysis['groups']
    if groups['n_trained'] > 0 and groups['n_random'] > 0:
        print(f"\n🎯 GROUP ANALYSIS:")
        print(f"   Trained models: {groups['trained_models']}")
        print(f"   Random models: {groups['random_models']}")
        
        if 'mean_within_trained' in groups and groups['mean_within_trained'] is not None:
            print(f"   Mean distance within trained: {groups['mean_within_trained']:.6f}")
        if 'mean_within_random' in groups and groups['mean_within_random'] is not None:
            print(f"   Mean distance within random: {groups['mean_within_random']:.6f}")
        if 'mean_cross_group' in groups and groups['mean_cross_group'] is not None:
            print(f"   Mean distance trained ↔ random: {groups['mean_cross_group']:.6f}")
    
    print("\n" + "="*80)


def main():
    """Main analysis function."""
    # Analysis timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🧬 EIGENVALUE EVOLUTION ISW DISTANCE ANALYSIS")
    print("=" * 50)
    
    # ISW parameters as specified
    isw_params = {
        'p': 1,
        'n_quantiles': 199,
        'tail_trim': 0.02,
        'time_weight_alpha': 3.8,
        'eps': 1e-3,
        'log_scale': True,
        'normalize': 'p95',
        'K': 200
    }
    
    try:
        # 1. Find eigenvalue files
        eigenvalue_files = find_eigenvalue_files()
        
        # 2. Extract model information
        model_info = extract_model_info(eigenvalue_files)
        
        # 3. Compute ISW distances
        distance_matrix, model_names = compute_isw_distances(eigenvalue_files, isw_params)
        
        # 4. Save results
        output_files = save_results(distance_matrix, model_names, model_info, timestamp)
        
        # 5. Create visualizations
        viz_files = create_visualizations(distance_matrix, model_names, timestamp)
        
        # 6. Analyze results
        analysis = analyze_results(distance_matrix, model_names, model_info)
        
        # 7. Print summary
        print_summary(analysis, model_info)
        
        # 8. Report output files
        print(f"\n📁 OUTPUT FILES:")
        all_files = {**output_files, **viz_files}
        for file_type, filepath in all_files.items():
            print(f"   {file_type}: {filepath}")
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()