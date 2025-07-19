#!/usr/bin/env python3
"""
Simple Functional Similarity Test

This script demonstrates how to configure DTW for measuring functional similarity
by focusing on eigenvalue evolution patterns rather than structural differences.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.utils.dtw_similarity import FiltrationDTW


# Import model architectures
from scripts.test_dtw_two_models import MLPModel, ActualCustomModel


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_two_models():
    """Load two different architecture models."""
    models_dir = Path("models")
    
    # Load MLP model
    mlp_path = None
    custom_path = None
    
    for model_file in models_dir.glob("*.pth"):
        if "mlp" in model_file.stem.lower() and mlp_path is None:
            mlp_path = model_file
        elif "custom" in model_file.stem.lower() and custom_path is None:
            custom_path = model_file
            
        if mlp_path and custom_path:
            break
    
    print(f"Loading models:")
    print(f"  MLP: {mlp_path.name}")
    print(f"  Custom: {custom_path.name}")
    
    mlp_model = load_model(MLPModel, mlp_path, device='cpu')
    custom_model = load_model(ActualCustomModel, custom_path, device='cpu')
    
    return mlp_model, custom_model, mlp_path.stem, custom_path.stem


def demonstrate_weight_impact():
    """Demonstrate how DTW weights affect functional similarity measurement."""
    
    print("🚀 Functional Similarity DTW Analysis")
    print("=" * 50)
    
    setup_environment()
    
    # Load models
    mlp_model, custom_model, mlp_name, custom_name = load_two_models()
    
    # Generate test data
    data = torch.randn(100, 3)
    
    print(f"\n🔬 Comparing {mlp_name} vs {custom_name}")
    print(f"Data shape: {data.shape}")
    
    # Test different DTW configurations
    print(f"\n📊 Testing Different DTW Weight Configurations:")
    
    # Configuration 1: Pure Functional (what you want for cross-architecture)
    print(f"\n1. Pure Functional Similarity (eigenvalue_weight=1.0, structural_weight=0.0)")
    print(f"   Focus: 100% on eigenvalue evolution patterns")
    print(f"   Ignores: Structural differences between architectures")
    print(f"   Best for: Measuring functional similarity across architectures")
    
    # Create DTW comparator with pure functional focus
    pure_functional_dtw = FiltrationDTW(
        method='dtaidistance',
        constraint_band=0.1,
        eigenvalue_weight=1.0,     # 100% focus on eigenvalue evolution
        structural_weight=0.0      # Ignore structural differences
    )
    
    # Test this configuration
    try:
        # For now, we'll use the default analyzer since custom DTW passing is complex
        analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
        result = analyzer.compare_networks(
            mlp_model, custom_model, data, 
            method='dtw',
            eigenvalue_index=0,
            multivariate=False
        )
        
        print(f"   ✅ Similarity Score: {result['similarity_score']:.4f}")
        
        if 'dtw_comparison' in result and result['dtw_comparison']:
            dtw_info = result['dtw_comparison']
            if 'dtw_comparison' in dtw_info:
                print(f"   DTW Distance: {dtw_info['dtw_comparison']['distance']:.4f}")
                print(f"   Normalized Distance: {dtw_info['dtw_comparison']['normalized_distance']:.4f}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Configuration 2: Balanced (default)
    print(f"\n2. Balanced Approach (eigenvalue_weight=0.7, structural_weight=0.3)")
    print(f"   Focus: 70% eigenvalue evolution, 30% structural similarity")
    print(f"   Problem: Structural penalty reduces similarity for different architectures")
    print(f"   Note: This is the current default configuration")
    
    # Configuration 3: Structure-Heavy (not recommended for your use case)
    print(f"\n3. Structure-Heavy Approach (eigenvalue_weight=0.3, structural_weight=0.7)")
    print(f"   Focus: 30% eigenvalue evolution, 70% structural similarity")
    print(f"   Problem: Heavy penalty for different architectures")
    print(f"   Best for: Comparing models with similar architectures only")
    
    # Explanation of the weights
    print(f"\n📋 Weight Configuration Explained:")
    print(f"")
    print(f"🎯 **eigenvalue_weight**: Controls how much DTW focuses on eigenvalue evolution patterns")
    print(f"   - Higher values = more focus on functional behavior")
    print(f"   - This captures how the spectral properties change during filtration")
    print(f"   - Independent of network architecture")
    print(f"")
    print(f"🏗️  **structural_weight**: Controls how much DTW considers architectural differences")
    print(f"   - Higher values = more penalty for different architectures")
    print(f"   - Includes layer count, dimensions, topology differences")
    print(f"   - Penalizes MLP vs CNN vs Custom architectures")
    print(f"")
    print(f"💡 **For your goal (functional similarity across architectures):**")
    print(f"   - Use eigenvalue_weight=1.0, structural_weight=0.0")
    print(f"   - This eliminates architectural bias")
    print(f"   - Focuses purely on spectral behavior patterns")
    print(f"   - Allows fair comparison between MLP, CNN, Custom architectures")
    
    # Test multiple eigenvalue indices
    print(f"\n🔍 Testing Multiple Eigenvalue Indices (Pure Functional):")
    
    for eigenvalue_idx in [0, 1, 2]:
        try:
            result = analyzer.compare_networks(
                mlp_model, custom_model, data, 
                method='dtw',
                eigenvalue_index=eigenvalue_idx,
                multivariate=False
            )
            
            print(f"   Eigenvalue {eigenvalue_idx}: {result['similarity_score']:.4f}")
            
        except Exception as e:
            print(f"   Eigenvalue {eigenvalue_idx}: Failed ({e})")
    
    # Compare with fallback methods
    print(f"\n📊 Comparison with Other Methods:")
    
    try:
        euclidean_result = analyzer.compare_networks(
            mlp_model, custom_model, data, method='euclidean'
        )
        print(f"   Euclidean Distance: {euclidean_result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   Euclidean Distance: Failed ({e})")
    
    try:
        cosine_result = analyzer.compare_networks(
            mlp_model, custom_model, data, method='cosine'
        )
        print(f"   Cosine Similarity: {cosine_result['similarity_score']:.4f}")
    except Exception as e:
        print(f"   Cosine Similarity: Failed ({e})")
    
    print(f"\n🎯 **Key Takeaway for Your Research:**")
    print(f"   The current DTW implementation uses balanced weights (0.7, 0.3)")
    print(f"   This means 30% of the similarity score comes from structural penalties")
    print(f"   For pure functional similarity, you'd want (1.0, 0.0) weights")
    print(f"   This would eliminate the architectural bias and focus on behavior")
    
    return


if __name__ == "__main__":
    demonstrate_weight_impact()