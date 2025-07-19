#!/usr/bin/env python3
"""
DTW Fix Validation - Test the complete pipeline with filtration fix

This script validates that the filtration threshold fix resolves the DTW homogenization issue
by testing the complete neural network comparison pipeline.
"""

import torch
import numpy as np
import os
from pathlib import Path
import sys

# Set environment
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add to path
sys.path.append(str(Path(__file__).parent))

# Import working model classes
from scripts.multivariate_dtw_similarity_analysis import MLPModel, ActualCustomModel
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model


def test_dtw_fix_two_models():
    """Test DTW with two different models to see if they now produce meaningful differences."""
    print("🔧 Testing DTW Fix with Two Neural Network Models")
    print("=" * 60)
    
    # Test data
    data = torch.randn(100, 3)
    
    # Load two different MLP models (if available)
    model_paths = [
        "models/torch_mlp_acc_0.9857_epoch_100.pth",
        "models/torch_mlp_acc_1.0000_epoch_200.pth"
    ]
    
    models = {}
    
    # Load available models
    for path in model_paths:
        if Path(path).exists():
            try:
                model = load_model(MLPModel, path, device="cpu")
                models[Path(path).stem] = model
                print(f"✅ Loaded {Path(path).stem}")
            except Exception as e:
                print(f"❌ Failed to load {path}: {e}")
    
    if len(models) < 2:
        print("❌ Need at least 2 models for comparison")
        return False
    
    # Create analyzer
    analyzer = NeurosheafAnalyzer(device='cpu')
    
    # Compare the models using DTW
    model_names = list(models.keys())
    model1_name, model2_name = model_names[0], model_names[1]
    model1, model2 = models[model1_name], models[model2_name]
    
    print(f"\n🔬 Comparing {model1_name} vs {model2_name}")
    
    try:
        # Perform DTW comparison
        result = analyzer.compare_networks(
            model1, model2, data,
            method='dtw',
            eigenvalue_index=0,  # Single eigenvalue first
            multivariate=False,
            filtration_type='threshold',
            n_steps=20  # Fewer steps for faster testing
        )
        
        print(f"\n📊 DTW Comparison Results:")
        print(f"   Similarity Score: {result['similarity_score']:.6f}")
        print(f"   DTW Distance: {result['dtw_comparison'].get('distance', 'N/A')}")
        print(f"   Normalized Distance: {result['dtw_comparison'].get('normalized_distance', 'N/A')}")
        print(f"   Method: {result['method']}")
        
        # Test multivariate DTW
        print(f"\n🔬 Testing Multivariate DTW:")
        result_mv = analyzer.compare_networks(
            model1, model2, data,
            method='dtw',
            multivariate=True,
            filtration_type='threshold',
            n_steps=15
        )
        
        print(f"   Multivariate Similarity: {result_mv['similarity_score']:.6f}")
        print(f"   Multivariate Distance: {result_mv['dtw_comparison'].get('normalized_distance', 'N/A')}")
        
        # Validate results
        print(f"\n✅ Validation Results:")
        
        # Check if we get meaningful (non-zero, non-infinite) distances
        # Updated to check for meaningful similarity scores instead of distances
        univariate_ok = (
            0.001 < result['similarity_score'] < 1.0 and
            not np.isnan(result['similarity_score']) and
            not np.isinf(result['similarity_score'])
        )
        
        multivariate_ok = (
            0.001 < result_mv['similarity_score'] < 1.0 and
            not np.isnan(result_mv['similarity_score']) and
            not np.isinf(result_mv['similarity_score'])
        )
        
        print(f"   Univariate DTW: {'✅ FIXED' if univariate_ok else '❌ Still broken'}")
        print(f"   Multivariate DTW: {'✅ FIXED' if multivariate_ok else '❌ Still broken'}")
        
        # Check that models are distinguishable (similarity not too close to 1.0)
        distinguishable = result['similarity_score'] < 0.99
        print(f"   Models Distinguishable: {'✅ YES' if distinguishable else '❌ Still homogenized'}")
        
        return univariate_ok and multivariate_ok and distinguishable
        
    except Exception as e:
        print(f"❌ DTW comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eigenvalue_evolution_diversity():
    """Test that eigenvalue evolution now shows proper diversity."""
    print(f"\n🔍 Testing Eigenvalue Evolution Diversity")
    print("=" * 50)
    
    data = torch.randn(80, 3)
    
    # Load one model for evolution testing
    try:
        model = load_model(MLPModel, "models/torch_mlp_acc_0.9857_epoch_100.pth", device="cpu")
        analyzer = NeurosheafAnalyzer(device='cpu')
        
        # Analyze with detailed spectral tracking
        analysis = analyzer.analyze(model, data)
        sheaf = analysis['sheaf']
        
        # Run spectral analysis
        from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
        spectral_analyzer = PersistentSpectralAnalyzer()
        
        result = spectral_analyzer.analyze(
            sheaf,
            filtration_type='threshold',
            n_steps=15
        )
        
        eigenvalue_sequences = result['persistence_result']['eigenvalue_sequences']
        filtration_params = result['filtration_params']
        
        print(f"📈 Eigenvalue Evolution Analysis:")
        print(f"{'Step':<4} {'Param':<8} {'NonZero':<8} {'Max':<10} {'Diversity':<10}")
        print("-" * 50)
        
        diversities = []
        non_zero_counts = []
        
        for i, (param, eigenvals) in enumerate(zip(filtration_params, eigenvalue_sequences)):
            non_zero = torch.sum(eigenvals > 1e-6).item()
            max_eig = torch.max(eigenvals).item() if len(eigenvals) > 0 else 0.0
            diversity = torch.std(eigenvals).item() if len(eigenvals) > 1 else 0.0
            
            diversities.append(diversity)
            non_zero_counts.append(non_zero)
            
            print(f"{i:<4} {param:<8.3f} {non_zero:<8} {max_eig:<10.4f} {diversity:<10.4f}")
        
        # Analyze evolution quality
        print(f"\n📊 Evolution Quality Analysis:")
        
        # Check if diversity increases over time (proper behavior)
        diversity_trend = np.polyfit(range(len(diversities)), diversities, 1)[0]
        print(f"   Diversity Trend: {diversity_trend:.6f} (positive = good)")
        
        # Check if non-zero eigenvalues increase over time
        nonzero_trend = np.polyfit(range(len(non_zero_counts)), non_zero_counts, 1)[0]
        print(f"   Non-zero Trend: {nonzero_trend:.6f} (positive = good)")
        
        # Check final diversity
        final_diversity = diversities[-1]
        print(f"   Final Diversity: {final_diversity:.6f} (>1.0 = good)")
        
        # Overall assessment
        evolution_fixed = (
            diversity_trend > 0 and
            nonzero_trend > 0 and
            final_diversity > 1.0
        )
        
        print(f"   Evolution Quality: {'✅ FIXED' if evolution_fixed else '❌ Still problematic'}")
        
        return evolution_fixed
        
    except Exception as e:
        print(f"❌ Evolution testing failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🔧 DTW Fix Validation")
    print("=" * 40)
    
    # Test 1: DTW comparison between models
    dtw_fixed = test_dtw_fix_two_models()
    
    # Test 2: Eigenvalue evolution diversity
    evolution_fixed = test_eigenvalue_evolution_diversity()
    
    # Summary
    print(f"\n📋 Fix Validation Summary:")
    print(f"   DTW Comparison: {'✅ FIXED' if dtw_fixed else '❌ Still broken'}")
    print(f"   Evolution Diversity: {'✅ FIXED' if evolution_fixed else '❌ Still broken'}")
    
    if dtw_fixed and evolution_fixed:
        print(f"\n🎉 SUCCESS: Filtration threshold fix resolved the homogenization issue!")
        print(f"   Neural networks now produce meaningful DTW distances")
        print(f"   Eigenvalue evolution shows proper persistence behavior")
    else:
        print(f"\n⚠️  PARTIAL: Some issues remain, may need additional fixes")
    
    return dtw_fixed and evolution_fixed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)