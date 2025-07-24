#!/usr/bin/env python3
"""Comprehensive examples of Gromov-Wasserstein API usage.

This module demonstrates how to use the Gromov-Wasserstein sheaf construction
method through the high-level NeurosheafAnalyzer API, including:

1. Basic GW sheaf construction
2. Custom GW configuration
3. Network comparison using GW method
4. Mixed method comparisons
5. Performance comparison between methods

Examples work with standard PyTorch models and demonstrate the key advantages
of GW-based sheaf construction for cross-architecture analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

# Import the main API
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.sheaf.core.gw_config import GWConfig


def create_simple_models():
    """Create simple test models with different architectures."""
    
    class SimpleNet1(nn.Module):
        """Small network with different layer sizes."""
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(20, 15)
            self.layer2 = nn.Linear(15, 10)
            self.layer3 = nn.Linear(10, 5)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)
    
    class SimpleNet2(nn.Module):
        """Different architecture with same input/output."""
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(20, 12)
            self.layer2 = nn.Linear(12, 8)
            self.layer3 = nn.Linear(8, 5)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)
    
    class WideNet(nn.Module):
        """Wider network architecture."""
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(20, 25)
            self.layer2 = nn.Linear(25, 15)
            self.layer3 = nn.Linear(15, 5)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)
    
    return SimpleNet1(), SimpleNet2(), WideNet()


def example_1_basic_gw_analysis():
    """Example 1: Basic GW sheaf construction."""
    print("=" * 60)
    print("Example 1: Basic GW Sheaf Construction")
    print("=" * 60)
    
    # Create analyzer
    analyzer = NeurosheafAnalyzer()
    
    # Create simple model and data
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
        nn.ReLU(),
        nn.Linear(6, 4)
    )
    data = torch.randn(50, 10)
    
    print("Standard Procrustes Analysis:")
    # Standard analysis (default)
    standard_results = analyzer.analyze(model, data, method='procrustes')
    print(f"  Method: {standard_results['construction_method']}")
    print(f"  Nodes: {len(standard_results['sheaf'].stalks)}")
    print(f"  Edges: {len(standard_results['sheaf'].restrictions)}")
    print(f"  Construction time: {standard_results['construction_time']:.3f}s")
    
    print("\\nGromov-Wasserstein Analysis:")
    # GW analysis with default configuration
    gw_results = analyzer.analyze(model, data, method='gromov_wasserstein')
    print(f"  Method: {gw_results['construction_method']}")
    print(f"  Nodes: {len(gw_results['sheaf'].stalks)}")
    print(f"  Edges: {len(gw_results['sheaf'].restrictions)}")
    print(f"  Construction time: {gw_results['construction_time']:.3f}s")
    print(f"  GW config epsilon: {gw_results['gw_config']['epsilon']}")
    
    # Access sheaf properties
    gw_sheaf = gw_results['sheaf']
    print(f"  GW sheaf type: {gw_sheaf.is_gw_sheaf()}")
    print(f"  Filtration semantics: {gw_sheaf.get_filtration_semantics()}")
    
    if 'gw_costs' in gw_sheaf.metadata:
        gw_costs = gw_sheaf.metadata['gw_costs']
        print(f"  GW costs: {list(gw_costs.values())}")
    
    print()


def example_2_custom_gw_configuration():
    """Example 2: Custom GW configuration."""
    print("=" * 60)
    print("Example 2: Custom GW Configuration")
    print("=" * 60)
    
    analyzer = NeurosheafAnalyzer()
    model = nn.Sequential(
        nn.Linear(15, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(), 
        nn.Linear(8, 3)
    )
    data = torch.randn(40, 15)
    
    # Fast configuration (good for prototyping)
    print("Fast GW Configuration:")
    fast_config = GWConfig.default_fast()
    fast_results = analyzer.analyze(model, data, 
                                   method='gromov_wasserstein',
                                   gw_config=fast_config)
    print(f"  Epsilon: {fast_config.epsilon}")
    print(f"  Max iterations: {fast_config.max_iter}")
    print(f"  Construction time: {fast_results['construction_time']:.3f}s")
    
    # Accurate configuration (good for final analysis)
    print("\\nAccurate GW Configuration:")
    accurate_config = GWConfig.default_accurate()
    accurate_results = analyzer.analyze(model, data,
                                       method='gromov_wasserstein', 
                                       gw_config=accurate_config)
    print(f"  Epsilon: {accurate_config.epsilon}")
    print(f"  Max iterations: {accurate_config.max_iter}")
    print(f"  Construction time: {accurate_results['construction_time']:.3f}s")
    
    # Custom configuration
    print("\\nCustom GW Configuration:")
    custom_config = GWConfig(
        epsilon=0.05,           # Medium regularization
        max_iter=800,           # Moderate iterations  
        tolerance=1e-8,         # Tight convergence
        use_gpu=False,          # Use CPU for reproducibility
        validate_couplings=True  # Full validation
    )
    custom_results = analyzer.analyze(model, data,
                                     method='gromov_wasserstein',
                                     gw_config=custom_config)
    print(f"  Epsilon: {custom_config.epsilon}")
    print(f"  Max iterations: {custom_config.max_iter}")
    print(f"  Construction time: {custom_results['construction_time']:.3f}s")
    
    print()


def example_3_network_comparison():
    """Example 3: Network comparison using GW method."""
    print("=" * 60)
    print("Example 3: Network Comparison with GW")
    print("=" * 60)
    
    analyzer = NeurosheafAnalyzer()
    model1, model2, model3 = create_simple_models()
    data = torch.randn(100, 20)
    
    print("Comparing networks with different methods:")
    
    # Compare with standard Procrustes method
    print("\\n1. Standard Procrustes Comparison:")
    try:
        procrustes_comparison = analyzer.compare_networks(
            model1, model2, data,
            comparison_method='dtw',
            sheaf_method='procrustes'
        )
        print(f"   Similarity: {procrustes_comparison['similarity_score']:.4f}")
        print(f"   Method: {procrustes_comparison['sheaf_method']}")
    except Exception as e:
        print(f"   Note: DTW comparison may not work with standard sheaves: {e}")
        
        # Fallback to simple comparison
        simple_comparison = analyzer.compare_networks(
            model1, model2, data,
            comparison_method='euclidean',
            sheaf_method='procrustes'
        )
        print(f"   Euclidean similarity: {simple_comparison['similarity_score']:.4f}")
    
    # Compare with GW method
    print("\\n2. Gromov-Wasserstein Comparison:")
    gw_comparison = analyzer.compare_networks(
        model1, model2, data,
        comparison_method='dtw', 
        sheaf_method='gromov_wasserstein'
    )
    print(f"   Similarity: {gw_comparison['similarity_score']:.4f}")
    print(f"   Method: {gw_comparison['sheaf_method']}")
    print(f"   GW epsilon: {gw_comparison['gw_config']['epsilon']}")
    
    # Multiple network comparison
    print("\\n3. Multiple Network Comparison:")
    models = [model1, model2, model3]
    multi_comparison = analyzer.compare_multiple_networks(
        models, data,
        comparison_method='euclidean',  # Use simple method for reliability
        sheaf_method='gromov_wasserstein'
    )
    
    print(f"   Number of models: {multi_comparison['comparison_metadata']['n_models']}")
    print(f"   Distance matrix shape: {multi_comparison['distance_matrix'].shape}")
    print(f"   Sheaf method: {multi_comparison['sheaf_method']}")
    
    # Print similarity rankings
    for i, ranking in enumerate(multi_comparison['similarity_rankings']):
        print(f"   Model {i} most similar to: Model {ranking['most_similar'][0]['model_index']} "
              f"(distance: {ranking['most_similar'][0]['distance']:.4f})")
    
    print()


def example_4_mixed_method_comparison():
    """Example 4: Compare different sheaf construction methods."""
    print("=" * 60)
    print("Example 4: Mixed Method Comparison")
    print("=" * 60)
    
    analyzer = NeurosheafAnalyzer()
    model = nn.Sequential(
        nn.Linear(12, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 6)
    )
    data = torch.randn(80, 12)
    
    methods = ['procrustes', 'gromov_wasserstein', 'whitened_procrustes']
    results = {}
    
    print("Analyzing same model with different methods:")
    
    for method in methods:
        print(f"\\n{method.upper()} Method:")
        try:
            if method == 'gromov_wasserstein':
                # Use fast config for demonstration
                gw_config = GWConfig.default_fast()
                result = analyzer.analyze(model, data, 
                                        method=method,
                                        gw_config=gw_config)
            else:
                result = analyzer.analyze(model, data, method=method)
            
            results[method] = result
            print(f"   Construction time: {result['construction_time']:.3f}s")
            print(f"   Nodes: {len(result['sheaf'].stalks)}")
            print(f"   Edges: {len(result['sheaf'].restrictions)}")
            
            if method == 'gromov_wasserstein':
                print(f"   GW epsilon: {result['gw_config']['epsilon']}")
                sheaf = result['sheaf']
                if sheaf.is_gw_sheaf():
                    print(f"   Filtration semantics: {sheaf.get_filtration_semantics()}")
                    
        except Exception as e:
            print(f"   Error with {method}: {e}")
    
    # Compare sheaf properties
    if len(results) > 1:
        print("\\nSheaf Property Comparison:")
        for method, result in results.items():
            sheaf = result['sheaf']
            print(f"   {method}:")
            print(f"     Stalk dimensions: {[stalk.shape for stalk in sheaf.stalks.values()]}")
            print(f"     Restriction shapes: {[r.shape for r in sheaf.restrictions.values()]}")
    
    print()


def example_5_performance_comparison():
    """Example 5: Performance comparison between methods."""
    print("=" * 60)
    print("Example 5: Performance Comparison")
    print("=" * 60)
    
    analyzer = NeurosheafAnalyzer()
    
    # Create models of different sizes
    small_model = nn.Sequential(
        nn.Linear(8, 6),
        nn.ReLU(),
        nn.Linear(6, 4)
    )
    
    medium_model = nn.Sequential(
        nn.Linear(15, 12),
        nn.ReLU(), 
        nn.Linear(12, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 5)
    )
    
    models = {'small': small_model, 'medium': medium_model}
    data_sizes = {'small': torch.randn(60, 8), 'medium': torch.randn(100, 15)}
    
    print("Performance comparison across model sizes:")
    
    for size_name, model in models.items():
        data = data_sizes[size_name]
        print(f"\\n{size_name.upper()} Model:")
        
        # Standard method
        try:
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            standard_result = analyzer.analyze(model, data, method='procrustes')
            print(f"   Procrustes time: {standard_result['construction_time']:.3f}s")
        except Exception as e:
            print(f"   Procrustes failed: {e}")
        
        # GW method with fast config
        try:
            gw_config = GWConfig.default_fast()
            gw_result = analyzer.analyze(model, data, 
                                       method='gromov_wasserstein',
                                       gw_config=gw_config)
            print(f"   GW (fast) time: {gw_result['construction_time']:.3f}s")
            
            # Speed ratio
            if 'standard_result' in locals():
                ratio = gw_result['construction_time'] / standard_result['construction_time']
                print(f"   GW/Procrustes ratio: {ratio:.2f}x")
                
        except Exception as e:
            print(f"   GW failed: {e}")
    
    print()


def example_6_advanced_gw_features():
    """Example 6: Advanced GW features and configuration."""
    print("=" * 60)
    print("Example 6: Advanced GW Features")
    print("=" * 60)
    
    analyzer = NeurosheafAnalyzer()
    model = nn.Sequential(
        nn.Linear(16, 14),
        nn.ReLU(),
        nn.Linear(14, 12),
        nn.ReLU(),
        nn.Linear(12, 8)
    )
    data = torch.randn(70, 16)
    
    print("Advanced GW configuration options:")
    
    # Configuration with different regularization levels
    epsilon_values = [0.01, 0.05, 0.1, 0.2]
    
    for epsilon in epsilon_values:
        print(f"\\nEpsilon = {epsilon}:")
        config = GWConfig(
            epsilon=epsilon,
            max_iter=500,
            tolerance=1e-8,
            validate_couplings=True
        )
        
        try:
            result = analyzer.analyze(model, data,
                                    method='gromov_wasserstein',
                                    gw_config=config)
            print(f"   Construction time: {result['construction_time']:.3f}s")
            
            # Extract GW costs if available
            sheaf = result['sheaf']
            if 'gw_costs' in sheaf.metadata:
                costs = list(sheaf.metadata['gw_costs'].values())
                print(f"   GW costs: min={min(costs):.4f}, max={max(costs):.4f}, mean={np.mean(costs):.4f}")
                
        except Exception as e:
            print(f"   Failed: {e}")
    
    # Debugging configuration
    print("\\nDebugging Configuration:")
    debug_config = GWConfig.default_debugging()
    try:
        debug_result = analyzer.analyze(model, data,
                                      method='gromov_wasserstein',
                                      gw_config=debug_config)
        print(f"   Debug time: {debug_result['construction_time']:.3f}s")
        print(f"   Validation enabled: {debug_config.validate_couplings}")
        print(f"   GPU disabled: {not debug_config.use_gpu}")
    except Exception as e:
        print(f"   Debug failed: {e}")
    
    print()


def main():
    """Run all examples."""
    print("Gromov-Wasserstein API Usage Examples")
    print("=====================================")
    print("Demonstrating GW sheaf construction through NeurosheafAnalyzer API\\n")
    
    try:
        example_1_basic_gw_analysis()
        example_2_custom_gw_configuration() 
        example_3_network_comparison()
        example_4_mixed_method_comparison()
        example_5_performance_comparison()
        example_6_advanced_gw_features()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()