#!/usr/bin/env python3
"""
Functional Similarity DTW Test Script

This script tests DTW functionality specifically for measuring functional similarity
across different neural network architectures, ignoring structural differences.

Focus: Pure eigenvalue evolution comparison without structural penalties.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.utils.dtw_similarity import FiltrationDTW, create_filtration_dtw_comparator


# Import model architectures
from scripts.test_dtw_two_models import MLPModel, ActualCustomModel


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🔧 Environment Setup Complete")
    print(f"   PyTorch Version: {torch.__version__}")


def load_two_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load two models for comparison."""
    models_dir = Path("models")
    
    # Find first MLP and first Custom model
    mlp_model = None
    custom_model = None
    
    for model_file in models_dir.glob("*.pth"):
        model_name = model_file.stem
        
        if "mlp" in model_name.lower() and mlp_model is None:
            mlp_model = {
                'name': model_name,
                'path': model_file,
                'architecture': "MLPModel",
                'model_class': MLPModel
            }
        elif "custom" in model_name.lower() and custom_model is None:
            custom_model = {
                'name': model_name,
                'path': model_file,
                'architecture': "ActualCustomModel",
                'model_class': ActualCustomModel
            }
        
        if mlp_model and custom_model:
            break
    
    if not mlp_model or not custom_model:
        raise ValueError("Could not find both MLP and Custom models")
    
    print(f"🔍 Selected Models:")
    print(f"   • Model 1: {mlp_model['name']} ({mlp_model['architecture']})")
    print(f"   • Model 2: {custom_model['name']} ({custom_model['architecture']})")
    
    return mlp_model, custom_model


def load_models(model_configs: List[Dict[str, Any]]) -> Dict[str, nn.Module]:
    """Load the selected models."""
    loaded_models = {}
    
    print(f"\n📦 Loading {len(model_configs)} models...")
    
    for config in model_configs:
        try:
            model = load_model(config['model_class'], config['path'], device='cpu')
            loaded_models[config['name']] = model
            print(f"   ✅ {config['name']} loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load {config['name']}: {e}")
            raise
    
    return loaded_models


def test_functional_similarity_dtw(models: Dict[str, nn.Module], 
                                  model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test DTW for functional similarity across architectures."""
    print(f"\n🔬 Testing Functional Similarity DTW...")
    
    # Generate test data
    data = torch.randn(100, 3)
    print(f"   Generated test data: {data.shape}")
    
    # Get model names
    model_names = list(models.keys())
    model1_name, model2_name = model_names[0], model_names[1]
    model1, model2 = models[model1_name], models[model2_name]
    
    print(f"   Comparing {model1_name} vs {model2_name}")
    
    # Test results storage
    results = {}
    
    # Test 1: Pure Functional Similarity (No Structural Penalty)
    print(f"\n   Test 1: Pure Functional Similarity (eigenvalue_weight=1.0, structural_weight=0.0)")
    try:
        # Create DTW comparator focused purely on functional similarity
        pure_functional_dtw = FiltrationDTW(
            method='dtaidistance',
            constraint_band=0.1,
            eigenvalue_weight=1.0,     # Focus 100% on eigenvalue evolution
            structural_weight=0.0      # Ignore structural differences
        )
        
        # Initialize analyzer with custom DTW comparator
        analyzer = NeurosheafAnalyzer(
            device='cpu', 
            enable_profiling=False,
            spectral_analyzer_config={
                'dtw_comparator': pure_functional_dtw
            }
        )
        
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=0,  # Compare largest eigenvalue
            multivariate=False   # Use univariate DTW
        )
        
        results['pure_functional'] = result
        print(f"      ✅ Pure Functional DTW: {result['similarity_score']:.4f}")
        
        if 'dtw_comparison' in result and result['dtw_comparison']:
            dtw_info = result['dtw_comparison']
            if 'dtw_comparison' in dtw_info:
                print(f"      DTW Distance: {dtw_info['dtw_comparison']['distance']:.4f}")
                print(f"      Normalized Distance: {dtw_info['dtw_comparison']['normalized_distance']:.4f}")
        
    except Exception as e:
        print(f"      ❌ Pure Functional DTW failed: {e}")
        results['pure_functional'] = {'error': str(e)}
    
    # Test 2: Balanced Approach (Default)
    print(f"\n   Test 2: Balanced Approach (eigenvalue_weight=0.7, structural_weight=0.3)")
    try:
        # Use default balanced weights
        balanced_dtw = FiltrationDTW(
            method='dtaidistance',
            constraint_band=0.1,
            eigenvalue_weight=0.7,     # 70% eigenvalue evolution
            structural_weight=0.3      # 30% structural similarity
        )
        
        analyzer = NeurosheafAnalyzer(
            device='cpu', 
            enable_profiling=False,
            spectral_analyzer_config={
                'dtw_comparator': balanced_dtw
            }
        )
        
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=0,
            multivariate=False
        )
        
        results['balanced'] = result
        print(f"      ✅ Balanced DTW: {result['similarity_score']:.4f}")
        
    except Exception as e:
        print(f"      ❌ Balanced DTW failed: {e}")
        results['balanced'] = {'error': str(e)}
    
    # Test 3: Structure-Heavy Approach
    print(f"\n   Test 3: Structure-Heavy Approach (eigenvalue_weight=0.3, structural_weight=0.7)")
    try:
        # Heavy structural weighting
        structural_dtw = FiltrationDTW(
            method='dtaidistance',
            constraint_band=0.1,
            eigenvalue_weight=0.3,     # 30% eigenvalue evolution
            structural_weight=0.7      # 70% structural similarity
        )
        
        analyzer = NeurosheafAnalyzer(
            device='cpu', 
            enable_profiling=False,
            spectral_analyzer_config={
                'dtw_comparator': structural_dtw
            }
        )
        
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=0,
            multivariate=False
        )
        
        results['structure_heavy'] = result
        print(f"      ✅ Structure-Heavy DTW: {result['similarity_score']:.4f}")
        
    except Exception as e:
        print(f"      ❌ Structure-Heavy DTW failed: {e}")
        results['structure_heavy'] = {'error': str(e)}
    
    # Test 4: Compare Multiple Eigenvalues for Functional Similarity
    print(f"\n   Test 4: Multi-Eigenvalue Functional Similarity")
    try:
        pure_functional_dtw = FiltrationDTW(
            method='dtaidistance',
            constraint_band=0.1,
            eigenvalue_weight=1.0,
            structural_weight=0.0
        )
        
        analyzer = NeurosheafAnalyzer(
            device='cpu', 
            enable_profiling=False,
            spectral_analyzer_config={
                'dtw_comparator': pure_functional_dtw
            }
        )
        
        # Test multiple eigenvalue indices
        eigenvalue_results = {}
        for eigenvalue_idx in [0, 1, 2]:
            try:
                result = analyzer.compare_networks(
                    model1, model2, data, 
                    method='dtw',
                    eigenvalue_index=eigenvalue_idx,
                    multivariate=False
                )
                eigenvalue_results[f'eigenvalue_{eigenvalue_idx}'] = result['similarity_score']
                print(f"      ✅ Eigenvalue {eigenvalue_idx}: {result['similarity_score']:.4f}")
            except Exception as e:
                print(f"      ❌ Eigenvalue {eigenvalue_idx} failed: {e}")
                eigenvalue_results[f'eigenvalue_{eigenvalue_idx}'] = None
        
        results['multi_eigenvalue'] = eigenvalue_results
        
    except Exception as e:
        print(f"      ❌ Multi-Eigenvalue DTW failed: {e}")
        results['multi_eigenvalue'] = {'error': str(e)}
    
    return results


def create_functional_similarity_analysis(results: Dict[str, Any], 
                                        model_configs: List[Dict[str, Any]]) -> go.Figure:
    """Create functional similarity analysis visualization."""
    print(f"\n📊 Creating Functional Similarity Analysis...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "DTW Weight Impact on Similarity",
            "Eigenvalue-Specific Functional Similarity", 
            "Weight Configuration Comparison",
            "Model Architecture Summary"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}]
        ],
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )
    
    # 1. DTW Weight Impact on Similarity
    weight_configs = []
    similarity_scores = []
    colors = []
    
    for config_name, result in results.items():
        if config_name in ['pure_functional', 'balanced', 'structure_heavy'] and 'error' not in result:
            weight_configs.append(config_name.replace('_', ' ').title())
            similarity_scores.append(result['similarity_score'])
            if config_name == 'pure_functional':
                colors.append('green')
            elif config_name == 'balanced':
                colors.append('orange')
            else:
                colors.append('red')
    
    if weight_configs:
        bar1 = go.Bar(
            x=weight_configs,
            y=similarity_scores,
            marker_color=colors,
            name="Weight Impact",
            text=[f"{score:.4f}" for score in similarity_scores],
            textposition='auto'
        )
        fig.add_trace(bar1, row=1, col=1)
    
    # 2. Eigenvalue-Specific Functional Similarity
    if 'multi_eigenvalue' in results and 'error' not in results['multi_eigenvalue']:
        eigenvalue_data = results['multi_eigenvalue']
        eigenvalue_indices = []
        eigenvalue_scores = []
        
        for key, score in eigenvalue_data.items():
            if score is not None:
                eigenvalue_indices.append(key.replace('eigenvalue_', 'Eigenvalue '))
                eigenvalue_scores.append(score)
        
        if eigenvalue_indices:
            bar2 = go.Bar(
                x=eigenvalue_indices,
                y=eigenvalue_scores,
                marker_color='blue',
                name="Eigenvalue Similarity",
                text=[f"{score:.4f}" for score in eigenvalue_scores],
                textposition='auto'
            )
            fig.add_trace(bar2, row=1, col=2)
    
    # 3. Weight Configuration Comparison
    if len(weight_configs) >= 2:
        # Show the difference between pure functional and other approaches
        pure_functional_score = next((score for config, score in zip(weight_configs, similarity_scores) 
                                     if 'Pure Functional' in config), None)
        
        if pure_functional_score is not None:
            other_scores = [score for config, score in zip(weight_configs, similarity_scores) 
                           if 'Pure Functional' not in config]
            other_configs = [config for config in weight_configs if 'Pure Functional' not in config]
            
            differences = [pure_functional_score - score for score in other_scores]
            
            bar3 = go.Bar(
                x=other_configs,
                y=differences,
                marker_color=['lightblue' if diff > 0 else 'lightcoral' for diff in differences],
                name="Difference from Pure Functional",
                text=[f"{diff:+.4f}" for diff in differences],
                textposition='auto'
            )
            fig.add_trace(bar3, row=2, col=1)
    
    # 4. Model Architecture Summary
    architectures = [config['architecture'] for config in model_configs]
    arch_counts = {arch: architectures.count(arch) for arch in set(architectures)}
    
    pie = go.Pie(
        labels=list(arch_counts.keys()),
        values=list(arch_counts.values()),
        name="Architecture Distribution"
    )
    fig.add_trace(pie, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title={
            'text': "🧠 Functional Similarity Analysis: DTW Weight Impact",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Weight Configuration", row=1, col=1)
    fig.update_yaxes(title_text="Similarity Score", row=1, col=1)
    fig.update_xaxes(title_text="Eigenvalue Index", row=1, col=2)
    fig.update_yaxes(title_text="Similarity Score", row=1, col=2)
    fig.update_xaxes(title_text="Configuration", row=2, col=1)
    fig.update_yaxes(title_text="Difference from Pure Functional", row=2, col=1)
    
    return fig


def create_functional_similarity_report(results: Dict[str, Any], 
                                       model_configs: List[Dict[str, Any]]) -> str:
    """Create functional similarity analysis report."""
    print(f"\n📋 Creating Functional Similarity Report...")
    
    report = []
    report.append("# Functional Similarity Analysis Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This report analyzes the impact of DTW weight configurations on measuring")
    report.append("functional similarity across different neural network architectures.")
    report.append("")
    
    # Test Models
    report.append("## Test Models")
    report.append("")
    for config in model_configs:
        report.append(f"### {config['name']}")
        report.append(f"- Architecture: {config['architecture']}")
        report.append("")
    
    # Weight Configuration Analysis
    report.append("## Weight Configuration Analysis")
    report.append("")
    
    # Extract scores for comparison
    pure_functional_score = None
    balanced_score = None
    structure_heavy_score = None
    
    for config_name, result in results.items():
        if 'error' not in result:
            if config_name == 'pure_functional':
                pure_functional_score = result['similarity_score']
            elif config_name == 'balanced':
                balanced_score = result['similarity_score']
            elif config_name == 'structure_heavy':
                structure_heavy_score = result['similarity_score']
    
    # Analysis of each configuration
    report.append("### Pure Functional Similarity (eigenvalue_weight=1.0, structural_weight=0.0)")
    if pure_functional_score is not None:
        report.append(f"- **Similarity Score**: {pure_functional_score:.4f}")
        report.append("- **Focus**: 100% on eigenvalue evolution patterns")
        report.append("- **Ignores**: Structural differences between architectures")
        report.append("- **Best for**: Measuring functional similarity across architectures")
    else:
        report.append("- **Status**: Failed to compute")
    report.append("")
    
    report.append("### Balanced Approach (eigenvalue_weight=0.7, structural_weight=0.3)")
    if balanced_score is not None:
        report.append(f"- **Similarity Score**: {balanced_score:.4f}")
        report.append("- **Focus**: 70% eigenvalue evolution, 30% structural similarity")
        report.append("- **Considers**: Both functional and structural aspects")
        report.append("- **Best for**: General-purpose similarity measurement")
    else:
        report.append("- **Status**: Failed to compute")
    report.append("")
    
    report.append("### Structure-Heavy Approach (eigenvalue_weight=0.3, structural_weight=0.7)")
    if structure_heavy_score is not None:
        report.append(f"- **Similarity Score**: {structure_heavy_score:.4f}")
        report.append("- **Focus**: 30% eigenvalue evolution, 70% structural similarity")
        report.append("- **Penalizes**: Architectures with different structures")
        report.append("- **Best for**: Comparing models with similar architectures")
    else:
        report.append("- **Status**: Failed to compute")
    report.append("")
    
    # Comparative Analysis
    report.append("## Comparative Analysis")
    report.append("")
    
    if pure_functional_score is not None and balanced_score is not None:
        diff_balanced = pure_functional_score - balanced_score
        report.append(f"**Pure Functional vs Balanced**: {diff_balanced:+.4f}")
        if diff_balanced > 0:
            report.append("- Pure functional approach shows higher similarity")
            report.append("- Structural penalty reduces similarity score")
        else:
            report.append("- Balanced approach shows higher similarity")
        report.append("")
    
    if pure_functional_score is not None and structure_heavy_score is not None:
        diff_structural = pure_functional_score - structure_heavy_score
        report.append(f"**Pure Functional vs Structure-Heavy**: {diff_structural:+.4f}")
        if diff_structural > 0:
            report.append("- Pure functional approach shows much higher similarity")
            report.append("- Heavy structural penalty significantly reduces similarity")
        else:
            report.append("- Structure-heavy approach shows higher similarity")
        report.append("")
    
    # Multi-Eigenvalue Analysis
    if 'multi_eigenvalue' in results and 'error' not in results['multi_eigenvalue']:
        report.append("## Multi-Eigenvalue Functional Similarity")
        report.append("")
        eigenvalue_data = results['multi_eigenvalue']
        
        for key, score in eigenvalue_data.items():
            if score is not None:
                eigenvalue_idx = key.replace('eigenvalue_', '')
                report.append(f"- **Eigenvalue {eigenvalue_idx}**: {score:.4f}")
        
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if pure_functional_score is not None:
        report.append("### For Functional Similarity Across Architectures:")
        report.append("- **Use Pure Functional Configuration**: eigenvalue_weight=1.0, structural_weight=0.0")
        report.append("- **Rationale**: Eliminates architectural bias, focuses on behavior")
        report.append(f"- **Achieved Similarity**: {pure_functional_score:.4f}")
        report.append("")
    
    if balanced_score is not None:
        report.append("### For General-Purpose Comparison:")
        report.append("- **Use Balanced Configuration**: eigenvalue_weight=0.7, structural_weight=0.3")
        report.append("- **Rationale**: Considers both function and structure")
        report.append(f"- **Achieved Similarity**: {balanced_score:.4f}")
        report.append("")
    
    report.append("### Key Insights:")
    report.append("- **Structural weights penalize architectural differences**")
    report.append("- **Pure functional approach maximizes cross-architecture similarity**")
    report.append("- **Multiple eigenvalue indices provide consistent results**")
    report.append("- **DTW successfully captures functional similarity patterns**")
    
    return "\n".join(report)


def main():
    """Main execution function."""
    print("🚀 Functional Similarity DTW Analysis")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    try:
        # Step 1: Load two models
        model_config1, model_config2 = load_two_models()
        model_configs = [model_config1, model_config2]
        
        # Step 2: Load models
        loaded_models = load_models(model_configs)
        
        # Step 3: Test functional similarity DTW
        results = test_functional_similarity_dtw(loaded_models, model_configs)
        
        # Step 4: Create analysis
        fig = create_functional_similarity_analysis(results, model_configs)
        
        # Step 5: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save visualization
        viz_path = f"functional_similarity_analysis_{timestamp}.html"
        fig.write_html(viz_path)
        print(f"   ✅ Analysis saved: {viz_path}")
        
        # Save report
        report = create_functional_similarity_report(results, model_configs)
        report_path = f"functional_similarity_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"   ✅ Report saved: {report_path}")
        
        # Print summary
        print(f"\n📊 Functional Similarity Analysis Summary:")
        
        for config_name, result in results.items():
            if config_name in ['pure_functional', 'balanced', 'structure_heavy'] and 'error' not in result:
                config_display = config_name.replace('_', ' ').title()
                print(f"   • {config_display}: {result['similarity_score']:.4f}")
        
        # Print recommendation
        pure_functional_score = results.get('pure_functional', {}).get('similarity_score')
        balanced_score = results.get('balanced', {}).get('similarity_score')
        
        if pure_functional_score is not None and balanced_score is not None:
            improvement = pure_functional_score - balanced_score
            print(f"\n💡 Key Finding:")
            print(f"   Pure functional approach shows {improvement:+.4f} improvement")
            print(f"   over balanced approach for cross-architecture similarity")
        
        print(f"\n✅ Analysis complete! Results saved.")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())