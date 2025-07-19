#!/usr/bin/env python3
"""
Simple DTW Test Script - Compare Two Models

This script tests DTW functionality by comparing just two models from the models/ folder.
Focuses on individual eigenvalue DTW (not multivariate) to avoid library compatibility issues.

Usage:
    python scripts/test_dtw_two_models.py
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
from neurosheaf.utils import load_model, list_model_info
from neurosheaf.visualization.spectral import SpectralVisualizer
from neurosheaf.utils.dtw_similarity import FiltrationDTW


# Define model architectures locally
class MLPModel(nn.Module):
    """MLP model architecture."""
    def __init__(
        self,
        input_dim: int = 3,
        num_hidden_layers: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 1,
        activation_fn_name: str = 'relu',
        output_activation_fn_name: str = 'sigmoid',
        dropout_rate: float = 0.0012
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        layers_list = []
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(self.output_activation_fn)
        
        self.layers = nn.Sequential(*layers_list)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'softmax': nn.Softmax(dim=-1),
            'none': nn.Identity()
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        return activations[name.lower()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ActualCustomModel(nn.Module):
    """Custom model class matching the actual saved weights structure."""
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(3, 32),                                    # layers.0
            nn.ReLU(),                                           # layers.1
            nn.Linear(32, 32),                                   # layers.2
            nn.ReLU(),                                           # layers.3
            nn.Dropout(0.0),                                     # layers.4
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.5
            nn.ReLU(),                                           # layers.6
            nn.Dropout(0.0),                                     # layers.7
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.8
            nn.ReLU(),                                           # layers.9
            nn.Dropout(0.0),                                     # layers.10
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.11
            nn.ReLU(),                                           # layers.12
            nn.Dropout(0.0),                                     # layers.13
            nn.Linear(32, 1),                                    # layers.14
            nn.Sigmoid()                                         # layers.15
        )
    
    def forward(self, x):
        # Input: [batch_size, 3]
        x = self.layers[1](self.layers[0](x))  # [batch_size, 32]
        x = self.layers[4](self.layers[3](self.layers[2](x)))  # [batch_size, 32]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        x = self.layers[7](self.layers[6](self.layers[5](x)))  # [batch_size, 32, 1]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        x = self.layers[10](self.layers[9](self.layers[8](x)))  # [batch_size, 32, 1]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        x = self.layers[13](self.layers[12](self.layers[11](x)))  # [batch_size, 32, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        x = self.layers[15](self.layers[14](x))  # [batch_size, 1]
        return x


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🔧 Environment Setup Complete")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")


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


def test_dtw_comparison(models: Dict[str, nn.Module], 
                       model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test DTW comparison between two models."""
    print(f"\n🔬 Testing DTW Comparison...")
    
    # Generate test data
    data = torch.randn(100, 3)
    print(f"   Generated test data: {data.shape}")
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Get model names
    model_names = list(models.keys())
    model1_name, model2_name = model_names[0], model_names[1]
    model1, model2 = models[model1_name], models[model2_name]
    
    print(f"   Comparing {model1_name} vs {model2_name}")
    
    # Test individual eigenvalue DTW (not multivariate)
    test_results = {}
    
    # Test eigenvalue index 0 (largest eigenvalue)
    print(f"   Testing eigenvalue index 0 (largest eigenvalue)...")
    try:
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=0,  # Compare largest eigenvalue only
            multivariate=False   # Use univariate DTW
        )
        
        test_results['eigenvalue_0'] = result
        print(f"   ✅ Eigenvalue 0 DTW: Success")
        print(f"      Similarity Score: {result['similarity_score']:.4f}")
        
        if 'dtw_comparison' in result and result['dtw_comparison']:
            dtw_info = result['dtw_comparison']
            if 'dtw_comparison' in dtw_info:
                print(f"      DTW Distance: {dtw_info['dtw_comparison']['distance']:.4f}")
                print(f"      Normalized Distance: {dtw_info['dtw_comparison']['normalized_distance']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Eigenvalue 0 DTW failed: {e}")
        test_results['eigenvalue_0'] = {'error': str(e)}
    
    # Test eigenvalue index 1 (second largest eigenvalue)
    print(f"   Testing eigenvalue index 1 (second largest eigenvalue)...")
    try:
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=1,  # Compare second largest eigenvalue
            multivariate=False   # Use univariate DTW
        )
        
        test_results['eigenvalue_1'] = result
        print(f"   ✅ Eigenvalue 1 DTW: Success")
        print(f"      Similarity Score: {result['similarity_score']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Eigenvalue 1 DTW failed: {e}")
        test_results['eigenvalue_1'] = {'error': str(e)}
    
    # Test DTW with different parameters
    print(f"   Testing DTW with custom parameters...")
    try:
        result = analyzer.compare_networks(
            model1, model2, data, 
            method='dtw',
            eigenvalue_index=0,
            multivariate=False,
            dtw_constraint_band=0.1  # Custom constraint
        )
        
        test_results['custom_params'] = result
        print(f"   ✅ Custom DTW parameters: Success")
        print(f"      Similarity Score: {result['similarity_score']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Custom DTW parameters failed: {e}")
        test_results['custom_params'] = {'error': str(e)}
    
    # Test fallback methods for comparison
    print(f"   Testing fallback methods...")
    try:
        euclidean_result = analyzer.compare_networks(
            model1, model2, data, method='euclidean'
        )
        test_results['euclidean_fallback'] = euclidean_result
        print(f"   ✅ Euclidean fallback: {euclidean_result['similarity_score']:.4f}")
        
        cosine_result = analyzer.compare_networks(
            model1, model2, data, method='cosine'
        )
        test_results['cosine_fallback'] = cosine_result
        print(f"   ✅ Cosine fallback: {cosine_result['similarity_score']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Fallback methods failed: {e}")
    
    return test_results


def create_dtw_visualization(results: Dict[str, Any], 
                           model_configs: List[Dict[str, Any]]) -> go.Figure:
    """Create DTW test visualization."""
    print(f"\n📊 Creating DTW Test Visualization...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "DTW Similarity Scores",
            "DTW vs Fallback Methods", 
            "DTW Distance Analysis",
            "Model Architectures"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}]
        ],
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )
    
    # Extract results
    methods = []
    scores = []
    colors = []
    
    for method, result in results.items():
        if 'error' not in result:
            methods.append(method)
            scores.append(result['similarity_score'])
            if 'dtw' in method.lower():
                colors.append('blue')
            else:
                colors.append('red')
    
    # 1. DTW Similarity Scores
    if methods:
        bar1 = go.Bar(
            x=methods,
            y=scores,
            marker_color=colors,
            name="Similarity Scores"
        )
        fig.add_trace(bar1, row=1, col=1)
    
    # 2. DTW vs Fallback comparison
    dtw_scores = [score for method, score in zip(methods, scores) if 'dtw' in method.lower()]
    fallback_scores = [score for method, score in zip(methods, scores) if 'dtw' not in method.lower()]
    
    if dtw_scores and fallback_scores:
        scatter2 = go.Scatter(
            x=dtw_scores,
            y=fallback_scores,
            mode='markers',
            marker=dict(size=12, color='purple'),
            name="DTW vs Fallback"
        )
        fig.add_trace(scatter2, row=1, col=2)
        
        # Add diagonal line
        max_score = max(max(dtw_scores), max(fallback_scores))
        min_score = min(min(dtw_scores), min(fallback_scores))
        fig.add_shape(
            type="line",
            x0=min_score, y0=min_score,
            x1=max_score, y1=max_score,
            line=dict(color="gray", dash="dash"),
            row=1, col=2
        )
    
    # 3. DTW Distance Analysis
    dtw_distances = []
    dtw_methods = []
    
    for method, result in results.items():
        if 'error' not in result and 'dtw' in method.lower():
            if 'dtw_comparison' in result and result['dtw_comparison']:
                dtw_info = result['dtw_comparison']
                if 'dtw_comparison' in dtw_info:
                    dtw_distances.append(dtw_info['dtw_comparison']['distance'])
                    dtw_methods.append(method)
    
    if dtw_distances:
        bar3 = go.Bar(
            x=dtw_methods,
            y=dtw_distances,
            marker_color='green',
            name="DTW Distances"
        )
        fig.add_trace(bar3, row=2, col=1)
    
    # 4. Model Architecture Distribution
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
            'text': "🧠 DTW Neural Network Comparison Test Results",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Method", row=1, col=1)
    fig.update_yaxes(title_text="Similarity Score", row=1, col=1)
    fig.update_xaxes(title_text="DTW Scores", row=1, col=2)
    fig.update_yaxes(title_text="Fallback Scores", row=1, col=2)
    fig.update_xaxes(title_text="DTW Method", row=2, col=1)
    fig.update_yaxes(title_text="DTW Distance", row=2, col=1)
    
    return fig


def create_test_report(results: Dict[str, Any], 
                      model_configs: List[Dict[str, Any]]) -> str:
    """Create DTW test report."""
    print(f"\n📋 Creating DTW Test Report...")
    
    report = []
    report.append("# DTW Neural Network Comparison Test Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Model Summary
    report.append("## Test Models")
    report.append("")
    for config in model_configs:
        report.append(f"### {config['name']}")
        report.append(f"- Architecture: {config['architecture']}")
        report.append("")
    
    # DTW Test Results
    report.append("## DTW Test Results")
    report.append("")
    
    dtw_successes = 0
    dtw_failures = 0
    
    for method, result in results.items():
        if 'dtw' in method.lower():
            if 'error' in result:
                dtw_failures += 1
                report.append(f"### {method} - ❌ FAILED")
                report.append(f"Error: {result['error']}")
            else:
                dtw_successes += 1
                report.append(f"### {method} - ✅ SUCCESS")
                report.append(f"- Similarity Score: {result['similarity_score']:.4f}")
                report.append(f"- Method: {result['method']}")
                
                if 'dtw_comparison' in result and result['dtw_comparison']:
                    dtw_info = result['dtw_comparison']
                    if 'dtw_comparison' in dtw_info:
                        dtw_data = dtw_info['dtw_comparison']
                        report.append(f"- DTW Distance: {dtw_data['distance']:.4f}")
                        report.append(f"- Normalized Distance: {dtw_data['normalized_distance']:.4f}")
            
            report.append("")
    
    # Summary Statistics
    report.append("## Summary")
    report.append("")
    report.append(f"- DTW Tests Passed: {dtw_successes}")
    report.append(f"- DTW Tests Failed: {dtw_failures}")
    if dtw_successes + dtw_failures > 0:
        report.append(f"- Success Rate: {dtw_successes/(dtw_successes+dtw_failures)*100:.1f}%")
    else:
        report.append(f"- Success Rate: N/A")
    
    # Comparison with Fallback Methods
    report.append("")
    report.append("## Comparison with Fallback Methods")
    report.append("")
    
    for method, result in results.items():
        if 'fallback' in method.lower() and 'error' not in result:
            report.append(f"- {method}: {result['similarity_score']:.4f}")
    
    return "\n".join(report)


def main():
    """Main execution function."""
    print("🚀 DTW Neural Network Comparison Test")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    try:
        # Step 1: Load two models
        model_config1, model_config2 = load_two_models()
        model_configs = [model_config1, model_config2]
        
        # Step 2: Load models
        loaded_models = load_models(model_configs)
        
        # Step 3: Test DTW comparison
        results = test_dtw_comparison(loaded_models, model_configs)
        
        # Step 4: Create visualization
        fig = create_dtw_visualization(results, model_configs)
        
        # Step 5: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save visualization
        viz_path = f"dtw_test_results_{timestamp}.html"
        fig.write_html(viz_path)
        print(f"   ✅ Visualization saved: {viz_path}")
        
        # Save report
        report = create_test_report(results, model_configs)
        report_path = f"dtw_test_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"   ✅ Report saved: {report_path}")
        
        # Print summary
        print(f"\n📊 Test Summary:")
        dtw_successes = sum(1 for method, result in results.items() 
                           if 'dtw' in method.lower() and 'error' not in result)
        dtw_failures = sum(1 for method, result in results.items() 
                          if 'dtw' in method.lower() and 'error' in result)
        
        print(f"   • DTW Tests Passed: {dtw_successes}")
        print(f"   • DTW Tests Failed: {dtw_failures}")
        if dtw_successes + dtw_failures > 0:
            print(f"   • Success Rate: {dtw_successes/(dtw_successes+dtw_failures)*100:.1f}%")
        else:
            print(f"   • Success Rate: N/A")
        
        # Print successful DTW results
        print(f"\n✅ Successful DTW Results:")
        for method, result in results.items():
            if 'dtw' in method.lower() and 'error' not in result:
                print(f"   • {method}: {result['similarity_score']:.4f}")
        
        print(f"\n✅ Test complete! Results saved.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())