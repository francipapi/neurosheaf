#!/usr/bin/env python3
"""
Comprehensive DTW Model Comparison Script

This script compares all saved models in the models/ folder using Dynamic Time Warping (DTW)
applied to their full eigenvalue evolution during spectral analysis.

Features:
- Loads all models from models/ folder using correct architectures
- Performs full eigenvalue evolution analysis using multivariate DTW
- Creates interactive HTML dashboard with distance matrix, evolution plots, and clustering
- Handles both MLPModel and ActualCustomModel architectures
- Generates comprehensive analysis reports

Usage:
    python scripts/compare_all_models_dtw.py

Requirements:
    - neurosheaf package with DTW functionality
    - dtaidistance>=2.3.10 and tslearn>=0.6.0
    - All model files in models/ folder
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model, list_model_info
from neurosheaf.visualization.spectral import SpectralVisualizer
from neurosheaf.utils.dtw_similarity import FiltrationDTW


# Define model architectures locally (copied from test_all.py)
class MLPModel(nn.Module):
    """MLP model architecture matching the configuration."""
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
        
        # Store configuration
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Get activation functions
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        # Build the network
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(self.output_activation_fn)
        
        self.layers = nn.Sequential(*layers_list)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        """Get activation function by name."""
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
        
        # Layer 0: Linear(3 -> 32) + ReLU
        x = self.layers[1](self.layers[0](x))  # [batch_size, 32]
        
        # Layer 2: Linear(32 -> 32) + ReLU + Dropout
        x = self.layers[4](self.layers[3](self.layers[2](x)))  # [batch_size, 32]
        
        # Reshape for Conv1D: [batch_size, 32] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 5: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[7](self.layers[6](self.layers[5](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 8: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[10](self.layers[9](self.layers[8](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 11: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[13](self.layers[12](self.layers[11](x)))  # [batch_size, 32, 1]
        
        # Flatten for final layer: [batch_size, 32, 1] -> [batch_size, 32]
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        
        # Layer 14: Linear(32 -> 1) + Sigmoid
        x = self.layers[15](self.layers[14](x))  # [batch_size, 1]
        
        return x


def setup_environment():
    """Set up the environment for reproducible analysis."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set OpenMP settings for macOS compatibility
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🔧 Environment Setup Complete")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")


def discover_models(models_dir: Path) -> List[Dict[str, Any]]:
    """Discover all model files and determine their architectures."""
    models = []
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    print(f"🔍 Discovering models in {models_dir}")
    
    for model_file in models_dir.glob("*.pth"):
        model_name = model_file.stem
        
        # Determine architecture based on filename
        if "mlp" in model_name.lower():
            architecture = "MLPModel"
            model_class = MLPModel
        elif "custom" in model_name.lower():
            architecture = "ActualCustomModel"
            model_class = ActualCustomModel
        else:
            print(f"⚠️  Unknown architecture for {model_name}, skipping...")
            continue
        
        # Get model info
        try:
            info = list_model_info(model_file)
            models.append({
                'name': model_name,
                'path': model_file,
                'architecture': architecture,
                'model_class': model_class,
                'parameters': info.get('model_parameters', 'Unknown'),
                'file_size_mb': info.get('file_size_mb', 0),
                'epoch': info.get('epoch', 'Unknown'),
                'accuracy': info.get('accuracy', 'Unknown')
            })
            print(f"   ✅ {model_name} ({architecture})")
        except Exception as e:
            print(f"   ❌ Failed to load info for {model_name}: {e}")
    
    return models


def load_all_models(model_configs: List[Dict[str, Any]]) -> Dict[str, nn.Module]:
    """Load all discovered models."""
    loaded_models = {}
    
    print(f"\n📦 Loading {len(model_configs)} models...")
    
    for config in model_configs:
        try:
            model = load_model(config['model_class'], config['path'], device='cpu')
            loaded_models[config['name']] = model
            print(f"   ✅ {config['name']} loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load {config['name']}: {e}")
    
    return loaded_models


def generate_test_data(input_dim: int = 3, batch_size: int = 100) -> torch.Tensor:
    """Generate test data for model analysis."""
    print(f"\n🎲 Generating test data (batch_size={batch_size}, input_dim={input_dim})")
    
    # Create diverse test data that exercises different model behaviors
    data = torch.randn(batch_size, input_dim)
    
    # Add some structure to make eigenvalue evolution more interesting
    data[:batch_size//3] *= 0.5  # Small values
    data[batch_size//3:2*batch_size//3] *= 1.5  # Large values
    # Middle third remains standard normal
    
    return data


def perform_dtw_analysis(models: Dict[str, nn.Module], 
                        data: torch.Tensor) -> Dict[str, Any]:
    """Perform comprehensive DTW analysis on all models."""
    print(f"\n🔬 Performing DTW Analysis on {len(models)} models...")
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Convert to list for multiple network comparison
    model_names = list(models.keys())
    model_objects = [models[name] for name in model_names]
    
    print("   Phase 1: Individual model analysis...")
    individual_analyses = {}
    
    for name, model in models.items():
        try:
            print(f"     Analyzing {name}...")
            analysis = analyzer.analyze(model, data)
            individual_analyses[name] = analysis
            print(f"     ✅ {name} complete")
        except Exception as e:
            print(f"     ❌ {name} failed: {e}")
            individual_analyses[name] = None
    
    print("   Phase 2: DTW comparison using multivariate approach...")
    try:
        comparison_result = analyzer.compare_multiple_networks(
            model_objects, 
            data, 
            method='dtw',
            eigenvalue_index=None,  # Compare all eigenvalues
            multivariate=True  # Use multivariate DTW for full eigenvalue evolution
        )
        
        # Add model names to results
        comparison_result['model_names'] = model_names
        comparison_result['individual_analyses'] = individual_analyses
        
        print("   ✅ DTW comparison complete")
        return comparison_result
        
    except Exception as e:
        print(f"   ❌ DTW comparison failed: {e}")
        print("   🔄 Falling back to euclidean distance...")
        
        # Fallback to euclidean distance
        try:
            fallback_result = analyzer.compare_multiple_networks(
                model_objects, 
                data, 
                method='euclidean'
            )
            fallback_result['model_names'] = model_names
            fallback_result['individual_analyses'] = individual_analyses
            fallback_result['fallback_method'] = True
            
            print("   ✅ Fallback comparison complete")
            return fallback_result
            
        except Exception as e2:
            print(f"   ❌ Fallback comparison also failed: {e2}")
            raise


def create_comprehensive_dashboard(results: Dict[str, Any], 
                                 model_configs: List[Dict[str, Any]]) -> go.Figure:
    """Create comprehensive interactive dashboard."""
    print("\n📊 Creating comprehensive dashboard...")
    
    model_names = results['model_names']
    distance_matrix = results['distance_matrix']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Distance Matrix Heatmap",
            "Model Architecture Distribution", 
            "Clustering Analysis",
            "DTW Evolution Comparison"
        ],
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # 1. Distance Matrix Heatmap
    heatmap = go.Heatmap(
        z=distance_matrix,
        x=model_names,
        y=model_names,
        colorscale='Viridis',
        text=distance_matrix,
        texttemplate="%{text:.3f}",
        colorbar=dict(title="DTW Distance", x=0.46)
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # 2. Model Architecture Distribution
    architectures = [config['architecture'] for config in model_configs]
    arch_counts = pd.Series(architectures).value_counts()
    
    pie = go.Pie(
        labels=arch_counts.index,
        values=arch_counts.values,
        name="Architecture Distribution"
    )
    fig.add_trace(pie, row=1, col=2)
    
    # 3. Clustering Analysis
    if 'cluster_analysis' in results and results['cluster_analysis'].get('status') != 'sklearn_not_available':
        cluster_info = results['cluster_analysis']
        cluster_labels = cluster_info.get('labels', [])
        
        if len(cluster_labels) > 0:
            scatter = go.Scatter(
                x=list(range(len(model_names))),
                y=cluster_labels,
                mode='markers+text',
                text=model_names,
                textposition='top center',
                marker=dict(
                    size=12,
                    color=cluster_labels,
                    colorscale='viridis'
                ),
                name="Cluster Assignment"
            )
            fig.add_trace(scatter, row=2, col=1)
        else:
            # Add placeholder if no clustering
            fig.add_annotation(
                text="Clustering analysis not available",
                x=0.5, y=0.5,
                xref="x3", yref="y3",
                showarrow=False,
                font=dict(size=14)
            )
    else:
        fig.add_annotation(
            text="Clustering requires sklearn",
            x=0.5, y=0.5,
            xref="x3", yref="y3",
            showarrow=False,
            font=dict(size=14)
        )
    
    # 4. DTW Evolution Comparison (if available)
    if 'individual_analyses' in results and results.get('method') == 'dtw':
        # Plot eigenvalue evolution for each model
        for i, (name, analysis) in enumerate(results['individual_analyses'].items()):
            if analysis is not None and 'spectral_results' in analysis:
                spectral_results = analysis['spectral_results']
                if 'eigenvalue_sequences' in spectral_results['persistence_result']:
                    eigenval_seqs = spectral_results['persistence_result']['eigenvalue_sequences']
                    filtration_params = spectral_results['filtration_params']
                    
                    # Plot first eigenvalue evolution
                    if len(eigenval_seqs) > 0 and len(eigenval_seqs[0]) > 0:
                        evolution = [seq[0].item() if len(seq) > 0 else 0 for seq in eigenval_seqs]
                        
                        line = go.Scatter(
                            x=filtration_params,
                            y=evolution,
                            mode='lines',
                            name=name,
                            line=dict(width=2)
                        )
                        fig.add_trace(line, row=2, col=2)
    else:
        fig.add_annotation(
            text="DTW evolution data not available",
            x=0.5, y=0.5,
            xref="x4", yref="y4",
            showarrow=False,
            font=dict(size=14)
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "🧠 Neural Network DTW Comparison Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model Index", row=2, col=1)
    fig.update_yaxes(title_text="Cluster", row=2, col=1)
    fig.update_xaxes(title_text="Filtration Parameter", row=2, col=2)
    fig.update_yaxes(title_text="Eigenvalue", row=2, col=2, type='log')
    
    return fig


def create_detailed_comparison_report(results: Dict[str, Any], 
                                    model_configs: List[Dict[str, Any]]) -> str:
    """Create detailed comparison report."""
    print("\n📋 Creating detailed comparison report...")
    
    report = []
    report.append("# Neural Network DTW Comparison Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Model Summary
    report.append("## Model Summary")
    report.append("")
    for config in model_configs:
        report.append(f"### {config['name']}")
        report.append(f"- Architecture: {config['architecture']}")
        report.append(f"- Parameters: {config['parameters']:,}")
        report.append(f"- File Size: {config['file_size_mb']:.2f} MB")
        report.append(f"- Training Epoch: {config['epoch']}")
        report.append(f"- Accuracy: {config['accuracy']}")
        report.append("")
    
    # DTW Analysis Results
    report.append("## DTW Analysis Results")
    report.append("")
    report.append(f"- Method: {results.get('method', 'Unknown')}")
    report.append(f"- Number of Models: {len(results['model_names'])}")
    report.append(f"- Distance Matrix Shape: {results['distance_matrix'].shape}")
    
    if results.get('fallback_method'):
        report.append("- **Note**: DTW libraries not available, used Euclidean distance fallback")
    
    report.append("")
    
    # Distance Matrix
    report.append("### Distance Matrix")
    report.append("")
    model_names = results['model_names']
    distance_matrix = results['distance_matrix']
    
    # Create markdown table
    header = "| Model | " + " | ".join(model_names) + " |"
    separator = "|-------|" + "-------|" * len(model_names)
    report.append(header)
    report.append(separator)
    
    for i, name in enumerate(model_names):
        row = f"| {name} | " + " | ".join([f"{distance_matrix[i,j]:.3f}" for j in range(len(model_names))]) + " |"
        report.append(row)
    
    report.append("")
    
    # Similarity Rankings
    if 'similarity_rankings' in results:
        report.append("### Similarity Rankings")
        report.append("")
        
        for ranking in results['similarity_rankings']:
            model_idx = ranking.get('sheaf_index', ranking.get('index', 0))
            model_name = model_names[model_idx] if model_idx < len(model_names) else f"Model {model_idx}"
            report.append(f"**{model_name}** is most similar to:")
            
            most_similar = ranking.get('most_similar', [])
            for similar in most_similar[:3]:  # Top 3
                sim_idx = similar.get('sheaf_index', similar.get('index', 0))
                sim_name = model_names[sim_idx] if sim_idx < len(model_names) else f"Model {sim_idx}"
                similarity = similar.get('similarity', 1 - similar.get('distance', 0))
                report.append(f"1. {sim_name}: {similarity:.3f}")
            
            report.append("")
    
    # Clustering Analysis
    if 'cluster_analysis' in results and results['cluster_analysis'].get('status') != 'sklearn_not_available':
        cluster_info = results['cluster_analysis']
        report.append("### Clustering Analysis")
        report.append("")
        report.append(f"- Number of Clusters: {cluster_info.get('n_clusters', 'Unknown')}")
        report.append(f"- Silhouette Score: {cluster_info.get('silhouette_score', 'Unknown'):.3f}")
        report.append("")
        
        if 'cluster_assignments' in cluster_info:
            report.append("**Cluster Assignments:**")
            for cluster_name, members in cluster_info['cluster_assignments'].items():
                member_names = [model_names[i] for i in members]
                report.append(f"- {cluster_name}: {', '.join(member_names)}")
            report.append("")
    
    # Comparison Metadata
    if 'comparison_metadata' in results:
        metadata = results['comparison_metadata']
        report.append("### Analysis Metadata")
        report.append("")
        report.append(f"- Data Shape: {metadata.get('data_shape', 'Unknown')}")
        report.append(f"- Device: {metadata.get('device', 'Unknown')}")
        report.append(f"- Eigenvalue Index: {metadata.get('eigenvalue_index', 'All')}")
        report.append(f"- Multivariate DTW: {metadata.get('multivariate', 'Unknown')}")
        report.append("")
    
    return "\n".join(report)


def save_results(results: Dict[str, Any], 
                model_configs: List[Dict[str, Any]],
                output_dir: Path = None) -> None:
    """Save all results to files."""
    if output_dir is None:
        output_dir = Path.cwd()
    
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 Saving results to {output_dir}")
    
    # 1. Save comprehensive dashboard
    dashboard_fig = create_comprehensive_dashboard(results, model_configs)
    dashboard_path = output_dir / f"dtw_model_comparison_dashboard_{timestamp}.html"
    dashboard_fig.write_html(str(dashboard_path))
    print(f"   ✅ Dashboard: {dashboard_path.name}")
    
    # 2. Save detailed report
    report = create_detailed_comparison_report(results, model_configs)
    report_path = output_dir / f"dtw_model_comparison_report_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   ✅ Report: {report_path.name}")
    
    # 3. Save raw results as JSON
    json_results = {
        'model_names': results['model_names'],
        'distance_matrix': results['distance_matrix'].tolist(),
        'method': results.get('method', 'unknown'),
        'timestamp': timestamp,
        'model_configs': [
            {k: v for k, v in config.items() if k not in ['model_class', 'path']}
            for config in model_configs
        ]
    }
    
    if 'similarity_rankings' in results:
        json_results['similarity_rankings'] = results['similarity_rankings']
    
    if 'cluster_analysis' in results:
        cluster_info = results['cluster_analysis']
        if cluster_info.get('status') != 'sklearn_not_available':
            json_results['cluster_analysis'] = {
                'n_clusters': cluster_info.get('n_clusters'),
                'silhouette_score': cluster_info.get('silhouette_score'),
                'cluster_assignments': cluster_info.get('cluster_assignments')
            }
    
    json_path = output_dir / f"dtw_model_comparison_data_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   ✅ Data: {json_path.name}")
    
    # 4. Create individual distance matrix visualization
    fig_distance = go.Figure(data=go.Heatmap(
        z=results['distance_matrix'],
        x=results['model_names'],
        y=results['model_names'],
        colorscale='Viridis',
        text=results['distance_matrix'],
        texttemplate="%{text:.3f}",
        colorbar=dict(title="DTW Distance")
    ))
    
    fig_distance.update_layout(
        title="Neural Network DTW Distance Matrix",
        xaxis_title="Model",
        yaxis_title="Model",
        template='plotly_white'
    )
    
    distance_path = output_dir / f"dtw_distance_matrix_{timestamp}.html"
    fig_distance.write_html(str(distance_path))
    print(f"   ✅ Distance Matrix: {distance_path.name}")


def main():
    """Main execution function."""
    print("🚀 Neural Network DTW Comparison Analysis")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    # Configuration
    models_dir = Path("models")
    output_dir = Path("dtw_analysis_results")
    
    try:
        # Step 1: Discover models
        model_configs = discover_models(models_dir)
        if not model_configs:
            print("❌ No models found! Please ensure models are in the 'models/' directory.")
            return
        
        print(f"\n📋 Found {len(model_configs)} models:")
        for config in model_configs:
            print(f"   • {config['name']} ({config['architecture']})")
        
        # Step 2: Load models
        loaded_models = load_all_models(model_configs)
        if not loaded_models:
            print("❌ No models could be loaded!")
            return
        
        # Step 3: Generate test data
        test_data = generate_test_data(input_dim=3, batch_size=200)
        
        # Step 4: Perform DTW analysis
        results = perform_dtw_analysis(loaded_models, test_data)
        
        # Step 5: Save results
        save_results(results, model_configs, output_dir)
        
        # Step 6: Print summary
        print("\n📊 Analysis Summary:")
        print(f"   • Models analyzed: {len(loaded_models)}")
        print(f"   • Method used: {results.get('method', 'Unknown')}")
        print(f"   • Distance matrix shape: {results['distance_matrix'].shape}")
        
        if results.get('fallback_method'):
            print("   • Note: DTW libraries not available, used Euclidean distance")
        
        # Print top similarities
        if 'similarity_rankings' in results:
            print("\n🔍 Top Similarities:")
            for ranking in results['similarity_rankings'][:3]:  # Top 3 models
                model_idx = ranking.get('sheaf_index', ranking.get('index', 0))
                model_name = results['model_names'][model_idx] if model_idx < len(results['model_names']) else f"Model {model_idx}"
                most_similar_list = ranking.get('most_similar', [])
                if most_similar_list:
                    most_similar = most_similar_list[0]  # Most similar
                    sim_idx = most_similar.get('sheaf_index', most_similar.get('index', 0))
                    sim_name = results['model_names'][sim_idx] if sim_idx < len(results['model_names']) else f"Model {sim_idx}"
                    similarity = most_similar.get('similarity', 1 - most_similar.get('distance', 0))
                    print(f"   • {model_name} ↔ {sim_name}: {similarity:.3f}")
        
        print(f"\n✅ Analysis complete! Results saved to '{output_dir}'")
        print("\nGenerated files:")
        print(f"   • Interactive dashboard: dtw_model_comparison_dashboard_*.html")
        print(f"   • Detailed report: dtw_model_comparison_report_*.md")
        print(f"   • Raw data: dtw_model_comparison_data_*.json")
        print(f"   • Distance matrix: dtw_distance_matrix_*.html")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())