#!/usr/bin/env python3
"""
Multivariate DTW Neural Network Similarity Analysis

This script measures functional similarity between neural networks using multivariate Dynamic Time Warping (DTW) 
applied to their full eigenvalue evolution during spectral analysis. It computes pairwise similarities for all 
available models and generates comprehensive interactive dashboards and reports.

Key Features:
- Pure multivariate DTW analysis (eigenvalue_index=None, multivariate=True)
- Architecture-agnostic functional similarity measurement
- Symmetric pairwise analysis of all model combinations
- Interactive HTML dashboards with clustering and evolution plots
- Comprehensive analysis reports with similarity rankings

Usage:
    python scripts/multivariate_dtw_similarity_analysis.py

Requirements:
    - neurosheaf package with DTW functionality
    - tslearn>=0.6.0 (for multivariate DTW)
    - dtaidistance>=2.3.10 (fallback DTW implementation)
    - All model files in models/ folder
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import json
import time

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.utils import load_model
from neurosheaf.utils.dtw_similarity import FiltrationDTW


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
    """Custom model class with Linear + Conv1D architecture."""
    
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
        
        # Get basic model info
        try:
            file_size_mb = model_file.stat().st_size / (1024 * 1024)
            
            # Extract info from filename
            epoch = "Unknown"
            accuracy = "Unknown"
            
            if "epoch" in model_name:
                try:
                    epoch_part = model_name.split("epoch_")[1].split("_")[0].split(".")[0]
                    epoch = int(epoch_part)
                except:
                    pass
                    
            if "acc_" in model_name:
                try:
                    acc_part = model_name.split("acc_")[1].split("_")[0]
                    accuracy = float(acc_part)
                except:
                    pass
            
            models.append({
                'name': model_name,
                'path': model_file,
                'architecture': architecture,
                'model_class': model_class,
                'file_size_mb': file_size_mb,
                'epoch': epoch,
                'accuracy': accuracy
            })
            print(f"   ✅ {model_name} ({architecture})")
            
        except Exception as e:
            print(f"   ❌ Failed to process {model_name}: {e}")
    
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


def generate_test_data(input_dim: int = 3, batch_size: int = 150) -> torch.Tensor:
    """Generate diverse test data for model analysis."""
    print(f"\n🎲 Generating test data (batch_size={batch_size}, input_dim={input_dim})")
    
    # Create diverse test data that exercises different model behaviors
    data = torch.randn(batch_size, input_dim)
    
    # Add structure to make eigenvalue evolution more interesting
    data[:batch_size//3] *= 0.5  # Small values
    data[batch_size//3:2*batch_size//3] *= 1.5  # Large values
    # Middle third remains standard normal
    
    return data


def perform_pairwise_multivariate_dtw_analysis(models: Dict[str, nn.Module], 
                                              data: torch.Tensor) -> Dict[str, Any]:
    """Perform pairwise multivariate DTW analysis on all model combinations."""
    print(f"\n🔬 Performing Pairwise Multivariate DTW Analysis on {len(models)} models...")
    
    model_names = list(models.keys())
    n_models = len(model_names)
    
    # Initialize analyzer
    analyzer = NeurosheafAnalyzer(device='cpu', enable_profiling=False)
    
    # Storage for results
    distance_matrix = np.zeros((n_models, n_models))
    similarity_matrix = np.zeros((n_models, n_models))
    individual_analyses = {}
    pairwise_details = {}
    
    print(f"   Computing {n_models * (n_models - 1) // 2} unique pairwise comparisons...")
    
    # Phase 1: Individual model analyses
    print("   Phase 1: Individual model spectral analyses...")
    for i, (name, model) in enumerate(models.items()):
        try:
            print(f"     Analyzing {name} ({i+1}/{n_models})...")
            start_time = time.time()
            
            analysis = analyzer.analyze(model, data)
            individual_analyses[name] = analysis
            
            elapsed = time.time() - start_time
            print(f"     ✅ {name} complete ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"     ❌ {name} failed: {e}")
            individual_analyses[name] = None
    
    # Phase 2: Pairwise multivariate DTW comparisons
    print("\n   Phase 2: Pairwise multivariate DTW comparisons...")
    comparison_count = 0
    total_comparisons = n_models * (n_models - 1) // 2
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            comparison_count += 1
            model_name_i = model_names[i]
            model_name_j = model_names[j]
            
            print(f"     [{comparison_count}/{total_comparisons}] {model_name_i} ↔ {model_name_j}")
            
            try:
                start_time = time.time()
                
                # Perform multivariate DTW comparison
                result = analyzer.compare_networks(
                    models[model_name_i], 
                    models[model_name_j], 
                    data, 
                    method='dtw',
                    eigenvalue_index=None,  # All eigenvalues
                    multivariate=True  # Multivariate DTW
                )
                
                elapsed = time.time() - start_time
                
                # Extract the combined similarity score directly
                similarity = result['similarity_score']  # This is the combined similarity score
                
                # For display purposes, also get the raw DTW distance if available
                if 'dtw_comparison' in result and result['dtw_comparison']:
                    dtw_distance = result['dtw_comparison'].get('normalized_distance', similarity)
                else:
                    dtw_distance = 1.0 - similarity  # Approximate distance for display
                
                # Store symmetric results - using DTW distance for distance matrix, similarity for similarity matrix
                distance_matrix[i, j] = dtw_distance
                distance_matrix[j, i] = dtw_distance
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                
                # Store detailed results
                pairwise_details[f"{model_name_i}_vs_{model_name_j}"] = {
                    'dtw_distance': dtw_distance,
                    'similarity': similarity,
                    'time_elapsed': elapsed,
                    'result_details': result
                }
                
                print(f"       ✅ Distance: {dtw_distance:.4f}, Similarity: {similarity:.4f} ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"       ❌ Comparison failed: {e}")
                
                # Set default values for failed comparisons
                distance_matrix[i, j] = 1.0  # Maximum distance
                distance_matrix[j, i] = 1.0
                similarity_matrix[i, j] = 0.0  # Minimum similarity
                similarity_matrix[j, i] = 0.0
                
                pairwise_details[f"{model_name_i}_vs_{model_name_j}"] = {
                    'dtw_distance': 1.0,
                    'similarity': 0.0,
                    'time_elapsed': 0.0,
                    'error': str(e)
                }
    
    # Phase 3: Compute similarity rankings
    print("\n   Phase 3: Computing similarity rankings...")
    similarity_rankings = []
    
    for i, model_name in enumerate(model_names):
        # Get similarities for this model (excluding self)
        model_similarities = []
        for j, other_model in enumerate(model_names):
            if i != j:
                model_similarities.append({
                    'model_name': other_model,
                    'model_index': j,
                    'similarity': similarity_matrix[i, j],
                    'distance': distance_matrix[i, j]
                })
        
        # Sort by similarity (descending)
        model_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        similarity_rankings.append({
            'model_name': model_name,
            'model_index': i,
            'most_similar': model_similarities
        })
    
    # Phase 4: Clustering analysis (optional, if sklearn available)
    cluster_analysis = {}
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        if n_models >= 3:  # Need at least 3 models for meaningful clustering
            n_clusters = min(3, n_models - 1)  # Reasonable number of clusters
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(distance_matrix)
            
            if len(set(cluster_labels)) > 1:  # Only compute if we have multiple clusters
                silhouette_avg = silhouette_score(distance_matrix, cluster_labels)
                
                # Group models by cluster
                cluster_assignments = {}
                for i, label in enumerate(cluster_labels):
                    cluster_name = f"Cluster_{label}"
                    if cluster_name not in cluster_assignments:
                        cluster_assignments[cluster_name] = []
                    cluster_assignments[cluster_name].append(i)
                
                cluster_analysis = {
                    'status': 'success',
                    'n_clusters': n_clusters,
                    'labels': cluster_labels.tolist(),
                    'silhouette_score': silhouette_avg,
                    'cluster_assignments': cluster_assignments
                }
                
                print(f"       ✅ Clustering: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
            else:
                cluster_analysis = {'status': 'single_cluster'}
        else:
            cluster_analysis = {'status': 'insufficient_models'}
            
    except ImportError:
        cluster_analysis = {'status': 'sklearn_not_available'}
        print("       ⚠️  sklearn not available, skipping clustering")
    except Exception as e:
        cluster_analysis = {'status': 'clustering_failed', 'error': str(e)}
        print(f"       ❌ Clustering failed: {e}")
    
    return {
        'model_names': model_names,
        'distance_matrix': distance_matrix,
        'similarity_matrix': similarity_matrix,
        'individual_analyses': individual_analyses,
        'pairwise_details': pairwise_details,
        'similarity_rankings': similarity_rankings,
        'cluster_analysis': cluster_analysis,
        'method': 'multivariate_dtw',
        'comparison_metadata': {
            'data_shape': list(data.shape),
            'device': 'cpu',
            'eigenvalue_index': None,
            'multivariate': True,
            'total_comparisons': total_comparisons
        }
    }


def create_multivariate_dtw_dashboard(results: Dict[str, Any], 
                                     model_configs: List[Dict[str, Any]]) -> go.Figure:
    """Create comprehensive interactive dashboard for multivariate DTW analysis."""
    print("\n📊 Creating multivariate DTW dashboard...")
    
    model_names = results['model_names']
    distance_matrix = results['distance_matrix']
    similarity_matrix = results['similarity_matrix']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "🔥 Similarity Matrix (Multivariate DTW)",
            "🏗️ Model Architecture Distribution", 
            "🎯 Clustering Analysis",
            "📈 Top Similarity Pairs"
        ],
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # 1. Similarity Matrix Heatmap (inverted colorscale for similarity)
    similarity_heatmap = go.Heatmap(
        z=similarity_matrix,
        x=model_names,
        y=model_names,
        colorscale='RdYlBu',  # Red for low similarity, Blue for high similarity
        text=similarity_matrix,
        texttemplate="%{text:.3f}",
        colorbar=dict(title="Similarity Score", x=0.46),
        zmin=0,
        zmax=1
    )
    fig.add_trace(similarity_heatmap, row=1, col=1)
    
    # 2. Model Architecture Distribution
    architectures = [config['architecture'] for config in model_configs]
    arch_counts = pd.Series(architectures).value_counts()
    
    pie = go.Pie(
        labels=arch_counts.index,
        values=arch_counts.values,
        name="Architecture Distribution",
        marker_colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    )
    fig.add_trace(pie, row=1, col=2)
    
    # 3. Clustering Analysis
    if 'cluster_analysis' in results and results['cluster_analysis'].get('status') == 'success':
        cluster_info = results['cluster_analysis']
        cluster_labels = cluster_info.get('labels', [])
        
        scatter = go.Scatter(
            x=list(range(len(model_names))),
            y=cluster_labels,
            mode='markers+text',
            text=model_names,
            textposition='top center',
            marker=dict(
                size=15,
                color=cluster_labels,
                colorscale='viridis',
                line=dict(width=2, color='white')
            ),
            name="Cluster Assignment"
        )
        fig.add_trace(scatter, row=2, col=1)
    else:
        # Add placeholder text
        fig.add_annotation(
            text="Clustering analysis not available<br>(requires sklearn and ≥3 models)",
            x=0.5, y=0.5,
            xref="x3", yref="y3",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    # 4. Top Similarity Pairs
    if 'similarity_rankings' in results:
        # Extract top similarity pairs
        top_pairs = []
        for ranking in results['similarity_rankings']:
            model_name = ranking['model_name']
            most_similar = ranking['most_similar']
            if most_similar:
                top_similar = most_similar[0]  # Most similar
                top_pairs.append({
                    'pair': f"{model_name} ↔ {top_similar['model_name']}",
                    'similarity': top_similar['similarity']
                })
        
        # Sort by similarity and take top pairs
        top_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        top_pairs = top_pairs[:min(10, len(top_pairs))]  # Top 10
        
        if top_pairs:
            bar = go.Bar(
                x=[pair['similarity'] for pair in top_pairs],
                y=[pair['pair'] for pair in top_pairs],
                orientation='h',
                marker_color='lightblue',
                text=[f"{pair['similarity']:.3f}" for pair in top_pairs],
                textposition='auto'
            )
            fig.add_trace(bar, row=2, col=2)
        else:
            fig.add_annotation(
                text="No similarity pairs to display",
                x=0.5, y=0.5,
                xref="x4", yref="y4",
                showarrow=False,
                font=dict(size=12)
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "🧠 Multivariate DTW Neural Network Similarity Analysis Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22}
        },
        height=900,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model Index", row=2, col=1)
    fig.update_yaxes(title_text="Cluster ID", row=2, col=1)
    fig.update_xaxes(title_text="Similarity Score", row=2, col=2)
    fig.update_yaxes(title_text="Model Pair", row=2, col=2)
    
    return fig


def create_detailed_analysis_report(results: Dict[str, Any], 
                                  model_configs: List[Dict[str, Any]]) -> str:
    """Create detailed multivariate DTW analysis report."""
    print("\n📋 Creating detailed analysis report...")
    
    report = []
    report.append("# Multivariate DTW Neural Network Similarity Analysis Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Analysis Overview")
    report.append("")
    report.append("This report presents the results of a comprehensive multivariate Dynamic Time Warping (DTW)")
    report.append("analysis comparing functional similarity between neural networks based on their full")
    report.append("eigenvalue evolution during spectral analysis.")
    report.append("")
    
    # Model Summary
    report.append("## Model Summary")
    report.append("")
    
    architectures = {}
    for config in model_configs:
        arch = config['architecture']
        if arch not in architectures:
            architectures[arch] = []
        architectures[arch].append(config)
    
    for arch, models in architectures.items():
        report.append(f"### {arch}")
        report.append(f"**{len(models)} model(s):**")
        report.append("")
        
        for config in models:
            report.append(f"- **{config['name']}**")
            report.append(f"  - File Size: {config['file_size_mb']:.2f} MB")
            if config['epoch'] != "Unknown":
                report.append(f"  - Training Epoch: {config['epoch']}")
            if config['accuracy'] != "Unknown":
                report.append(f"  - Accuracy: {config['accuracy']:.4f}")
            report.append("")
    
    # Analysis Configuration
    report.append("## Analysis Configuration")
    report.append("")
    metadata = results.get('comparison_metadata', {})
    report.append(f"- **Method**: Multivariate DTW (Dynamic Time Warping)")
    report.append(f"- **Eigenvalue Analysis**: Full eigenvalue evolution (all eigenvalues)")
    report.append(f"- **Data Shape**: {metadata.get('data_shape', 'Unknown')}")
    report.append(f"- **Device**: {metadata.get('device', 'Unknown')}")
    report.append(f"- **Total Pairwise Comparisons**: {metadata.get('total_comparisons', 'Unknown')}")
    report.append("")
    
    # Similarity Matrix
    report.append("## Similarity Matrix")
    report.append("")
    report.append("**Multivariate DTW Similarity Scores** (1.0 = identical functional behavior, 0.0 = completely different)")
    report.append("")
    
    model_names = results['model_names']
    similarity_matrix = results['similarity_matrix']
    
    # Create markdown table
    header = "| Model | " + " | ".join([name[:12] + "..." if len(name) > 15 else name for name in model_names]) + " |"
    separator = "|-------|" + "-------|" * len(model_names)
    report.append(header)
    report.append(separator)
    
    for i, name in enumerate(model_names):
        short_name = name[:12] + "..." if len(name) > 15 else name
        row_values = []
        for j in range(len(model_names)):
            if i == j:
                row_values.append("1.000")  # Self-similarity
            else:
                row_values.append(f"{similarity_matrix[i,j]:.3f}")
        row = f"| {short_name} | " + " | ".join(row_values) + " |"
        report.append(row)
    
    report.append("")
    
    # Top Similarity Pairs
    report.append("## Top Similarity Pairs")
    report.append("")
    report.append("**Most functionally similar model pairs based on multivariate DTW:**")
    report.append("")
    
    if 'similarity_rankings' in results:
        # Extract all pairwise similarities
        all_pairs = []
        for i, ranking in enumerate(results['similarity_rankings']):
            model_name = ranking['model_name']
            for similar in ranking['most_similar']:
                # Avoid duplicates by only including pairs where i < j
                sim_idx = similar['model_index']
                if i < sim_idx:
                    all_pairs.append({
                        'model1': model_name,
                        'model2': similar['model_name'],
                        'similarity': similar['similarity'],
                        'distance': similar['distance']
                    })
        
        # Sort by similarity and take top 10
        all_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        top_pairs = all_pairs[:min(10, len(all_pairs))]
        
        for i, pair in enumerate(top_pairs, 1):
            report.append(f"{i}. **{pair['model1']} ↔ {pair['model2']}**")
            report.append(f"   - Similarity: {pair['similarity']:.4f}")
            report.append(f"   - DTW Distance: {pair['distance']:.4f}")
            report.append("")
    
    # Architecture-Based Analysis
    report.append("## Architecture-Based Analysis")
    report.append("")
    
    # Cross-architecture vs same-architecture comparisons
    cross_arch_similarities = []
    same_arch_similarities = []
    
    config_by_name = {config['name']: config for config in model_configs}
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:  # Avoid duplicates
                arch1 = config_by_name.get(name1, {}).get('architecture', 'Unknown')
                arch2 = config_by_name.get(name2, {}).get('architecture', 'Unknown')
                similarity = similarity_matrix[i, j]
                
                if arch1 == arch2:
                    same_arch_similarities.append(similarity)
                else:
                    cross_arch_similarities.append(similarity)
    
    if same_arch_similarities:
        avg_same_arch = np.mean(same_arch_similarities)
        report.append(f"### Same Architecture Comparisons")
        report.append(f"- **Average Similarity**: {avg_same_arch:.4f}")
        report.append(f"- **Count**: {len(same_arch_similarities)} pairs")
        report.append(f"- **Range**: [{min(same_arch_similarities):.4f}, {max(same_arch_similarities):.4f}]")
        report.append("")
    
    if cross_arch_similarities:
        avg_cross_arch = np.mean(cross_arch_similarities)
        report.append(f"### Cross-Architecture Comparisons")
        report.append(f"- **Average Similarity**: {avg_cross_arch:.4f}")
        report.append(f"- **Count**: {len(cross_arch_similarities)} pairs")
        report.append(f"- **Range**: [{min(cross_arch_similarities):.4f}, {max(cross_arch_similarities):.4f}]")
        report.append("")
        
        if same_arch_similarities and cross_arch_similarities:
            if avg_cross_arch > avg_same_arch:
                report.append("**🔍 Insight**: Cross-architecture models show higher functional similarity than same-architecture models!")
                report.append("This suggests that functional behavior is more influenced by training than architecture.")
            else:
                report.append("**🔍 Insight**: Same-architecture models show higher functional similarity than cross-architecture models.")
                report.append("This suggests that architecture plays a significant role in functional behavior.")
            report.append("")
    
    # Clustering Analysis
    if 'cluster_analysis' in results and results['cluster_analysis'].get('status') == 'success':
        cluster_info = results['cluster_analysis']
        report.append("## Clustering Analysis")
        report.append("")
        report.append(f"**Unsupervised clustering based on DTW similarity patterns:**")
        report.append("")
        report.append(f"- **Number of Clusters**: {cluster_info.get('n_clusters', 'Unknown')}")
        report.append(f"- **Silhouette Score**: {cluster_info.get('silhouette_score', 'Unknown'):.4f}")
        report.append("  (Higher scores indicate better-defined clusters)")
        report.append("")
        
        if 'cluster_assignments' in cluster_info:
            report.append("### Cluster Assignments")
            report.append("")
            for cluster_name, member_indices in cluster_info['cluster_assignments'].items():
                member_names = [model_names[i] for i in member_indices]
                report.append(f"**{cluster_name}**: {', '.join(member_names)}")
            report.append("")
    
    # Technical Details
    report.append("## Technical Details")
    report.append("")
    report.append("### Multivariate DTW Analysis")
    report.append("")
    report.append("- **Method**: Dynamic Time Warping applied to full eigenvalue evolution sequences")
    report.append("- **Multivariate**: All eigenvalues considered simultaneously (not just single eigenvalue)")
    report.append("- **Eigenvalue Evolution**: Spectral analysis across filtration parameters")
    report.append("- **Similarity Conversion**: Similarity = 1 - normalized_DTW_distance")
    report.append("- **Architecture Agnostic**: Pure functional similarity ignoring structural differences")
    report.append("")
    
    report.append("### Interpretation Guidelines")
    report.append("")
    report.append("- **High Similarity (>0.8)**: Models exhibit very similar functional behavior patterns")
    report.append("- **Moderate Similarity (0.4-0.8)**: Models share some functional characteristics")
    report.append("- **Low Similarity (<0.4)**: Models have distinctly different functional behaviors")
    report.append("- **Cross-Architecture**: DTW enables fair comparison across different architectures")
    report.append("")
    
    # Analysis Summary
    report.append("## Summary")
    report.append("")
    if similarity_matrix.size > 0:
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        max_similarity = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        min_similarity = np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        
        report.append(f"- **Overall Average Similarity**: {avg_similarity:.4f}")
        report.append(f"- **Maximum Similarity**: {max_similarity:.4f}")
        report.append(f"- **Minimum Similarity**: {min_similarity:.4f}")
        report.append(f"- **Models Analyzed**: {len(model_names)}")
        report.append(f"- **Successful Comparisons**: {metadata.get('total_comparisons', 'Unknown')}")
        
        if avg_similarity > 0.7:
            report.append("\n**Overall Assessment**: Models show high functional similarity despite architectural differences.")
        elif avg_similarity > 0.4:
            report.append("\n**Overall Assessment**: Models show moderate functional similarity with some shared patterns.")
        else:
            report.append("\n**Overall Assessment**: Models show diverse functional behaviors with limited similarity.")
    
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
    
    # 1. Save interactive dashboard
    dashboard_fig = create_multivariate_dtw_dashboard(results, model_configs)
    dashboard_path = output_dir / f"multivariate_dtw_dashboard_{timestamp}.html"
    dashboard_fig.write_html(str(dashboard_path))
    print(f"   ✅ Dashboard: {dashboard_path.name}")
    
    # 2. Save detailed report
    report = create_detailed_analysis_report(results, model_configs)
    report_path = output_dir / f"multivariate_dtw_report_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   ✅ Report: {report_path.name}")
    
    # 3. Save raw results as JSON
    json_results = {
        'model_names': results['model_names'],
        'distance_matrix': results['distance_matrix'].tolist(),
        'similarity_matrix': results['similarity_matrix'].tolist(),
        'method': results.get('method', 'multivariate_dtw'),
        'timestamp': timestamp,
        'model_configs': [
            {k: v for k, v in config.items() if k not in ['model_class', 'path']}
            for config in model_configs
        ],
        'comparison_metadata': results.get('comparison_metadata', {})
    }
    
    if 'similarity_rankings' in results:
        json_results['similarity_rankings'] = results['similarity_rankings']
    
    if 'cluster_analysis' in results:
        cluster_info = results['cluster_analysis']
        if cluster_info.get('status') == 'success':
            json_results['cluster_analysis'] = {
                'n_clusters': cluster_info.get('n_clusters'),
                'silhouette_score': cluster_info.get('silhouette_score'),
                'cluster_assignments': cluster_info.get('cluster_assignments')
            }
    
    json_path = output_dir / f"multivariate_dtw_data_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   ✅ Data: {json_path.name}")
    
    # 4. Create standalone similarity matrix visualization
    fig_similarity = go.Figure(data=go.Heatmap(
        z=results['similarity_matrix'],
        x=results['model_names'],
        y=results['model_names'],
        colorscale='RdYlBu',
        text=results['similarity_matrix'],
        texttemplate="%{text:.3f}",
        colorbar=dict(title="Similarity Score"),
        zmin=0,
        zmax=1
    ))
    
    fig_similarity.update_layout(
        title="Multivariate DTW Neural Network Similarity Matrix",
        xaxis_title="Model",
        yaxis_title="Model",
        template='plotly_white'
    )
    
    similarity_path = output_dir / f"multivariate_dtw_similarity_matrix_{timestamp}.html"
    fig_similarity.write_html(str(similarity_path))
    print(f"   ✅ Similarity Matrix: {similarity_path.name}")


def main():
    """Main execution function."""
    print("🚀 Multivariate DTW Neural Network Similarity Analysis")
    print("=" * 70)
    
    # Setup
    setup_environment()
    
    # Configuration
    models_dir = Path("models")
    output_dir = Path("multivariate_dtw_results")
    
    try:
        # Step 1: Discover models
        model_configs = discover_models(models_dir)
        if not model_configs:
            print("❌ No models found! Please ensure models are in the 'models/' directory.")
            return 1
        
        print(f"\n📋 Found {len(model_configs)} models:")
        for config in model_configs:
            print(f"   • {config['name']} ({config['architecture']})")
        
        # Step 2: Load models
        loaded_models = load_all_models(model_configs)
        if not loaded_models:
            print("❌ No models could be loaded!")
            return 1
        
        print(f"\n✅ Successfully loaded {len(loaded_models)} models")
        
        # Step 3: Generate test data
        test_data = generate_test_data(input_dim=3, batch_size=150)
        
        # Step 4: Perform multivariate DTW analysis
        results = perform_pairwise_multivariate_dtw_analysis(loaded_models, test_data)
        
        # Step 5: Save results
        save_results(results, model_configs, output_dir)
        
        # Step 6: Print summary
        print("\n📊 Analysis Summary:")
        print(f"   • Models analyzed: {len(loaded_models)}")
        print(f"   • Method: {results.get('method', 'Unknown')}")
        print(f"   • Similarity matrix shape: {results['similarity_matrix'].shape}")
        print(f"   • Pairwise comparisons: {results['comparison_metadata'].get('total_comparisons', 'Unknown')}")
        
        # Print architecture breakdown
        architectures = {}
        for config in model_configs:
            arch = config['architecture']
            architectures[arch] = architectures.get(arch, 0) + 1
        
        print(f"\n🏗️  Architecture Breakdown:")
        for arch, count in architectures.items():
            print(f"   • {arch}: {count} model(s)")
        
        # Print top similarities
        if 'similarity_rankings' in results:
            print("\n🔍 Top Functional Similarities:")
            
            # Extract top pairs
            all_pairs = []
            for i, ranking in enumerate(results['similarity_rankings']):
                model_name = ranking['model_name']
                for similar in ranking['most_similar']:
                    sim_idx = similar['model_index']
                    if i < sim_idx:  # Avoid duplicates
                        all_pairs.append({
                            'model1': model_name,
                            'model2': similar['model_name'],
                            'similarity': similar['similarity']
                        })
            
            all_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            top_pairs = all_pairs[:min(5, len(all_pairs))]
            
            for i, pair in enumerate(top_pairs, 1):
                print(f"   {i}. {pair['model1']} ↔ {pair['model2']}: {pair['similarity']:.4f}")
        
        # Analysis insights
        if results['similarity_matrix'].size > 0:
            avg_similarity = np.mean(results['similarity_matrix'][np.triu_indices_from(results['similarity_matrix'], k=1)])
            print(f"\n🧠 Overall Average Similarity: {avg_similarity:.4f}")
            
            if avg_similarity > 0.7:
                print("   📈 High functional similarity across models!")
            elif avg_similarity > 0.4:
                print("   📊 Moderate functional similarity detected.")
            else:
                print("   📉 Low functional similarity - diverse behaviors.")
        
        print(f"\n✅ Analysis complete! Results saved to '{output_dir}'")
        print("\n📁 Generated Files:")
        print(f"   • Interactive Dashboard: multivariate_dtw_dashboard_*.html")
        print(f"   • Detailed Report: multivariate_dtw_report_*.md")
        print(f"   • Raw Data: multivariate_dtw_data_*.json")
        print(f"   • Similarity Matrix: multivariate_dtw_similarity_matrix_*.html")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())