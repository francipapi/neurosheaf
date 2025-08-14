#!/usr/bin/env python3
"""
Minimal test for normalized Laplacian and global section tracking of GW sheaves.

This test:
1. Loads models from the models folder
2. Builds GW sheaves with normalized Laplacian
3. Tests global section (H⁰) tracking
4. Creates clear visualizations
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

# Set environment
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Core imports
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.spectral.static_laplacian_unified import UnifiedStaticLaplacian
from neurosheaf.io.config import H0Config
from neurosheaf.visualization.enhanced_factory import EnhancedVisualizationFactory
from neurosheaf.visualization.persistence import PersistenceVisualizer
from neurosheaf.utils.simple_model_loader import load_model

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# Model Architectures (copied from test_all.py)
# ============================================================================

class MLPModel(nn.Module):
    """MLP model architecture matching saved models."""
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
        
        # Get activation functions
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        # Build network
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
            'none': nn.Identity()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CustomModel(nn.Module):
    """Model class that matches the actual saved weights structure with Conv1D layers."""
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(3, 32),                                    # layers.0
            nn.ReLU(),                                           # layers.1 (activation)
            nn.Linear(32, 32),                                   # layers.2
            nn.ReLU(),                                           # layers.3 (activation)
            nn.Dropout(0.0),                                     # layers.4 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.5
            nn.ReLU(),                                           # layers.6 (activation)
            nn.Dropout(0.0),                                     # layers.7 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.8
            nn.ReLU(),                                           # layers.9 (activation)
            nn.Dropout(0.0),                                     # layers.10 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.11
            nn.ReLU(),                                           # layers.12 (activation)
            nn.Dropout(0.0),                                     # layers.13 (dropout)
            nn.Linear(32, 1),                                    # layers.14
            nn.Sigmoid()                                         # layers.15 (activation)
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


# ============================================================================
# Test Functions
# ============================================================================

def load_model_from_path(model_class, model_path: str) -> nn.Module:
    """Load a saved model using the utils loader."""
    return load_model(model_class, model_path, device='cpu')


def test_normalized_laplacian_bounds_detailed(sheaf, gw_builder):
    """Test normalized Laplacian eigenvalues with detailed analysis."""
    active_edges = list(sheaf.restrictions.keys())
    print(f"Testing with {len(active_edges)} active edges")
    
    if len(active_edges) == 0:
        print("❌ No active edges found in sheaf")
        return None
    
    # Determine appropriate number of eigenvalues to compute
    # Get the first stalk to estimate problem size
    first_stalk = next(iter(sheaf.stalks.values()))
    stalk_dims = [stalk.shape[0] for stalk in sheaf.stalks.values()]
    total_dim = sum(stalk_dims)
    k = min(30, total_dim // 2, 100)  # Reasonable number of eigenvalues
    
    print(f"Sheaf structure: {len(sheaf.stalks)} stalks, total dimension: {total_dim}")
    print(f"Computing {k} eigenvalues...")
    
    try:
        # Compute eigenvalues using generalized solver
        eigenvals, eigenvecs = gw_builder.solve_generalized_robust(
            sheaf, active_edges, k=k, use_matrix_free=False
        )
        
        if eigenvals is None or len(eigenvals) == 0:
            print("❌ No eigenvalues computed")
            return None
        
        # Detailed eigenvalue analysis
        min_eig = np.min(eigenvals)
        max_eig = np.max(eigenvals)
        
        print(f"\n✅ Eigenvalue Computation Results:")
        print(f"  Number computed: {len(eigenvals)}")
        print(f"  Range: [{min_eig:.6e}, {max_eig:.6e}]")
        print(f"  First 10: {eigenvals[:10]}")
        
        # Count eigenvalues in different ranges
        zero_count = np.sum(eigenvals < 1e-10)
        small_count = np.sum((eigenvals >= 1e-10) & (eigenvals < 1e-6))
        medium_count = np.sum((eigenvals >= 1e-6) & (eigenvals < 0.1))
        large_count = np.sum(eigenvals >= 0.1)
        
        print(f"\n  Eigenvalue Distribution:")
        print(f"    Near zero (< 1e-10): {zero_count}")
        print(f"    Small (1e-10 to 1e-6): {small_count}")  
        print(f"    Medium (1e-6 to 0.1): {medium_count}")
        print(f"    Large (> 0.1): {large_count}")
        
        # Verify normalized Laplacian bounds [0, 2]
        negative_count = np.sum(eigenvals < -1e-10)
        above_two_count = np.sum(eigenvals > 2.0 + 1e-10)
        
        if negative_count > 0:
            print(f"⚠️  {negative_count} significantly negative eigenvalues (numerical issue)")
        if above_two_count > 0:
            print(f"❌ {above_two_count} eigenvalues above 2 (not normalized Laplacian)")
        
        if negative_count == 0 and above_two_count == 0:
            print("✅ All eigenvalues within normalized Laplacian bounds [0, 2]")
        
        # Test eigenvalue classification
        classification = gw_builder.classify_eigenvalues_downstream(
            eigenvals, is_normalized=True
        )
        
        print(f"\n  Advanced Classification:")
        print(f"    Numerical errors: {np.sum(classification['numerical_zeros'])}")
        print(f"    True zeros: {np.sum(classification['true_zeros'])}")
        print(f"    Positive spectrum: {np.sum(classification['positive_spectrum'])}")
        print(f"    Effective threshold: {classification['effective_threshold']:.2e}")
        
        return eigenvals
        
    except Exception as e:
        print(f"❌ Eigenvalue computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_global_section_tracking_direct(sheaf, n_steps=20):
    """Test H⁰ global section tracking using direct PersistentSpectralAnalyzer."""
    print("\n" + "="*60)
    print("TEST 2: Global Section (H⁰) Tracking - Direct")
    print("="*60)
    
    # Create H⁰ configuration for normalized Laplacian
    h0_config = H0Config.for_normalized_laplacian()
    print(f"H⁰ Config: c_in={h0_config.c_in}, gap={h0_config.gap}")
    
    # Create static Laplacian with normalized support
    static_laplacian = UnifiedStaticLaplacian(
        use_generalized_normalization=True,
        enable_caching=True
    )
    
    spectral_analyzer = PersistentSpectralAnalyzer(
        static_laplacian=static_laplacian,
        default_n_steps=n_steps
    )
    
    print(f"Running spectral analysis with {n_steps} filtration steps...")
    
    result = spectral_analyzer.analyze(
        sheaf,
        filtration_type='threshold',
        n_steps=n_steps,
        h0_config=h0_config
    )
    
    # Display results
    features = result['features']
    print(f"\nGlobal Section Tracking Results:")
    print(f"  Birth events: {features.get('num_birth_events', 0)}")
    print(f"  Death events: {features.get('num_death_events', 0)}")
    print(f"  Persistent paths: {features.get('num_persistent_paths', 0)}")
    print(f"  Max Betti number (β₀): {features.get('max_betti_number', 0)}")
    print(f"  Mean lifetime: {features.get('mean_lifetime', 0):.6f}")
    
    # Check if H⁰ tracking was used
    if 'h0_result' in result.get('persistence_result', {}):
        h0_result = result['persistence_result']['h0_result']
        print(f"\n✅ H⁰ persistence tracking was used")
        print(f"  Total intervals: {len(h0_result.intervals)}")
        print(f"  Total computation time: {h0_result.total_time:.3f}s")
    else:
        print("\n⚠️  Standard eigenvalue tracking was used (not H⁰)")
    
    return result


def create_visualizations(sheaf, analysis_result, model_name):
    """Create and save visualizations."""
    print("\n" + "="*60)
    print("TEST 3: Creating Visualizations")
    print("="*60)
    
    # Create visualization factory
    viz_factory = EnhancedVisualizationFactory()
    persistence_viz = PersistenceVisualizer()
    
    # Extract data
    eigenvalue_sequences = analysis_result['persistence_result'].get('eigenvalue_sequences', [])
    filtration_params = analysis_result['filtration_params']
    diagrams = analysis_result['diagrams']
    
    # 1. Create persistence diagram
    print("Creating persistence diagram...")
    persistence_fig = persistence_viz.create_persistence_diagram(
        diagrams,
        title=f"Persistence Diagram - {model_name}",
        show_infinite=True
    )
    persistence_fig.write_html(f"persistence_diagram_{model_name}.html")
    print(f"  Saved: persistence_diagram_{model_name}.html")
    
    # 2. Create persistence barcode
    print("Creating persistence barcode...")
    barcode_fig = persistence_viz.create_persistence_barcode(
        diagrams,
        filtration_params,
        title=f"Persistence Barcode - {model_name}"
    )
    barcode_fig.write_html(f"persistence_barcode_{model_name}.html")
    print(f"  Saved: persistence_barcode_{model_name}.html")
    
    # 3. Create eigenvalue evolution (if available)
    if eigenvalue_sequences:
        print("Creating eigenvalue evolution...")
        import plotly.graph_objects as go
        
        fig = go.Figure()
        n_eigenvalues = min(10, len(eigenvalue_sequences[0]))
        
        for i in range(n_eigenvalues):
            eigenval_evolution = [seq[i].item() if i < len(seq) else 0 
                                 for seq in eigenvalue_sequences]
            fig.add_trace(go.Scatter(
                x=filtration_params,
                y=eigenval_evolution,
                mode='lines+markers',
                name=f'λ_{i}',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"Eigenvalue Evolution - {model_name}",
            xaxis_title="Filtration Parameter",
            yaxis_title="Eigenvalue",
            yaxis_type="log",
            height=500,
            showlegend=True
        )
        
        fig.write_html(f"eigenvalue_evolution_{model_name}.html")
        print(f"  Saved: eigenvalue_evolution_{model_name}.html")
    
    # 4. Print summary statistics
    stats = persistence_viz.create_summary_stats(diagrams)
    print(f"\nPersistence Summary Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def test_global_section_enhanced(sheaf, n_steps=25):
    """Enhanced H⁰ global section tracking with detailed analysis."""
    print(f"Running enhanced global section analysis with {n_steps} filtration steps...")
    
    # Create H⁰ configuration for normalized Laplacian
    h0_config = H0Config.for_normalized_laplacian()
    print(f"H⁰ Config: c_in={h0_config.c_in}, gap={h0_config.gap}")
    
    # Create static Laplacian with normalized support
    static_laplacian = UnifiedStaticLaplacian(
        use_generalized_normalization=True,
        enable_caching=True
    )
    
    spectral_analyzer = PersistentSpectralAnalyzer(
        static_laplacian=static_laplacian,
        default_n_steps=n_steps
    )
    
    print(f"Running spectral analysis with {n_steps} filtration steps...")
    
    try:
        result = spectral_analyzer.analyze(
            sheaf,
            filtration_type='threshold',
            n_steps=n_steps,
            h0_config=h0_config
        )
        
        # Extract detailed results
        features = result['features']
        persistence_result = result.get('persistence_result', {})
        eigenvalue_sequences = persistence_result.get('eigenvalue_sequences', [])
        filtration_params = result.get('filtration_params', [])
        
        print(f"\n✅ Global Section Tracking Results:")
        print(f"  Filtration steps completed: {len(filtration_params)}")
        print(f"  Birth events: {features.get('num_birth_events', 0)}")
        print(f"  Death events: {features.get('num_death_events', 0)}")
        print(f"  Persistent paths: {features.get('num_persistent_paths', 0)}")
        print(f"  Max Betti number (β₀): {features.get('max_betti_number', 0)}")
        print(f"  Mean lifetime: {features.get('mean_lifetime', 0):.6f}")
        print(f"  Eigenvalue sequences collected: {len(eigenvalue_sequences)}")
        
        # Analyze eigenvalue evolution
        if eigenvalue_sequences:
            print(f"\n  Eigenvalue Evolution Analysis:")
            for i, seq in enumerate(eigenvalue_sequences[:5]):  # Show first 5 filtration steps
                if len(seq) > 0:
                    print(f"    Step {i}: {len(seq)} eigenvalues, range [{np.min(seq):.6e}, {np.max(seq):.6e}]")
        
        # Check if H⁰ tracking was used
        if 'h0_result' in persistence_result:
            h0_result = persistence_result['h0_result']
            print(f"\n✅ H⁰ persistence tracking was used")
            print(f"  Total intervals: {len(h0_result.intervals)}")
            print(f"  Total computation time: {h0_result.total_time:.3f}s")
            
            # Analyze birth/death pattern
            births = [interval.birth for interval in h0_result.intervals]
            deaths = [interval.death for interval in h0_result.intervals if interval.death != float('inf')]
            
            print(f"  Birth parameters: min={np.min(births):.6f}, max={np.max(births):.6f}")
            if deaths:
                print(f"  Death parameters: min={np.min(deaths):.6f}, max={np.max(deaths):.6f}")
                print(f"  Finite intervals: {len(deaths)}/{len(h0_result.intervals)}")
            else:
                print(f"  No finite deaths detected (all intervals persistent)")
                
        else:
            print(f"\n⚠️  Standard eigenvalue tracking was used (not H⁰)")
        
        # Validate we have meaningful results
        has_births = features.get('num_birth_events', 0) > 0
        has_deaths = features.get('num_death_events', 0) > 0
        has_eigenvalue_sequences = len(eigenvalue_sequences) > 0
        
        if has_births and has_deaths and has_eigenvalue_sequences:
            print(f"\n✅ All requirements satisfied: births, deaths, and eigenvalue sequences")
        else:
            print(f"\n⚠️  Missing components: births={has_births}, deaths={has_deaths}, eigenvalues={has_eigenvalue_sequences}")
        
        return result
        
    except Exception as e:
        print(f"❌ Enhanced global section analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_detailed_visualizations(sheaf, analysis_result, model_name, eigenvals):
    """Create detailed visualizations for the analysis results."""
    print(f"Creating detailed visualizations for {model_name}...")
    
    try:
        # Create visualization factory
        viz_factory = EnhancedVisualizationFactory()
        persistence_viz = PersistenceVisualizer()
        
        # Extract data from analysis results
        persistence_result = analysis_result.get('persistence_result', {})
        eigenvalue_sequences = persistence_result.get('eigenvalue_sequences', [])
        filtration_params = analysis_result.get('filtration_params', [])
        diagrams = analysis_result.get('diagrams', [])
        
        print(f"  Data extracted: {len(eigenvalue_sequences)} eigenvalue sequences, {len(diagrams)} diagrams")
        
        # 1. Create persistence diagram
        if diagrams:
            print("  Creating persistence diagram...")
            persistence_fig = persistence_viz.create_persistence_diagram(
                diagrams,
                title=f"Persistence Diagram - {model_name}",
                show_infinite=True
            )
            persistence_fig.write_html(f"persistence_diagram_{model_name}.html")
            print(f"    Saved: persistence_diagram_{model_name}.html")
        
        # 2. Create persistence barcode
        if diagrams and filtration_params:
            print("  Creating persistence barcode...")
            barcode_fig = persistence_viz.create_persistence_barcode(
                diagrams,
                filtration_params,
                title=f"Persistence Barcode - {model_name}"
            )
            barcode_fig.write_html(f"persistence_barcode_{model_name}.html")
            print(f"    Saved: persistence_barcode_{model_name}.html")
        
        # 3. Create eigenvalue evolution plot
        if eigenvalue_sequences and filtration_params:
            print("  Creating eigenvalue evolution...")
            import plotly.graph_objects as go
            
            fig = go.Figure()
            n_eigenvalues = min(10, len(eigenvalue_sequences[0]) if eigenvalue_sequences else 0)
            
            for i in range(n_eigenvalues):
                eigenval_evolution = []
                for seq in eigenvalue_sequences:
                    if i < len(seq):
                        val = seq[i].item() if hasattr(seq[i], 'item') else float(seq[i])
                        eigenval_evolution.append(val)
                    else:
                        eigenval_evolution.append(0)
                
                fig.add_trace(go.Scatter(
                    x=filtration_params[:len(eigenval_evolution)],
                    y=eigenval_evolution,
                    mode='lines+markers',
                    name=f'λ_{i}',
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f"Eigenvalue Evolution - {model_name}",
                xaxis_title="Filtration Parameter",
                yaxis_title="Eigenvalue",
                yaxis_type="log",
                height=500,
                showlegend=True
            )
            
            fig.write_html(f"eigenvalue_evolution_{model_name}.html")
            print(f"    Saved: eigenvalue_evolution_{model_name}.html")
        
        # 4. Create eigenvalue distribution histogram
        if eigenvals is not None and len(eigenvals) > 0:
            print("  Creating eigenvalue distribution...")
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Create histogram
            fig.add_trace(go.Histogram(
                x=eigenvals,
                nbinsx=50,
                name="Eigenvalue Distribution",
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f"Eigenvalue Distribution - {model_name}",
                xaxis_title="Eigenvalue",
                yaxis_title="Count",
                height=400
            )
            
            fig.write_html(f"eigenvalue_distribution_{model_name}.html")
            print(f"    Saved: eigenvalue_distribution_{model_name}.html")
        
        # 5. Create sheaf structure visualization
        print("  Creating sheaf structure visualization...")
        import plotly.graph_objects as go
        
        # Create network graph of sheaf structure
        node_names = list(sheaf.stalks.keys())
        edge_list = list(sheaf.restrictions.keys())
        
        # Node data
        node_dims = [sheaf.stalks[node].shape[0] for node in node_names]
        
        fig = go.Figure()
        
        # Add bar chart of node dimensions
        fig.add_trace(go.Bar(
            x=[str(node) for node in node_names],
            y=node_dims,
            name="Stalk Dimensions",
            text=node_dims,
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Sheaf Structure - {model_name}",
            xaxis_title="Nodes (Layers)",
            yaxis_title="Stalk Dimension",
            height=400
        )
        
        fig.write_html(f"sheaf_structure_{model_name}.html")
        print(f"    Saved: sheaf_structure_{model_name}.html")
        
        # 6. Print summary statistics
        if diagrams:
            stats = persistence_viz.create_summary_stats(diagrams)
            print(f"\n  Persistence Summary Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")
        
        print(f"✅ All visualizations created for {model_name}")
        
    except Exception as e:
        print(f"❌ Visualization creation failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()


def test_single_model(model_path: str, model_class, model_name: str):
    """Test a single model with enhanced numerical stability."""
    print("\n" + "="*100)
    print(f"FOCUSED TEST: {model_name}")
    print("="*100)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = load_model_from_path(model_class, model_path)
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Generate sample data with larger batch size
    batch_size = 100  # Larger batch size as requested
    input_dim = 3
    X = 8*torch.randn(batch_size, input_dim)
    print(f"✅ Generated input data: {X.shape}")
    
    # Improved GW configuration for numerical stability
    print("\nConfiguring GW sheaf construction...")
    from neurosheaf.sheaf.assembly.builder import SheafBuilder
    
    gw_config = GWConfig(
        epsilon=0.05,      # Smaller epsilon for better convergence
        max_iter=1000,     # More iterations for stability
        tolerance=1e-9,    # Tighter tolerance
        use_gpu=False      # CPU for numerical stability
    )
    
    print(f"GW Config: epsilon={gw_config.epsilon}, max_iter={gw_config.max_iter}")
    
    # Build sheaf with exclude_final_single_output for numerical stability
    print("Building GW sheaf with exclude_final_single_output=True...")
    builder = SheafBuilder(
        restriction_method='gromov_wasserstein',
        exclude_final_single_output=True  # KEY: Exclude final layer for stability
    )
    
    try:
        sheaf = builder.build_from_activations(
            model, X, 
            validate=True, 
            gw_config=gw_config
        )
        print(f"✅ Sheaf built successfully: {len(sheaf.stalks)} nodes, {len(sheaf.restrictions)} edges")
        
        # Print sheaf structure details
        print(f"\nSheaf Structure:")
        node_dims = [sheaf.stalks[node].shape[0] for node in sheaf.stalks]
        print(f"  Node dimensions: {node_dims[:5]}{'...' if len(node_dims) > 5 else ''}")
        print(f"  Total stalk dimension: {sum(node_dims)}")
        
    except Exception as e:
        print(f"❌ Sheaf construction failed: {e}")
        return None, None, None
    
    # Create GW Laplacian builder for eigenvalue testing
    gw_builder = GWLaplacianBuilder(
        validate_properties=True,
        enable_caching=True
    )
    
    # Test 1: Normalized Laplacian bounds with detailed output
    print(f"\n" + "="*60)
    print("TEST 1: Normalized Laplacian Eigenvalue Analysis")
    print("="*60)
    
    try:
        eigenvals = test_normalized_laplacian_bounds_detailed(sheaf, gw_builder)
        if eigenvals is not None:
            print(f"✅ Eigenvalue computation successful: {len(eigenvals)} eigenvalues")
        else:
            print("❌ Eigenvalue computation failed")
            return sheaf, None, None
    except Exception as e:
        print(f"❌ Eigenvalue test failed: {e}")
        return sheaf, None, None
    
    # Test 2: Global section tracking with enhanced parameters
    print(f"\n" + "="*60)
    print("TEST 2: Global Section (H⁰) Tracking Analysis")
    print("="*60)
    
    try:
        analysis_result = test_global_section_enhanced(sheaf, n_steps=25)
        if analysis_result:
            print(f"✅ Global section analysis successful")
        else:
            print("❌ Global section analysis failed")
            return sheaf, eigenvals, None
    except Exception as e:
        print(f"❌ Global section test failed: {e}")
        return sheaf, eigenvals, None
    
    # Test 3: Create detailed visualizations
    print(f"\n" + "="*60)
    print("TEST 3: Visualization Generation")
    print("="*60)
    
    try:
        create_detailed_visualizations(sheaf, analysis_result, model_name, eigenvals)
        print(f"✅ Visualizations created successfully")
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
    
    # Clear cache
    gw_builder.clear_cache()
    
    print(f"\n" + "="*100)
    print(f"COMPLETED: {model_name} analysis finished")
    print("="*100)
    
    return sheaf, eigenvals, analysis_result


def main():
    """Main test function - focus on single model at a time."""
    print("="*80)
    print("SINGLE MODEL NORMALIZED LAPLACIAN TEST")
    print("="*80)
    
    # Define model to test (focusing on one at a time as requested)
    model_to_test = {
        'path': 'models/random_mlp_net_000_default_seed_42.pth',
        'class': MLPModel,
        'name': 'random_mlp'
    }
    
    print(f"Testing single model: {model_to_test['name']}")
    
    try:
        sheaf, eigenvals, analysis = test_single_model(
            model_to_test['path'],
            model_to_test['class'],
            model_to_test['name']
        )
        
        if eigenvals is not None and analysis is not None:
            print(f"\n✅ SUCCESS: {model_to_test['name']} analysis completed")
            print(f"   Nodes: {len(sheaf.stalks)}, Edges: {len(sheaf.restrictions)}")
            print(f"   Eigenvalues: {len(eigenvals)} computed")
            print(f"   Range: [{np.min(eigenvals):.6e}, {np.max(eigenvals):.6e}]")
            
            # Check features from analysis
            features = analysis['features']
            print(f"   Birth events: {features.get('num_birth_events', 0)}")
            print(f"   Death events: {features.get('num_death_events', 0)}")
        else:
            print(f"\n❌ FAILED: {model_to_test['name']} analysis failed")
            
    except Exception as e:
        print(f"\n❌ Error testing {model_to_test['name']}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Single model test completed. Check HTML files for visualizations.")
    print("="*80)


if __name__ == "__main__":
    main()