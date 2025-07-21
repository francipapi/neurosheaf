#!/usr/bin/env python3
"""
Compare results with preserve_eigenvalues=True vs False using the exact same setup as test_all_directed_simple.py
"""
import torch
import torch.nn as nn 
import numpy as np
import os

# Set environment for CPU usage  
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.directed_sheaf import DirectedSheafAdapter

# MLPModel class from test_all_directed_simple.py
class MLPModel(nn.Module):
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
        
        # Use 'layers' as the attribute name to match saved weights
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

def run_analysis(preserve_eigenvalues):
    """Run the exact same analysis as test_all_directed_simple.py with specified preserve_eigenvalues."""
    print(f"\n=== Running with preserve_eigenvalues={preserve_eigenvalues} ===")
    
    # Load model (same as original test)
    mlp_path = "models/torch_mlp_acc_1.0000_epoch_200.pth"
    try:
        model = load_model(MLPModel, mlp_path, device="cpu")
        print(f"✅ Successfully loaded MLP model with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Error loading MLP model: {e}")
        return None
    
    # Generate same data (same as original test)
    batch_size = 50
    data = 8*torch.randn(batch_size, 3)  # Same scaling and size
    
    # Initialize analyzer (same as original test)  
    analyzer = NeurosheafAnalyzer(device='cpu')
    
    # Configure regularization (same as original test)
    directionality_parameter = 0.25
    regularization_config = {
        'strategy': 'moderate',
        'target_condition': 1e8,
        'min_regularization': 1e-10,
        'max_regularization': 1e-4
    }
    
    # Perform analysis (same as original test, just changing preserve_eigenvalues)
    analysis = analyzer.analyze(
        model, 
        data, 
        directed=True, 
        directionality_parameter=directionality_parameter,
        use_gram_regularization=True,
        preserve_eigenvalues=preserve_eigenvalues,  # This is what we're testing
        regularization_config=regularization_config
    )
    
    directed_sheaf = analysis['directed_sheaf']
    
    print(f"Directed sheaf constructed: {len(directed_sheaf.complex_stalks)} complex stalks")
    
    # Check base sheaf properties
    base_sheaf = directed_sheaf.base_sheaf
    eigenvalue_meta = base_sheaf.eigenvalue_metadata.preserve_eigenvalues if base_sheaf.eigenvalue_metadata else None
    print(f"Base sheaf eigenvalue preservation: {eigenvalue_meta}")
    
    # Count stalk types in base sheaf
    identity_count = 0
    eigenvalue_count = 0
    for node_id, stalk in base_sheaf.stalks.items():
        if torch.allclose(stalk, torch.eye(stalk.shape[0]), atol=1e-6):
            identity_count += 1
        else:
            eigenvalue_count += 1
    
    print(f"Base sheaf stalks: {identity_count} identity, {eigenvalue_count} eigenvalue matrices")
    
    # Run spectral analysis (same as original test)
    spectral_analyzer = PersistentSpectralAnalyzer(
        default_n_steps=15,  # Reduced for speed
        default_filtration_type='threshold'
    )
    
    # Create compatibility sheaf for spectral analysis  
    adapter = DirectedSheafAdapter()
    compatibility_sheaf = adapter.create_compatibility_sheaf(directed_sheaf)
    
    # Run spectral analysis
    results = spectral_analyzer.analyze(
        compatibility_sheaf,
        filtration_type='threshold',
        n_steps=15
    )
    
    # Extract eigenvalue sequences
    eigenval_seqs = results['persistence_result']['eigenvalue_sequences']
    if eigenval_seqs and len(eigenval_seqs[0]) > 0:
        first_step = eigenval_seqs[0][:5]
        last_step = eigenval_seqs[-1][:5]
        print(f"First step eigenvalues (first 5): {[f'{x:.6f}' for x in first_step.tolist()]}")
        print(f"Last step eigenvalues (first 5): {[f'{x:.6f}' for x in last_step.tolist()]}")
        return {
            'first_eigenvals': first_step.tolist(),
            'last_eigenvals': last_step.tolist(),
            'identity_stalks': identity_count,
            'eigenvalue_stalks': eigenvalue_count
        }
    else:
        print("No eigenvalue sequences found!")
        return None

def main():
    print("=== Verification: test_all_directed_simple.py with preserve_eigenvalues comparison ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(5670)
    np.random.seed(5670)
    
    # Run both analyses
    results_true = run_analysis(preserve_eigenvalues=True)
    results_false = run_analysis(preserve_eigenvalues=False)
    
    if results_true and results_false:
        print("\n" + "="*70)
        print("VERIFICATION RESULTS")
        print("="*70)
        
        # Compare eigenvalue evolution
        first_diff = np.array(results_true['first_eigenvals']) - np.array(results_false['first_eigenvals'])
        last_diff = np.array(results_true['last_eigenvals']) - np.array(results_false['last_eigenvals'])
        
        print(f"\nEigenvalue evolution differences:")
        print(f"First step: max abs diff = {np.max(np.abs(first_diff)):.8f}")
        print(f"Last step:  max abs diff = {np.max(np.abs(last_diff)):.8f}")
        
        # Compare stalk types
        print(f"\nBase sheaf stalk types:")
        print(f"preserve_eigenvalues=True:  {results_true['identity_stalks']:2d} identity, {results_true['eigenvalue_stalks']:2d} eigenvalue")
        print(f"preserve_eigenvalues=False: {results_false['identity_stalks']:2d} identity, {results_false['eigenvalue_stalks']:2d} eigenvalue")
        
        # Final assessment
        eigenvals_different = np.max(np.abs(first_diff)) > 1e-6 or np.max(np.abs(last_diff)) > 1e-6
        stalks_correct = (results_true['eigenvalue_stalks'] > 0 and results_false['identity_stalks'] > 0)
        
        print(f"\n" + "="*70)
        print("FINAL ASSESSMENT")
        print("="*70)
        
        if eigenvals_different:
            print("🎉 SUCCESS: preserve_eigenvalues flag produces DIFFERENT eigenvalue evolution!")
        else:
            print("⚠️  Eigenvalue evolution appears similar")
            
        if stalks_correct:
            print("🎉 SUCCESS: Base sheaf stalk types are correctly different!")
        else:
            print("❌ Base sheaf stalk types are incorrect")
            
        if eigenvals_different and stalks_correct:
            print("\n✅ COMPLETE SUCCESS: The preserve_eigenvalues fix is working perfectly!")
            print("   test_all_directed_simple.py will now produce different results when")
            print("   you change the preserve_eigenvalues flag!")
        else:
            print("\n❌ Issues remain with the preserve_eigenvalues propagation")
    else:
        print("❌ Could not complete comparison")

if __name__ == "__main__":
    main()