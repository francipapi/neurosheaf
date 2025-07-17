"""Example script showing how to load and process saved models using neurosheaf.

This script demonstrates:
1. Loading models from saved weights
2. Processing them through the neurosheaf pipeline
3. Analyzing the results
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Import neurosheaf utilities
from neurosheaf.utils import load_model, load_checkpoint, infer_model_architecture
from neurosheaf.sheaf import SheafBuilder, build_sheaf_laplacian
from neurosheaf.spectral import PersistentSpectralAnalyzer


# Define your model architectures
# You'll need to match these to the architectures of your saved models

class CustomModel(nn.Module):
    """Example custom model architecture.
    
    Adjust this to match your torch_custom_acc_1.0000_epoch_200.pth model.
    """
    def __init__(self, input_size=3*32*32, hidden_size=256, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.features(x)


class MLPModel(nn.Module):
    """Example MLP model architecture.
    
    Adjust this to match your torch_mlp_acc_1.0000_epoch_200.pth model.
    """
    def __init__(self, input_size=3*32*32, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)


def analyze_saved_model(model_path, model_class, model_name="Model"):
    """Load and analyze a saved model using neurosheaf pipeline."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")
    
    # First, let's infer the architecture
    print("\n1. Inferring model architecture...")
    arch_info = infer_model_architecture(model_path)
    print(f"   - Total parameters: {arch_info['total_parameters']:,}")
    print(f"   - Number of layers: {arch_info['num_layers']}")
    print(f"   - First few layers: {list(arch_info['layer_names'][:5])}")
    
    # Load the model
    print("\n2. Loading model...")
    try:
        model = load_model(model_class, model_path)
        print(f"   - Model loaded successfully")
        print(f"   - Device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"   - Error loading model: {e}")
        print("   - You may need to adjust the model architecture to match your saved weights")
        return None
    
    # Generate sample data
    print("\n3. Generating sample data...")
    batch_size = 50
    data = torch.randn(batch_size, 3, 32, 32)  # CIFAR-10 like data
    print(f"   - Data shape: {data.shape}")
    
    # Build sheaf
    print("\n4. Building sheaf from activations...")
    builder = SheafBuilder()
    sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=True)
    print(f"   - Sheaf constructed: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
    
    # Build Laplacian
    print("\n5. Building sheaf Laplacian...")
    laplacian, metadata = build_sheaf_laplacian(sheaf, validate=True)
    print(f"   - Laplacian shape: {laplacian.shape}")
    print(f"   - Non-zero elements: {laplacian.nnz}")
    
    # Spectral analysis
    print("\n6. Running spectral persistence analysis...")
    analyzer = PersistentSpectralAnalyzer(
        default_n_steps=15,
        default_filtration_type='threshold'
    )
    
    results = analyzer.analyze(
        sheaf,
        filtration_type='threshold',
        n_steps=20,
        param_range=(0.0, 5.0)
    )
    
    print(f"   - Birth events: {results['features']['num_birth_events']}")
    print(f"   - Death events: {results['features']['num_death_events']}")
    print(f"   - Crossing events: {results['features']['num_crossings']}")
    print(f"   - Persistent paths: {results['features']['num_persistent_paths']}")
    
    return model, sheaf, results


def main():
    """Main function to load and analyze saved models."""
    
    # Paths to your saved models
    models_dir = Path("models")
    custom_model_path = models_dir / "torch_custom_acc_1.0000_epoch_200.pth"
    mlp_model_path = models_dir / "torch_mlp_acc_1.0000_epoch_200.pth"
    
    # Check if model files exist
    print("Checking for saved models...")
    print(f"Custom model exists: {custom_model_path.exists()}")
    print(f"MLP model exists: {mlp_model_path.exists()}")
    
    results = {}
    
    # Analyze custom model
    if custom_model_path.exists():
        custom_results = analyze_saved_model(
            custom_model_path, 
            CustomModel, 
            "Custom Model"
        )
        if custom_results:
            results['custom'] = custom_results
    
    # Analyze MLP model
    if mlp_model_path.exists():
        mlp_results = analyze_saved_model(
            mlp_model_path, 
            MLPModel, 
            "MLP Model"
        )
        if mlp_results:
            results['mlp'] = mlp_results
    
    # Create comparison visualization if both models loaded
    if len(results) == 2:
        print("\n\nCreating comparison visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (name, (model, sheaf, analysis)) in enumerate(results.items()):
            ax = axes[idx]
            
            # Plot persistence diagram
            diagrams = analysis['diagrams']
            birth_death_pairs = diagrams['birth_death_pairs']
            
            if birth_death_pairs:
                births = [pair['birth'] for pair in birth_death_pairs]
                deaths = [pair['death'] for pair in birth_death_pairs]
                lifetimes = [pair['lifetime'] for pair in birth_death_pairs]
                
                scatter = ax.scatter(births, deaths, c=lifetimes, cmap='hot', 
                                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
                
                # Plot diagonal
                max_val = max(deaths) if deaths else 1.0
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'{name.capitalize()} Model - Persistence Diagram')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Comparison - Spectral Persistence')
        plt.tight_layout()
        plt.savefig('model_comparison_persistence.png', dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to model_comparison_persistence.png")
        plt.show()
    
    print("\n\nAnalysis complete!")
    
    # If you need to adjust the model architectures:
    print("\n" + "="*60)
    print("NOTE: If the models failed to load, you need to:")
    print("1. Adjust the CustomModel and MLPModel classes to match your saved architectures")
    print("2. Use infer_model_architecture() to get hints about the architecture")
    print("3. Check the original training code to get the exact model definitions")
    

if __name__ == "__main__":
    main()