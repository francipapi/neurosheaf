"""Enhanced example script showing how to load and process saved models using neurosheaf.

This script demonstrates:
1. Auto-detection of model formats (TorchScript, torch.export, ONNX, state_dict)
2. Loading models without needing class definitions (when possible)
3. Converting models to different formats
4. Processing through the neurosheaf pipeline
5. Comparing different model formats
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Import neurosheaf utilities
from neurosheaf.utils import (
    load_any_model,
    detect_model_format,
    infer_model_architecture,
    convert_to_torchscript,
    convert_to_torch_export,
    convert_to_onnx,
    save_model_optimally
)
from neurosheaf.sheaf import SheafBuilder, build_sheaf_laplacian
from neurosheaf.spectral import PersistentSpectralAnalyzer


# Example model architectures (only needed for state_dict format)
class CustomModel(nn.Module):
    """Example custom model architecture."""
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
        x = x.view(x.size(0), -1)
        return self.features(x)


class MLPModel(nn.Module):
    """Example MLP model architecture."""
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
        x = x.view(x.size(0), -1)
        return self.model(x)


def demonstrate_model_conversion(model_class, model_name="TestModel"):
    """Demonstrate converting a model to different formats."""
    
    print(f"\\n{'='*60}")
    print(f"Model Format Conversion Demo - {model_name}")
    print(f"{'='*60}")
    
    # Create a model instance
    model = model_class()
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 32, 32)
    
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Convert to all formats
    print("\\nConverting to multiple formats...")
    try:
        saved_paths = save_model_optimally(
            model, 
            example_input, 
            f"demo_{model_name.lower()}", 
            formats=['torchscript', 'torch_export', 'onnx']
        )
        
        print(f"Saved formats: {list(saved_paths.keys())}")
        
        # Test loading from each format
        for format_name, path in saved_paths.items():
            print(f"\\nTesting {format_name} format...")
            try:
                loaded_model = load_any_model(path)
                
                # Test inference
                with torch.no_grad():
                    output = loaded_model(example_input)
                    print(f"  - Loaded successfully, output shape: {output.shape}")
                    
            except Exception as e:
                print(f"  - Error loading {format_name}: {e}")
        
        return saved_paths
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return {}


def analyze_model_with_auto_detection(model_path, expected_model_class=None):
    """Analyze a model using auto-detection."""
    
    model_path = Path(model_path)
    
    print(f"\\n{'='*60}")
    print(f"Auto-Detection Analysis - {model_path.name}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return None
    
    # Detect format
    print("\\n1. Detecting model format...")
    try:
        format_type = detect_model_format(model_path)
        print(f"   - Detected format: {format_type}")
    except Exception as e:
        print(f"   - Error detecting format: {e}")
        return None
    
    # Load model
    print("\\n2. Loading model...")
    try:
        if format_type == 'state_dict' and expected_model_class:
            # For state_dict format, we need the class
            model = load_any_model(model_path, model_class=expected_model_class)
        else:
            # For other formats, no class needed
            model = load_any_model(model_path)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   - Model loaded successfully")
        print(f"   - Parameters: {param_count:,}")
        
    except Exception as e:
        print(f"   - Error loading model: {e}")
        if format_type == 'state_dict':
            print("   - For state_dict format, you need to provide the model class")
        return None
    
    # Analyze with neurosheaf
    print("\\n3. Analyzing with neurosheaf...")
    try:
        # Generate sample data
        batch_size = 50
        data = torch.randn(batch_size, 3, 32, 32)
        
        # Build sheaf
        builder = SheafBuilder()
        sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=True)
        
        print(f"   - Sheaf built: {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
        
        # Build Laplacian
        laplacian, metadata = build_sheaf_laplacian(sheaf, validate=True)
        print(f"   - Laplacian: {laplacian.shape}, {laplacian.nnz} non-zeros")
        
        # Spectral analysis
        analyzer = PersistentSpectralAnalyzer(
            default_n_steps=10,
            default_filtration_type='threshold'
        )
        
        results = analyzer.analyze(
            sheaf,
            filtration_type='threshold',
            n_steps=15,
            param_range=(0.0, 3.0)
        )
        
        print(f"   - Spectral analysis complete:")
        print(f"     * Birth events: {results['features']['num_birth_events']}")
        print(f"     * Death events: {results['features']['num_death_events']}")
        print(f"     * Persistent paths: {results['features']['num_persistent_paths']}")
        
        return model, sheaf, results
        
    except Exception as e:
        print(f"   - Error in neurosheaf analysis: {e}")
        return model, None, None


def main():
    """Main function demonstrating enhanced model loading capabilities."""
    
    print("Enhanced Model Loading Demo for Neurosheaf")
    print("="*60)
    
    # 1. Model format conversion demo
    print("\\n🔄 PART 1: Model Format Conversion Demo")
    custom_paths = demonstrate_model_conversion(CustomModel, "CustomModel")
    mlp_paths = demonstrate_model_conversion(MLPModel, "MLPModel")
    
    # 2. Auto-detection with converted models
    print("\\n\\n🔍 PART 2: Auto-Detection with Converted Models")
    
    results = {}
    
    # Test different formats
    for model_name, paths in [("CustomModel", custom_paths), ("MLPModel", mlp_paths)]:
        for format_name, path in paths.items():
            print(f"\\n--- Testing {model_name} in {format_name} format ---")
            result = analyze_model_with_auto_detection(path)
            if result:
                results[f"{model_name}_{format_name}"] = result
    
    # 3. Analyze your original saved models
    print("\\n\\n📁 PART 3: Analyzing Your Original Saved Models")
    
    models_dir = Path("models")
    original_models = [
        ("torch_custom_acc_1.0000_epoch_200.pth", CustomModel),
        ("torch_mlp_acc_1.0000_epoch_200.pth", MLPModel)
    ]
    
    for model_file, model_class in original_models:
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"\\n--- Analyzing {model_file} ---")
            
            # First, analyze the architecture
            try:
                arch_info = infer_model_architecture(model_path)
                print(f"Architecture info:")
                print(f"  - Parameters: {arch_info['total_parameters']:,}")
                print(f"  - Layers: {arch_info['num_layers']}")
                
                # Try to load and analyze
                result = analyze_model_with_auto_detection(model_path, model_class)
                if result:
                    results[f"original_{model_file}"] = result
                    
            except Exception as e:
                print(f"Error analyzing {model_file}: {e}")
        else:
            print(f"\\n--- {model_file} not found ---")
    
    # 4. Create comparison visualization
    if len(results) > 0:
        print("\\n\\n📊 PART 4: Results Comparison")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, (model, sheaf, analysis)) in enumerate(results.items()):
            if idx >= 4 or analysis is None:
                continue
                
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
            ax.set_title(f'{name}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(results), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Model Format Comparison - Persistence Diagrams', fontsize=14)
        plt.tight_layout()
        plt.savefig('enhanced_model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\\nSaved comparison visualization to enhanced_model_comparison.png")
        plt.show()
    
    # 5. Summary and recommendations
    print("\\n\\n✅ PART 5: Summary and Recommendations")
    print("="*60)
    
    print("\\n🎯 Key Findings:")
    print("1. TorchScript models load without needing class definitions")
    print("2. torch.export provides modern PyTorch 2.0+ deployment")
    print("3. ONNX enables cross-framework compatibility")
    print("4. State_dict format still requires original class definitions")
    
    print("\\n📋 Recommendations:")
    print("1. For deployment: Use TorchScript or torch.export formats")
    print("2. For collaboration: Convert state_dict models to TorchScript")
    print("3. For cross-framework: Use ONNX format")
    print("4. For development: Keep state_dict for debugging")
    
    print("\\n🔧 Usage Examples:")
    print("```python")
    print("# Simple auto-loading (works with all formats)")
    print("from neurosheaf.utils import load_any_model")
    print("model = load_any_model('path/to/model.pt')  # Auto-detects format")
    print("")
    print("# Convert existing model to TorchScript")
    print("from neurosheaf.utils import convert_to_torchscript")
    print("convert_to_torchscript(model, example_input, 'model.pt')")
    print("")
    print("# Save in multiple formats")
    print("from neurosheaf.utils import save_model_optimally")
    print("save_model_optimally(model, example_input, 'my_model')")
    print("```")
    
    print("\\n✨ Enhanced model loading complete!")


if __name__ == "__main__":
    main()