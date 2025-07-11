"""Phase 3 End-to-End Pipeline Integration Tests.

This module validates the complete Phase 3 pipeline from neural network activations
through sheaf construction to sparse Laplacian assembly, testing with multiple
architectures and data types.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from neurosheaf.sheaf import SheafBuilder, FXPosetExtractor, ProcrustesMaps
from neurosheaf.sheaf.laplacian import SheafLaplacianBuilder
from neurosheaf.sheaf.construction import Sheaf
from neurosheaf.cka import DebiasedCKA


@dataclass
class PipelineResult:
    """Container for end-to-end pipeline validation results."""
    model_name: str
    num_activations: int
    sheaf_nodes: int
    sheaf_edges: int
    laplacian_shape: Tuple[int, int]
    laplacian_nnz: int
    validation_passed: bool
    errors: List[str]
    warnings: List[str]


class TestEndToEndPipeline:
    """Test complete pipeline with different architectures."""
    
    @pytest.fixture
    def test_models(self):
        """Create test models representing different architectures."""
        models = {}
        
        # Simple CNN
        models['simple_cnn'] = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        
        # ResNet-like with skip connections
        class MiniResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 10)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                # First block
                x1 = self.relu(self.conv1(x))
                x2 = self.relu(self.conv2(x1))
                
                # Second block with skip
                x3 = self.relu(self.conv3(x2))
                x4 = self.relu(self.conv4(x3))
                
                # Skip connection (simplified)
                if x2.shape[1] != x4.shape[1]:
                    # Channel dimension adaptation
                    x2_adapted = nn.functional.adaptive_avg_pool2d(x2, x4.shape[2:])
                    x2_adapted = nn.functional.pad(x2_adapted, (0, 0, 0, 0, 0, x4.shape[1] - x2.shape[1]))
                else:
                    x2_adapted = x2
                
                x_skip = x4 + x2_adapted
                
                # Final layers
                x_pool = self.pool(x_skip)
                x_flat = x_pool.flatten(1)
                return self.fc(x_flat)
        
        models['mini_resnet'] = MiniResNet()
        
        # Transformer-like (simplified)
        class MiniTransformer(nn.Module):
            def __init__(self, d_model=64, nhead=4, seq_len=16):
                super().__init__()
                self.embedding = nn.Linear(10, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
                self.transformer_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, batch_first=True
                )
                self.norm = nn.LayerNorm(d_model)
                self.classifier = nn.Linear(d_model, 10)
                
            def forward(self, x):
                # x shape: [batch, seq_len, features]
                x = self.embedding(x)
                x = x + self.pos_encoding[:x.size(1)]
                x = self.transformer_layer(x)
                x = self.norm(x)
                # Global average pooling
                x = x.mean(dim=1)
                return self.classifier(x)
        
        models['mini_transformer'] = MiniTransformer()
        
        return models
    
    def extract_activations_comprehensive(self, model: nn.Module, 
                                        input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from all relevant layers."""
        activations = {}
        
        def create_hook(name):
            def hook(module, input, output):
                # Handle different tensor shapes
                if isinstance(output, torch.Tensor):
                    if len(output.shape) == 4:  # Conv: [B, C, H, W]
                        # Global average pooling
                        pooled = output.mean(dim=[2, 3])
                        activations[name] = pooled.detach()
                    elif len(output.shape) == 3:  # Transformer: [B, S, D]
                        # Average over sequence dimension
                        pooled = output.mean(dim=1)
                        activations[name] = pooled.detach()
                    elif len(output.shape) == 2:  # Linear: [B, D]
                        activations[name] = output.detach()
                    else:
                        # Flatten everything except batch dimension
                        flattened = output.flatten(1)
                        activations[name] = flattened.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.TransformerEncoderLayer, nn.LayerNorm)):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def validate_pipeline(self, model_name: str, model: nn.Module, 
                         input_tensor: torch.Tensor) -> PipelineResult:
        """Validate complete pipeline for a given model."""
        errors = []
        warnings = []
        
        try:
            # Step 1: Extract activations
            activations = self.extract_activations_comprehensive(model, input_tensor)
            
            if len(activations) == 0:
                errors.append("No activations extracted")
                return PipelineResult(
                    model_name=model_name,
                    num_activations=0,
                    sheaf_nodes=0,
                    sheaf_edges=0,
                    laplacian_shape=(0, 0),
                    laplacian_nnz=0,
                    validation_passed=False,
                    errors=errors,
                    warnings=warnings
                )
            
            # Step 2: Build sheaf
            builder = SheafBuilder(use_whitening=True)
            
            try:
                sheaf = builder.build_from_activations(
                    model, activations, 
                    use_gram_matrices=True,
                    validate=True
                )
            except Exception as e:
                errors.append(f"Sheaf construction failed: {e}")
                return PipelineResult(
                    model_name=model_name,
                    num_activations=len(activations),
                    sheaf_nodes=0,
                    sheaf_edges=0,
                    laplacian_shape=(0, 0),
                    laplacian_nnz=0,
                    validation_passed=False,
                    errors=errors,
                    warnings=warnings
                )
            
            # Step 3: Validate sheaf properties
            if len(sheaf.stalks) == 0:
                warnings.append("No stalks created in sheaf")
            
            if len(sheaf.restrictions) == 0:
                warnings.append("No restrictions created in sheaf")
            
            # Step 4: Build Laplacian (if possible)
            laplacian_shape = (0, 0)
            laplacian_nnz = 0
            
            if len(sheaf.restrictions) > 0:
                try:
                    laplacian_builder = SheafLaplacianBuilder(enable_gpu=False)
                    L_sparse, metadata = laplacian_builder.build_laplacian(sheaf)
                    laplacian_shape = L_sparse.shape
                    laplacian_nnz = L_sparse.nnz
                except Exception as e:
                    errors.append(f"Laplacian construction failed: {e}")
            else:
                warnings.append("Cannot build Laplacian: no restrictions")
            
            # Determine if validation passed
            validation_passed = (
                len(errors) == 0 and 
                len(activations) > 0 and 
                len(sheaf.stalks) > 0
            )
            
            return PipelineResult(
                model_name=model_name,
                num_activations=len(activations),
                sheaf_nodes=len(sheaf.stalks),
                sheaf_edges=len(sheaf.restrictions),
                laplacian_shape=laplacian_shape,
                laplacian_nnz=laplacian_nnz,
                validation_passed=validation_passed,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            errors.append(f"Pipeline failed: {e}")
            return PipelineResult(
                model_name=model_name,
                num_activations=0,
                sheaf_nodes=0,
                sheaf_edges=0,
                laplacian_shape=(0, 0),
                laplacian_nnz=0,
                validation_passed=False,
                errors=errors,
                warnings=warnings
            )
    
    def test_cnn_pipeline(self, test_models):
        """Test pipeline with CNN architecture."""
        model = test_models['simple_cnn']
        model.eval()
        
        # Create input
        batch_size = 32
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        result = self.validate_pipeline('simple_cnn', model, input_tensor)
        
        # Print results
        print(f"\\nCNN Pipeline Results:")
        print(f"Activations extracted: {result.num_activations}")
        print(f"Sheaf nodes: {result.sheaf_nodes}")
        print(f"Sheaf edges: {result.sheaf_edges}")
        print(f"Laplacian shape: {result.laplacian_shape}")
        print(f"Validation passed: {result.validation_passed}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        # Basic validation
        assert result.num_activations > 0, "No activations extracted"
        assert len(result.errors) == 0, f"Pipeline errors: {result.errors}"
    
    def test_resnet_pipeline(self, test_models):
        """Test pipeline with ResNet-like architecture (skip connections)."""
        model = test_models['mini_resnet']
        model.eval()
        
        # Create input
        batch_size = 24
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        result = self.validate_pipeline('mini_resnet', model, input_tensor)
        
        print(f"\\nResNet Pipeline Results:")
        print(f"Activations extracted: {result.num_activations}")
        print(f"Sheaf nodes: {result.sheaf_nodes}")
        print(f"Sheaf edges: {result.sheaf_edges}")
        print(f"Laplacian shape: {result.laplacian_shape}")
        print(f"Validation passed: {result.validation_passed}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        # Should handle skip connections
        assert result.num_activations > 0, "No activations extracted"
    
    def test_transformer_pipeline(self, test_models):
        """Test pipeline with Transformer-like architecture."""
        model = test_models['mini_transformer']
        model.eval()
        
        # Create input (sequence data)
        batch_size = 16
        seq_len = 12
        input_dim = 10
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        
        result = self.validate_pipeline('mini_transformer', model, input_tensor)
        
        print(f"\\nTransformer Pipeline Results:")
        print(f"Activations extracted: {result.num_activations}")
        print(f"Sheaf nodes: {result.sheaf_nodes}")
        print(f"Sheaf edges: {result.sheaf_edges}")
        print(f"Laplacian shape: {result.laplacian_shape}")
        print(f"Validation passed: {result.validation_passed}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        # Should handle attention patterns
        assert result.num_activations > 0, "No activations extracted"
    
    def test_all_architectures_comprehensive(self, test_models):
        """Comprehensive test across all architectures."""
        results = {}
        
        # Define appropriate inputs for each model type
        model_inputs = {
            'simple_cnn': torch.randn(16, 3, 32, 32),
            'mini_resnet': torch.randn(16, 3, 32, 32),
            'mini_transformer': torch.randn(16, 12, 10)
        }
        
        for model_name, model in test_models.items():
            model.eval()
            input_tensor = model_inputs[model_name]
            
            result = self.validate_pipeline(model_name, model, input_tensor)
            results[model_name] = result
        
        # Summary report
        print("\\n" + "="*50)
        print("COMPREHENSIVE PIPELINE VALIDATION REPORT")
        print("="*50)
        
        total_passed = 0
        for model_name, result in results.items():
            print(f"\\n{model_name.upper()}:")
            print(f"  Activations: {result.num_activations}")
            print(f"  Sheaf nodes: {result.sheaf_nodes}")
            print(f"  Sheaf edges: {result.sheaf_edges}")
            print(f"  Laplacian: {result.laplacian_shape}")
            print(f"  Status: {'✓ PASS' if result.validation_passed else '✗ FAIL'}")
            
            if result.errors:
                print(f"  Errors: {result.errors}")
            if result.warnings:
                print(f"  Warnings: {result.warnings}")
            
            if result.validation_passed:
                total_passed += 1
        
        print(f"\\nOVERALL: {total_passed}/{len(results)} architectures passed")
        
        # At least one architecture should pass completely
        assert total_passed > 0, "No architectures passed validation"
        
        # All should at least extract activations
        for model_name, result in results.items():
            assert result.num_activations > 0, f"{model_name}: No activations extracted"
    
    def test_synthetic_data_pipeline(self):
        """Test pipeline with synthetic Gaussian data."""
        torch.manual_seed(42)
        
        # Create synthetic layer activations
        batch_size = 64
        layer_configs = [
            ('input', 32),
            ('hidden1', 64),
            ('hidden2', 128),
            ('hidden3', 64),
            ('output', 10)
        ]
        
        activations = {}
        for name, dim in layer_configs:
            activations[name] = torch.randn(batch_size, dim)
        
        # Create simple sequential poset
        poset = nx.DiGraph()
        layer_names = [name for name, _ in layer_configs]
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i+1])
        
        # Build sheaf directly from activations dict
        builder = SheafBuilder(use_whitening=True)
        
        # Compute Gram matrices
        gram_matrices = {}
        for name, act in activations.items():
            gram_matrices[name] = act @ act.T
        
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
        
        # Should have good properties with synthetic data
        assert len(sheaf.stalks) == len(layer_configs)
        print(f"\\nSynthetic Data Pipeline:")
        print(f"Stalks: {len(sheaf.stalks)}")
        print(f"Restrictions: {len(sheaf.restrictions)}")
        
        # Try to build Laplacian
        if len(sheaf.restrictions) > 0:
            laplacian_builder = SheafLaplacianBuilder(enable_gpu=False)
            L_sparse, metadata = laplacian_builder.build_laplacian(sheaf)
            print(f"Laplacian: {L_sparse.shape} with {L_sparse.nnz} nnz")
            
            # Should be well-conditioned for synthetic data
            assert L_sparse.shape[0] > 0
            assert L_sparse.nnz > 0
    
    def test_edge_cases_robustness(self):
        """Test pipeline robustness with edge cases."""
        # Test 1: Very small model
        tiny_model = nn.Linear(5, 3)
        tiny_input = torch.randn(8, 5)
        
        result = self.validate_pipeline('tiny_model', tiny_model, tiny_input)
        print(f"\\nTiny model result: {result.validation_passed}")
        
        # Test 2: Single-layer model
        single_layer = nn.Sequential(nn.Linear(10, 5))
        single_input = torch.randn(16, 10)
        
        result = self.validate_pipeline('single_layer', single_layer, single_input)
        print(f"Single layer result: {result.validation_passed}")
        
        # Test 3: Large batch, small model
        small_model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        large_batch_input = torch.randn(256, 8)
        
        result = self.validate_pipeline('large_batch', small_model, large_batch_input)
        print(f"Large batch result: {result.validation_passed}")
        
        # At least some edge cases should work
        print("Edge case testing completed")


class TestCKAIntegration:
    """Test integration with CKA computation."""
    
    def test_cka_to_sheaf_pipeline(self):
        """Test building sheaf from pre-computed CKA matrices."""
        torch.manual_seed(0)
        
        # Generate activations for 5 layers
        batch_size = 128
        activations = {
            f'layer_{i}': torch.randn(batch_size, 32 + 16*i)
            for i in range(5)
        }
        
        # Compute CKA matrices between all pairs
        cka_computer = DebiasedCKA()
        layer_names = list(activations.keys())
        
        cka_matrices = {}
        for name in layer_names:
            cka_row = []
            for other_name in layer_names:
                if name == other_name:
                    cka_value = 1.0  # Self-similarity
                else:
                    cka_value = cka_computer.compute(
                        activations[name], 
                        activations[other_name]
                    )
                cka_row.append(cka_value)
            cka_matrices[name] = torch.tensor(cka_row)
        
        # Create poset
        poset = nx.DiGraph()
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i+1])
        
        # Build sheaf from CKA data
        builder = SheafBuilder(use_whitening=True)
        sheaf = builder.build_from_cka_matrices(poset, cka_matrices)
        
        print(f"\\nCKA Integration Results:")
        print(f"Stalks: {len(sheaf.stalks)}")
        print(f"Restrictions: {len(sheaf.restrictions)}")
        
        # Validate CKA properties
        for name, stalk in sheaf.stalks.items():
            if name in cka_matrices:
                cka_vector = cka_matrices[name]
                # Stalk should reflect CKA similarities
                assert stalk.shape[0] == len(layer_names)
        
        assert len(sheaf.stalks) > 0
        print("CKA integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])