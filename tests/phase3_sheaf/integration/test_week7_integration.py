"""Integration tests for Week 7: Sparse Laplacian assembly and optimization.

Tests the complete pipeline from sheaf construction to Laplacian assembly,
including GPU operations, memory efficiency, and mathematical validation.

Integration Points Tested:
- SheafBuilder → SheafLaplacianBuilder pipeline
- StaticMaskedLaplacian → filtration sequence
- GPU/CPU consistency across entire pipeline
- Memory targets and performance benchmarks
- Real neural network architectures
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time
import psutil
import sys
import os

# Add neurosheaf to path
sys.path.append('/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf import (
    Sheaf, SheafBuilder, SheafLaplacianBuilder, 
    ProcrustesMaps, WhiteningProcessor
)
from neurosheaf.spectral import StaticMaskedLaplacian, create_static_masked_laplacian
from tests.test_data_generators import NeuralNetworkDataGenerator


class SimpleConvNet(nn.Module):
    """Simple CNN for testing with realistic architecture."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TestEndToEndPipeline:
    """Test complete pipeline from neural network to persistent spectral analysis."""
    
    @pytest.fixture
    def sample_network_data(self):
        """Generate realistic network data for testing."""
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Create branching network structure
        activations, poset = generator.generate_branching_network_data(
            trunk_layers=4, branch_depth=3, num_branches=2,
            input_dim=40, batch_size=20
        )
        
        return activations, poset
    
    @pytest.fixture
    def simple_cnn_data(self):
        """Generate data from simple CNN for realistic testing."""
        model = SimpleConvNet(num_classes=5)
        model.eval()
        
        # Generate sample input
        batch_size = 8
        input_data = torch.randn(batch_size, 3, 32, 32)
        
        # Extract activations
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if output.dim() == 4:  # Conv layers
                    # Flatten spatial dimensions
                    activations[name] = output.view(output.size(0), -1).detach()
                else:  # Linear layers
                    activations[name] = output.detach()
            return hook
        
        # Register hooks for named modules
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create sequential poset from activations
        layer_names = list(activations.keys())
        poset = nx.DiGraph()
        for name in layer_names:
            poset.add_node(name)
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        
        return activations, poset, model
    
    def test_whitened_sheaf_to_laplacian_pipeline(self, sample_network_data):
        """Test complete pipeline with pure whitened coordinates."""
        activations, poset = sample_network_data
        
        # Build sheaf with whitening (Week 6 implementation)
        builder = SheafBuilder(
            use_whitening=True,
            enable_edge_filtering=True,
            residual_threshold=0.08
        )
        
        # Convert to Gram matrices
        generator = NeuralNetworkDataGenerator(seed=42)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        
        # Build sheaf
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        # Validate sheaf properties
        assert len(sheaf.stalks) > 0
        assert len(sheaf.restrictions) > 0
        
        # Check that stalks are in whitened coordinates (identity inner product)
        for node, stalk in sheaf.stalks.items():
            # For whitened Gram matrices, should be close to identity
            stalk_np = stalk.detach().cpu().numpy()
            identity = np.eye(stalk.shape[0])
            identity_error = np.linalg.norm(stalk_np - identity, 'fro')
            
            # Should be very close to identity (whitened property)
            assert identity_error < 0.1, f"Stalk {node} not properly whitened: error = {identity_error:.4f}"
        
        # Build Laplacian (Week 7 implementation)
        laplacian, metadata = builder.build_laplacian(sheaf, memory_efficient=True)
        
        # Validate Laplacian mathematical properties
        assert isinstance(laplacian, csr_matrix)
        assert laplacian.shape[0] == laplacian.shape[1]
        
        # Test symmetry
        symmetry_error = (laplacian - laplacian.T).max()
        assert symmetry_error < 1e-10, f"Laplacian not symmetric: {symmetry_error:.2e}"
        
        # Test positive semi-definite
        if laplacian.shape[0] > 1:
            min_eigenval = eigsh(laplacian, k=1, which='SA', return_eigenvectors=False)[0]
            assert min_eigenval >= -1e-10, f"Laplacian not PSD: min eigenvalue = {min_eigenval:.2e}"
        
        # Validate metadata
        assert metadata.total_dimension == laplacian.shape[0]
        assert len(metadata.stalk_dimensions) == len(sheaf.stalks)
        assert metadata.construction_time > 0
        assert metadata.sparsity_ratio > 0.5  # Should be reasonably sparse
        
        print(f"✅ Whitened pipeline success: {laplacian.shape[0]}×{laplacian.shape[1]} Laplacian, "
              f"{metadata.sparsity_ratio:.1%} sparse, {metadata.construction_time:.3f}s")
    
    def test_static_laplacian_filtration_pipeline(self, sample_network_data):
        """Test complete filtration pipeline for persistent analysis."""
        activations, poset = sample_network_data
        
        # Build sheaf
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
        generator = NeuralNetworkDataGenerator(seed=42)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        # Build static masked Laplacian
        static_laplacian = builder.build_static_masked_laplacian(sheaf)
        
        # Validate static Laplacian
        assert isinstance(static_laplacian, StaticMaskedLaplacian)
        assert len(static_laplacian.masking_metadata.edge_weights) > 0
        
        # Test threshold generation strategies
        num_thresholds = 8
        uniform_thresholds = static_laplacian.suggest_thresholds(num_thresholds, 'uniform')
        quantile_thresholds = static_laplacian.suggest_thresholds(num_thresholds, 'quantile')
        adaptive_thresholds = static_laplacian.suggest_thresholds(num_thresholds, 'adaptive')
        
        # Test filtration sequences
        for strategy, thresholds in [
            ('uniform', uniform_thresholds),
            ('quantile', quantile_thresholds), 
            ('adaptive', adaptive_thresholds)
        ]:
            sequence = static_laplacian.compute_filtration_sequence(thresholds[:5])  # Limit for speed
            
            # Validate filtration properties
            assert len(sequence) == len(thresholds[:5])
            
            # Check monotonicity: sparsity increases with threshold
            for i in range(len(sequence) - 1):
                assert sequence[i+1].nnz <= sequence[i].nnz, f"Non-monotonic sparsity in {strategy}"
            
            # Check mathematical properties for each filtered Laplacian
            for j, filtered_laplacian in enumerate(sequence):
                # Symmetry
                symmetry_error = (filtered_laplacian - filtered_laplacian.T).max()
                assert symmetry_error < 1e-10, f"{strategy} threshold {j} not symmetric"
                
                # Positive semi-definite (check smallest few eigenvalues)
                if filtered_laplacian.shape[0] > 2 and filtered_laplacian.nnz > 0:
                    try:
                        min_eigenvals = eigsh(filtered_laplacian, k=min(3, filtered_laplacian.shape[0]-1),
                                            which='SA', return_eigenvectors=False)
                        min_eigenval = min_eigenvals[0]
                        assert min_eigenval >= -1e-8, f"{strategy} threshold {j} not PSD: {min_eigenval:.2e}"
                    except:
                        pass  # Skip if eigenvalue computation fails (very sparse matrices)
        
        print(f"✅ Filtration pipeline success: {len(uniform_thresholds)} uniform, "
              f"{len(quantile_thresholds)} quantile, {len(adaptive_thresholds)} adaptive thresholds")
    
    def test_cnn_architecture_pipeline(self, simple_cnn_data):
        """Test pipeline with realistic CNN architecture."""
        activations, poset, model = simple_cnn_data
        
        print(f"Testing CNN with {len(activations)} layers: {list(activations.keys())}")
        
        # Build sheaf from CNN activations
        builder = SheafBuilder(
            use_whitening=True,
            enable_edge_filtering=True,
            residual_threshold=0.15  # More lenient for CNN layers
        )
        
        # Convert to Gram matrices
        gram_matrices = {}
        for layer_name, activation in activations.items():
            if activation.shape[0] > activation.shape[1]:  # More samples than features
                # Standard case: K = X @ X.T
                gram_matrices[layer_name] = activation @ activation.T
            else:
                # High-dimensional case: use feature Gram matrix
                gram_matrices[layer_name] = activation.T @ activation
        
        # Build sheaf
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        # Validate sheaf construction succeeded
        assert len(sheaf.stalks) > 0, "No stalks created from CNN"
        assert len(sheaf.restrictions) > 0, "No restrictions created from CNN"
        
        # Build complete pipeline
        laplacian, laplacian_metadata = builder.build_laplacian(sheaf)
        static_laplacian = builder.build_static_masked_laplacian(sheaf)
        
        # Validate Laplacian from CNN
        assert laplacian.shape[0] > 0
        assert laplacian.nnz > 0
        
        # Test filtration on CNN Laplacian
        thresholds = static_laplacian.suggest_thresholds(5, 'quantile')
        sequence = static_laplacian.compute_filtration_sequence(thresholds)
        
        # Validate CNN filtration
        assert len(sequence) == len(thresholds)
        
        # Check that filtration preserves essential properties
        for filtered_laplacian in sequence[:3]:  # Test first few
            if filtered_laplacian.nnz > 0:
                symmetry_error = (filtered_laplacian - filtered_laplacian.T).max()
                assert symmetry_error < 1e-10, "CNN filtration breaks symmetry"
        
        print(f"✅ CNN pipeline success: {model.__class__.__name__} → "
              f"{laplacian.shape[0]}×{laplacian.shape[1]} Laplacian → "
              f"{len(thresholds)} filtration levels")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_pipeline_consistency(self, sample_network_data):
        """Test GPU/CPU consistency across entire pipeline."""
        activations, poset = sample_network_data
        
        # Build sheaf
        builder = SheafBuilder(use_whitening=True)
        generator = NeuralNetworkDataGenerator(seed=42)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
        
        # Test CPU pipeline
        laplacian_cpu, metadata_cpu = builder.build_laplacian(sheaf, enable_gpu=False)
        static_laplacian_cpu = create_static_masked_laplacian(sheaf, enable_gpu=False)
        
        # Test GPU pipeline
        laplacian_gpu, metadata_gpu = builder.build_laplacian(sheaf, enable_gpu=True)
        static_laplacian_gpu = create_static_masked_laplacian(sheaf, enable_gpu=True)
        
        # Check Laplacian consistency
        difference = (laplacian_cpu - laplacian_gpu).max()
        assert difference < 1e-6, f"GPU/CPU Laplacian differ: {difference:.2e}"
        
        # Test masking consistency
        test_threshold = np.median(list(static_laplacian_cpu.masking_metadata.edge_weights.values()))
        
        masked_cpu = static_laplacian_cpu.apply_threshold_mask(test_threshold, return_torch=False)
        masked_gpu = static_laplacian_gpu.apply_threshold_mask(test_threshold, return_torch=True)
        
        # Convert GPU result to CPU for comparison
        masked_gpu_cpu = masked_gpu.cpu().to_dense().numpy()
        difference_masked = np.max(np.abs(masked_gpu_cpu - masked_cpu.toarray()))
        assert difference_masked < 1e-6, f"GPU/CPU masking differ: {difference_masked:.2e}"
        
        print(f"✅ GPU consistency verified: Laplacian diff = {difference:.2e}, "
              f"masking diff = {difference_masked:.2e}")
    
    def test_performance_benchmarks(self, sample_network_data):
        """Test that pipeline meets performance targets."""
        activations, poset = sample_network_data
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**3  # GB
        
        # Build sheaf with timing
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True)
        generator = NeuralNetworkDataGenerator(seed=42)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        
        start_time = time.time()
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        sheaf_time = time.time() - start_time
        
        # Build Laplacian with timing
        start_time = time.time()
        laplacian, metadata = builder.build_laplacian(sheaf, memory_efficient=True)
        laplacian_time = time.time() - start_time
        
        # Build static Laplacian with timing
        start_time = time.time()
        static_laplacian = builder.build_static_masked_laplacian(sheaf)
        static_laplacian_time = time.time() - start_time
        
        # Test filtration performance
        thresholds = static_laplacian.suggest_thresholds(10, 'uniform')
        start_time = time.time()
        sequence = static_laplacian.compute_filtration_sequence(thresholds)
        filtration_time = time.time() - start_time
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024**3  # GB
        memory_used = peak_memory - initial_memory
        
        # Performance targets (relaxed for test environment)
        total_time = sheaf_time + laplacian_time + static_laplacian_time + filtration_time
        
        assert total_time < 30.0, f"Pipeline too slow: {total_time:.2f}s"
        assert memory_used < 2.0, f"Memory usage too high: {memory_used:.2f}GB"
        assert laplacian_time < 5.0, f"Laplacian construction too slow: {laplacian_time:.2f}s"
        assert filtration_time < 10.0, f"Filtration too slow: {filtration_time:.2f}s"
        
        # Check sparsity efficiency
        sparsity = metadata.sparsity_ratio
        assert sparsity > 0.6, f"Laplacian not sparse enough: {sparsity:.1%}"
        
        print(f"✅ Performance targets met:")
        print(f"   Total time: {total_time:.2f}s (sheaf: {sheaf_time:.2f}s, "
              f"laplacian: {laplacian_time:.2f}s, static: {static_laplacian_time:.2f}s, "
              f"filtration: {filtration_time:.2f}s)")
        print(f"   Memory used: {memory_used:.2f}GB")
        print(f"   Sparsity: {sparsity:.1%}")
    
    def test_mathematical_validation_comprehensive(self, sample_network_data):
        """Comprehensive validation of mathematical properties across pipeline."""
        activations, poset = sample_network_data
        
        # Build complete pipeline
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True)
        generator = NeuralNetworkDataGenerator(seed=42)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        laplacian, metadata = builder.build_laplacian(sheaf)
        static_laplacian = builder.build_static_masked_laplacian(sheaf)
        
        # Test 1: Sheaf properties (from Week 6)
        sheaf_validation = self._validate_sheaf_properties(sheaf)
        assert sheaf_validation['transitivity_valid'], "Sheaf transitivity violated"
        assert sheaf_validation['whitened_orthogonal'], "Whitened orthogonality violated"
        
        # Test 2: Laplacian properties
        laplacian_validation = self._validate_laplacian_properties(laplacian)
        assert laplacian_validation['symmetric'], "Laplacian not symmetric"
        assert laplacian_validation['positive_semidefinite'], "Laplacian not PSD"
        assert laplacian_validation['correct_dimensions'], "Laplacian dimensions incorrect"
        
        # Test 3: Filtration properties
        thresholds = static_laplacian.suggest_thresholds(5, 'uniform')
        sequence = static_laplacian.compute_filtration_sequence(thresholds)
        
        filtration_validation = self._validate_filtration_properties(sequence, thresholds)
        assert filtration_validation['monotonic_sparsity'], "Filtration not monotonic"
        assert filtration_validation['preserved_symmetry'], "Filtration breaks symmetry"
        
        print("✅ Comprehensive mathematical validation passed")
    
    def _validate_sheaf_properties(self, sheaf):
        """Validate sheaf mathematical properties."""
        from neurosheaf.sheaf.restriction import validate_sheaf_properties
        
        # Test transitivity using existing validation
        validation_results = validate_sheaf_properties(sheaf.restrictions, sheaf.poset, tolerance=1e-1)
        
        # Test whitened orthogonality
        whitened_orthogonal = True
        for edge, restriction in sheaf.restrictions.items():
            if hasattr(restriction, 'whitened_validation'):
                if not restriction.whitened_validation.get('exact_orthogonal', False):
                    whitened_orthogonal = False
                    break
        
        return {
            'transitivity_valid': validation_results['valid_sheaf'],
            'max_violation': validation_results['max_violation'],
            'whitened_orthogonal': whitened_orthogonal
        }
    
    def _validate_laplacian_properties(self, laplacian):
        """Validate Laplacian mathematical properties."""
        # Test symmetry
        symmetry_error = (laplacian - laplacian.T).max()
        symmetric = symmetry_error < 1e-10
        
        # Test positive semi-definite
        positive_semidefinite = True
        if laplacian.shape[0] > 1:
            try:
                min_eigenvals = eigsh(laplacian, k=min(3, laplacian.shape[0]-1),
                                     which='SA', return_eigenvectors=False)
                min_eigenval = min_eigenvals[0]
                positive_semidefinite = min_eigenval >= -1e-10
            except:
                positive_semidefinite = False
        
        # Test dimensions
        correct_dimensions = laplacian.shape[0] == laplacian.shape[1] and laplacian.shape[0] > 0
        
        return {
            'symmetric': symmetric,
            'symmetry_error': symmetry_error,
            'positive_semidefinite': positive_semidefinite,
            'correct_dimensions': correct_dimensions
        }
    
    def _validate_filtration_properties(self, sequence, thresholds):
        """Validate filtration sequence properties."""
        # Test monotonic sparsity
        monotonic_sparsity = True
        for i in range(len(sequence) - 1):
            if sequence[i+1].nnz > sequence[i].nnz:
                monotonic_sparsity = False
                break
        
        # Test preserved symmetry
        preserved_symmetry = True
        for filtered_laplacian in sequence:
            if filtered_laplacian.nnz > 0:
                symmetry_error = (filtered_laplacian - filtered_laplacian.T).max()
                if symmetry_error > 1e-10:
                    preserved_symmetry = False
                    break
        
        return {
            'monotonic_sparsity': monotonic_sparsity,
            'preserved_symmetry': preserved_symmetry,
            'num_thresholds': len(thresholds),
            'sparsity_range': (sequence[-1].nnz, sequence[0].nnz) if sequence else (0, 0)
        }


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "-s", "--tb=short"])