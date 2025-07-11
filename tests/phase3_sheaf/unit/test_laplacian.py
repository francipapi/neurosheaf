"""Unit tests for sparse sheaf Laplacian construction.

Tests the mathematical correctness, performance, and GPU compatibility
of the SheafLaplacianBuilder and StaticMaskedLaplacian classes.

Mathematical Properties Tested:
- Symmetry: Δ = Δ^T
- Positive semi-definite: eigenvalues ≥ 0
- Correct block structure from whitened restriction maps
- Edge masking integrity
- GPU/CPU consistency
"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import sys
import os

# Add neurosheaf to path
sys.path.append('/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf import (
    Sheaf, SheafBuilder, SheafLaplacianBuilder, LaplacianMetadata,
    ProcrustesMaps, WhiteningProcessor
)
from neurosheaf.spectral import StaticMaskedLaplacian, create_static_masked_laplacian
from tests.test_data_generators import NeuralNetworkDataGenerator


class TestSheafLaplacianBuilder:
    """Test SheafLaplacianBuilder for mathematical correctness and performance."""
    
    @pytest.fixture
    def sample_sheaf(self):
        """Create a small test sheaf with whitened data."""
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Generate structured activations
        activations = generator.generate_linear_transformation_sequence(
            num_layers=3, input_dim=20, batch_size=15,
            transformation_strength=0.6, noise_level=0.05
        )
        
        # Create simple poset
        poset = nx.DiGraph()
        layer_names = list(activations.keys())
        for name in layer_names:
            poset.add_node(name)
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        
        # Build sheaf with whitening
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        return sheaf
    
    def test_laplacian_construction_basic(self, sample_sheaf):
        """Test basic Laplacian construction functionality."""
        builder = SheafLaplacianBuilder(enable_gpu=False, memory_efficient=True)
        
        laplacian, metadata = builder.build_laplacian(sample_sheaf)
        
        # Check output types
        assert isinstance(laplacian, csr_matrix)
        assert isinstance(metadata, LaplacianMetadata)
        
        # Check matrix properties
        assert laplacian.shape[0] == laplacian.shape[1]  # Square matrix
        assert laplacian.nnz > 0  # Non-trivial sparsity
        
        # Check metadata
        assert metadata.total_dimension == laplacian.shape[0]
        assert len(metadata.stalk_dimensions) == len(sample_sheaf.stalks)
        assert metadata.construction_time > 0
    
    def test_laplacian_symmetry(self, sample_sheaf):
        """Test that Laplacian is symmetric."""
        builder = SheafLaplacianBuilder(enable_gpu=False, validate_properties=True)
        
        laplacian, metadata = builder.build_laplacian(sample_sheaf)
        
        # Check symmetry: Δ = Δ^T
        laplacian_T = laplacian.T
        symmetry_error = (laplacian - laplacian_T).max()
        
        assert symmetry_error < 1e-10, f"Laplacian not symmetric: error = {symmetry_error}"
    
    def test_laplacian_positive_semidefinite(self, sample_sheaf):
        """Test that Laplacian is positive semi-definite."""
        builder = SheafLaplacianBuilder(enable_gpu=False, validate_properties=True)
        
        laplacian, metadata = builder.build_laplacian(sample_sheaf)
        
        # Check smallest eigenvalues
        if laplacian.shape[0] > 1:
            min_eigenvals = eigsh(laplacian, k=min(5, laplacian.shape[0]-1), 
                                 which='SA', return_eigenvectors=False)
            min_eigenval = min_eigenvals[0]
            
            assert min_eigenval >= -1e-10, f"Laplacian not PSD: min eigenvalue = {min_eigenval}"
    
    def test_whitened_dimension_consistency(self, sample_sheaf):
        """Test that Laplacian dimensions match whitened stalk dimensions."""
        builder = SheafLaplacianBuilder(enable_gpu=False)
        
        laplacian, metadata = builder.build_laplacian(sample_sheaf)
        
        # Check that total dimension equals sum of whitened stalk dimensions
        expected_dim = sum(metadata.stalk_dimensions.values())
        assert laplacian.shape[0] == expected_dim
        
        # Check that each stalk dimension is smaller than original (due to whitening)
        for node, stalk in sample_sheaf.stalks.items():
            whitened_dim = metadata.stalk_dimensions[node]
            original_dim = stalk.shape[0]
            
            # Whitened dimension should be ≤ original (rank reduction)
            assert whitened_dim <= original_dim
    
    def test_edge_position_cache(self, sample_sheaf):
        """Test that edge position cache is correctly constructed."""
        builder = SheafLaplacianBuilder(enable_gpu=False)
        
        laplacian, metadata = builder.build_laplacian(sample_sheaf)
        
        # Check edge positions are cached
        assert hasattr(metadata, 'edge_positions')
        assert len(metadata.edge_positions) == len(sample_sheaf.restrictions)
        
        # Check that all positions are within matrix bounds
        for edge, positions in metadata.edge_positions.items():
            for row, col in positions:
                assert 0 <= row < laplacian.shape[0]
                assert 0 <= col < laplacian.shape[1]
    
    def test_memory_efficient_vs_standard(self, sample_sheaf):
        """Test consistency between memory-efficient and standard assembly."""
        builder_efficient = SheafLaplacianBuilder(enable_gpu=False, memory_efficient=True)
        builder_standard = SheafLaplacianBuilder(enable_gpu=False, memory_efficient=False)
        
        laplacian_efficient, _ = builder_efficient.build_laplacian(sample_sheaf)
        laplacian_standard, _ = builder_standard.build_laplacian(sample_sheaf)
        
        # Check that both methods produce equivalent results
        difference = (laplacian_efficient - laplacian_standard).max()
        assert difference < 1e-12, f"Memory-efficient and standard methods differ: {difference}"
    
    def test_torch_sparse_conversion(self, sample_sheaf):
        """Test conversion to torch sparse tensors."""
        builder = SheafLaplacianBuilder(enable_gpu=False)
        
        laplacian, metadata = builder.build_laplacian(sample_sheaf)
        torch_sparse = builder.to_torch_sparse(laplacian)
        
        # Check torch sparse tensor properties
        assert torch_sparse.is_sparse
        assert torch_sparse.shape == laplacian.shape
        assert torch_sparse.dtype == torch.float32
        
        # Check numerical consistency
        torch_dense = torch_sparse.to_dense().numpy()
        scipy_dense = laplacian.toarray()
        
        difference = np.max(np.abs(torch_dense - scipy_dense))
        assert difference < 1e-6, f"Torch conversion inconsistent: {difference}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_operations(self, sample_sheaf):
        """Test GPU-enabled Laplacian construction."""
        builder = SheafLaplacianBuilder(enable_gpu=True)
        
        laplacian, metadata = builder.build_laplacian(sample_sheaf)
        
        # Convert to torch sparse and move to GPU
        torch_sparse = builder.to_torch_sparse(laplacian)
        
        assert torch_sparse.is_cuda, "Torch sparse tensor not on GPU"
        
        # Test GPU computation consistency with CPU
        builder_cpu = SheafLaplacianBuilder(enable_gpu=False)
        laplacian_cpu, _ = builder_cpu.build_laplacian(sample_sheaf)
        
        torch_sparse_cpu = torch_sparse.cpu()
        difference = (torch_sparse_cpu.to_dense().numpy() - laplacian_cpu.toarray()).max()
        assert difference < 1e-6, f"GPU/CPU results differ: {difference}"


class TestStaticMaskedLaplacian:
    """Test StaticMaskedLaplacian for edge masking and filtration."""
    
    @pytest.fixture
    def sample_static_laplacian(self):
        """Create StaticMaskedLaplacian from test sheaf."""
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Generate test data with varied edge weights
        activations = generator.generate_linear_transformation_sequence(
            num_layers=4, input_dim=16, batch_size=12,
            transformation_strength=0.5, noise_level=0.1
        )
        
        # Create chain poset
        poset = nx.DiGraph()
        layer_names = list(activations.keys())
        for name in layer_names:
            poset.add_node(name)
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        
        # Build sheaf
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        # Create static masked Laplacian
        static_laplacian = create_static_masked_laplacian(sheaf, enable_gpu=False)
        
        return static_laplacian, sheaf
    
    def test_static_laplacian_creation(self, sample_static_laplacian):
        """Test basic StaticMaskedLaplacian creation."""
        static_laplacian, sheaf = sample_static_laplacian
        
        assert isinstance(static_laplacian, StaticMaskedLaplacian)
        assert static_laplacian.L_static.shape[0] == static_laplacian.L_static.shape[1]
        assert len(static_laplacian.masking_metadata.edge_weights) == len(sheaf.restrictions)
    
    def test_threshold_masking(self, sample_static_laplacian):
        """Test threshold-based edge masking."""
        static_laplacian, _ = sample_static_laplacian
        
        weights = list(static_laplacian.masking_metadata.edge_weights.values())
        min_weight = min(weights)
        max_weight = max(weights)
        
        # Test masking at different thresholds
        threshold_low = min_weight + 0.1 * (max_weight - min_weight)
        threshold_high = min_weight + 0.8 * (max_weight - min_weight)
        
        laplacian_low = static_laplacian.apply_threshold_mask(threshold_low)
        laplacian_high = static_laplacian.apply_threshold_mask(threshold_high)
        
        # Higher threshold should result in sparser matrix
        assert laplacian_high.nnz <= laplacian_low.nnz
        
        # Both should be symmetric
        assert (laplacian_low - laplacian_low.T).max() < 1e-10
        assert (laplacian_high - laplacian_high.T).max() < 1e-10
    
    def test_filtration_sequence(self, sample_static_laplacian):
        """Test computing filtration sequence."""
        static_laplacian, _ = sample_static_laplacian
        
        # Generate thresholds
        thresholds = static_laplacian.suggest_thresholds(num_thresholds=5, strategy='uniform')
        
        # Compute filtration sequence
        sequence = static_laplacian.compute_filtration_sequence(thresholds)
        
        assert len(sequence) == len(thresholds)
        
        # Check monotonicity: higher thresholds should give sparser matrices
        for i in range(len(sequence) - 1):
            assert sequence[i+1].nnz <= sequence[i].nnz
    
    def test_weight_distribution(self, sample_static_laplacian):
        """Test edge weight distribution analysis."""
        static_laplacian, _ = sample_static_laplacian
        
        bin_centers, counts = static_laplacian.get_weight_distribution(num_bins=10)
        
        assert len(bin_centers) == len(counts)
        assert all(count >= 0 for count in counts)
        assert sum(counts) == len(static_laplacian.masking_metadata.edge_weights)
    
    def test_threshold_strategies(self, sample_static_laplacian):
        """Test different threshold generation strategies."""
        static_laplacian, _ = sample_static_laplacian
        
        num_thresholds = 8
        
        # Test all strategies
        uniform_thresholds = static_laplacian.suggest_thresholds(num_thresholds, 'uniform')
        quantile_thresholds = static_laplacian.suggest_thresholds(num_thresholds, 'quantile')
        adaptive_thresholds = static_laplacian.suggest_thresholds(num_thresholds, 'adaptive')
        
        # Check lengths
        assert len(uniform_thresholds) <= num_thresholds  # May be deduplicated
        assert len(quantile_thresholds) <= num_thresholds
        assert len(adaptive_thresholds) <= num_thresholds
        
        # Check ordering
        assert uniform_thresholds == sorted(uniform_thresholds)
        assert quantile_thresholds == sorted(quantile_thresholds)
        assert adaptive_thresholds == sorted(adaptive_thresholds)
    
    def test_masking_validation(self, sample_static_laplacian):
        """Test masking integrity validation."""
        static_laplacian, _ = sample_static_laplacian
        
        weights = list(static_laplacian.masking_metadata.edge_weights.values())
        test_threshold = np.median(weights)
        
        validation_results = static_laplacian.validate_masking_integrity(test_threshold)
        
        assert 'threshold' in validation_results
        assert 'symmetric' in validation_results
        assert 'shape' in validation_results
        assert validation_results['threshold'] == test_threshold
        
        # Should pass symmetry test
        assert validation_results['symmetric'] is True
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_masking(self, sample_static_laplacian):
        """Test GPU-based masking operations."""
        # Create GPU-enabled version
        static_laplacian_cpu, sheaf = sample_static_laplacian
        static_laplacian_gpu = create_static_masked_laplacian(sheaf, enable_gpu=True)
        
        weights = list(static_laplacian_gpu.masking_metadata.edge_weights.values())
        test_threshold = np.median(weights)
        
        # Test both CPU and GPU masking
        laplacian_cpu = static_laplacian_cpu.apply_threshold_mask(test_threshold, return_torch=False)
        laplacian_gpu = static_laplacian_gpu.apply_threshold_mask(test_threshold, return_torch=True)
        
        # Check GPU tensor properties
        assert laplacian_gpu.is_sparse
        assert laplacian_gpu.is_cuda
        
        # Check consistency
        laplacian_gpu_numpy = laplacian_gpu.cpu().to_dense().numpy()
        difference = np.max(np.abs(laplacian_gpu_numpy - laplacian_cpu.toarray()))
        assert difference < 1e-6, f"GPU/CPU masking differs: {difference}"
    
    def test_memory_usage_tracking(self, sample_static_laplacian):
        """Test memory usage monitoring."""
        static_laplacian, _ = sample_static_laplacian
        
        memory_stats = static_laplacian.get_memory_usage()
        
        assert 'static_laplacian_gb' in memory_stats
        assert 'edge_cache_gb' in memory_stats
        assert 'total_gb' in memory_stats
        
        # Check reasonable values
        assert memory_stats['total_gb'] > 0
        assert memory_stats['static_laplacian_gb'] > 0


class TestIntegrationLaplacian:
    """Integration tests for Laplacian construction with SheafBuilder."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from activations to masked Laplacian."""
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Generate realistic test data
        activations = generator.generate_linear_transformation_sequence(
            num_layers=5, input_dim=24, batch_size=18
        )
        
        # Create branching poset
        layer_names = list(activations.keys())
        poset = nx.DiGraph()
        for name in layer_names:
            poset.add_node(name)
        
        # Create chain + one branch
        for i in range(len(layer_names) - 2):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        # Add branch from middle layer
        if len(layer_names) >= 4:
            poset.add_edge(layer_names[1], layer_names[-1])
        
        # Build sheaf with whitening
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=True,
                              residual_threshold=0.1)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        # Build Laplacian
        laplacian, metadata = builder.build_laplacian(sheaf)
        
        # Build static masked Laplacian
        static_laplacian = builder.build_static_masked_laplacian(sheaf)
        
        # Validate complete pipeline
        assert isinstance(laplacian, csr_matrix)
        assert isinstance(static_laplacian, StaticMaskedLaplacian)
        assert laplacian.shape == static_laplacian.L_static.shape
        
        # Test filtration
        thresholds = static_laplacian.suggest_thresholds(3, 'uniform')
        sequence = static_laplacian.compute_filtration_sequence(thresholds)
        
        assert len(sequence) == len(thresholds)
        
        # Validate mathematical properties for each threshold
        for i, filtered_laplacian in enumerate(sequence):
            # Check symmetry
            symmetry_error = (filtered_laplacian - filtered_laplacian.T).max()
            assert symmetry_error < 1e-10, f"Threshold {i} not symmetric"
            
            # Check sparsity increases with threshold
            if i > 0:
                assert filtered_laplacian.nnz <= sequence[i-1].nnz
    
    def test_performance_benchmarks(self):
        """Test performance meets targets for medium-sized networks."""
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Create medium-sized test case (representative of small networks)
        activations = generator.generate_linear_transformation_sequence(
            num_layers=10, input_dim=32, batch_size=24
        )
        
        # Create chain poset
        layer_names = list(activations.keys())
        poset = nx.DiGraph()
        for name in layer_names:
            poset.add_node(name)
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        
        # Build sheaf
        builder = SheafBuilder(use_whitening=True)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices)
        
        # Time Laplacian construction
        import time
        start_time = time.time()
        laplacian, metadata = builder.build_laplacian(sheaf, memory_efficient=True)
        construction_time = time.time() - start_time
        
        # Performance targets (relaxed for test environment)
        assert construction_time < 5.0, f"Laplacian construction too slow: {construction_time:.2f}s"
        assert metadata.memory_usage < 1.0, f"Memory usage too high: {metadata.memory_usage:.2f}GB"
        
        # Check sparsity efficiency
        sparsity = 1.0 - (laplacian.nnz / (laplacian.shape[0] * laplacian.shape[1]))
        assert sparsity > 0.7, f"Laplacian not sparse enough: {sparsity:.1%} sparse"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])