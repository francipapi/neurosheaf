"""Unit tests for adaptive sampling strategies."""

import pytest
import torch
import numpy as np

from neurosheaf.cka.sampling import AdaptiveSampler
from neurosheaf.utils.exceptions import ValidationError


class TestAdaptiveSampler:
    """Test the AdaptiveSampler class."""
    
    def test_initialization(self):
        """Test sampler initialization and validation."""
        # Valid initialization
        sampler = AdaptiveSampler(min_samples=512, max_samples=4096)
        assert sampler.min_samples == 512
        assert sampler.max_samples == 4096
        
        # Invalid: min_samples < 4
        with pytest.raises(ValidationError, match="at least 4"):
            AdaptiveSampler(min_samples=3)
        
        # Invalid: max_samples < min_samples
        with pytest.raises(ValidationError, match="must be >="):
            AdaptiveSampler(min_samples=1000, max_samples=500)
    
    def test_determine_sample_size_small_data(self):
        """Test sample size determination with small datasets."""
        sampler = AdaptiveSampler(min_samples=100, max_samples=1000)
        
        # Very small dataset - use all
        size = sampler.determine_sample_size(
            n_total=50,
            n_features=128,
            available_memory_mb=1000
        )
        assert size == 50
    
    def test_determine_sample_size_memory_constrained(self):
        """Test sample size determination under memory constraints."""
        sampler = AdaptiveSampler(min_samples=100, max_samples=1000)
        
        # Memory constraint should limit sample size
        # With 10MB and float32, we can fit roughly sqrt(10*1024*1024/4/2.5) ≈ 1024 samples
        size = sampler.determine_sample_size(
            n_total=10000,
            n_features=512,
            available_memory_mb=10
        )
        
        # Verify memory usage
        memory_mb = (size * size * 4 * 2.5) / (1024 * 1024)
        assert memory_mb <= 10
        assert size >= sampler.min_samples
    
    def test_determine_sample_size_sufficient_memory(self):
        """Test sample size with sufficient memory."""
        sampler = AdaptiveSampler(min_samples=100, max_samples=1000)
        
        # Plenty of memory - should use max_samples
        size = sampler.determine_sample_size(
            n_total=10000,
            n_features=512,
            available_memory_mb=1000
        )
        assert size == sampler.max_samples
    
    def test_stratified_sampling_uniform(self):
        """Test uniform random sampling (no labels)."""
        sampler = AdaptiveSampler(random_seed=42)
        
        n_total = 1000
        n_samples = 200
        
        indices = sampler.stratified_sample(n_total, n_samples)
        
        # Check properties
        assert len(indices) == n_samples
        assert len(torch.unique(indices)) == n_samples  # No duplicates
        assert torch.all(indices >= 0)
        assert torch.all(indices < n_total)
        
        # Test reproducibility
        sampler2 = AdaptiveSampler(random_seed=42)
        indices2 = sampler2.stratified_sample(n_total, n_samples)
        assert torch.all(indices == indices2)
    
    def test_stratified_sampling_with_labels(self):
        """Test stratified sampling with class labels."""
        sampler = AdaptiveSampler(random_seed=42)
        
        # Create imbalanced labels
        labels = torch.cat([
            torch.zeros(200),    # Class 0: 200 samples
            torch.ones(300),     # Class 1: 300 samples  
            torch.full((100,), 2),  # Class 2: 100 samples
            torch.full((250,), 3),  # Class 3: 250 samples
            torch.full((150,), 4),  # Class 4: 150 samples
        ])
        
        n_total = 1000
        n_samples = 200
        
        indices = sampler.stratified_sample(n_total, n_samples, labels=labels)
        
        # Check basic properties
        assert len(indices) == n_samples
        assert len(torch.unique(indices)) == n_samples
        
        # Check class balance
        sampled_labels = labels[indices]
        for class_id in range(5):
            class_count = (sampled_labels == class_id).sum().item()
            expected = n_samples // 5  # Should be roughly balanced
            assert abs(class_count - expected) <= 2  # Allow small deviation
    
    def test_stratified_sampling_edge_cases(self):
        """Test edge cases in stratified sampling."""
        sampler = AdaptiveSampler()
        
        # Sample size = total size
        indices = sampler.stratified_sample(100, 100)
        assert len(indices) == 100
        assert torch.all(indices == torch.arange(100))
        
        # Invalid: sample more than total
        with pytest.raises(ValidationError):
            sampler.stratified_sample(100, 150)
        
        # With mask return
        indices, mask = sampler.stratified_sample(100, 50, return_mask=True)
        assert len(indices) == 50
        assert mask.sum() == 50
        assert torch.all(mask[indices])
    
    def test_progressive_sampling(self):
        """Test progressive sampling with overlap."""
        sampler = AdaptiveSampler(random_seed=42)
        
        n_total = 1000
        batch_sizes = [300, 300, 300]
        overlap = 0.1
        
        batches = sampler.progressive_sampling(n_total, batch_sizes, overlap)
        
        assert len(batches) == 3
        
        # Check each batch
        for i, batch in enumerate(batches):
            assert len(batch) <= batch_sizes[i]
            assert torch.all(batch >= 0)
            assert torch.all(batch < n_total)
            
            # Check for duplicates within batch
            assert len(torch.unique(batch)) == len(batch)
        
        # Check overlap between consecutive batches
        for i in range(len(batches) - 1):
            overlap_count = len(set(batches[i].tolist()) & set(batches[i+1].tolist()))
            expected_overlap = int(batch_sizes[i+1] * overlap)
            assert overlap_count <= expected_overlap + 10  # Allow some flexibility
    
    def test_progressive_sampling_no_overlap(self):
        """Test progressive sampling without overlap."""
        sampler = AdaptiveSampler(random_seed=42)
        
        batches = sampler.progressive_sampling(1000, [200, 300, 400], overlap=0.0)
        
        # With no overlap, batches should be disjoint
        all_indices = []
        for batch in batches:
            all_indices.extend(batch.tolist())
        
        # All indices should be unique
        assert len(set(all_indices)) == len(all_indices)
    
    def test_estimate_required_samples(self):
        """Test sample size estimation for target error."""
        sampler = AdaptiveSampler(min_samples=100, max_samples=5000)
        
        # Low error requires more samples
        n_low_error = sampler.estimate_required_samples(target_error=0.001)
        n_high_error = sampler.estimate_required_samples(target_error=0.1)
        
        assert n_low_error > n_high_error
        assert sampler.min_samples <= n_low_error <= sampler.max_samples
        assert sampler.min_samples <= n_high_error <= sampler.max_samples
    
    def test_memory_aware_sampling_scaling(self):
        """Test that sample size scales appropriately with memory."""
        sampler = AdaptiveSampler(min_samples=100, max_samples=10000)
        
        memory_limits = [10, 50, 100, 500, 1000]
        sizes = []
        
        for mem_mb in memory_limits:
            size = sampler.determine_sample_size(
                n_total=50000,
                n_features=1024,
                available_memory_mb=mem_mb
            )
            sizes.append(size)
        
        # Sizes should increase with available memory
        for i in range(len(sizes) - 1):
            assert sizes[i] <= sizes[i+1]
        
        # Verify memory constraints are respected
        for size, mem_mb in zip(sizes, memory_limits):
            kernel_memory_mb = (size * size * 4 * 2.5) / (1024 * 1024)
            assert kernel_memory_mb <= mem_mb * 1.1  # Allow 10% tolerance