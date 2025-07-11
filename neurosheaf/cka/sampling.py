"""Adaptive sampling strategies for memory-efficient CKA computation.

This module provides sampling strategies to handle large-scale neural network
activations while maintaining CKA accuracy and respecting memory constraints.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union, List
import warnings

from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError
from ..utils.validation import validate_sample_indices, validate_memory_limit


logger = setup_logger(__name__)


class AdaptiveSampler:
    """Adaptive sampling strategy for CKA computation.
    
    This class implements memory-aware sampling strategies to enable CKA
    computation on large datasets while respecting memory constraints.
    """
    
    def __init__(
        self,
        min_samples: int = 512,
        max_samples: int = 4096,
        target_variance: float = 0.01,
        random_seed: Optional[int] = None
    ):
        """Initialize the adaptive sampler.
        
        Args:
            min_samples: Minimum number of samples to use
            max_samples: Maximum number of samples to use
            target_variance: Target variance for adaptive sampling
            random_seed: Random seed for reproducibility
        """
        if min_samples < 4:
            raise ValidationError("min_samples must be at least 4 for unbiased HSIC")
        
        if max_samples < min_samples:
            raise ValidationError(f"max_samples ({max_samples}) must be >= min_samples ({min_samples})")
        
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.target_variance = target_variance
        self.random_seed = random_seed
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        logger.info(f"Initialized AdaptiveSampler: min={min_samples}, max={max_samples}")
    
    def determine_sample_size(
        self,
        n_total: int,
        n_features: int,
        available_memory_mb: float,
        dtype: torch.dtype = torch.float32
    ) -> int:
        """Determine optimal sample size based on constraints.
        
        The sample size is chosen to maximize accuracy while respecting:
        1. Memory constraints (for kernel matrix computation)
        2. Minimum samples for statistical validity
        3. Maximum samples to limit computation time
        
        Args:
            n_total: Total number of samples available
            n_features: Number of features (not directly used but kept for API)
            available_memory_mb: Available memory in megabytes
            dtype: Data type for computation
            
        Returns:
            int: Optimal number of samples to use
        """
        validate_memory_limit(available_memory_mb)
        
        # If total samples is small enough, use all
        if n_total <= self.min_samples:
            logger.warning(f"Total samples ({n_total}) <= min_samples ({self.min_samples})")
            return n_total
        
        # Memory requirement: O(n^2) for kernel matrices
        # We need memory for at least 2 kernel matrices (K and L)
        bytes_per_element = 4 if dtype == torch.float32 else 8
        
        # Memory for one n×n kernel matrix
        memory_per_kernel_mb = lambda n: (n * n * bytes_per_element) / (1024 * 1024)
        
        # We need at least 2 kernels + some overhead (use 2.5x for safety)
        total_memory_mb = lambda n: 2.5 * memory_per_kernel_mb(n)
        
        # Check if we can use all samples
        if total_memory_mb(n_total) <= available_memory_mb:
            selected = min(n_total, self.max_samples)
            logger.info(f"Memory sufficient for {selected} samples (using all available)")
            return selected
        
        # Binary search for largest feasible sample size
        left, right = self.min_samples, min(self.max_samples, n_total)
        
        while left < right:
            mid = (left + right + 1) // 2
            required_mb = total_memory_mb(mid)
            
            if required_mb <= available_memory_mb:
                left = mid
            else:
                right = mid - 1
        
        selected = left
        logger.info(
            f"Selected sample size: {selected} (from {n_total} total) "
            f"using {total_memory_mb(selected):.1f}MB of {available_memory_mb:.1f}MB available"
        )
        
        return selected
    
    def stratified_sample(
        self,
        n_total: int,
        n_samples: int,
        labels: Optional[torch.Tensor] = None,
        return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Create stratified sample indices.
        
        If labels are provided, performs stratified sampling to maintain
        class balance. Otherwise, performs uniform random sampling.
        
        Args:
            n_total: Total number of samples
            n_samples: Number of samples to select
            labels: Optional class labels for stratified sampling
            return_mask: If True, also return boolean mask
            
        Returns:
            Sample indices, or (indices, mask) if return_mask=True
        """
        if n_samples > n_total:
            raise ValidationError(f"Cannot sample {n_samples} from {n_total} total")
        
        if n_samples == n_total:
            indices = torch.arange(n_total)
            if return_mask:
                mask = torch.ones(n_total, dtype=torch.bool)
                return indices, mask
            return indices
        
        if labels is None:
            # Random sampling
            perm = torch.randperm(n_total)
            indices = perm[:n_samples]
        else:
            # Stratified sampling
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            
            if len(labels) != n_total:
                raise ValidationError(f"Labels length ({len(labels)}) != n_total ({n_total})")
            
            unique_labels = torch.unique(labels)
            n_classes = len(unique_labels)
            
            # Calculate samples per class
            samples_per_class = n_samples // n_classes
            extra_samples = n_samples % n_classes
            
            indices_list = []
            
            for i, label in enumerate(unique_labels):
                # Find indices for this class
                class_mask = labels == label
                class_indices = torch.where(class_mask)[0]
                n_class_total = len(class_indices)
                
                # Determine number of samples for this class
                n_class_samples = samples_per_class
                if i < extra_samples:
                    n_class_samples += 1
                
                # Sample from this class
                if n_class_samples >= n_class_total:
                    # Use all samples from this class
                    sampled = class_indices
                else:
                    # Random sample within class
                    perm = torch.randperm(n_class_total)
                    sampled = class_indices[perm[:n_class_samples]]
                
                indices_list.append(sampled)
            
            indices = torch.cat(indices_list)
            
            # Shuffle the final indices
            final_perm = torch.randperm(len(indices))
            indices = indices[final_perm]
        
        # Validate indices
        indices = validate_sample_indices(indices, n_total, expected_size=n_samples)
        
        if return_mask:
            mask = torch.zeros(n_total, dtype=torch.bool)
            mask[indices] = True
            return indices, mask
        
        return indices
    
    def progressive_sampling(
        self,
        n_total: int,
        batch_sizes: List[int],
        overlap: float = 0.1
    ) -> List[torch.Tensor]:
        """Generate progressive sample batches with optional overlap.
        
        Useful for computing CKA in multiple passes when memory is very limited.
        
        Args:
            n_total: Total number of samples
            batch_sizes: List of batch sizes for each pass
            overlap: Fraction of samples to overlap between batches
            
        Returns:
            List of sample indices for each batch
        """
        if overlap < 0 or overlap >= 1:
            raise ValidationError(f"Overlap must be in [0, 1), got {overlap}")
        
        batches = []
        used_indices = set()
        
        for i, batch_size in enumerate(batch_sizes):
            if batch_size > n_total:
                logger.warning(f"Batch {i} size {batch_size} > total {n_total}, using all")
                batch_size = n_total
            
            # Calculate overlap size
            overlap_size = int(batch_size * overlap) if i > 0 else 0
            new_size = batch_size - overlap_size
            
            # Get available indices
            all_indices = set(range(n_total))
            available = list(all_indices - used_indices)
            
            if len(available) < new_size:
                # Use all remaining
                new_indices = available
                # Fill rest with random from used
                if len(new_indices) < batch_size:
                    fill_size = batch_size - len(new_indices)
                    fill_from = list(used_indices)
                    if fill_from:
                        fill_indices = np.random.choice(
                            fill_from, 
                            size=min(fill_size, len(fill_from)),
                            replace=False
                        )
                        new_indices.extend(fill_indices)
            else:
                # Random sample from available
                new_indices = np.random.choice(available, size=new_size, replace=False)
                
                # Add overlap from previous batch if needed
                if overlap_size > 0 and i > 0 and batches:
                    prev_batch = batches[-1].numpy()
                    overlap_indices = np.random.choice(
                        prev_batch,
                        size=min(overlap_size, len(prev_batch)),
                        replace=False
                    )
                    new_indices = np.concatenate([overlap_indices, new_indices])
            
            batch_indices = torch.tensor(new_indices, dtype=torch.long)
            batches.append(batch_indices)
            used_indices.update(new_indices)
        
        return batches
    
    def estimate_required_samples(
        self,
        target_error: float = 0.01,
        confidence: float = 0.95
    ) -> int:
        """Estimate required sample size for target CKA estimation error.
        
        This is a heuristic based on statistical sampling theory.
        
        Args:
            target_error: Desired maximum estimation error
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Estimated number of samples needed
        """
        # Using simplified formula based on sampling theory
        # For CKA, we assume worst-case variance of 0.25 (when p=0.5)
        # Formula: n = (z^2 * variance) / error^2
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        variance = 0.25  # Conservative estimate
        
        n_estimated = int(np.ceil((z_score ** 2 * variance) / (target_error ** 2)))
        
        # Apply constraints
        n_estimated = max(self.min_samples, min(n_estimated, self.max_samples))
        
        logger.info(
            f"Estimated {n_estimated} samples needed for "
            f"{target_error:.1%} error at {confidence:.0%} confidence"
        )
        
        return n_estimated