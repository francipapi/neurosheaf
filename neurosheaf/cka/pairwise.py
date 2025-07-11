"""Pairwise CKA computation with memory efficiency and checkpointing.

This module provides efficient computation of pairwise CKA matrices with
support for checkpointing, progress tracking, and memory monitoring.
"""

import torch
import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
from tqdm import tqdm

from .debiased import DebiasedCKA
from .sampling import AdaptiveSampler
from ..utils.memory import MemoryMonitor
from ..utils.logging import setup_logger
from ..utils.exceptions import ValidationError, ComputationError, MemoryError


logger = setup_logger(__name__)


class PairwiseCKA:
    """Compute pairwise CKA matrix efficiently.
    
    Features:
    - Memory-aware computation with monitoring
    - Checkpoint/resume functionality
    - Progress tracking
    - Parallel computation support (future)
    """
    
    def __init__(
        self,
        cka_computer: Optional[DebiasedCKA] = None,
        memory_limit_mb: float = 1024,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 10,
        use_nystrom: bool = False,
        nystrom_landmarks: int = 256
    ):
        """Initialize PairwiseCKA computer.
        
        Args:
            cka_computer: CKA computer instance (creates default if None)
            memory_limit_mb: Memory limit for computation
            checkpoint_dir: Directory for saving checkpoints
            checkpoint_frequency: Save checkpoint every N pairs
            use_nystrom: Whether to use Nyström approximation by default
            nystrom_landmarks: Number of landmarks for Nyström approximation
        """
        self.cka_computer = cka_computer or DebiasedCKA(use_unbiased=True)
        self.memory_limit_mb = memory_limit_mb
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.use_nystrom = use_nystrom
        self.nystrom_landmarks = nystrom_landmarks
        self.memory_monitor = MemoryMonitor(device=self.cka_computer.device)
        
        # Import NystromCKA lazily to avoid circular imports
        self._nystrom_cka = None
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpointing enabled at: {checkpoint_dir}")
        
        if use_nystrom:
            logger.info(f"Nyström approximation enabled with {nystrom_landmarks} landmarks")
    
    def compute_matrix(
        self,
        activations: Dict[str, torch.Tensor],
        layer_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        adaptive_sampling: bool = False,
        sampler: Optional[AdaptiveSampler] = None,
        use_nystrom: Optional[bool] = None
    ) -> torch.Tensor:
        """Compute full CKA matrix between layers.
        
        Args:
            activations: Dict mapping layer names to activation tensors
                        MUST be raw activations (not centered!)
            layer_names: Subset of layers to compute (default: all)
            progress_callback: Called with (current, total) progress
            adaptive_sampling: Whether to use adaptive sampling for large activations
            sampler: AdaptiveSampler instance (creates default if None and adaptive_sampling=True)
            use_nystrom: Override default Nyström setting for this computation
            
        Returns:
            CKA matrix of shape [n_layers, n_layers]
        """
        if not activations:
            raise ValidationError("Activations dictionary cannot be empty")
        
        if layer_names is None:
            layer_names = list(activations.keys())
        
        n_layers = len(layer_names)
        if n_layers < 2:
            raise ValidationError("Need at least 2 layers to compute CKA matrix")
        
        # Validate all requested layers exist
        for name in layer_names:
            if name not in activations:
                raise ValidationError(f"Layer '{name}' not found in activations")
        
        # Initialize CKA matrix
        cka_matrix = torch.zeros(n_layers, n_layers, device=self.cka_computer.device)
        
        # Check if we can resume from checkpoint
        start_i, start_j, cka_matrix = self._load_checkpoint(layer_names, cka_matrix)
        
        # Prepare for adaptive sampling if requested
        if adaptive_sampling and sampler is None:
            sampler = AdaptiveSampler()
        
        # Determine whether to use Nyström approximation
        use_nystrom = use_nystrom if use_nystrom is not None else self.use_nystrom
        
        # Compute total number of pairs (upper triangle only due to symmetry)
        total_pairs = n_layers * (n_layers + 1) // 2
        current_pair = self._get_pair_index(start_i, start_j, n_layers)
        
        logger.info(f"Computing CKA matrix for {n_layers} layers ({total_pairs} pairs)")
        if current_pair > 0:
            logger.info(f"Resuming from pair {current_pair}/{total_pairs}")
        
        # Progress tracking
        with tqdm(total=total_pairs, initial=current_pair, desc="Computing CKA") as pbar:
            for i in range(n_layers):
                for j in range(i, n_layers):
                    # Skip if already computed (when resuming)
                    if i < start_i or (i == start_i and j < start_j):
                        continue
                    
                    # Check memory before computation
                    self._check_memory()
                    
                    # Get activations
                    act_i = activations[layer_names[i]]
                    act_j = activations[layer_names[j]]
                    
                    # Apply adaptive sampling if needed
                    if adaptive_sampling and sampler:
                        sample_indices = self._get_sample_indices(
                            act_i, act_j, sampler
                        )
                        if sample_indices is not None:
                            act_i = act_i[sample_indices]
                            act_j = act_j[sample_indices]
                    
                    # Compute CKA (exact or Nyström)
                    try:
                        if use_nystrom:
                            cka_value = self._compute_nystrom_cka(act_i, act_j)
                        else:
                            cka_value = self.cka_computer.compute(act_i, act_j)
                        
                        cka_matrix[i, j] = cka_value
                        cka_matrix[j, i] = cka_value  # Symmetry
                    except Exception as e:
                        logger.error(
                            f"Failed to compute CKA for layers "
                            f"{layer_names[i]} and {layer_names[j]}: {str(e)}"
                        )
                        raise ComputationError(
                            f"CKA computation failed at ({i}, {j}): {str(e)}"
                        )
                    
                    # Update progress
                    current_pair += 1
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(current_pair, total_pairs)
                    
                    # Checkpoint periodically
                    if (self.checkpoint_dir and 
                        current_pair % self.checkpoint_frequency == 0):
                        self._save_checkpoint(cka_matrix, i, j, layer_names)
        
        # Final checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint(cka_matrix, n_layers-1, n_layers-1, layer_names, final=True)
        
        logger.info("CKA matrix computation completed")
        return cka_matrix
    
    def _check_memory(self) -> None:
        """Check memory and clear cache if needed."""
        available_mb = self.memory_monitor.available_mb()
        
        if available_mb < 100:  # Less than 100MB available
            logger.warning(f"Low memory ({available_mb:.1f}MB), clearing cache")
            self.memory_monitor.clear_cache()
            
            # Re-check after clearing
            available_mb = self.memory_monitor.available_mb()
            if available_mb < 50:  # Still very low
                logger.error(f"Critical memory level: {available_mb:.1f}MB")
                raise MemoryError(
                    f"Insufficient memory: {available_mb:.1f}MB available"
                )
    
    def _get_sample_indices(
        self,
        act_i: torch.Tensor,
        act_j: torch.Tensor,
        sampler: AdaptiveSampler
    ) -> Optional[torch.Tensor]:
        """Get sample indices for adaptive sampling if needed."""
        n_samples = act_i.shape[0]
        
        # Estimate memory requirement for full computation
        kernel_memory_mb = (n_samples * n_samples * 4 * 2) / (1024**2)
        
        if kernel_memory_mb > self.memory_limit_mb:
            # Need to subsample
            sample_size = sampler.determine_sample_size(
                n_total=n_samples,
                n_features=max(act_i.shape[1], act_j.shape[1]),
                available_memory_mb=self.memory_limit_mb
            )
            
            if sample_size < n_samples:
                logger.info(
                    f"Subsampling from {n_samples} to {sample_size} samples "
                    f"(memory: {kernel_memory_mb:.1f}MB -> "
                    f"{(sample_size * sample_size * 4 * 2) / (1024**2):.1f}MB)"
                )
                return sampler.stratified_sample(n_samples, sample_size)
        
        return None  # Use all samples
    
    def _get_pair_index(self, i: int, j: int, n_layers: int) -> int:
        """Convert (i, j) indices to linear pair index."""
        # Count pairs in completed rows
        pairs_before = sum(n_layers - k for k in range(i))
        # Add pairs in current row
        pairs_current = j - i
        return pairs_before + pairs_current
    
    def _save_checkpoint(
        self,
        cka_matrix: torch.Tensor,
        current_i: int,
        current_j: int,
        layer_names: List[str],
        final: bool = False
    ) -> None:
        """Save checkpoint to disk."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = Path(self.checkpoint_dir) / "cka_checkpoint.pkl"
        
        checkpoint = {
            'cka_matrix': cka_matrix.cpu().numpy(),
            'current_i': current_i,
            'current_j': current_j,
            'layer_names': layer_names,
            'completed': final
        }
        
        # Save atomically by writing to temp file first
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Atomic rename
        temp_path.replace(checkpoint_path)
        
        if final:
            logger.info("Saved final CKA matrix")
        else:
            logger.debug(f"Saved checkpoint at ({current_i}, {current_j})")
    
    def _load_checkpoint(
        self,
        layer_names: List[str],
        cka_matrix: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor]:
        """Load checkpoint from disk if available.
        
        Returns:
            (start_i, start_j, cka_matrix) - indices to resume from and loaded matrix
        """
        if not self.checkpoint_dir:
            return 0, 0, cka_matrix
        
        checkpoint_path = Path(self.checkpoint_dir) / "cka_checkpoint.pkl"
        
        if not checkpoint_path.exists():
            return 0, 0, cka_matrix
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Validate checkpoint
            saved_names = checkpoint['layer_names']
            if saved_names != layer_names:
                logger.warning(
                    "Checkpoint layer names don't match, starting fresh"
                )
                return 0, 0, cka_matrix
            
            # Load matrix
            loaded_matrix = torch.from_numpy(checkpoint['cka_matrix'])
            loaded_matrix = loaded_matrix.to(self.cka_computer.device)
            
            if checkpoint.get('completed', False):
                logger.info("Found completed CKA matrix in checkpoint")
                return len(layer_names), 0, loaded_matrix
            
            # Find where to resume
            i = checkpoint['current_i']
            j = checkpoint['current_j'] + 1  # Start from next pair
            
            # Handle row overflow
            n_layers = len(layer_names)
            if j >= n_layers:
                i += 1
                j = i
            
            logger.info(f"Loaded checkpoint, resuming from ({i}, {j})")
            return i, j, loaded_matrix
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)}, starting fresh")
            return 0, 0, cka_matrix
    
    def clear_checkpoint(self) -> None:
        """Clear any existing checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = Path(self.checkpoint_dir) / "cka_checkpoint.pkl"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Cleared checkpoint")
    
    def _compute_nystrom_cka(self, act_i: torch.Tensor, act_j: torch.Tensor) -> float:
        """Compute CKA using Nyström approximation.
        
        Args:
            act_i: First activation tensor
            act_j: Second activation tensor
            
        Returns:
            CKA value computed using Nyström approximation
        """
        # Lazy import to avoid circular imports
        if self._nystrom_cka is None:
            from .nystrom import NystromCKA
            self._nystrom_cka = NystromCKA(
                n_landmarks=self.nystrom_landmarks,
                device=self.cka_computer.device
            )
        
        return self._nystrom_cka.compute(act_i, act_j)
    
    def estimate_computation_method(self, activations: Dict[str, torch.Tensor]) -> str:
        """Estimate the best computation method based on data size and available memory.
        
        Args:
            activations: Dictionary of activation tensors
            
        Returns:
            Recommended method: 'exact', 'nystrom', or 'sampling'
        """
        # Find the largest activation
        max_samples = max(act.shape[0] for act in activations.values())
        max_features = max(act.shape[1] for act in activations.values())
        
        # Estimate memory for exact computation
        exact_memory_mb = self.memory_monitor.estimate_tensor_memory(
            (max_samples, max_samples), torch.float32
        ) * 2  # Two kernel matrices
        
        # Estimate memory for Nyström computation
        nystrom_memory_mb = self.memory_monitor.estimate_tensor_memory(
            (max_samples, self.nystrom_landmarks), torch.float32
        ) * 2  # Two cross-kernel matrices
        
        # Use memory limit rather than available memory for method selection
        memory_limit = self.memory_limit_mb
        
        if exact_memory_mb <= memory_limit * 0.8:
            return 'exact'
        elif nystrom_memory_mb <= memory_limit * 0.8:
            return 'nystrom'
        else:
            return 'sampling'
    
    def auto_configure(self, activations: Dict[str, torch.Tensor]) -> None:
        """Automatically configure computation method based on data size.
        
        Args:
            activations: Dictionary of activation tensors
        """
        method = self.estimate_computation_method(activations)
        
        if method == 'exact':
            self.use_nystrom = False
            logger.info("Using exact CKA computation")
        elif method == 'nystrom':
            self.use_nystrom = True
            logger.info(f"Using Nyström approximation with {self.nystrom_landmarks} landmarks")
        else:
            self.use_nystrom = True
            logger.info("Using Nyström approximation with adaptive sampling")
    
    def get_memory_usage_estimate(self, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get memory usage estimates for different computation methods.
        
        Args:
            activations: Dictionary of activation tensors
            
        Returns:
            Dictionary with memory estimates in MB for each method
        """
        max_samples = max(act.shape[0] for act in activations.values())
        
        # Exact CKA memory (two kernel matrices)
        exact_mb = self.memory_monitor.estimate_tensor_memory(
            (max_samples, max_samples), torch.float32
        ) * 2
        
        # Nyström CKA memory
        nystrom_mb = self.memory_monitor.estimate_tensor_memory(
            (max_samples, self.nystrom_landmarks), torch.float32
        ) * 2
        
        # Sampling memory (estimate with adaptive sample size, typically smaller)
        # Use a reasonable default that would be smaller than Nyström
        typical_sample_size = min(256, int(np.sqrt(self.memory_limit_mb * 1024 * 1024 / 16)))
        sampling_mb = self.memory_monitor.estimate_tensor_memory(
            (typical_sample_size, typical_sample_size), torch.float32
        ) * 2
        
        return {
            'exact_mb': exact_mb,
            'nystrom_mb': nystrom_mb,
            'sampling_mb': sampling_mb,
            'available_mb': self.memory_monitor.available_mb()
        }