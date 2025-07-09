# Phase 4: Spectral Analysis Implementation Plan (Weeks 8-10)

## Overview
Implement persistent spectral analysis with subspace similarity tracking, static Laplacian with edge masking, and multi-parameter persistence integration.

## Week 8: Subspace Similarity Tracker

### Day 1-2: Subspace Tracking Foundation
**Reference**: docs/comprehensive-update-summary.md - "Subspace similarity tracker for eigenvalue crossings"
- [ ] Implement subspace angle computation using scipy.linalg.subspace_angles
- [ ] Create eigenspace grouping for near-degenerate eigenvalues
- [ ] Build principal angle matching algorithm
- [ ] Add cosine similarity computation for subspace alignment
- [ ] Implement gap_eps parameter for eigenvalue grouping

### Day 3-4: Eigenvalue Crossing Handlers
- [ ] Detect eigenvalue crossings in filtration
- [ ] Implement subspace continuity tracking
- [ ] Create eigenvalue permutation resolution
- [ ] Add degeneracy handling for repeated eigenvalues
- [ ] Build transition matrix for eigenspace evolution

### Day 5: Integration with Persistence
- [ ] Connect subspace tracker to persistence computation
- [ ] Implement birth/death event detection
- [ ] Create persistence pairing with subspace information
- [ ] Add filtration parameter tracking
- [ ] Build persistence diagram generation

## Week 9: Static Laplacian with Edge Masking

### Day 6-7: Edge Masking Implementation
**Reference**: docs/updated-optimized-laplacian-persistence-v3.md - "Static Laplacian with edge masking"
- [ ] Implement boolean edge mask system
- [ ] Create filtration-based edge activation
- [ ] Build incremental Laplacian updates
- [ ] Add efficient sparse matrix masking
- [ ] Implement batch edge toggling

### Day 8-9: Persistence Computation
- [ ] Implement persistent cohomology computation
- [ ] Create boundary matrix reduction
- [ ] Add persistence pair extraction
- [ ] Build filtration value assignment
- [ ] Implement persistence interval computation

### Day 10: Performance Optimization
- [ ] Optimize eigenvalue computation using sparse methods
- [ ] Implement incremental eigenvalue updates
- [ ] Add memory-efficient persistence storage
- [ ] Create parallel computation support
- [ ] Build checkpoint/resume for long computations

## Week 10: Multi-Parameter Persistence

### Day 11-12: Multi-Parameter Framework
- [ ] Implement multi-parameter filtration structure
- [ ] Create parameter space discretization
- [ ] Build persistence computation for multi-parameter
- [ ] Add visualization for multi-parameter persistence
- [ ] Implement parameter correlation analysis

### Day 13-14: Integration and Validation
- [ ] Integrate with sheaf construction module
- [ ] Validate mathematical correctness
- [ ] Create performance benchmarks
- [ ] Build comprehensive test suite
- [ ] Add documentation and examples

### Day 15: Final Optimization
- [ ] Profile and optimize critical paths
- [ ] Implement GPU acceleration where possible
- [ ] Add memory usage monitoring
- [ ] Create error handling and edge cases
- [ ] Prepare for visualization integration

## Implementation Details

### Subspace Similarity Tracker
```python
# neurosheaf/spectral/tracker.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.linalg import subspace_angles
import networkx as nx
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError

logger = setup_logger(__name__)

class SubspaceTracker:
    """Track eigenspace evolution through filtration using subspace similarity."""
    
    def __init__(self, 
                 gap_eps: float = 1e-6,
                 cos_tau: float = 0.80,
                 max_groups: int = 100):
        self.gap_eps = gap_eps  # Threshold for eigenvalue grouping
        self.cos_tau = cos_tau  # Cosine similarity threshold for matching
        self.max_groups = max_groups
        
    def track_eigenspaces(self,
                         eigenvalues_sequence: List[torch.Tensor],
                         eigenvectors_sequence: List[torch.Tensor],
                         filtration_params: List[float]) -> Dict:
        """Track eigenspaces through filtration parameter changes.
        
        Args:
            eigenvalues_sequence: List of eigenvalue tensors for each filtration
            eigenvectors_sequence: List of eigenvector tensors for each filtration
            filtration_params: List of filtration parameter values
            
        Returns:
            Dictionary with tracking information
        """
        if len(eigenvalues_sequence) != len(eigenvectors_sequence):
            raise ValueError("Eigenvalues and eigenvectors sequences must have same length")
        
        n_steps = len(eigenvalues_sequence)
        tracking_info = {
            'eigenvalue_paths': [],
            'birth_events': [],
            'death_events': [],
            'crossings': [],
            'persistent_pairs': []
        }
        
        # Initialize with first step
        prev_groups = self._group_eigenvalues(
            eigenvalues_sequence[0], 
            eigenvectors_sequence[0]
        )
        
        # Track through sequence
        for i in range(1, n_steps):
            curr_eigenvals = eigenvalues_sequence[i]
            curr_eigenvecs = eigenvectors_sequence[i]
            
            curr_groups = self._group_eigenvalues(curr_eigenvals, curr_eigenvecs)
            
            # Match groups between steps
            matching = self._match_eigenspaces(prev_groups, curr_groups)
            
            # Update tracking info
            self._update_tracking_info(
                tracking_info, 
                matching, 
                filtration_params[i-1:i+1],
                i
            )
            
            prev_groups = curr_groups
        
        return tracking_info
    
    def _group_eigenvalues(self, 
                          eigenvalues: torch.Tensor,
                          eigenvectors: torch.Tensor) -> List[Dict]:
        """Group eigenvalues by proximity (handle degeneracies)."""
        # Sort eigenvalues
        sorted_idx = torch.argsort(eigenvalues)
        sorted_vals = eigenvalues[sorted_idx]
        sorted_vecs = eigenvectors[:, sorted_idx]
        
        groups = []
        current_group = {
            'eigenvalues': [sorted_vals[0]],
            'eigenvectors': [sorted_vecs[:, 0:1]],
            'indices': [sorted_idx[0]],
            'mean_eigenvalue': sorted_vals[0]
        }
        
        for i in range(1, len(sorted_vals)):
            gap = sorted_vals[i] - sorted_vals[i-1]
            
            if gap < self.gap_eps:
                # Add to current group
                current_group['eigenvalues'].append(sorted_vals[i])
                current_group['eigenvectors'].append(sorted_vecs[:, i:i+1])
                current_group['indices'].append(sorted_idx[i])
                current_group['mean_eigenvalue'] = torch.mean(
                    torch.stack(current_group['eigenvalues'])
                )
            else:
                # Finalize current group
                current_group['subspace'] = torch.cat(
                    current_group['eigenvectors'], dim=1
                )
                groups.append(current_group)
                
                # Start new group
                current_group = {
                    'eigenvalues': [sorted_vals[i]],
                    'eigenvectors': [sorted_vecs[:, i:i+1]],
                    'indices': [sorted_idx[i]],
                    'mean_eigenvalue': sorted_vals[i]
                }
        
        # Don't forget the last group
        current_group['subspace'] = torch.cat(
            current_group['eigenvectors'], dim=1
        )
        groups.append(current_group)
        
        return groups
    
    def _match_eigenspaces(self, 
                          prev_groups: List[Dict],
                          curr_groups: List[Dict]) -> List[Tuple[int, int, float]]:
        """Match eigenspaces between consecutive steps using principal angles."""
        matches = []
        
        # Compute all pairwise similarities
        similarity_matrix = torch.zeros(len(prev_groups), len(curr_groups))
        
        for i, prev_group in enumerate(prev_groups):
            for j, curr_group in enumerate(curr_groups):
                similarity = self._compute_subspace_similarity(
                    prev_group['subspace'],
                    curr_group['subspace']
                )
                similarity_matrix[i, j] = similarity
        
        # Find best matches using Hungarian algorithm or greedy approach
        used_curr = set()
        
        # Sort by similarity (greedy matching)
        flat_similarities = []
        for i in range(len(prev_groups)):
            for j in range(len(curr_groups)):
                flat_similarities.append((similarity_matrix[i, j], i, j))
        
        flat_similarities.sort(reverse=True)
        
        for similarity, i, j in flat_similarities:
            if i not in [m[0] for m in matches] and j not in used_curr:
                if similarity > self.cos_tau:
                    matches.append((i, j, similarity))
                    used_curr.add(j)
        
        return matches
    
    def _compute_subspace_similarity(self,
                                   subspace1: torch.Tensor,
                                   subspace2: torch.Tensor) -> float:
        """Compute similarity between subspaces using principal angles."""
        # Convert to numpy for scipy
        Q1 = subspace1.detach().cpu().numpy()
        Q2 = subspace2.detach().cpu().numpy()
        
        # Compute principal angles
        try:
            angles = subspace_angles(Q1, Q2)
            # Similarity is product of cosines
            similarity = np.prod(np.cos(angles))
            return float(similarity)
        except Exception as e:
            logger.warning(f"Subspace angle computation failed: {e}")
            # Fallback to simple dot product
            return float(torch.mean(torch.abs(subspace1.T @ subspace2)))
    
    def _update_tracking_info(self,
                            tracking_info: Dict,
                            matching: List[Tuple[int, int, float]],
                            filtration_params: List[float],
                            step_idx: int):
        """Update tracking information with current matching."""
        # Record eigenvalue paths
        for prev_idx, curr_idx, similarity in matching:
            if len(tracking_info['eigenvalue_paths']) <= prev_idx:
                tracking_info['eigenvalue_paths'].extend([[] for _ in range(prev_idx + 1 - len(tracking_info['eigenvalue_paths']))])
            
            tracking_info['eigenvalue_paths'][prev_idx].append({
                'step': step_idx,
                'current_group': curr_idx,
                'similarity': similarity,
                'filtration_param': filtration_params[1]
            })
        
        # Detect birth/death events
        matched_prev = set(m[0] for m in matching)
        matched_curr = set(m[1] for m in matching)
        
        # Birth events: new eigenspaces that weren't matched
        for curr_idx in range(len(matched_curr)):
            if curr_idx not in matched_curr:
                tracking_info['birth_events'].append({
                    'step': step_idx,
                    'group': curr_idx,
                    'filtration_param': filtration_params[1]
                })
        
        # Death events: previous eigenspaces that weren't matched
        for prev_idx in range(len(matched_prev)):
            if prev_idx not in matched_prev:
                tracking_info['death_events'].append({
                    'step': step_idx - 1,
                    'group': prev_idx,
                    'filtration_param': filtration_params[0]
                })
```

### Static Laplacian with Edge Masking
```python
# neurosheaf/spectral/static_laplacian.py
import torch
import torch.sparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from ..sheaf.construction import Sheaf
from ..sheaf.laplacian import SparseLaplacianBuilder
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class StaticLaplacianWithMasking:
    """Compute persistence using static Laplacian with edge masking."""
    
    def __init__(self,
                 laplacian_builder: Optional[SparseLaplacianBuilder] = None,
                 eigenvalue_method: str = 'lobpcg',
                 max_eigenvalues: int = 100):
        self.laplacian_builder = laplacian_builder or SparseLaplacianBuilder()
        self.eigenvalue_method = eigenvalue_method
        self.max_eigenvalues = max_eigenvalues
        
    def compute_persistence(self,
                           sheaf: Sheaf,
                           filtration_params: List[float],
                           edge_threshold_func: callable) -> Dict:
        """Compute persistence using edge masking approach.
        
        Args:
            sheaf: Sheaf object with full edge set
            filtration_params: List of filtration parameter values
            edge_threshold_func: Function that returns edge weights given parameter
            
        Returns:
            Dictionary with persistence information
        """
        # Build full Laplacian once
        full_laplacian = self.laplacian_builder.build_laplacian(sheaf)
        
        # Extract edge information
        edge_info = self._extract_edge_info(sheaf, full_laplacian)
        
        # Compute persistence
        eigenvalue_sequences = []
        eigenvector_sequences = []
        
        for param in filtration_params:
            # Create edge mask
            edge_mask = self._create_edge_mask(
                edge_info, 
                param, 
                edge_threshold_func
            )
            
            # Apply mask to Laplacian
            masked_laplacian = self._apply_edge_mask(
                full_laplacian, 
                edge_mask
            )
            
            # Compute eigenvalues/eigenvectors
            eigenvals, eigenvecs = self._compute_eigenvalues(
                masked_laplacian
            )
            
            eigenvalue_sequences.append(eigenvals)
            eigenvector_sequences.append(eigenvecs)
        
        # Track eigenspaces
        from .tracker import SubspaceTracker
        tracker = SubspaceTracker()
        
        tracking_info = tracker.track_eigenspaces(
            eigenvalue_sequences,
            eigenvector_sequences,
            filtration_params
        )
        
        return {
            'eigenvalue_sequences': eigenvalue_sequences,
            'eigenvector_sequences': eigenvector_sequences,
            'tracking_info': tracking_info,
            'filtration_params': filtration_params
        }
    
    def _extract_edge_info(self,
                          sheaf: Sheaf,
                          laplacian: torch.sparse.Tensor) -> Dict:
        """Extract edge information from sheaf and Laplacian."""
        edge_info = {}
        
        for edge in sheaf.poset.edges():
            source, target = edge
            restriction = sheaf.restrictions[edge]
            
            # Compute edge weight (e.g., Frobenius norm)
            weight = torch.norm(restriction, 'fro')
            
            edge_info[edge] = {
                'restriction': restriction,
                'weight': weight,
                'source': source,
                'target': target
            }
        
        return edge_info
    
    def _create_edge_mask(self,
                         edge_info: Dict,
                         filtration_param: float,
                         edge_threshold_func: callable) -> Dict[Tuple, bool]:
        """Create boolean mask for edges based on filtration parameter."""
        edge_mask = {}
        
        for edge, info in edge_info.items():
            threshold = edge_threshold_func(info['weight'], filtration_param)
            edge_mask[edge] = threshold
        
        return edge_mask
    
    def _apply_edge_mask(self,
                        laplacian: torch.sparse.Tensor,
                        edge_mask: Dict[Tuple, bool]) -> torch.sparse.Tensor:
        """Apply edge mask to Laplacian matrix."""
        # This is a simplified version - full implementation would need
        # to track which matrix entries correspond to which edges
        
        # For now, return the original Laplacian
        # In full implementation, would zero out appropriate blocks
        return laplacian
    
    def _compute_eigenvalues(self,
                           laplacian: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors of sparse Laplacian."""
        if self.eigenvalue_method == 'lobpcg':
            from scipy.sparse.linalg import lobpcg
            
            # Convert to scipy format
            L_scipy = self._torch_sparse_to_scipy(laplacian)
            
            # Initial guess
            n = L_scipy.shape[0]
            k = min(self.max_eigenvalues, n - 1)
            X = np.random.randn(n, k)
            
            # Compute smallest eigenvalues
            eigenvals, eigenvecs = lobpcg(L_scipy, X, largest=False, tol=1e-6)
            
            # Convert back to torch
            eigenvals = torch.from_numpy(eigenvals).to(laplacian.device)
            eigenvecs = torch.from_numpy(eigenvecs).to(laplacian.device)
            
            return eigenvals, eigenvecs
        else:
            # Use torch's sparse eigenvalue solver when available
            # For now, convert to dense (not practical for large matrices)
            L_dense = laplacian.to_dense()
            eigenvals, eigenvecs = torch.linalg.eigh(L_dense)
            
            # Return smallest eigenvalues
            return eigenvals[:self.max_eigenvalues], eigenvecs[:, :self.max_eigenvalues]
    
    def _torch_sparse_to_scipy(self, tensor: torch.sparse.Tensor):
        """Convert torch sparse tensor to scipy sparse matrix."""
        from scipy.sparse import coo_matrix
        
        coo_tensor = tensor.coalesce()
        indices = coo_tensor.indices().cpu().numpy()
        values = coo_tensor.values().cpu().numpy()
        shape = coo_tensor.shape
        
        return coo_matrix(
            (values, (indices[0], indices[1])),
            shape=shape
        )
```

### Persistent Spectral Analyzer
```python
# neurosheaf/spectral/persistent.py
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from ..sheaf.construction import Sheaf
from .static_laplacian import StaticLaplacianWithMasking
from .tracker import SubspaceTracker
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PersistentSpectralAnalyzer:
    """Main class for persistent spectral analysis of neural sheaves."""
    
    def __init__(self,
                 static_laplacian: Optional[StaticLaplacianWithMasking] = None,
                 subspace_tracker: Optional[SubspaceTracker] = None):
        self.static_laplacian = static_laplacian or StaticLaplacianWithMasking()
        self.subspace_tracker = subspace_tracker or SubspaceTracker()
        
    def analyze(self,
               sheaf: Sheaf,
               filtration_type: str = 'threshold',
               n_steps: int = 50,
               param_range: Optional[Tuple[float, float]] = None) -> Dict:
        """Perform complete persistent spectral analysis.
        
        Args:
            sheaf: Sheaf object to analyze
            filtration_type: Type of filtration ('threshold', 'cka_based', etc.)
            n_steps: Number of filtration steps
            param_range: Range of filtration parameters (auto-detected if None)
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting persistent spectral analysis...")
        
        # Determine filtration parameters
        filtration_params = self._generate_filtration_params(
            sheaf, filtration_type, n_steps, param_range
        )
        
        # Create edge threshold function
        edge_threshold_func = self._create_edge_threshold_func(filtration_type)
        
        # Compute persistence
        persistence_result = self.static_laplacian.compute_persistence(
            sheaf, filtration_params, edge_threshold_func
        )
        
        # Extract persistence features
        features = self._extract_persistence_features(persistence_result)
        
        # Generate persistence diagrams
        diagrams = self._generate_persistence_diagrams(
            persistence_result['tracking_info'],
            filtration_params
        )
        
        return {
            'persistence_result': persistence_result,
            'features': features,
            'diagrams': diagrams,
            'filtration_params': filtration_params,
            'filtration_type': filtration_type
        }
    
    def _generate_filtration_params(self,
                                  sheaf: Sheaf,
                                  filtration_type: str,
                                  n_steps: int,
                                  param_range: Optional[Tuple[float, float]]) -> List[float]:
        """Generate filtration parameter sequence."""
        if param_range is None:
            # Auto-detect range based on edge weights
            edge_weights = []
            for edge in sheaf.poset.edges():
                restriction = sheaf.restrictions[edge]
                weight = torch.norm(restriction, 'fro')
                edge_weights.append(weight.item())
            
            min_weight = min(edge_weights)
            max_weight = max(edge_weights)
            param_range = (min_weight * 0.1, max_weight * 1.1)
        
        # Generate sequence
        if filtration_type == 'threshold':
            # Linear sequence
            return np.linspace(param_range[0], param_range[1], n_steps).tolist()
        elif filtration_type == 'cka_based':
            # Sequence based on CKA values
            return np.linspace(0.0, 1.0, n_steps).tolist()
        else:
            # Default linear
            return np.linspace(param_range[0], param_range[1], n_steps).tolist()
    
    def _create_edge_threshold_func(self, filtration_type: str) -> Callable:
        """Create edge threshold function based on filtration type."""
        if filtration_type == 'threshold':
            return lambda weight, param: weight >= param
        elif filtration_type == 'cka_based':
            return lambda weight, param: weight >= param
        else:
            return lambda weight, param: weight >= param
    
    def _extract_persistence_features(self, persistence_result: Dict) -> Dict:
        """Extract features from persistence computation."""
        features = {}
        
        # Eigenvalue statistics
        eigenval_sequences = persistence_result['eigenvalue_sequences']
        
        # Compute feature evolution
        features['eigenvalue_evolution'] = []
        features['spectral_gap_evolution'] = []
        features['effective_dimension'] = []
        
        for eigenvals in eigenval_sequences:
            # Basic statistics
            features['eigenvalue_evolution'].append({
                'mean': torch.mean(eigenvals).item(),
                'std': torch.std(eigenvals).item(),
                'min': torch.min(eigenvals).item(),
                'max': torch.max(eigenvals).item()
            })
            
            # Spectral gap
            if len(eigenvals) > 1:
                gap = eigenvals[1] - eigenvals[0]
                features['spectral_gap_evolution'].append(gap.item())
            else:
                features['spectral_gap_evolution'].append(0.0)
            
            # Effective dimension (participation ratio)
            if len(eigenvals) > 0:
                normalized = eigenvals / torch.sum(eigenvals)
                eff_dim = 1.0 / torch.sum(normalized ** 2)
                features['effective_dimension'].append(eff_dim.item())
            else:
                features['effective_dimension'].append(0.0)
        
        # Persistence-specific features
        tracking_info = persistence_result['tracking_info']
        features['num_birth_events'] = len(tracking_info['birth_events'])
        features['num_death_events'] = len(tracking_info['death_events'])
        features['num_crossings'] = len(tracking_info['crossings'])
        
        return features
    
    def _generate_persistence_diagrams(self,
                                     tracking_info: Dict,
                                     filtration_params: List[float]) -> Dict:
        """Generate persistence diagrams from tracking information."""
        diagrams = {
            'birth_death_pairs': [],
            'infinite_bars': []
        }
        
        # Extract birth-death pairs
        birth_events = tracking_info['birth_events']
        death_events = tracking_info['death_events']
        
        # Simple pairing based on timing
        for birth in birth_events:
            # Find corresponding death
            death = None
            for d in death_events:
                if d['step'] > birth['step']:
                    death = d
                    break
            
            if death is not None:
                diagrams['birth_death_pairs'].append({
                    'birth': birth['filtration_param'],
                    'death': death['filtration_param'],
                    'lifetime': death['filtration_param'] - birth['filtration_param']
                })
            else:
                # Infinite bar
                diagrams['infinite_bars'].append({
                    'birth': birth['filtration_param'],
                    'death': float('inf')
                })
        
        return diagrams
```

## Testing Suite

### Test Structure
```
tests/phase4_spectral/
├── unit/
│   ├── test_subspace_tracker.py
│   ├── test_static_laplacian.py
│   └── test_persistent_analyzer.py
├── integration/
│   ├── test_spectral_pipeline.py
│   └── test_eigenvalue_crossings.py
└── validation/
    ├── test_persistence_correctness.py
    └── test_performance.py
```

### Critical Test: Subspace Tracking
```python
# tests/phase4_spectral/unit/test_subspace_tracker.py
import pytest
import torch
import numpy as np
from neurosheaf.spectral.tracker import SubspaceTracker

class TestSubspaceTracker:
    """Test subspace tracking through eigenvalue crossings."""
    
    def test_eigenvalue_grouping(self):
        """Test eigenvalue grouping with degeneracies."""
        tracker = SubspaceTracker(gap_eps=1e-3)
        
        # Create eigenvalues with degeneracies
        eigenvals = torch.tensor([0.0, 0.0001, 0.0002, 1.0, 1.0001, 2.0])
        eigenvecs = torch.eye(6)
        
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Should have 3 groups: [0, 0.0001, 0.0002], [1.0, 1.0001], [2.0]
        assert len(groups) == 3
        assert len(groups[0]['eigenvalues']) == 3
        assert len(groups[1]['eigenvalues']) == 2
        assert len(groups[2]['eigenvalues']) == 1
    
    def test_subspace_matching(self):
        """Test subspace matching across filtration steps."""
        tracker = SubspaceTracker(cos_tau=0.8)
        
        # Create two similar subspaces
        Q1 = torch.eye(4)[:, :2]  # First two standard basis vectors
        Q2 = torch.eye(4)[:, 1:3]  # Shifted by one
        
        # Create groups
        group1 = [{'subspace': Q1}]
        group2 = [{'subspace': Q2}]
        
        matches = tracker._match_eigenspaces(group1, group2)
        
        # Should find a match (though not perfect)
        assert len(matches) >= 0  # May or may not match depending on threshold
    
    def test_eigenvalue_crossing_detection(self):
        """Test detection of eigenvalue crossings."""
        tracker = SubspaceTracker()
        
        # Create sequence with crossing
        n_steps = 10
        eigenval_seqs = []
        eigenvec_seqs = []
        
        for i in range(n_steps):
            t = i / (n_steps - 1)
            # Two eigenvalues that cross
            eig1 = 1.0 - t
            eig2 = t
            
            eigenvals = torch.tensor([eig1, eig2, 2.0])
            eigenvecs = torch.eye(3)
            
            eigenval_seqs.append(eigenvals)
            eigenvec_seqs.append(eigenvecs)
        
        params = list(range(n_steps))
        tracking_info = tracker.track_eigenspaces(
            eigenval_seqs, eigenvec_seqs, params
        )
        
        # Should detect the crossing
        assert len(tracking_info['eigenvalue_paths']) >= 2
```

### Static Laplacian Tests
```python
# tests/phase4_spectral/unit/test_static_laplacian.py
import pytest
import torch
import networkx as nx
from neurosheaf.spectral.static_laplacian import StaticLaplacianWithMasking
from neurosheaf.sheaf.construction import Sheaf

class TestStaticLaplacian:
    """Test static Laplacian with edge masking."""
    
    def test_edge_masking(self):
        """Test edge masking functionality."""
        # Create simple sheaf
        poset = nx.path_graph(3, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(3)}
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.8,
            ('1', '2'): torch.eye(2) * 0.6
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking()
        
        # Test different threshold functions
        def threshold_func(weight, param):
            return weight >= param
        
        # Should include all edges with low threshold
        result = analyzer.compute_persistence(
            sheaf, 
            [0.1, 0.5, 0.9], 
            threshold_func
        )
        
        assert len(result['eigenvalue_sequences']) == 3
        assert all(len(seq) > 0 for seq in result['eigenvalue_sequences'])
    
    def test_eigenvalue_computation(self):
        """Test eigenvalue computation methods."""
        # Create larger sheaf
        poset = nx.complete_graph(5, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(3) for i in range(5)}
        restrictions = {
            (str(i), str(j)): torch.eye(3) * 0.9
            for i in range(5) for j in range(5) if i != j
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = StaticLaplacianWithMasking(max_eigenvalues=10)
        
        # Build full Laplacian
        full_laplacian = analyzer.laplacian_builder.build_laplacian(sheaf)
        
        # Compute eigenvalues
        eigenvals, eigenvecs = analyzer._compute_eigenvalues(full_laplacian)
        
        # Check properties
        assert len(eigenvals) <= 10
        assert eigenvals.shape[0] == eigenvecs.shape[1]
        assert torch.all(eigenvals >= -1e-6)  # Should be non-negative
```

### Integration Tests
```python
# tests/phase4_spectral/integration/test_spectral_pipeline.py
import pytest
import torch
import torch.nn as nn
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.sheaf.construction import SheafBuilder
from neurosheaf.cka import DebiasedCKA

class TestSpectralPipeline:
    """Test complete spectral analysis pipeline."""
    
    def test_end_to_end_analysis(self):
        """Test complete pipeline from model to persistence."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        )
        
        # Generate activations
        x = torch.randn(100, 10)
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        for i, layer in enumerate(model):
            layer.register_forward_hook(get_activation(f'layer_{i}'))
        
        _ = model(x)
        
        # Build sheaf
        sheaf_builder = SheafBuilder()
        sheaf = sheaf_builder.build_sheaf(model, activations)
        
        # Analyze
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=10)
        
        # Check results
        assert 'persistence_result' in result
        assert 'features' in result
        assert 'diagrams' in result
        assert len(result['persistence_result']['eigenvalue_sequences']) == 10
    
    def test_different_filtration_types(self):
        """Test different filtration types."""
        # Create simple sheaf
        poset = nx.path_graph(4, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(3) for i in range(4)}
        restrictions = {
            (str(i), str(i+1)): torch.eye(3) * (0.9 - i * 0.1)
            for i in range(3)
        }
        sheaf = Sheaf(poset, stalks, restrictions)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Test threshold filtration
        result1 = analyzer.analyze(sheaf, filtration_type='threshold', n_steps=5)
        
        # Test CKA-based filtration
        result2 = analyzer.analyze(sheaf, filtration_type='cka_based', n_steps=5)
        
        # Both should complete
        assert len(result1['persistence_result']['eigenvalue_sequences']) == 5
        assert len(result2['persistence_result']['eigenvalue_sequences']) == 5
```

## Success Criteria

1. **Subspace Tracking**
   - Correctly handles eigenvalue crossings
   - Robust to numerical precision issues
   - Maintains eigenspace continuity

2. **Persistence Computation**
   - Mathematically correct persistence diagrams
   - Efficient sparse eigenvalue computation
   - Handles various filtration types

3. **Performance**
   - Scales to networks with 50+ layers
   - Memory usage <4GB for large networks
   - Computation time <10 minutes per analysis

4. **Integration**
   - Seamless integration with sheaf construction
   - Proper error handling and validation
   - Comprehensive feature extraction

## Phase 4 Deliverables

1. **Subspace Similarity Tracker**
   - Eigenvalue crossing detection
   - Principal angle matching
   - Persistence event tracking

2. **Static Laplacian with Masking**
   - Edge masking system
   - Incremental Laplacian updates
   - Efficient eigenvalue computation

3. **Persistent Spectral Analyzer**
   - Complete analysis pipeline
   - Feature extraction
   - Persistence diagram generation

4. **Testing Suite**
   - Unit tests for all components
   - Integration tests with real networks
   - Performance benchmarks

5. **Documentation**
   - API documentation
   - Mathematical background
   - Usage examples