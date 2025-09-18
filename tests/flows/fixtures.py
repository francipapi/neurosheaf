"""Common fixtures and helpers for alpha and t flow testing.

This module provides shared test infrastructure including synthetic sheaves,
mock builders, and validation utilities for comprehensive flow testing.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import torch
import torch.nn as nn
import pytest
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import copy

from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.spectral.flows.alpha_flow import AlphaGroupingPolicy
from neurosheaf.spectral.utils_numerical import heat_trace_slq

# Import the main analyzer - it should be available
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.sheaf.core.gw_config import GWConfig


# ============================================================================
# Small Neural Networks for Testing
# ============================================================================

class SmallPathNet(nn.Module):
    """Small path-like network: input -> fc1 -> fc2 -> fc3 -> output."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 6) 
        self.fc3 = nn.Linear(6, 2)
        
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3


class SmallStarNet(nn.Module):
    """Small star-like network with multiple branches converging."""
    
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(4, 6)
        self.branch2 = nn.Linear(4, 6)
        self.branch3 = nn.Linear(4, 6)
        self.fusion = nn.Linear(18, 8)  # 3 branches * 6 features
        self.output = nn.Linear(8, 2)
        
    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b2 = torch.relu(self.branch2(x))
        b3 = torch.relu(self.branch3(x))
        combined = torch.cat([b1, b2, b3], dim=1)
        fused = torch.relu(self.fusion(combined))
        output = self.output(fused)
        return output


# ============================================================================
# Test Sheaf Fixtures
# ============================================================================

@pytest.fixture
def small_path_sheaf():
    """Create a small path sheaf using NeurosheafAnalyzer with deterministic data."""
    # Create small network
    net = SmallPathNet()
    
    # Generate deterministic test data
    torch.manual_seed(42)
    batch_size = 32
    input_dim = 4
    x = torch.randn(batch_size, input_dim)
    
    # Force CPU device for compatibility (avoids MPS float64 issues)
    device = torch.device('cpu')
    net = net.to(device)
    x = x.to(device)
    
    # Create analyzer with CPU device and GW config
    analyzer = NeurosheafAnalyzer(device='cpu')
    gw_config = GWConfig()  # Can use default float64 on CPU
    
    # Build GW sheaf using analyze method
    result = analyzer.analyze(
        model=net,
        data=x,
        method='gromov_wasserstein',
        gw_config=gw_config
    )
    
    sheaf = result['sheaf']
    
    # Add test-specific metadata for flow testing
    if not hasattr(sheaf, 'metadata') or sheaf.metadata is None:
        sheaf.metadata = {}
        
    sheaf.metadata.update({
        'construction_info': {'type': 'test_path'},
        'edge_tags': {},  # No tags by default
    })
    
    return sheaf


@pytest.fixture 
def small_star_sheaf():
    """Create a small star sheaf using NeurosheafAnalyzer with deterministic data."""
    # Create star-like network
    net = SmallStarNet()
    
    # Generate deterministic test data
    torch.manual_seed(123)
    batch_size = 32
    input_dim = 4
    x = torch.randn(batch_size, input_dim)
    
    # Force CPU device for compatibility (avoids MPS float64 issues)
    device = torch.device('cpu')
    net = net.to(device)
    x = x.to(device)
    
    # Create analyzer with CPU device and GW config
    analyzer = NeurosheafAnalyzer(device='cpu')
    gw_config = GWConfig()  # Can use default float64 on CPU
    
    # Build GW sheaf using analyze method
    result = analyzer.analyze(
        model=net,
        data=x,
        method='gromov_wasserstein',
        gw_config=gw_config
    )
    
    sheaf = result['sheaf']
    
    # Add test-specific metadata
    if not hasattr(sheaf, 'metadata') or sheaf.metadata is None:
        sheaf.metadata = {}
        
    # Add edge tags for testing by_tag partitioning
    if hasattr(sheaf, 'restrictions'):
        edges = list(sheaf.restrictions.keys())
        edge_tags = {}
        for i, edge in enumerate(edges):
            if i < len(edges) // 2:
                edge_tags[edge] = 'primary'
            else:
                edge_tags[edge] = 'secondary'
        
        sheaf.metadata.update({
            'construction_info': {'type': 'test_star'},
            'edge_tags': edge_tags
        })
    
    return sheaf


# ============================================================================
# Fallback fixtures for when NeurosheafAnalyzer is not available
# ============================================================================

def create_minimal_sheaf_fallback(name='fallback'):
    """Create minimal sheaf structure for testing when analyzer is unavailable."""
    import networkx as nx
    
    # Create deterministic test data
    torch.manual_seed(42 if name == 'path' else 123)
    
    # Create poset structure
    poset = nx.DiGraph()
    poset.add_edge('layer1', 'layer2')
    poset.add_edge('layer2', 'layer3')
    
    # Create stalks (small tensors) - these represent network activations
    stalks = {
        'layer1': torch.randn(32, 8),  # batch_size=32, dim=8
        'layer2': torch.randn(32, 6),  # batch_size=32, dim=6
        'layer3': torch.randn(32, 4)   # batch_size=32, dim=4
    }
    
    # Create restrictions (should match stalk dimensions)
    restrictions = {
        ('layer1', 'layer2'): torch.randn(6, 8),  # target_dim x source_dim
        ('layer2', 'layer3'): torch.randn(4, 6)   # target_dim x source_dim
    }
    
    # Create GW metadata to make it a valid GW sheaf
    metadata = {
        'construction_method': 'gromov_wasserstein',  # Required for is_gw_sheaf()
        'is_gw_sheaf': True,
        'gw_costs': {
            ('layer1', 'layer2'): 0.3,
            ('layer2', 'layer3'): 0.7
        },
        'edge_tags': {
            ('layer1', 'layer2'): 'primary',
            ('layer2', 'layer3'): 'secondary'
        },
        'construction_info': {'type': f'test_{name}_fallback'},
        'num_nodes': 3,
        'num_edges': 2
    }
    
    # Create sheaf using dataclass constructor
    sheaf = Sheaf(
        poset=poset,
        stalks=stalks,
        restrictions=restrictions,
        metadata=metadata
    )
    
    return sheaf


@pytest.fixture
def fallback_path_sheaf():
    """Fallback fixture when NeurosheafAnalyzer is not available."""
    return create_minimal_sheaf_fallback('path')


@pytest.fixture  
def fallback_star_sheaf():
    """Fallback fixture when NeurosheafAnalyzer is not available.""" 
    return create_minimal_sheaf_fallback('star')


@pytest.fixture 
def joined_sheaf_pair(small_path_sheaf, small_star_sheaf):
    """Create a pair of sheaves that can be used for cross-architecture testing."""
    return small_path_sheaf, small_star_sheaf


def permute_sheaf(sheaf: Sheaf, permutation: Dict[str, str]) -> Sheaf:
    """Return a node-permuted copy of the sheaf.
    
    Args:
        sheaf: Original sheaf
        permutation: Dict mapping old node names to new node names
        
    Returns:
        New sheaf with permuted nodes and updated restrictions
    """
    import networkx as nx
    
    # Create new poset with permuted nodes
    new_poset = nx.DiGraph()
    for u, v in sheaf.poset.edges():
        new_u = permutation.get(u, u)
        new_v = permutation.get(v, v)
        new_poset.add_edge(new_u, new_v)
    
    # Add any isolated nodes
    for node in sheaf.poset.nodes():
        new_node = permutation.get(node, node)
        if new_node not in new_poset:
            new_poset.add_node(new_node)
    
    # Create permuted stalks
    new_stalks = {}
    for old_node, stalk in sheaf.stalks.items():
        new_node = permutation.get(old_node, old_node)
        new_stalks[new_node] = stalk.clone()
    
    # Create permuted restrictions
    new_restrictions = {}
    for (u, v), restriction in sheaf.restrictions.items():
        new_u = permutation.get(u, u)
        new_v = permutation.get(v, v)
        new_restrictions[(new_u, new_v)] = restriction.clone()
    
    # Update metadata with permuted keys
    new_metadata = {}
    if sheaf.metadata:
        new_metadata = copy.deepcopy(sheaf.metadata)
        
        # Update GW costs
        if 'gw_costs' in new_metadata:
            old_costs = new_metadata['gw_costs']
            new_costs = {}
            for (u, v), cost in old_costs.items():
                new_u = permutation.get(u, u)
                new_v = permutation.get(v, v)
                new_costs[(new_u, new_v)] = cost
            new_metadata['gw_costs'] = new_costs
            
        # Update edge tags  
        if 'edge_tags' in new_metadata:
            old_tags = new_metadata['edge_tags']
            new_tags = {}
            for (u, v), tag in old_tags.items():
                new_u = permutation.get(u, u)
                new_v = permutation.get(v, v)
                new_tags[(new_u, new_v)] = tag
            new_metadata['edge_tags'] = new_tags
    
    # Create new sheaf using dataclass constructor
    new_sheaf = Sheaf(
        poset=new_poset,
        stalks=new_stalks,
        restrictions=new_restrictions,
        metadata=new_metadata
    )
    
    return new_sheaf


def random_gw_costs(sheaf: Sheaf, seed: int = 42, cost_range: Tuple[float, float] = (0.0, 1.0)) -> Sheaf:
    """Add random GW costs to a sheaf with reproducible seed.
    
    Args:
        sheaf: Input sheaf
        seed: Random seed for reproducibility
        cost_range: (min_cost, max_cost) range for cost generation
        
    Returns:
        New sheaf with random costs spanning the specified range
    """
    rng = np.random.default_rng(seed)
    new_sheaf = copy.deepcopy(sheaf)
    
    # Generate random costs spanning wide quantile range
    edges = list(sheaf.restrictions.keys())
    n_edges = len(edges)
    
    # Use uniform random on [0,1] then transform to ensure wide quantile coverage
    raw_costs = rng.uniform(0, 1, size=n_edges)
    raw_costs.sort()  # Ensure we span quantiles evenly
    
    # Transform to target range
    min_cost, max_cost = cost_range
    costs = min_cost + (max_cost - min_cost) * raw_costs
    
    # Assign to edges
    gw_costs = {edge: float(cost) for edge, cost in zip(edges, costs)}
    
    # Update metadata
    if not new_sheaf.metadata:
        new_sheaf.metadata = {}
    new_sheaf.metadata.update({
        'is_gw_sheaf': True,
        'gw_costs': gw_costs,
        'cost_generation_seed': seed
    })
    
    return new_sheaf


# ============================================================================
# Mock GW Laplacian Builder
# ============================================================================

class FakeGWLaplacianBuilder:
    """Mock GW Laplacian builder for controlled testing.
    
    This fake builder allows precise control over returned matrices for testing
    flow algorithms without depending on the full GW pipeline complexity.
    """
    
    def __init__(self, default_size: int = 8, default_dtype: str = 'float64'):
        """Initialize fake builder.
        
        Args:
            default_size: Default matrix size for generated operators
            default_dtype: Default data type for matrices
        """
        self.default_size = default_size
        self.default_dtype = default_dtype
        self._call_count = 0
    
    def build_laplacian(self, sheaf: Sheaf, sparse: bool = True, 
                       mass_mode: str = 'fixed', return_mass: bool = False):
        """Build a fake Laplacian for testing.
        
        Returns a simple tridiagonal Laplacian and optionally a mass matrix.
        """
        self._call_count += 1
        n = self.default_size
        dtype = getattr(np, self.default_dtype)
        
        # Create simple tridiagonal Laplacian: -1 on off-diagonals, 2 on diagonal
        data = np.array([-1, 2, -1], dtype=dtype)
        offsets = np.array([-1, 0, 1])
        L = sp.diags(data, offsets, shape=(n, n), format='csr')
        
        # Ensure positive semi-definite (fix boundary conditions)
        L[0, 0] = 1  # Neumann boundary
        L[-1, -1] = 1
        
        if return_mass:
            # Create simple mass matrix (slightly perturbed identity)
            D_data = 1.0 + 0.1 * np.sin(np.arange(n))  # Positive diagonal
            D = sp.diags(D_data, format='csr')
            return L, D
        else:
            return L
    
    def build_laplacian_grouped(self, sheaf: Sheaf, grouping: AlphaGroupingPolicy,
                               base_edges: List, resid_edges: List, 
                               mass_mode: str = 'fixed', as_linear_operator: bool = True):
        """Build grouped Laplacians for alpha flow testing.
        
        Returns L_base, L_resid, D, and metadata for controlled alpha flow testing.
        """
        n = self.default_size
        dtype = getattr(np, self.default_dtype)
        
        # Create base Laplacian (identity-like, representing "confident" structure)
        L_base_csr = sp.eye(n, format='csr', dtype=dtype)
        
        # Create residual Laplacian (tridiagonal, representing "uncertain" structure)  
        L_resid_csr = sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr', dtype=dtype)
        L_resid_csr = L_resid_csr.T @ L_resid_csr  # Make PSD
        
        if as_linear_operator:
            # Wrap as LinearOperators
            L_base = LinearOperator((n, n), matvec=L_base_csr.dot, dtype=dtype)
            L_resid = LinearOperator((n, n), matvec=L_resid_csr.dot, dtype=dtype)
        else:
            L_base = L_base_csr
            L_resid = L_resid_csr
        
        # Create mass matrix
        D = sp.eye(n, format='csr', dtype=dtype) + 0.01 * sp.diags(np.ones(n), format='csr')
        
        # Create metadata
        metadata = {
            'base_edges': base_edges,
            'resid_edges': resid_edges,
            'builder_call_count': self._call_count,
            'matrix_size': n,
            'construction_time': 0.001,  # Fake timing
        }
        
        return L_base, L_resid, D, metadata


# ============================================================================
# Validation Helpers
# ============================================================================

def linear_operator_equals_sparse(LO: LinearOperator, A: sp.spmatrix, 
                                 k: int = 10, rng=None, atol: float = 1e-10) -> bool:
    """Test if LinearOperator matches sparse matrix via random probes.
    
    Args:
        LO: LinearOperator to test
        A: Sparse matrix to compare against
        k: Number of random probes
        rng: Random number generator
        atol: Absolute tolerance for comparison
        
    Returns:
        True if operators match within tolerance
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    n = LO.shape[0]
    assert A.shape == (n, n), f"Shape mismatch: LO {LO.shape}, A {A.shape}"
    
    for _ in range(k):
        # Generate random probe vector
        v = rng.standard_normal(n)
        
        # Compare matvec results
        y_lo = LO @ v
        y_sparse = A @ v
        
        # Check relative error
        rel_error = np.linalg.norm(y_lo - y_sparse) / (np.linalg.norm(y_sparse) + 1e-15)
        
        if rel_error > atol:
            return False
    
    return True


def trace_hutchinson(LO: LinearOperator, power: int = 1, probes: int = 256, 
                    seed: int = 0) -> float:
    """Estimate Tr(LO^power) using Hutchinson estimator.
    
    This provides an independent implementation for testing against library code.
    
    Args:
        LO: LinearOperator
        power: Matrix power (1 for trace, 2 for Tr(LO^2), etc.)
        probes: Number of Hutchinson probes
        seed: Random seed for reproducibility
        
    Returns:
        Estimated trace
    """
    rng = np.random.default_rng(seed)
    n = LO.shape[0]
    
    # Generate Rademacher probes  
    Z = rng.integers(0, 2, size=(n, probes)) * 2 - 1  # {-1, +1}
    
    trace_estimates = []
    for i in range(probes):
        z = Z[:, i]
        
        # Compute LO^power @ z iteratively
        y = z.copy().astype(float)
        for _ in range(power):
            y = LO @ y
            
        # Trace contribution
        trace_estimates.append(z.T @ y)
    
    return float(np.mean(trace_estimates))


def is_psd(A: Union[np.ndarray, sp.spmatrix], tol: float = 1e-12) -> bool:
    """Check if matrix is positive semi-definite.
    
    For small matrices only (uses dense eigenvalue computation).
    """
    if sp.issparse(A):
        A = A.toarray()
    
    if A.shape[0] > 100:
        raise ValueError("is_psd only for small matrices (n <= 100)")
    
    try:
        eigs = np.linalg.eigvals(A)
        return np.min(eigs.real) >= -tol
    except Exception:
        return False


def eigs_dense(A: Union[np.ndarray, sp.spmatrix], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dense eigenvalue computation for tiny matrices (fallback for verification).
    
    Returns:
        (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
    """
    if sp.issparse(A):
        A = A.toarray()
        
    if A.shape[0] > 64:
        raise ValueError("eigs_dense only for tiny matrices (n <= 64)")
    
    eigs, vecs = np.linalg.eigh(A)
    
    # Sort by eigenvalue magnitude and return first k
    idx = np.argsort(eigs)
    return eigs[idx[:k]], vecs[:, idx[:k]]


# ============================================================================
# Parametrization Helpers
# ============================================================================

@pytest.fixture(params=['path', 'star'])
def sheaf_topology(request, small_path_sheaf, small_star_sheaf):
    """Parametrized fixture providing different sheaf topologies."""
    if request.param == 'path':
        return small_path_sheaf
    elif request.param == 'star':
        return small_star_sheaf
    else:
        raise ValueError(f"Unknown topology: {request.param}")


@pytest.fixture(params=[42, 123, 999])
def random_seed(request):
    """Parametrized random seeds for determinism testing."""
    return request.param


@pytest.fixture(params=['float32', 'float64'])
def computation_dtype(request):
    """Parametrized computation dtypes."""
    return request.param