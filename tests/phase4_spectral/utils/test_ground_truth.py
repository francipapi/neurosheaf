# tests/phase4_spectral/utils/test_ground_truth.py
"""Ground truth generators for persistent homology testing.

This module provides synthetic test cases with known theoretical results
for validating the persistent spectral analysis implementation against
established mathematical ground truth.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from neurosheaf.sheaf.construction import Sheaf


class GroundTruthGenerator:
    """Generate synthetic test cases with known persistent homology results."""
    
    def __init__(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
    
    def linear_chain_sheaf(self, n_nodes: int, stalk_dim: int = 2) -> Tuple[Sheaf, Dict]:
        """Generate linear chain (path graph) sheaf with known spectral properties.
        
        Args:
            n_nodes: Number of nodes in the chain
            stalk_dim: Dimension of each stalk
            
        Returns:
            Tuple of (Sheaf object, Expected properties dict)
        """
        # Create path graph
        poset = nx.path_graph(n_nodes, create_using=nx.DiGraph)
        
        # Create stalks (all identical) - use integer keys to match poset nodes
        stalks = {}
        for i in range(n_nodes):
            stalks[i] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype)
        
        # Create restrictions (uniform weights) - use integer keys to match poset edges
        restrictions = {}
        weight = 0.8  # Fixed weight for reproducibility
        for i in range(n_nodes - 1):
            edge = (i, i + 1)
            restrictions[edge] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype) * weight
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Expected properties for path graph Laplacian
        expected_eigenvalues = []
        for k in range(n_nodes):
            # Standard graph Laplacian eigenvalues for path graph
            # λ_k = 2(1 - cos(kπ/(n+1))) for k = 1, ..., n
            if k == 0:
                eigenval = 0.0  # Always one zero eigenvalue
            else:
                eigenval = 2 * (1 - np.cos(k * np.pi / (n_nodes + 1)))
            expected_eigenvalues.append(eigenval * stalk_dim)  # Scale by stalk dimension
        
        expected = {
            'type': 'linear_chain',
            'n_nodes': n_nodes,
            'stalk_dim': stalk_dim,
            'expected_eigenvalues': expected_eigenvalues,
            'expected_multiplicity': stalk_dim,  # Each eigenvalue has multiplicity = stalk_dim
            'expected_connected_components': 1,
            'expected_zero_eigenvalues': stalk_dim,  # Multiplicity of zero eigenvalue
            'expected_spectral_gap': expected_eigenvalues[1] if n_nodes > 1 else 0.0,
            'expected_infinite_bars': 1,  # One infinite bar for connectivity
            'expected_finite_bars_h0': 0,  # No finite bars in H_0 for connected graph
            'topology_type': 'contractible'  # Path is topologically trivial
        }
        
        return sheaf, expected
    
    def cycle_graph_sheaf(self, n_nodes: int, stalk_dim: int = 2) -> Tuple[Sheaf, Dict]:
        """Generate cycle graph sheaf with known loop topology.
        
        Args:
            n_nodes: Number of nodes in the cycle
            stalk_dim: Dimension of each stalk
            
        Returns:
            Tuple of (Sheaf object, Expected properties dict)
        """
        # Create cycle graph
        poset = nx.cycle_graph(n_nodes, create_using=nx.DiGraph)
        
        # Create stalks - use integer keys to match poset nodes
        stalks = {}
        for i in range(n_nodes):
            stalks[i] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype)
        
        # Create restrictions - use integer keys to match poset edges
        restrictions = {}
        weight = 0.8
        for i in range(n_nodes):
            edge = (i, (i + 1) % n_nodes)
            restrictions[edge] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype) * weight
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Expected eigenvalues for cycle graph
        expected_eigenvalues = []
        for k in range(n_nodes):
            # λ_k = 2(1 - cos(2πk/n)) for k = 0, 1, ..., n-1
            eigenval = 2 * (1 - np.cos(2 * np.pi * k / n_nodes))
            expected_eigenvalues.append(eigenval * stalk_dim)
        
        expected_eigenvalues.sort()
        
        expected = {
            'type': 'cycle_graph',
            'n_nodes': n_nodes,
            'stalk_dim': stalk_dim,
            'expected_eigenvalues': expected_eigenvalues,
            'expected_multiplicity': stalk_dim,
            'expected_connected_components': 1,
            'expected_zero_eigenvalues': stalk_dim,
            'expected_spectral_gap': expected_eigenvalues[1],
            'expected_infinite_bars': 1,  # One infinite bar for connectivity
            'expected_finite_bars_h1': 1,  # One finite bar for the loop
            'topology_type': 'circle',  # S^1 topology
            'first_betti_number': 1  # One independent loop
        }
        
        return sheaf, expected
    
    def complete_graph_sheaf(self, n_nodes: int, stalk_dim: int = 2) -> Tuple[Sheaf, Dict]:
        """Generate complete graph sheaf with maximal connectivity.
        
        Args:
            n_nodes: Number of nodes in complete graph
            stalk_dim: Dimension of each stalk
            
        Returns:
            Tuple of (Sheaf object, Expected properties dict)
        """
        # Create complete directed graph
        poset = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
        
        # Create stalks - use integer keys to match poset nodes
        stalks = {}
        for i in range(n_nodes):
            stalks[i] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype)
        
        # Create restrictions (all-to-all connectivity) - use integer keys
        restrictions = {}
        weight = 0.9
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge = (i, j)
                    restrictions[edge] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype) * weight
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Expected eigenvalues for complete graph
        # λ_0 = 0 (multiplicity 1), λ_1 = ... = λ_{n-1} = n (multiplicity n-1)
        expected_eigenvalues = [0.0] * stalk_dim  # Zero eigenvalue with multiplicity stalk_dim
        for _ in range((n_nodes - 1) * stalk_dim):
            expected_eigenvalues.append(n_nodes * stalk_dim)
        
        expected = {
            'type': 'complete_graph',
            'n_nodes': n_nodes,
            'stalk_dim': stalk_dim,
            'expected_eigenvalues': expected_eigenvalues,
            'expected_connected_components': 1,
            'expected_zero_eigenvalues': stalk_dim,
            'expected_spectral_gap': n_nodes * stalk_dim,
            'expected_infinite_bars': 1,
            'expected_finite_bars_h0': 0,
            'topology_type': 'complete',
            'algebraic_connectivity': n_nodes  # Very high connectivity
        }
        
        return sheaf, expected
    
    def tree_sheaf(self, depth: int, branching_factor: int = 2, stalk_dim: int = 2) -> Tuple[Sheaf, Dict]:
        """Generate tree sheaf (acyclic graph).
        
        Args:
            depth: Depth of the tree
            branching_factor: Number of children per node
            stalk_dim: Dimension of each stalk
            
        Returns:
            Tuple of (Sheaf object, Expected properties dict)
        """
        # Create balanced tree
        poset = nx.balanced_tree(branching_factor, depth, create_using=nx.DiGraph)
        n_nodes = poset.number_of_nodes()
        
        # Create stalks - use original node keys to match poset nodes
        stalks = {}
        for node in poset.nodes():
            stalks[node] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype)
        
        # Create restrictions - use original edge keys to match poset edges
        restrictions = {}
        weight = 0.85
        for edge in poset.edges():
            restrictions[edge] = (
                torch.eye(stalk_dim, device=self.device, dtype=self.dtype) * weight
            )
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        expected = {
            'type': 'tree',
            'n_nodes': n_nodes,
            'stalk_dim': stalk_dim,
            'depth': depth,
            'branching_factor': branching_factor,
            'expected_connected_components': 1,
            'expected_zero_eigenvalues': stalk_dim,
            'expected_infinite_bars': 1,
            'expected_finite_bars_h0': 0,
            'expected_finite_bars_h1': 0,  # Trees have no loops
            'topology_type': 'tree',
            'first_betti_number': 0,  # Trees are acyclic
            'euler_characteristic': 1  # χ = V - E = n - (n-1) = 1
        }
        
        return sheaf, expected
    
    def disconnected_components_sheaf(self, component_sizes: List[int], stalk_dim: int = 2) -> Tuple[Sheaf, Dict]:
        """Generate sheaf with multiple disconnected components.
        
        Args:
            component_sizes: List of sizes for each connected component
            stalk_dim: Dimension of each stalk
            
        Returns:
            Tuple of (Sheaf object, Expected properties dict)
        """
        # Create disjoint union of path graphs
        poset = nx.DiGraph()
        stalks = {}
        restrictions = {}
        
        node_offset = 0
        for comp_size in component_sizes:
            # Add path graph for this component
            for i in range(comp_size):
                node = node_offset + i
                poset.add_node(node)
                stalks[node] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype)
            
            # Add edges within component
            weight = 0.8
            for i in range(comp_size - 1):
                edge = (node_offset + i, node_offset + i + 1)
                poset.add_edge(*edge)
                restrictions[edge] = torch.eye(stalk_dim, device=self.device, dtype=self.dtype) * weight
            
            node_offset += comp_size
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        n_components = len(component_sizes)
        total_nodes = sum(component_sizes)
        
        expected = {
            'type': 'disconnected_components',
            'component_sizes': component_sizes,
            'n_components': n_components,
            'total_nodes': total_nodes,
            'stalk_dim': stalk_dim,
            'expected_connected_components': n_components,
            'expected_zero_eigenvalues': n_components * stalk_dim,  # One zero per component
            'expected_infinite_bars': n_components,  # One infinite bar per component
            'expected_finite_bars_h0': 0,
            'topology_type': 'disjoint_union'
        }
        
        return sheaf, expected
    
    def crossing_eigenvalues_sequence(self, n_steps: int = 20) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict]:
        """Generate eigenvalue sequence with known crossings for testing tracker.
        
        Args:
            n_steps: Number of filtration steps
            
        Returns:
            Tuple of (eigenvalue_sequences, eigenvector_sequences, expected_crossings)
        """
        eigenval_sequences = []
        eigenvec_sequences = []
        crossing_points = []
        
        for i in range(n_steps):
            t = i / (n_steps - 1)
            
            # Two eigenvalues that cross at t = 0.5
            eig1 = 1.0 - t  # Decreasing from 1 to 0
            eig2 = t        # Increasing from 0 to 1
            eig3 = 2.0      # Constant (no crossing)
            
            eigenvals = torch.tensor([eig1, eig2, eig3], dtype=self.dtype)
            
            # Simple identity eigenvectors (not realistic but sufficient for testing)
            eigenvecs = torch.eye(3, dtype=self.dtype)
            
            eigenval_sequences.append(eigenvals)
            eigenvec_sequences.append(eigenvecs)
            
            # Record crossing point
            if abs(t - 0.5) < 1.0 / (2 * n_steps):
                crossing_points.append(i)
        
        expected = {
            'type': 'crossing_eigenvalues',
            'n_steps': n_steps,
            'crossing_points': crossing_points,
            'expected_crossings': 1,
            'crossing_location': 0.5,
            'constant_eigenvalue': 2.0
        }
        
        return eigenval_sequences, eigenvec_sequences, expected


class PersistenceValidator:
    """Validate mathematical properties of persistence computations."""
    
    @staticmethod
    def validate_eigenvalue_properties(eigenvalues: torch.Tensor, tolerance: float = 1e-6) -> Dict[str, bool]:
        """Validate basic eigenvalue properties for Laplacian matrices.
        
        Args:
            eigenvalues: Computed eigenvalues
            tolerance: Numerical tolerance for validation
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Non-negativity (Laplacian is positive semi-definite)
        results['non_negative'] = torch.all(eigenvalues >= -tolerance).item()
        
        # Sorted order
        sorted_eigenvals = torch.sort(eigenvalues)[0]
        results['sorted'] = torch.allclose(eigenvalues, sorted_eigenvals, atol=tolerance)
        
        # At least one zero eigenvalue (connected component)
        results['has_zero'] = torch.any(eigenvalues < tolerance).item()
        
        # Finite values
        results['finite'] = torch.all(torch.isfinite(eigenvalues)).item()
        
        return results
    
    @staticmethod
    def validate_persistence_diagram(birth_death_pairs: List[Dict], tolerance: float = 1e-6) -> Dict[str, bool]:
        """Validate persistence diagram properties.
        
        Args:
            birth_death_pairs: List of persistence pairs
            tolerance: Numerical tolerance
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        if not birth_death_pairs:
            results['birth_death_ordering'] = True
            results['positive_lifetimes'] = True
            results['finite_births'] = True
            return results
        
        # Birth ≤ Death ordering
        birth_death_valid = all(
            pair['birth'] <= pair['death'] + tolerance
            for pair in birth_death_pairs
        )
        results['birth_death_ordering'] = birth_death_valid
        
        # Positive lifetimes
        positive_lifetimes = all(
            pair['lifetime'] >= -tolerance
            for pair in birth_death_pairs
        )
        results['positive_lifetimes'] = positive_lifetimes
        
        # Finite birth times
        finite_births = all(
            np.isfinite(pair['birth'])
            for pair in birth_death_pairs
        )
        results['finite_births'] = finite_births
        
        return results
    
    @staticmethod
    def validate_spectral_gap(eigenvalues: torch.Tensor, expected_gap: Optional[float] = None, 
                            tolerance: float = 1e-3, expected_zero_count: int = 1) -> Dict[str, bool]:
        """Validate spectral gap properties for sheaf Laplacians.
        
        Args:
            eigenvalues: Computed eigenvalues (sorted)
            expected_gap: Expected spectral gap value
            tolerance: Tolerance for comparison
            expected_zero_count: Expected number of zero eigenvalues (stalk dimension for connected sheaves)
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        if len(eigenvalues) < 2:
            results['gap_exists'] = False
            results['gap_matches_expected'] = expected_gap is None
            return results
        
        # Sort eigenvalues
        sorted_eigenvals = torch.sort(eigenvalues)[0]
        
        # Count zero eigenvalues
        zero_count = torch.sum(sorted_eigenvals < tolerance).item()
        results['zero_count'] = zero_count
        results['expected_zero_count'] = expected_zero_count
        results['zero_count_correct'] = zero_count == expected_zero_count
        
        # For sheaf Laplacians, spectral gap is between the last zero eigenvalue and first positive eigenvalue
        if zero_count < len(sorted_eigenvals):
            # Find first positive eigenvalue after zero eigenvalues
            first_positive_idx = zero_count
            gap = sorted_eigenvals[first_positive_idx] - sorted_eigenvals[zero_count - 1]
            
            results['gap_exists'] = gap > tolerance
            results['gap_value'] = gap.item()
            
            if expected_gap is not None:
                results['gap_matches_expected'] = abs(gap.item() - expected_gap) < tolerance
            else:
                results['gap_matches_expected'] = True
        else:
            # All eigenvalues are zero (disconnected or degenerate case)
            results['gap_exists'] = False
            results['gap_value'] = 0.0
            results['gap_matches_expected'] = expected_gap is None or expected_gap == 0.0
        
        return results


# Export main classes
__all__ = ['GroundTruthGenerator', 'PersistenceValidator']