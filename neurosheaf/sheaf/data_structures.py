"""Core data structures for cellular sheaves.

This module defines the fundamental data structures used throughout
the neurosheaf package for representing cellular sheaves constructed
from neural network activations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
import torch
import numpy as np

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)


@dataclass
class GWCouplingInfo:
    """Information about a Gromov-Wasserstein coupling for a specific edge.
    
    This structure stores both the transport plan and associated metadata
    for GW-based restriction map construction.
    
    Attributes:
        coupling: Transport plan π_{target→source} with marginal constraints
        cost: Scalar GW distortion cost for this edge
        convergence_info: Solver convergence diagnostics
        source_node: Source node identifier in the poset
        target_node: Target node identifier in the poset
        measures_uniform: Whether uniform measures were used
    """
    coupling: torch.Tensor              # Transport plan π
    cost: float                        # Scalar GW cost
    convergence_info: Dict[str, Any]    # Solver diagnostics
    source_node: str                   # Source node ID
    target_node: str                   # Target node ID  
    measures_uniform: bool = True      # Uniform vs non-uniform measures


@dataclass
class Sheaf:
    """Cellular sheaf data structure for neural network analysis.
    
    A sheaf consists of:
    - poset: Directed acyclic graph representing network structure
    - stalks: Data attached to each node (in whitened coordinate space)
    - restrictions: Linear maps between connected nodes
    
    This structure enables spectral analysis of neural network similarity
    patterns using persistent sheaf Laplacians.
    
    Mathematical Structure:
    The restriction maps implement the general sheaf Laplacian formulation
    where each restriction R: u → v for edge (u,v) represents the composition
    F_v→e ∘ F_u→e^(-1) of the underlying vertex-to-edge restriction maps.
    This allows the implementation to handle rectangular restriction maps
    between stalks of different dimensions correctly.
    
    Attributes:
        poset: NetworkX directed graph representing layer dependencies
        stalks: Dictionary mapping node names to tensor data in whitened coordinates
        restrictions: Dictionary mapping edges (u,v) to restriction map tensors R: u → v.
                     These maps can be rectangular when stalks have different dimensions,
                     enabling the general sheaf Laplacian formulation.
        metadata: Additional information about construction and validation
        whitening_maps: Dictionary mapping node names to whitening transformations
        eigenvalue_metadata: Optional metadata for eigenvalue-preserving operations
    """
    poset: nx.DiGraph = field(default_factory=nx.DiGraph)
    stalks: Dict[str, torch.Tensor] = field(default_factory=dict)
    restrictions: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    whitening_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    eigenvalue_metadata: Optional['EigenvalueMetadata'] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if not self.metadata:
            self.metadata = {
                'construction_method': 'whitened_activations',
                'num_nodes': len(self.poset.nodes()),
                'num_edges': len(self.poset.edges()),
                'validation_passed': False,
                'whitened_coordinates': True
            }
    
    def get_laplacian_structure(self) -> Dict[str, Any]:
        """Get information about the sparse Laplacian structure.
        
        Returns:
            Dictionary with Laplacian structure information
        """
        nodes = list(self.poset.nodes())
        edges = list(self.poset.edges())
        
        # Compute total dimension
        total_dim = sum(stalk.shape[-1] if stalk.ndim > 1 else stalk.shape[0] 
                       for stalk in self.stalks.values())
        
        # Estimate sparsity
        max_entries = total_dim ** 2
        actual_entries = len(edges) * 2  # Each edge contributes to 2 off-diagonal blocks
        for node in nodes:
            if node in self.stalks:
                stalk_dim = (self.stalks[node].shape[-1] if self.stalks[node].ndim > 1 
                           else self.stalks[node].shape[0])
                actual_entries += stalk_dim ** 2  # Diagonal block
        
        sparsity = 1.0 - (actual_entries / max_entries) if max_entries > 0 else 0.0
        
        return {
            'total_dimension': total_dim,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'estimated_sparsity': sparsity,
            'memory_savings': f"{sparsity * 100:.1f}%"
        }
    
    def summary(self) -> str:
        """Get a summary string of the sheaf structure."""
        laplacian_info = self.get_laplacian_structure()
        validation_status = "✓" if self.metadata.get('validation_passed', False) else "✗"
        
        return (f"Sheaf Summary:\n"
                f"  Nodes: {laplacian_info['num_nodes']}\n"
                f"  Edges: {laplacian_info['num_edges']}\n"
                f"  Total dimension: {laplacian_info['total_dimension']}\n"
                f"  Sparsity: {laplacian_info['memory_savings']}\n"
                f"  Validation: {validation_status}\n"
                f"  Method: {self.metadata.get('construction_method', 'unknown')}")
    
    def _compute_stalk_metrics(self, stalk: torch.Tensor) -> Dict[str, Any]:
        """Compute metrics for a single stalk.
        
        Args:
            stalk: Stalk tensor
            
        Returns:
            Dictionary with stalk metrics
        """
        # Move to CPU for computations
        stalk_cpu = stalk.detach().cpu()
        
        # Basic shape info
        shape = tuple(stalk.shape)
        total_elements = stalk.numel()
        
        # Compute norms
        frobenius_norm = torch.norm(stalk_cpu, 'fro').item()
        
        # Check if it's an identity matrix (common for whitened stalks)
        is_square = stalk.shape[0] == stalk.shape[1] if stalk.ndim == 2 else False
        is_identity = False
        if is_square and stalk.shape[0] > 0:
            identity = torch.eye(stalk.shape[0])
            is_identity = torch.allclose(stalk_cpu, identity, atol=1e-6)
        
        # Compute rank for 2D tensors
        rank = None
        if stalk.ndim == 2 and min(stalk.shape) > 0:
            try:
                rank = torch.linalg.matrix_rank(stalk_cpu).item()
            except:
                rank = None
        
        return {
            'shape': shape,
            'elements': total_elements,
            'frobenius_norm': frobenius_norm,
            'is_identity': is_identity,
            'rank': rank,
            'is_square': is_square
        }
    
    def _compute_restriction_metrics(self, restriction: torch.Tensor) -> Dict[str, Any]:
        """Compute metrics for a single restriction map.
        
        Args:
            restriction: Restriction map tensor
            
        Returns:
            Dictionary with restriction metrics
        """
        # Move to CPU for computations
        restriction_cpu = restriction.detach().cpu()
        
        # Basic info
        shape = tuple(restriction.shape)
        
        # Compute norms  
        frobenius_norm = torch.norm(restriction_cpu, 'fro').item()
        
        # Compute singular values for detailed analysis
        singular_values = None
        condition_number = None
        rank = None
        
        if restriction.ndim == 2 and min(restriction.shape) > 0:
            try:
                U, S, Vt = torch.linalg.svd(restriction_cpu, full_matrices=False)
                singular_values = S
                
                # Rank (number of significant singular values)
                rank = torch.sum(S > 1e-10).item()
                
                # Condition number
                if S[0] > 1e-10 and len(S) > 0:
                    condition_number = (S[0] / S[-1]).item() if S[-1] > 1e-10 else float('inf')
                
            except:
                pass
        
        # Sparsity
        num_nonzero = torch.count_nonzero(torch.abs(restriction_cpu) > 1e-10).item()
        sparsity = 1.0 - (num_nonzero / restriction.numel()) if restriction.numel() > 0 else 0.0
        
        return {
            'shape': shape,
            'frobenius_norm': frobenius_norm,
            'rank': rank,
            'condition_number': condition_number,
            'max_singular': singular_values[0].item() if singular_values is not None and len(singular_values) > 0 else None,
            'min_singular': singular_values[-1].item() if singular_values is not None and len(singular_values) > 0 else None,
            'sparsity': sparsity,
            'num_nonzero': num_nonzero
        }
    
    def print_detailed_summary(self, max_items: int = 5, verbosity: str = 'detailed'):
        """Print comprehensive sheaf information for distinguishing different sheaves.
        
        Args:
            max_items: Maximum number of stalks/restrictions to show in detail
            verbosity: 'basic' or 'detailed' level of information
        """
        print("\n" + "="*80)
        print("SHEAF DETAILED SUMMARY")
        print("="*80)
        
        # 1. Basic Structure
        print("\n1. BASIC STRUCTURE:")
        print("-"*40)
        laplacian_info = self.get_laplacian_structure()
        print(f"  Nodes: {len(self.poset.nodes())}")
        print(f"  Edges: {len(self.poset.edges())}")
        print(f"  Total Laplacian dimension: {laplacian_info['total_dimension']}")
        print(f"  Sparsity: {laplacian_info['memory_savings']}")
        print(f"  Construction method: {self.metadata.get('construction_method', 'unknown')}")
        print(f"  Validation: {'✓' if self.metadata.get('validation_passed', False) else '✗'}")
        print(f"  Whitened coordinates: {'Yes' if self.metadata.get('whitened_coordinates', False) else 'No'}")
        
        # 2. Graph Structure
        print("\n2. GRAPH STRUCTURE:")
        print("-"*40)
        # Compute layer information
        layers = {}
        for node in self.poset.nodes():
            if '_' in node:
                layer_type = node.split('_')[0]
                layers[layer_type] = layers.get(layer_type, 0) + 1
        
        print(f"  Layer distribution:")
        for layer_type, count in sorted(layers.items()):
            print(f"    {layer_type}: {count} nodes")
        
        # Longest path
        if nx.is_directed_acyclic_graph(self.poset):
            longest_path_length = nx.dag_longest_path_length(self.poset)
            print(f"  Longest path: {longest_path_length} edges")
        
        # 3. Stalk Metrics
        print("\n3. STALK METRICS:")
        print("-"*40)
        
        stalk_metrics = {}
        all_norms = []
        all_ranks = []
        identity_count = 0
        
        for node, stalk in self.stalks.items():
            metrics = self._compute_stalk_metrics(stalk)
            stalk_metrics[node] = metrics
            all_norms.append(metrics['frobenius_norm'])
            if metrics['rank'] is not None:
                all_ranks.append(metrics['rank'])
            if metrics['is_identity']:
                identity_count += 1
        
        # Compute restriction metrics early for use in detailed stalk display
        restriction_metrics = {}
        for edge, restriction in self.restrictions.items():
            metrics = self._compute_restriction_metrics(restriction)
            restriction_metrics[edge] = metrics
        
        # Summary statistics
        print(f"  Total stalks: {len(self.stalks)}")
        print(f"  Identity stalks: {identity_count}/{len(self.stalks)}")
        print(f"  Norm range: [{min(all_norms):.6f}, {max(all_norms):.6f}]")
        print(f"  Mean norm: {np.mean(all_norms):.6f} (std: {np.std(all_norms):.6f})")
        if all_ranks:
            print(f"  Rank range: [{min(all_ranks)}, {max(all_ranks)}]")
        
        if verbosity == 'detailed':
            # Show first few stalks
            print("\n  First few stalks:")
            sorted_nodes = sorted(self.stalks.keys())
            displayed_nodes = sorted_nodes[:max_items]
            
            for i, node in enumerate(displayed_nodes):
                metrics = stalk_metrics[node]
                print(f"    {node}: shape={metrics['shape']}, norm={metrics['frobenius_norm']:.4f}, "
                      f"rank={metrics['rank']}, identity={metrics['is_identity']}")
            
            if len(sorted_nodes) > max_items:
                print(f"    ... ({len(sorted_nodes) - max_items} more stalks)")
            
            # Show restrictions between displayed stalks
            print("\n  Restrictions between displayed stalks:")
            displayed_restrictions = []
            for edge, restriction in self.restrictions.items():
                if edge[0] in displayed_nodes and edge[1] in displayed_nodes:
                    displayed_restrictions.append((edge, restriction))
            
            if displayed_restrictions:
                for (source, target), restriction in sorted(displayed_restrictions):
                    source_shape = stalk_metrics[source]['shape']
                    target_shape = stalk_metrics[target]['shape']
                    restriction_shape = tuple(restriction.shape)
                    metrics = restriction_metrics[(source, target)]
                    
                    # Check dimensional consistency
                    dim_check = "✓" if (restriction_shape[1] == source_shape[0] and 
                                       restriction_shape[0] == target_shape[0]) else "✗"
                    
                    print(f"    {source}[{source_shape[0]}] → {target}[{target_shape[0]}]:")
                    print(f"      Restriction shape: {restriction_shape} (norm={metrics['frobenius_norm']:.4f})")
                    print(f"      Dimensional flow: {source_shape} → {restriction_shape} → {target_shape} {dim_check}")
            else:
                print("    (No restrictions between displayed stalks)")
        
        # 4. Restriction Map Metrics
        print("\n4. RESTRICTION MAP METRICS:")
        print("-"*40)
        
        # restriction_metrics already computed in section 3
        all_weights = []
        all_conditions = []
        all_restriction_ranks = []
        
        for edge, metrics in restriction_metrics.items():
            all_weights.append(metrics['frobenius_norm'])
            if metrics['condition_number'] is not None and metrics['condition_number'] < float('inf'):
                all_conditions.append(metrics['condition_number'])
            if metrics['rank'] is not None:
                all_restriction_ranks.append(metrics['rank'])
        
        # Summary statistics
        print(f"  Total restrictions: {len(self.restrictions)}")
        print(f"  Weight range: [{min(all_weights):.6f}, {max(all_weights):.6f}]")
        print(f"  Mean weight: {np.mean(all_weights):.6f} (std: {np.std(all_weights):.6f})")
        
        if all_conditions:
            print(f"  Condition number range: [{min(all_conditions):.2e}, {max(all_conditions):.2e}]")
        
        if all_restriction_ranks:
            print(f"  Rank distribution: min={min(all_restriction_ranks)}, "
                  f"max={max(all_restriction_ranks)}, mean={np.mean(all_restriction_ranks):.1f}")
        
        # Average sparsity
        all_sparsities = [m['sparsity'] for m in restriction_metrics.values()]
        print(f"  Average sparsity: {np.mean(all_sparsities):.1%}")
        
        if verbosity == 'detailed':
            # Show restrictions in layer order (topological order)
            print("\n  Restrictions by layer order:")
            
            # Sort edges by source node name, then target node name
            sorted_edges = sorted(restriction_metrics.items(), 
                                key=lambda x: (x[0][0], x[0][1]))
            
            # Group by source layer
            current_source = None
            shown_count = 0
            
            for edge, metrics in sorted_edges:
                if shown_count >= max_items * 2:  # Show more items since we're grouping
                    remaining = len(sorted_edges) - shown_count
                    if remaining > 0:
                        print(f"    ... ({remaining} more restrictions)")
                    break
                
                source, target = edge
                # Extract layer info
                source_layer = source.split('_')[0] if '_' in source else source
                
                if source_layer != current_source:
                    current_source = source_layer
                    print(f"\n    From {source_layer}:")
                
                # Get stalk dimensions for dimensional check
                source_dim = stalk_metrics[source]['shape'][0] if source in stalk_metrics else '?'
                target_dim = stalk_metrics[target]['shape'][0] if target in stalk_metrics else '?'
                
                # Show full source → target with dimensions
                print(f"      {source}[{source_dim}] → {target}[{target_dim}]:")
                print(f"        Restriction: shape={metrics['shape']}, "
                      f"weight={metrics['frobenius_norm']:.4f}, rank={metrics['rank']}, "
                      f"sparsity={metrics['sparsity']:.1%}")
                shown_count += 1
        
        # 5. Special Metadata
        print("\n5. SPECIAL METADATA:")
        print("-"*40)
        
        # Eigenvalue metadata
        if self.eigenvalue_metadata:
            print(f"  Eigenvalue preservation: {'ACTIVE' if self.eigenvalue_metadata.preserve_eigenvalues else 'INACTIVE'}")
            print(f"  Hodge formulation: {'ACTIVE' if self.eigenvalue_metadata.hodge_formulation_active else 'INACTIVE'}")
            print(f"  Eigenvalue matrices: {len(self.eigenvalue_metadata.eigenvalue_matrices)}")
            
            if self.eigenvalue_metadata.condition_numbers:
                cond_values = list(self.eigenvalue_metadata.condition_numbers.values())
                print(f"  Eigenvalue condition range: [{min(cond_values):.2e}, {max(cond_values):.2e}]")
        else:
            print(f"  Eigenvalue preservation: NOT CONFIGURED")
            # Check if it's in metadata instead
            if 'preserve_eigenvalues' in self.metadata:
                print(f"  (Note: preserve_eigenvalues={self.metadata['preserve_eigenvalues']} found in metadata)")
        
        # Whitening maps
        if self.whitening_maps:
            print(f"  Whitening maps: {len(self.whitening_maps)} stored")
        
        # 6. Distinguishing Features (Summary)
        print("\n6. DISTINGUISHING FEATURES:")
        print("-"*40)
        
        # Create a "fingerprint" of the sheaf
        # Check eigenvalue mode from multiple sources
        eigenvalue_mode = False
        if self.eigenvalue_metadata and hasattr(self.eigenvalue_metadata, 'preserve_eigenvalues'):
            eigenvalue_mode = self.eigenvalue_metadata.preserve_eigenvalues
        elif 'preserve_eigenvalues' in self.metadata:
            eigenvalue_mode = self.metadata['preserve_eigenvalues']
        
        fingerprint = {
            'num_nodes': len(self.stalks),
            'num_edges': len(self.restrictions),
            'total_dim': laplacian_info['total_dimension'],
            'identity_fraction': identity_count / len(self.stalks) if self.stalks else 0,
            'mean_stalk_norm': np.mean(all_norms),
            'mean_edge_weight': np.mean(all_weights),
            'weight_variance': np.var(all_weights),
            'eigenvalue_mode': eigenvalue_mode
        }
        
        print(f"  Fingerprint hash: {hash(tuple(sorted(fingerprint.items()))) % 10000:04d}")
        print(f"  Identity fraction: {fingerprint['identity_fraction']:.1%}")
        print(f"  Weight heterogeneity: {np.std(all_weights) / np.mean(all_weights):.3f}")
        print(f"  Dimension per node: {fingerprint['total_dim'] / fingerprint['num_nodes']:.1f}")
        
        print("\n" + "="*80)
    
    def is_gw_sheaf(self) -> bool:
        """Check if this sheaf was constructed using Gromov-Wasserstein method.
        
        Returns:
            True if sheaf uses GW construction method
        """
        return self.metadata.get('construction_method') == 'gromov_wasserstein'
    
    def get_gw_costs(self) -> Optional[Dict[Tuple[str, str], float]]:
        """Get GW distortion costs for all edges if available.
        
        Returns:
            Dictionary mapping edges to GW costs, or None if not a GW sheaf
        """
        if not self.is_gw_sheaf():
            return None
        return self.metadata.get('gw_costs', {})
    
    def get_gw_couplings(self) -> Optional[Dict[Tuple[str, str], torch.Tensor]]:
        """Get GW transport plans for all edges if available.
        
        Returns:
            Dictionary mapping edges to GW couplings, or None if not a GW sheaf
        """
        if not self.is_gw_sheaf():
            return None
        return self.metadata.get('gw_couplings', {})
    
    def get_gw_config(self) -> Optional[Dict[str, Any]]:
        """Get GW configuration used for construction if available.
        
        Returns:
            GW configuration dictionary, or None if not a GW sheaf
        """
        if not self.is_gw_sheaf():
            return None
        return self.metadata.get('gw_config', {})
    
    def validate_gw_quasi_sheaf_property(self, tolerance: float = 0.1) -> Optional[Dict[str, Any]]:
        """Validate quasi-sheaf property for GW-constructed sheaves.
        
        For GW sheaves, checks functoriality violations: ||ρ_{k→i} - ρ_{j→i} ∘ ρ_{k→j}||_F ≤ tolerance
        
        Args:
            tolerance: Maximum allowed functoriality violation (ε-sheaf threshold)
            
        Returns:
            Validation results dictionary, or None if not a GW sheaf
        """
        if not self.is_gw_sheaf():
            return None
        
        violations = []
        max_violation = 0.0
        
        # Find all 3-node paths for transitivity check
        for node_i in self.poset.nodes():
            for node_j in self.poset.successors(node_i):
                for node_k in self.poset.successors(node_j):
                    # Check path i -> j -> k
                    edge_ij = (node_i, node_j)
                    edge_jk = (node_j, node_k)
                    edge_ik = (node_i, node_k)
                    
                    if (edge_ij in self.restrictions and 
                        edge_jk in self.restrictions and 
                        edge_ik in self.restrictions):
                        
                        # Compute ρ_{k→i} - ρ_{j→i} ∘ ρ_{k→j}
                        R_ij = self.restrictions[edge_ij]  # ρ_{j→i}
                        R_jk = self.restrictions[edge_jk]  # ρ_{k→j}  
                        R_ik = self.restrictions[edge_ik]  # ρ_{k→i}
                        
                        # Composition: ρ_{j→i} ∘ ρ_{k→j}
                        composed = R_ij @ R_jk
                        
                        # Functoriality violation
                        violation = torch.norm(R_ik - composed, 'fro').item()
                        violations.append({
                            'path': f"{node_i} -> {node_j} -> {node_k}",
                            'violation': violation
                        })
                        max_violation = max(max_violation, violation)
        
        return {
            'max_violation': max_violation,
            'mean_violation': np.mean([v['violation'] for v in violations]) if violations else 0.0,
            'num_paths_checked': len(violations),
            'violations': violations,
            'satisfies_quasi_sheaf': max_violation <= tolerance,
            'tolerance_used': tolerance
        }
    
    def get_gw_edge_weights_for_filtration(self) -> Optional[Dict[Tuple[str, str], float]]:
        """Get edge weights appropriate for GW-based filtration.
        
        For GW sheaves, returns GW costs (lower = better match, increasing filtration).
        For standard sheaves, returns restriction norms (higher = stronger, decreasing filtration).
        
        Returns:
            Dictionary mapping edges to weights, or None if no weights available
        """
        if self.is_gw_sheaf():
            # GW costs: lower = better match = added first in increasing filtration
            return self.get_gw_costs()
        else:
            # Standard: restriction norms, higher = stronger = kept longest in decreasing filtration
            weights = {}
            for edge, restriction in self.restrictions.items():
                weights[edge] = torch.norm(restriction, 'fro').item()
            return weights
    
    def get_filtration_semantics(self) -> str:
        """Get appropriate filtration semantics for this sheaf type.
        
        Returns:
            'increasing' for GW sheaves, 'decreasing' for standard sheaves
        """
        return 'increasing' if self.is_gw_sheaf() else 'decreasing'


@dataclass
class SheafValidationResult:
    """Results from sheaf property validation.
    
    Attributes:
        valid_sheaf: Whether all validation tests passed
        transitivity_errors: List of transitivity violations
        restriction_errors: List of restriction map issues
        max_error: Maximum error found across all tests
        details: Detailed breakdown of validation results
    """
    valid_sheaf: bool
    transitivity_errors: List[str] = field(default_factory=list)
    restriction_errors: List[str] = field(default_factory=list)
    max_error: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get validation summary."""
        status = "PASS" if self.valid_sheaf else "FAIL"
        return (f"Validation Status: {status}\n"
                f"Maximum Error: {self.max_error:.6f}\n"
                f"Transitivity Errors: {len(self.transitivity_errors)}\n"
                f"Restriction Errors: {len(self.restriction_errors)}")


@dataclass  
class WhiteningInfo:
    """Information about whitening transformation applied to data.
    
    Attributes:
        whitening_matrix: The whitening transformation matrix W
        eigenvalues: Eigenvalues from the decomposition
        condition_number: Condition number of original matrix
        rank: Effective rank of the matrix
        explained_variance: Proportion of variance retained
        eigenvalue_diagonal: Diagonal eigenvalue matrix (if preserve_eigenvalues=True)
        preserve_eigenvalues: Whether eigenvalue preservation mode is active
    """
    whitening_matrix: torch.Tensor
    eigenvalues: torch.Tensor
    condition_number: float
    rank: int
    explained_variance: float = 1.0
    eigenvalue_diagonal: Optional[torch.Tensor] = None
    preserve_eigenvalues: bool = False
    
    def summary(self) -> str:
        """Get whitening summary."""
        eigenvalue_status = "ENABLED" if self.preserve_eigenvalues else "DISABLED"
        return (f"Whitening Info:\n"
                f"  Rank: {self.rank}\n" 
                f"  Condition Number: {self.condition_number:.2e}\n"
                f"  Explained Variance: {self.explained_variance:.3f}\n"
                f"  Eigenvalue Preservation: {eigenvalue_status}")


@dataclass
class EigenvalueMetadata:
    """Metadata for eigenvalue-preserving operations.
    
    This dataclass tracks information specific to eigenvalue-preserving whitening
    mode, including eigenvalue matrices for each stalk and regularization details.
    
    Attributes:
        eigenvalue_matrices: Dictionary mapping node names to their eigenvalue diagonal matrices
        condition_numbers: Condition numbers for each eigenvalue matrix
        regularization_applied: Whether regularization was applied to each matrix
        preserve_eigenvalues: Whether eigenvalue preservation mode is active
        hodge_formulation_active: Whether Hodge formulation is being used
    """
    eigenvalue_matrices: Dict[str, torch.Tensor] = field(default_factory=dict)
    condition_numbers: Dict[str, float] = field(default_factory=dict)
    regularization_applied: Dict[str, bool] = field(default_factory=dict)
    preserve_eigenvalues: bool = False
    hodge_formulation_active: bool = False
    
    def summary(self) -> str:
        """Get eigenvalue metadata summary."""
        mode_status = "ACTIVE" if self.preserve_eigenvalues else "INACTIVE"
        hodge_status = "ACTIVE" if self.hodge_formulation_active else "INACTIVE"
        num_stalks = len(self.eigenvalue_matrices)
        
        return (f"Eigenvalue Metadata:\n"
                f"  Preservation Mode: {mode_status}\n"
                f"  Hodge Formulation: {hodge_status}\n"
                f"  Number of Stalks: {num_stalks}")
    
    def get_regularization_summary(self) -> Dict[str, Any]:
        """Get summary of regularization applied."""
        total_stalks = len(self.eigenvalue_matrices)
        regularized_stalks = sum(self.regularization_applied.values())
        
        return {
            'total_stalks': total_stalks,
            'regularized_stalks': regularized_stalks,
            'regularization_fraction': regularized_stalks / total_stalks if total_stalks > 0 else 0.0,
            'condition_numbers': dict(self.condition_numbers)
        }