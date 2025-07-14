"""Core data structures for cellular sheaves.

This module defines the fundamental data structures used throughout
the neurosheaf package for representing cellular sheaves constructed
from neural network activations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
import torch

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)


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
    """
    poset: nx.DiGraph = field(default_factory=nx.DiGraph)
    stalks: Dict[str, torch.Tensor] = field(default_factory=dict)
    restrictions: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    whitening_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    
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
    """
    whitening_matrix: torch.Tensor
    eigenvalues: torch.Tensor
    condition_number: float
    rank: int
    explained_variance: float = 1.0
    
    def summary(self) -> str:
        """Get whitening summary."""
        return (f"Whitening Info:\n"
                f"  Rank: {self.rank}\n" 
                f"  Condition Number: {self.condition_number:.2e}\n"
                f"  Explained Variance: {self.explained_variance:.3f}")