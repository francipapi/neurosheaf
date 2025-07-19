"""Core data structures for directed cellular sheaves.

This module defines the fundamental data structures used for representing
directed cellular sheaves constructed from neural network activations.
These structures extend the base sheaf formulation to support complex-valued
stalks and directional encoding through complex phases.

Mathematical Foundation:
- Stalks: Complex vector spaces F(v) = C^{r_v} at each node v
- Directional Encoding: T^{(q)} = exp(i 2π q (A - A^T))
- Restriction Maps: Complex-valued with phase encoding
- Laplacian: Hermitian positive semi-definite

Key Features:
- Complex-valued stalks for directional information
- Hermitian Laplacian construction
- Real embedding for computational efficiency
- Backward compatibility with existing pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
import torch
import numpy as np

# Import base sheaf structures for compatibility
from ..sheaf.data_structures import Sheaf, SheafValidationResult

# Simple logging setup for this module
import logging
logger = logging.getLogger(__name__)


@dataclass
class DirectedSheaf:
    """Directed cellular sheaf with complex-valued stalks.
    
    Extends the base Sheaf structure to support asymmetric network analysis
    through complex-valued stalks and directional encoding. This structure
    enables the construction of Hermitian sheaf Laplacians that encode
    directional information through complex phases.
    
    Mathematical Structure:
    - Vertex stalks: F(v) = C^{r_v} (complex vector spaces)
    - Directional encoding: T^{(q)} = exp(i 2π q (A - A^T))
    - Directed restrictions: Complex-valued maps encoding asymmetry
    - Laplacian: Hermitian operator L^{F} = δ* δ
    
    The complex stalks are extended from real whitened stalks via:
    F(v) = R^{r_v} ⊗_R C, preserving the mathematical structure while
    enabling directional encoding through complex phases.
    
    Attributes:
        poset: NetworkX directed graph representing layer dependencies
        complex_stalks: Dictionary mapping node names to complex tensor data
        directed_restrictions: Dictionary mapping edges to complex restriction tensors
        directional_encoding: T^{(q)} matrix encoding directional information
        directionality_parameter: q parameter controlling directional strength
        base_sheaf: Reference to original real sheaf for compatibility
        metadata: Additional information about construction and validation
    """
    
    poset: nx.DiGraph = field(default_factory=nx.DiGraph)
    complex_stalks: Dict[str, torch.Tensor] = field(default_factory=dict)
    directed_restrictions: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    directional_encoding: Optional[torch.Tensor] = field(default=None)
    directionality_parameter: float = field(default=0.25)
    base_sheaf: Optional[Sheaf] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if not self.metadata:
            self.metadata = {
                'construction_method': 'directed_complex_extension',
                'num_nodes': len(self.poset.nodes()),
                'num_edges': len(self.poset.edges()),
                'validation_passed': False,
                'directed_sheaf': True,
                'directionality_parameter': self.directionality_parameter,
                'complex_stalks': True,
                'real_embedding_required': True
            }
        
        # Ensure directionality parameter is tracked
        self.metadata['directionality_parameter'] = self.directionality_parameter
    
    def get_complex_dimension(self) -> int:
        """Get total complex dimension of all stalks.
        
        Returns:
            Total complex dimension across all stalks
        """
        return sum(stalk.shape[-1] for stalk in self.complex_stalks.values())
    
    def get_real_dimension(self) -> int:
        """Get total real dimension (2× complex dimension for real embedding).
        
        The real dimension is twice the complex dimension due to the
        complex-to-real embedding: C^n → R^{2n}.
        
        Returns:
            Total real dimension for real embedded computation
        """
        return 2 * self.get_complex_dimension()
    
    def get_node_dimensions(self) -> Dict[str, int]:
        """Get complex dimensions for each node.
        
        Returns:
            Dictionary mapping node names to their complex dimensions
        """
        return {
            node: stalk.shape[-1] if stalk.ndim > 1 else stalk.shape[0]
            for node, stalk in self.complex_stalks.items()
        }
    
    def get_node_real_dimensions(self) -> Dict[str, int]:
        """Get real dimensions for each node (2× complex dimensions).
        
        Returns:
            Dictionary mapping node names to their real dimensions
        """
        complex_dims = self.get_node_dimensions()
        return {node: 2 * dim for node, dim in complex_dims.items()}
    
    def to_real_representation(self) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]]:
        """Convert complex stalks and restrictions to real representation.
        
        Converts complex matrices Z = X + iY to real representation:
        [[X, -Y], [Y, X]]
        
        This enables the use of existing real-valued numerical libraries
        while preserving all mathematical properties of the complex structure.
        
        Returns:
            Tuple of (real_stalks, real_restrictions)
        """
        real_stalks = {}
        real_restrictions = {}
        
        # Convert complex stalks to real representation
        for node, complex_stalk in self.complex_stalks.items():
            # Handle complex tensors
            if complex_stalk.is_complex():
                real_part = complex_stalk.real
                imag_part = complex_stalk.imag
                
                # Real embedding: [[real, -imag], [imag, real]]
                real_stalks[node] = torch.block_diag(
                    torch.cat([real_part, -imag_part], dim=-1),
                    torch.cat([imag_part, real_part], dim=-1)
                )
            else:
                # If tensor is real, embed in complex space as [[real, 0], [0, real]]
                zeros = torch.zeros_like(complex_stalk)
                real_stalks[node] = torch.block_diag(
                    torch.cat([complex_stalk, zeros], dim=-1),
                    torch.cat([zeros, complex_stalk], dim=-1)
                )
        
        # Convert complex restrictions to real representation
        for edge, complex_restriction in self.directed_restrictions.items():
            if complex_restriction.is_complex():
                real_part = complex_restriction.real
                imag_part = complex_restriction.imag
                
                # Real embedding for matrices
                real_restrictions[edge] = torch.cat([
                    torch.cat([real_part, -imag_part], dim=-1),
                    torch.cat([imag_part, real_part], dim=-1)
                ], dim=0)
            else:
                # If restriction is real, embed appropriately
                zeros = torch.zeros_like(complex_restriction)
                real_restrictions[edge] = torch.cat([
                    torch.cat([complex_restriction, zeros], dim=-1),
                    torch.cat([zeros, complex_restriction], dim=-1)
                ], dim=0)
        
        return real_stalks, real_restrictions
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get adjacency matrix from the poset.
        
        Returns:
            Adjacency matrix A where A[i,j] = 1 if edge (i,j) exists
        """
        nodes = list(self.poset.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create adjacency matrix
        adj = torch.zeros((n, n), dtype=torch.float32)
        for u, v in self.poset.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
        
        return adj
    
    def get_laplacian_structure(self) -> Dict[str, Any]:
        """Get information about the Hermitian Laplacian structure.
        
        Returns:
            Dictionary with Laplacian structure information
        """
        nodes = list(self.poset.nodes())
        edges = list(self.poset.edges())
        
        # Compute total complex and real dimensions
        complex_dim = self.get_complex_dimension()
        real_dim = self.get_real_dimension()
        
        # Estimate sparsity for Hermitian structure
        max_entries = real_dim ** 2
        # Each edge contributes to off-diagonal blocks (complex conjugate pairs)
        # Each node contributes to diagonal blocks
        actual_entries = len(edges) * 2  # Off-diagonal blocks
        for node in nodes:
            if node in self.complex_stalks:
                node_real_dim = self.get_node_real_dimensions()[node]
                actual_entries += node_real_dim ** 2  # Diagonal block
        
        sparsity = 1.0 - (actual_entries / max_entries) if max_entries > 0 else 0.0
        
        return {
            'total_complex_dimension': complex_dim,
            'total_real_dimension': real_dim,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'estimated_sparsity': sparsity,
            'memory_savings': f"{sparsity * 100:.1f}%",
            'laplacian_type': 'hermitian',
            'directionality_parameter': self.directionality_parameter,
            'real_embedding_overhead': 4.0  # 2x dimensions, 4x memory
        }
    
    def validate_complex_structure(self) -> Dict[str, Any]:
        """Validate the complex structure of the directed sheaf.
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        
        # Check that all stalks are properly defined
        for node, stalk in self.complex_stalks.items():
            if not isinstance(stalk, torch.Tensor):
                errors.append(f"Stalk for node {node} is not a torch.Tensor")
                continue
            
            # Check dimensions
            if stalk.ndim < 2:
                errors.append(f"Stalk for node {node} has insufficient dimensions: {stalk.shape}")
            
            # Check for complex dtype if applicable
            if stalk.is_complex() and stalk.dtype not in [torch.complex64, torch.complex128]:
                errors.append(f"Complex stalk for node {node} has incorrect dtype: {stalk.dtype}")
        
        # Check that restrictions are properly defined
        for edge, restriction in self.directed_restrictions.items():
            if not isinstance(restriction, torch.Tensor):
                errors.append(f"Restriction for edge {edge} is not a torch.Tensor")
                continue
            
            # Check dimensions compatibility
            u, v = edge
            if u in self.complex_stalks and v in self.complex_stalks:
                u_dim = self.get_node_dimensions()[u]
                v_dim = self.get_node_dimensions()[v]
                
                if restriction.shape[0] != v_dim or restriction.shape[1] != u_dim:
                    errors.append(f"Restriction for edge {edge} has incompatible dimensions: "
                                f"{restriction.shape} vs expected ({v_dim}, {u_dim})")
        
        # Check directional encoding matrix
        if self.directional_encoding is not None:
            n_nodes = len(self.poset.nodes())
            if self.directional_encoding.shape != (n_nodes, n_nodes):
                errors.append(f"Directional encoding matrix has wrong shape: "
                            f"{self.directional_encoding.shape} vs expected ({n_nodes}, {n_nodes})")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'num_stalks': len(self.complex_stalks),
            'num_restrictions': len(self.directed_restrictions),
            'directionality_parameter': self.directionality_parameter
        }
    
    def summary(self) -> str:
        """Get a summary string of the directed sheaf structure."""
        laplacian_info = self.get_laplacian_structure()
        validation_status = "✓" if self.metadata.get('validation_passed', False) else "✗"
        
        return (f"Directed Sheaf Summary:\n"
                f"  Nodes: {laplacian_info['num_nodes']}\n"
                f"  Edges: {laplacian_info['num_edges']}\n"
                f"  Complex dimension: {laplacian_info['total_complex_dimension']}\n"
                f"  Real dimension: {laplacian_info['total_real_dimension']}\n"
                f"  Sparsity: {laplacian_info['memory_savings']}\n"
                f"  Directionality (q): {self.directionality_parameter}\n"
                f"  Laplacian type: {laplacian_info['laplacian_type']}\n"
                f"  Validation: {validation_status}\n"
                f"  Method: {self.metadata.get('construction_method', 'unknown')}")


@dataclass
class DirectedSheafValidationResult:
    """Results from directed sheaf property validation.
    
    Extends the base validation framework to handle directed sheaf
    specific properties including Hermitian structure, complex stalks,
    and directional encoding validation.
    
    Attributes:
        valid_directed_sheaf: Whether all directed sheaf validation tests passed
        hermitian_errors: List of Hermitian structure violations
        complex_structure_errors: List of complex structure issues
        directional_encoding_errors: List of directional encoding issues
        max_error: Maximum error found across all tests
        base_validation: Results from base sheaf validation
        directionality_parameter: The q parameter used
        details: Detailed breakdown of validation results
    """
    
    valid_directed_sheaf: bool
    hermitian_errors: List[str] = field(default_factory=list)
    complex_structure_errors: List[str] = field(default_factory=list)
    directional_encoding_errors: List[str] = field(default_factory=list)
    max_error: float = 0.0
    base_validation: Optional[SheafValidationResult] = field(default=None)
    directionality_parameter: float = 0.25
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize details if not provided."""
        if not self.details:
            self.details = {
                'validation_type': 'directed_sheaf',
                'timestamp': torch.tensor(0.0),  # Will be set during validation
                'hermitian_validation': False,
                'complex_structure_validation': False,
                'directional_encoding_validation': False,
                'real_embedding_validation': False
            }
    
    @property
    def all_errors(self) -> List[str]:
        """Get all error messages combined."""
        all_errors = []
        all_errors.extend(self.hermitian_errors)
        all_errors.extend(self.complex_structure_errors)
        all_errors.extend(self.directional_encoding_errors)
        
        # Include base validation errors if available
        if self.base_validation:
            all_errors.extend(self.base_validation.transitivity_errors)
            all_errors.extend(self.base_validation.restriction_errors)
        
        return all_errors
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts by category."""
        summary = {
            'hermitian_errors': len(self.hermitian_errors),
            'complex_structure_errors': len(self.complex_structure_errors),
            'directional_encoding_errors': len(self.directional_encoding_errors),
            'total_errors': len(self.all_errors)
        }
        
        # Include base validation error counts
        if self.base_validation:
            summary['base_transitivity_errors'] = len(self.base_validation.transitivity_errors)
            summary['base_restriction_errors'] = len(self.base_validation.restriction_errors)
        
        return summary
    
    def summary(self) -> str:
        """Get validation summary."""
        status = "PASS" if self.valid_directed_sheaf else "FAIL"
        error_summary = self.get_error_summary()
        
        base_status = ""
        if self.base_validation:
            base_status = f"Base Validation: {'PASS' if self.base_validation.valid_sheaf else 'FAIL'}\n"
        
        return (f"Directed Sheaf Validation Status: {status}\n"
                f"{base_status}"
                f"Maximum Error: {self.max_error:.6f}\n"
                f"Directionality Parameter: {self.directionality_parameter}\n"
                f"Hermitian Errors: {error_summary['hermitian_errors']}\n"
                f"Complex Structure Errors: {error_summary['complex_structure_errors']}\n"
                f"Directional Encoding Errors: {error_summary['directional_encoding_errors']}\n"
                f"Total Errors: {error_summary['total_errors']}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary format."""
        result = {
            'valid_directed_sheaf': self.valid_directed_sheaf,
            'max_error': self.max_error,
            'directionality_parameter': self.directionality_parameter,
            'error_summary': self.get_error_summary(),
            'details': self.details
        }
        
        # Include base validation results if available
        if self.base_validation:
            result['base_validation'] = {
                'valid_sheaf': self.base_validation.valid_sheaf,
                'max_error': self.base_validation.max_error,
                'details': self.base_validation.details
            }
        
        return result


@dataclass
class DirectedWhiteningInfo:
    """Information about whitening transformation for directed sheaves.
    
    Extends the base WhiteningInfo to handle complex-valued transformations
    and directional encoding information.
    
    Attributes:
        complex_whitening_matrix: Complex whitening transformation matrix
        real_whitening_matrix: Real embedding of whitening matrix
        eigenvalues: Eigenvalues from the decomposition
        condition_number: Condition number of original matrix
        rank: Effective rank of the matrix
        explained_variance: Proportion of variance retained
        directionality_parameter: The q parameter used
        encoding_matrix: Directional encoding matrix T^{(q)}
    """
    
    complex_whitening_matrix: torch.Tensor
    real_whitening_matrix: torch.Tensor
    eigenvalues: torch.Tensor
    condition_number: float
    rank: int
    explained_variance: float = 1.0
    directionality_parameter: float = 0.25
    encoding_matrix: Optional[torch.Tensor] = field(default=None)
    
    def summary(self) -> str:
        """Get whitening summary for directed sheaf."""
        return (f"Directed Whitening Info:\n"
                f"  Rank: {self.rank}\n"
                f"  Condition Number: {self.condition_number:.2e}\n"
                f"  Explained Variance: {self.explained_variance:.3f}\n"
                f"  Directionality Parameter: {self.directionality_parameter}\n"
                f"  Complex Whitening: {self.complex_whitening_matrix.shape}\n"
                f"  Real Embedding: {self.real_whitening_matrix.shape}")