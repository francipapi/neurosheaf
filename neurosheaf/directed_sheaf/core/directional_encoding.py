"""Directional encoding computation for directed cellular sheaves.

This module implements the computation of the directional encoding matrix
T^{(q)} = exp(i 2π q (A - A^T)) as specified in the directed sheaf mathematical
formulation.

Mathematical Foundation:
- Directional Parameter: q ∈ [0, 1] controls directional strength
- Adjacency Matrix: A where A[i,j] = 1 if edge (i,j) exists
- Antisymmetric Part: A - A^T encodes directed structure
- Complex Exponential: T^{(q)} = exp(i 2π q (A - A^T))

Key Properties:
- T^{(q)} is a complex matrix with unit magnitude entries
- Diagonal entries are always 1 (no self-loops)
- For undirected graphs: T^{(q)} = I (identity matrix)
- For q = 0: T^{(q)} = I (reduces to undirected case)
- For q = 1/4: T^{(q)} gives quarter-turn phase encoding

This encoding enables the phase information that distinguishes directed
from undirected cellular sheaves.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import networkx as nx
import math

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


class DirectionalEncodingComputer:
    """Computes directional encoding matrix T^{(q)} for directed sheaves.
    
    This class implements the mathematical computation of the directional
    encoding matrix that enables directed cellular sheaf construction:
    
    T^{(q)} = exp(i 2π q (A - A^T))
    
    where:
    - q ∈ [0, 1] is the directionality parameter
    - A is the adjacency matrix of the directed graph
    - A - A^T is the antisymmetric part encoding direction
    
    Key Features:
    - Element-wise complex exponential computation
    - Automatic adjacency matrix extraction from NetworkX graphs
    - Comprehensive validation of mathematical properties
    - Support for different directionality parameters
    - Efficient computation using PyTorch operations
    
    The encoding matrix has unit magnitude entries and provides the phase
    information that distinguishes directed from undirected sheaves.
    """
    
    def __init__(self, q: float = 0.25, validate_properties: bool = True, 
                 tolerance: float = 1e-12, device: Optional[torch.device] = None):
        """Initialize the directional encoding computer.
        
        Args:
            q: Directionality parameter in [0, 1]
            validate_properties: Whether to validate mathematical properties
            tolerance: Tolerance for numerical validation
            device: PyTorch device for computations
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Directionality parameter q must be in [0, 1], got {q}")
        
        self.q = q
        self.validate_properties = validate_properties
        self.tolerance = tolerance
        self.device = device or torch.device('cpu')
        
        # Precompute constants
        self.two_pi_q = 2 * math.pi * q
        
        logger.debug(f"DirectionalEncodingComputer initialized with q={q}, device={device}")
    
    def compute_encoding_matrix(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute directional encoding matrix T^{(q)} = exp(i 2π q (A - A^T)).
        
        Mathematical Implementation:
        1. Compute antisymmetric part: A - A^T
        2. Scale by directionality parameter: 2π q (A - A^T)
        3. Apply complex exponential element-wise
        
        Args:
            adjacency: Adjacency matrix A of shape (n, n)
            
        Returns:
            Directional encoding matrix T^{(q)} of shape (n, n)
            
        Raises:
            ValueError: If adjacency matrix is invalid
            RuntimeError: If computation fails
        """
        # Validate adjacency matrix
        if not isinstance(adjacency, torch.Tensor):
            raise ValueError("Adjacency matrix must be a torch.Tensor")
        
        if adjacency.ndim != 2:
            raise ValueError(f"Adjacency matrix must be 2D, got shape {adjacency.shape}")
        
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adjacency.shape}")
        
        # Move to specified device
        adjacency = adjacency.to(self.device)
        
        # Compute antisymmetric part: A - A^T
        antisymmetric = adjacency - adjacency.T
        
        # Scale by directionality parameter: 2π q (A - A^T)
        scaled_antisymmetric = self.two_pi_q * antisymmetric
        
        # Apply complex exponential: exp(i * scaled_antisymmetric)
        # Using Euler's formula: exp(i θ) = cos(θ) + i sin(θ)
        real_part = torch.cos(scaled_antisymmetric)
        imag_part = torch.sin(scaled_antisymmetric)
        
        # Create complex tensor
        encoding_matrix = torch.complex(real_part, imag_part)
        
        # Validate properties if requested
        if self.validate_properties:
            self._validate_encoding_matrix(encoding_matrix, adjacency)
        
        logger.debug(f"Computed encoding matrix with shape {encoding_matrix.shape}, q={self.q}")
        return encoding_matrix
    
    def compute_from_poset(self, poset: nx.DiGraph, 
                          node_ordering: Optional[List[str]] = None) -> torch.Tensor:
        """Compute encoding matrix from NetworkX directed graph.
        
        Extracts the adjacency matrix from a NetworkX directed graph and
        computes the directional encoding matrix. Node ordering can be
        specified for consistent matrix indexing.
        
        Args:
            poset: NetworkX directed graph
            node_ordering: Optional list specifying node order for matrix indexing
            
        Returns:
            Directional encoding matrix T^{(q)}
            
        Raises:
            ValueError: If poset is invalid
        """
        if not isinstance(poset, nx.DiGraph):
            raise ValueError("Input must be a NetworkX DiGraph")
        
        if len(poset.nodes()) == 0:
            raise ValueError("Poset has no nodes")
        
        # Determine node ordering
        if node_ordering is None:
            node_ordering = list(poset.nodes())
        else:
            # Validate node ordering
            if set(node_ordering) != set(poset.nodes()):
                raise ValueError("Node ordering must contain exactly the nodes in the poset")
        
        # Extract adjacency matrix
        adjacency = self._extract_adjacency_matrix(poset, node_ordering)
        
        # Compute encoding matrix
        encoding_matrix = self.compute_encoding_matrix(adjacency)
        
        logger.debug(f"Computed encoding matrix from poset with {len(node_ordering)} nodes")
        return encoding_matrix
    
    def compute_with_metadata(self, adjacency: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute encoding matrix with detailed metadata.
        
        Args:
            adjacency: Adjacency matrix A
            
        Returns:
            Tuple of (encoding_matrix, metadata)
        """
        # Compute encoding matrix
        encoding_matrix = self.compute_encoding_matrix(adjacency)
        
        # Compute metadata
        antisymmetric = adjacency - adjacency.T
        
        metadata = {
            'directionality_parameter': self.q,
            'adjacency_shape': adjacency.shape,
            'encoding_shape': encoding_matrix.shape,
            'num_nodes': adjacency.shape[0],
            'num_edges': int(adjacency.sum().item()),
            'antisymmetric_norm': torch.norm(antisymmetric).item(),
            'encoding_properties': self._analyze_encoding_properties(encoding_matrix),
            'validation_passed': self.validate_properties,
            'tolerance': self.tolerance,
            'device': str(self.device)
        }
        
        return encoding_matrix, metadata
    
    def _extract_adjacency_matrix(self, poset: nx.DiGraph, node_ordering: List[str]) -> torch.Tensor:
        """Extract adjacency matrix from NetworkX directed graph.
        
        Args:
            poset: NetworkX directed graph
            node_ordering: List of node names for matrix indexing
            
        Returns:
            Adjacency matrix as torch.Tensor
        """
        n = len(node_ordering)
        node_to_idx = {node: i for i, node in enumerate(node_ordering)}
        
        # Initialize adjacency matrix
        adjacency = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        
        # Fill adjacency matrix
        for u, v in poset.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            adjacency[i, j] = 1.0
        
        return adjacency
    
    def _validate_encoding_matrix(self, encoding_matrix: torch.Tensor, adjacency: torch.Tensor) -> None:
        """Validate mathematical properties of the encoding matrix.
        
        Args:
            encoding_matrix: Computed encoding matrix
            adjacency: Original adjacency matrix
            
        Raises:
            RuntimeError: If validation fails
        """
        # Check complex dtype
        if not encoding_matrix.is_complex():
            raise RuntimeError("Encoding matrix must be complex")
        
        # Check unit magnitude for all entries
        magnitudes = torch.abs(encoding_matrix)
        magnitude_error = torch.abs(magnitudes - 1.0).max()
        if magnitude_error > self.tolerance:
            raise RuntimeError(f"Encoding matrix entries must have unit magnitude, max error: {magnitude_error}")
        
        # Check diagonal entries are 1 (no self-loops)
        diagonal = torch.diag(encoding_matrix)
        diagonal_error = torch.abs(diagonal - 1.0).max()
        if diagonal_error > self.tolerance:
            raise RuntimeError(f"Diagonal entries must be 1, max error: {diagonal_error}")
        
        # Check symmetry properties for undirected edges
        # For symmetric edges, encoding should be conjugate transpose
        symmetric_edges = self._find_symmetric_edges(adjacency)
        for i, j in symmetric_edges:
            expected_ji = torch.conj(encoding_matrix[i, j])
            if torch.abs(encoding_matrix[j, i] - expected_ji).max() > self.tolerance:
                raise RuntimeError(f"Symmetric edges must have conjugate encoding: ({i}, {j})")
        
        # Check reduction to all ones for q = 0
        if abs(self.q) < self.tolerance:
            ones_matrix = torch.ones(encoding_matrix.shape, dtype=encoding_matrix.dtype, device=encoding_matrix.device)
            ones_error = torch.abs(encoding_matrix - ones_matrix).max()
            if ones_error > self.tolerance:
                raise RuntimeError(f"For q=0, encoding matrix must be all ones, max error: {ones_error}")
        
        logger.debug("Encoding matrix validation passed")
    
    def _find_symmetric_edges(self, adjacency: torch.Tensor) -> List[Tuple[int, int]]:
        """Find symmetric edges in adjacency matrix.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            List of (i, j) pairs that are symmetric edges
        """
        symmetric_edges = []
        n = adjacency.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0 and adjacency[j, i] > 0:
                    symmetric_edges.append((i, j))
        
        return symmetric_edges
    
    def _analyze_encoding_properties(self, encoding_matrix: torch.Tensor) -> Dict[str, Any]:
        """Analyze mathematical properties of the encoding matrix.
        
        Args:
            encoding_matrix: Computed encoding matrix
            
        Returns:
            Dictionary with property analysis
        """
        properties = {}
        
        try:
            # Basic properties
            properties['is_complex'] = encoding_matrix.is_complex()
            properties['shape'] = encoding_matrix.shape
            properties['dtype'] = encoding_matrix.dtype
            
            # Magnitude analysis
            magnitudes = torch.abs(encoding_matrix)
            properties['magnitude_mean'] = magnitudes.mean().item()
            properties['magnitude_std'] = magnitudes.std().item()
            properties['magnitude_max'] = magnitudes.max().item()
            properties['magnitude_min'] = magnitudes.min().item()
            
            # Phase analysis
            phases = torch.angle(encoding_matrix)
            properties['phase_mean'] = phases.mean().item()
            properties['phase_std'] = phases.std().item()
            properties['phase_max'] = phases.max().item()
            properties['phase_min'] = phases.min().item()
            
            # Diagonal analysis
            diagonal = torch.diag(encoding_matrix)
            properties['diagonal_all_ones'] = torch.allclose(diagonal, torch.ones_like(diagonal), atol=self.tolerance)
            properties['diagonal_max_error'] = torch.abs(diagonal - 1.0).max().item()
            
            # Hermitian analysis
            hermitian_error = torch.abs(encoding_matrix - encoding_matrix.conj().T).max().item()
            properties['is_hermitian'] = hermitian_error <= self.tolerance
            properties['hermitian_error'] = hermitian_error
            
            # Unitary analysis
            product = encoding_matrix @ encoding_matrix.conj().T
            identity = torch.eye(encoding_matrix.shape[0], dtype=encoding_matrix.dtype, device=encoding_matrix.device)
            unitary_error = torch.abs(product - identity).max().item()
            properties['is_unitary'] = unitary_error <= self.tolerance
            properties['unitary_error'] = unitary_error
            
        except Exception as e:
            properties['analysis_error'] = str(e)
            logger.warning(f"Property analysis failed: {e}")
        
        return properties
    
    def validate_directionality_parameter(self, q: float) -> bool:
        """Validate directionality parameter.
        
        Args:
            q: Directionality parameter
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(q, (int, float)) and 0.0 <= q <= 1.0
    
    def compute_phase_matrix(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute phase matrix without exponential (for analysis).
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            Phase matrix 2π q (A - A^T)
        """
        antisymmetric = adjacency - adjacency.T
        return self.two_pi_q * antisymmetric
    
    def compare_directionality_parameters(self, adjacency: torch.Tensor, 
                                        q_values: List[float]) -> Dict[float, torch.Tensor]:
        """Compare encoding matrices for different directionality parameters.
        
        Args:
            adjacency: Adjacency matrix
            q_values: List of directionality parameters to compare
            
        Returns:
            Dictionary mapping q values to encoding matrices
        """
        results = {}
        
        for q in q_values:
            if not self.validate_directionality_parameter(q):
                logger.warning(f"Invalid directionality parameter: {q}")
                continue
            
            # Temporarily change q
            old_q = self.q
            self.q = q
            self.two_pi_q = 2 * math.pi * q
            
            try:
                encoding_matrix = self.compute_encoding_matrix(adjacency)
                results[q] = encoding_matrix
            except Exception as e:
                logger.error(f"Failed to compute encoding for q={q}: {e}")
            finally:
                # Restore original q
                self.q = old_q
                self.two_pi_q = 2 * math.pi * old_q
        
        return results
    
    def get_node_mapping(self, poset: nx.DiGraph, 
                        node_ordering: Optional[List[str]] = None) -> Dict[str, int]:
        """Get mapping from node names to matrix indices.
        
        Args:
            poset: NetworkX directed graph
            node_ordering: Optional node ordering
            
        Returns:
            Dictionary mapping node names to matrix indices
        """
        if node_ordering is None:
            node_ordering = list(poset.nodes())
        
        return {node: i for i, node in enumerate(node_ordering)}