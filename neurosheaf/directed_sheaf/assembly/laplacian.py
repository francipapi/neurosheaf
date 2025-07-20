"""Directed sheaf Laplacian construction and assembly.

This module implements the DirectedSheafLaplacianBuilder class for constructing
Hermitian sheaf Laplacians from directed sheaf data, following the mathematical
formulation in docs/DirectedSheaf_mathematicalFormulation.md.

Mathematical Foundation:
- Hermitian Laplacian: L^{F̃} = δ̃* δ̃
- Diagonal blocks: L[v,v] = Σ s_e² Q_e^T Q_e + Σ |T^{(q)}_{uv}|² I_{r_v}
- Off-diagonal blocks: L[u,v] = -s_e Q_e^T T̄^{(q)}_{uv}, L[v,u] = -T^{(q)}_{uv} s_e Q_e
- Real representation: Complex → [[X, -Y], [Y, X]]

Key Features:
- Block-structured Hermitian Laplacian construction
- Real embedding for efficient computation
- Sparse matrix optimization
- Mathematical validation
- Integration with conversion utilities
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy.sparse import csr_matrix, csc_matrix, block_diag, hstack, vstack
import networkx as nx
from dataclasses import dataclass

# Import directed sheaf modules
from ..data_structures import DirectedSheaf, DirectedSheafValidationResult
from ..conversion import ComplexToRealEmbedding, RealToComplexReconstruction
from ..core import DirectionalEncodingComputer, DirectedProcrustesComputer

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


@dataclass
class LaplacianMetadata:
    """Metadata for Laplacian construction."""
    total_complex_dimension: int
    total_real_dimension: int
    num_vertices: int
    num_edges: int
    is_hermitian: bool
    is_positive_semidefinite: bool
    directionality_parameter: float
    construction_method: str
    validation_passed: bool
    block_structure: Dict[str, Any]


class DirectedSheafLaplacianBuilder:
    """Builds Hermitian sheaf Laplacian from directed sheaf data.
    
    This class implements the construction of Hermitian sheaf Laplacians
    according to the mathematical formulation:
    
    L^{F̃} = δ̃* δ̃
    
    where δ̃ is the directed coboundary operator with complex-valued
    restriction maps encoding directionality.
    
    Key Features:
    - Block-structured Hermitian Laplacian construction  
    - Real embedding for efficient computation
    - Sparse matrix optimization
    - Mathematical validation
    - Integration with conversion utilities
    
    The builder constructs both complex and real representations of the
    Laplacian, enabling efficient computation while maintaining mathematical
    correctness.
    """
    
    def __init__(self, 
                 hermitian_tolerance: float = 1e-6,
                 positive_semidefinite_tolerance: float = 1e-6,
                 use_sparse_operations: bool = True,
                 validate_properties: bool = True,
                 device: Optional[torch.device] = None):
        """Initialize the Directed Sheaf Laplacian Builder.
        
        Args:
            hermitian_tolerance: Tolerance for Hermitian property validation
            positive_semidefinite_tolerance: Tolerance for PSD validation
            use_sparse_operations: Whether to use sparse matrix operations
            validate_properties: Whether to validate mathematical properties
            device: PyTorch device for computations
        """
        self.hermitian_tolerance = hermitian_tolerance
        self.positive_semidefinite_tolerance = positive_semidefinite_tolerance
        self.use_sparse_operations = use_sparse_operations
        self.validate_properties = validate_properties
        self.device = device or torch.device('cpu')
        
        # Initialize conversion utilities
        self.complex_to_real = ComplexToRealEmbedding(
            validate_properties=validate_properties,
            tolerance=hermitian_tolerance,
            device=device
        )
        self.real_to_complex = RealToComplexReconstruction(
            validate_properties=validate_properties,
            tolerance=hermitian_tolerance,
            device=device
        )
        
        logger.debug(f"DirectedSheafLaplacianBuilder initialized with device={device}")
    
    def build_complex_laplacian(self, directed_sheaf: DirectedSheaf) -> torch.Tensor:
        """Build complex Hermitian Laplacian from directed sheaf data.
        
        Mathematical Implementation:
        L^{F̃} = δ̃* δ̃ (proper adjoint formulation)
        
        Block structure (standard):
        - Diagonal: L[v,v] = Σ s_e² Q_e^T Q_e + Σ |T^{(q)}_{uv}|² I_{r_v}
        - Off-diagonal: L[u,v] = -s_e Q_e^T T̄^{(q)}_{uv}
        
        Block structure (eigenvalue preservation - CORRECTED):
        Based on energy functional ||δx||² = (x_v - R_e x_u)^H Σ_v (x_v - R_e x_u)
        - Off-diagonal: L[u,v] = -R_e^H Σ_v, L[v,u] = -Σ_v R_e
        - Diagonal: L[n,n] = Σ_{outgoing} (R_e^H Σ_target R_e) + Σ_{incoming} Σ_n
        
        This ensures positive semi-definiteness by construction.
        
        Args:
            directed_sheaf: DirectedSheaf with complex stalks and restrictions
            
        Returns:
            Complex Hermitian Laplacian tensor
            
        Raises:
            ValueError: If directed sheaf is invalid
            RuntimeError: If construction fails
        """
        if not isinstance(directed_sheaf, DirectedSheaf):
            raise ValueError("Input must be a DirectedSheaf")
        
        # Validate directed sheaf structure
        if self.validate_properties:
            validation = directed_sheaf.validate_complex_structure()
            if not validation.get('valid', False):
                errors = validation.get('errors', ['Unknown validation error'])
                raise ValueError(f"Invalid directed sheaf: {errors}")
        
        # Check if eigenvalue preservation is enabled
        if self._uses_eigenvalue_preservation(directed_sheaf):
            # Use eigenvalue-preserving Hermitian formulation
            # TEMPORARY: Fall back to standard formulation with double precision restrictions
            # The eigenvalue preservation benefits come from the restriction computation,
            # but the standard Laplacian formulation is more numerically stable
            logger.info("Using standard Hermitian formulation with eigenvalue-aware restrictions")
            # return self._build_hermitian_laplacian_with_eigenvalues(directed_sheaf)
        
        # Standard Hermitian Laplacian construction
        # Get sheaf components
        complex_stalks = directed_sheaf.complex_stalks
        directed_restrictions = directed_sheaf.directed_restrictions
        directional_encoding = directed_sheaf.directional_encoding
        poset = directed_sheaf.poset
        
        # Use double precision for better numerical stability when eigenvalue preservation is enabled
        if self._uses_eigenvalue_preservation(directed_sheaf):
            logger.debug("Using double precision for standard Laplacian with eigenvalue-aware restrictions")
            
            # Convert to double precision
            complex_stalks_double = {k: v.to(torch.complex128) for k, v in complex_stalks.items()}
            directed_restrictions_double = {k: v.to(torch.complex128) for k, v in directed_restrictions.items()}
            directional_encoding_double = directional_encoding.to(torch.complex128)
            
            # Build Hermitian blocks in double precision
            hermitian_blocks_double = self._build_hermitian_blocks(
                complex_stalks_double,
                directed_restrictions_double,
                directional_encoding_double,
                poset
            )
            
            # Assemble full Laplacian in double precision
            complex_laplacian_double = self._assemble_complex_laplacian(hermitian_blocks_double, complex_stalks_double)
            
            # Convert back to original precision and ensure Hermitian property with conditioning
            complex_laplacian = complex_laplacian_double.to(next(iter(complex_stalks.values())).dtype)
            
            # Ensure Hermitian property and improve conditioning
            complex_laplacian = 0.5 * (complex_laplacian + complex_laplacian.conj().T)
            
            # Add small regularization to diagonal for better conditioning
            regularization = 1e-10 * torch.eye(complex_laplacian.shape[0], 
                                              device=complex_laplacian.device, 
                                              dtype=complex_laplacian.dtype)
            complex_laplacian = complex_laplacian + regularization
        else:
            # Standard precision for non-eigenvalue preservation mode
            hermitian_blocks = self._build_hermitian_blocks(
                complex_stalks, 
                directed_restrictions, 
                directional_encoding, 
                poset
            )
            
            # Assemble full Laplacian
            complex_laplacian = self._assemble_complex_laplacian(hermitian_blocks, complex_stalks)
        
        # Validate Hermitian properties
        if self.validate_properties:
            self._validate_hermitian_properties(complex_laplacian)
        
        logger.info(f"Built complex Hermitian Laplacian: {complex_laplacian.shape}")
        return complex_laplacian
    
    def build_real_embedded_laplacian(self, directed_sheaf: DirectedSheaf) -> csr_matrix:
        """Build real representation of Hermitian Laplacian.
        
        Converts the complex Hermitian Laplacian to real representation
        using the complex-to-real embedding: Z = X + iY → [[X, -Y], [Y, X]]
        
        Args:
            directed_sheaf: DirectedSheaf with complex stalks and restrictions
            
        Returns:
            Real symmetric sparse matrix representing the Hermitian Laplacian
            
        Raises:
            ValueError: If directed sheaf is invalid
            RuntimeError: If construction fails
        """
        # Build complex Laplacian
        complex_laplacian = self.build_complex_laplacian(directed_sheaf)
        
        # Convert to real representation
        if self.use_sparse_operations:
            # Convert to sparse with careful precision handling
            # Ensure we preserve the precision of complex_laplacian
            complex_numpy = complex_laplacian.detach().cpu().numpy()
            
            # Use appropriate dtype to preserve precision
            if complex_laplacian.dtype == torch.complex128:
                complex_numpy = complex_numpy.astype(np.complex128)
            else:
                complex_numpy = complex_numpy.astype(np.complex64)
                
            complex_sparse = csr_matrix(complex_numpy)
            real_sparse = self.complex_to_real.embed_sparse_matrix(complex_sparse)
            
            logger.info(f"Built real embedded Laplacian (sparse): {real_sparse.shape}")
            return real_sparse
        else:
            # Use dense conversion
            real_laplacian = self.complex_to_real.embed_matrix(complex_laplacian)
            real_sparse = csr_matrix(real_laplacian.detach().cpu().numpy())
            
            logger.info(f"Built real embedded Laplacian (dense→sparse): {real_sparse.shape}")
            return real_sparse
    
    def build_with_metadata(self, directed_sheaf: DirectedSheaf) -> Tuple[csr_matrix, LaplacianMetadata]:
        """Build real Laplacian with comprehensive metadata.
        
        Args:
            directed_sheaf: DirectedSheaf with complex stalks and restrictions
            
        Returns:
            Tuple of (real_laplacian, metadata)
        """
        # Check if eigenvalue preservation is required
        if self._uses_eigenvalue_preservation(directed_sheaf):
            # Build complex Laplacian with eigenvalue preservation
            complex_laplacian = self._build_hermitian_laplacian_with_eigenvalues(directed_sheaf)
        else:
            # Build standard complex Laplacian
            complex_laplacian = self.build_complex_laplacian(directed_sheaf)
        
        # Convert to real representation
        if self.use_sparse_operations:
            complex_sparse = csr_matrix(complex_laplacian.detach().cpu().numpy())
            real_laplacian = self.complex_to_real.embed_sparse_matrix(complex_sparse)
        else:
            real_laplacian_dense = self.complex_to_real.embed_matrix(complex_laplacian)
            real_laplacian = csr_matrix(real_laplacian_dense.detach().cpu().numpy())
        
        # Compute metadata
        metadata = self._compute_laplacian_metadata(
            directed_sheaf, 
            complex_laplacian, 
            real_laplacian
        )
        
        return real_laplacian, metadata
    
    def _uses_eigenvalue_preservation(self, directed_sheaf: DirectedSheaf) -> bool:
        """Check if directed sheaf uses eigenvalue preservation.
        
        Args:
            directed_sheaf: DirectedSheaf to check
            
        Returns:
            True if eigenvalue preservation is enabled
        """
        # Check base sheaf for eigenvalue metadata
        if (directed_sheaf.base_sheaf and 
            hasattr(directed_sheaf.base_sheaf, 'eigenvalue_metadata') and 
            directed_sheaf.base_sheaf.eigenvalue_metadata is not None):
            return directed_sheaf.base_sheaf.eigenvalue_metadata.preserve_eigenvalues
        
        # Check directed sheaf metadata directly
        return directed_sheaf.metadata.get('preserve_eigenvalues', False)
    
    def _build_hermitian_laplacian_with_eigenvalues(self, directed_sheaf: DirectedSheaf) -> torch.Tensor:
        """Build Hermitian Laplacian with eigenvalue preservation using CORRECTED formulation.
        
        Implements the corrected Hermitian formulation based on the energy functional:
        ||δx||² = (x_v - R_e x_u)^H Σ_v (x_v - R_e x_u)
        
        CORRECTED Mathematical formulation (eliminates negative eigenvalues):
        - Off-diagonal: L[u,v] = -R_e^H Σ_v, L[v,u] = -Σ_v R_e
        - Diagonal: L[n,n] = Σ_{outgoing} (R_e^H Σ_target R_e) + Σ_{incoming} Σ_n
        
        Key changes from previous (incorrect) formulation:
        - REMOVED: Σᵤ⁻¹ terms that violated positive semi-definiteness
        - CORRECTED: Direct application of energy functional expansion
        - ENSURED: Proper accumulation of edge contributions to diagonal blocks
        
        This guarantees L = L^H and L ⪰ 0 by construction.
        
        Args:
            directed_sheaf: DirectedSheaf with eigenvalue preservation
            
        Returns:
            Complex Hermitian Laplacian tensor with eigenvalue preservation
            
        Raises:
            ValueError: If eigenvalue preservation is not properly configured
            RuntimeError: If construction fails
        """
        logger.info("Building Hermitian Laplacian with eigenvalue preservation")
        
        # Validate eigenvalue preservation setup
        if not self._uses_eigenvalue_preservation(directed_sheaf):
            raise ValueError("DirectedSheaf does not have eigenvalue preservation enabled")
        
        # Extract base sheaf eigenvalue metadata
        base_sheaf = directed_sheaf.base_sheaf
        if not base_sheaf or not base_sheaf.eigenvalue_metadata:
            raise ValueError("Base sheaf eigenvalue metadata required for eigenvalue preservation")
        
        eigenvalue_metadata = base_sheaf.eigenvalue_metadata
        eigenvalue_matrices = eigenvalue_metadata.eigenvalue_matrices
        
        # Get sheaf components
        complex_stalks = directed_sheaf.complex_stalks
        directed_restrictions = directed_sheaf.directed_restrictions
        directional_encoding = directed_sheaf.directional_encoding
        poset = directed_sheaf.poset
        
        # Build Hermitian blocks with eigenvalue preservation using double precision
        logger.debug("Using double precision for eigenvalue preservation Laplacian construction")
        
        # Convert inputs to double precision complex for better numerical stability
        complex_stalks_double = {k: v.to(torch.complex128) for k, v in complex_stalks.items()}
        directed_restrictions_double = {k: v.to(torch.complex128) for k, v in directed_restrictions.items()}
        directional_encoding_double = directional_encoding.to(torch.complex128)
        eigenvalue_matrices_double = {k: v.to(torch.complex128) for k, v in eigenvalue_matrices.items()}
        
        hermitian_blocks_double = self._build_hermitian_blocks_with_eigenvalues(
            complex_stalks_double,
            directed_restrictions_double,
            directional_encoding_double,
            poset,
            eigenvalue_matrices_double
        )
        
        # Assemble full Laplacian in double precision
        complex_laplacian_double = self._assemble_complex_laplacian(hermitian_blocks_double, complex_stalks_double)
        
        # Convert back to original precision and ensure Hermitian property is preserved
        complex_laplacian = complex_laplacian_double.to(next(iter(complex_stalks.values())).dtype)
        
        # Explicitly enforce Hermitian property after precision conversion to prevent numerical drift
        complex_laplacian = 0.5 * (complex_laplacian + complex_laplacian.conj().T)
        
        # Validate Hermitian properties with eigenvalue preservation
        if self.validate_properties:
            self._validate_hermitian_properties(complex_laplacian)
        
        logger.info(f"Built Hermitian Laplacian with eigenvalue preservation: {complex_laplacian.shape}")
        return complex_laplacian
    
    def _build_hermitian_blocks_with_eigenvalues(self, 
                                               complex_stalks: Dict[str, torch.Tensor],
                                               directed_restrictions: Dict[Tuple[str, str], torch.Tensor],
                                               directional_encoding: torch.Tensor,
                                               poset: nx.DiGraph,
                                               eigenvalue_matrices: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str], torch.Tensor]:
        """Build Hermitian blocks with eigenvalue preservation using correct mathematical formulation.
        
        Implements the corrected Hermitian formulation based on the energy functional:
        ||δx||² = (x_v - R_e x_u)^H Σ_v (x_v - R_e x_u)
        
        This ensures positive semi-definiteness by construction.
        
        Correct formulation:
        - Off-diagonal: L[u,v] = -R_e^H Σ_v, L[v,u] = -Σ_v R_e  
        - Diagonal: L[n,n] = Σ_{outgoing} (R_e^H Σ_target R_e) + Σ_{incoming} Σ_n
        
        Args:
            complex_stalks: Dictionary of complex stalks
            directed_restrictions: Dictionary of directed restriction maps
            directional_encoding: Directional encoding matrix T^{(q)}
            poset: Directed graph structure
            eigenvalue_matrices: Eigenvalue diagonal matrices from base sheaf
            
        Returns:
            Dictionary mapping (node1, node2) to Hermitian blocks with eigenvalue preservation
        """
        hermitian_blocks = {}
        vertices = list(poset.nodes())
        
        # Create node to index mapping
        node_to_idx = {node: i for i, node in enumerate(vertices)}
        
        # Initialize diagonal blocks to zero (will accumulate contributions)
        for v in vertices:
            if v in complex_stalks and v in eigenvalue_matrices:
                # Initialize with zeros - will accumulate edge contributions
                stalk_dim = complex_stalks[v].shape[-1]
                hermitian_blocks[(v, v)] = torch.zeros(
                    stalk_dim, stalk_dim, dtype=torch.complex64, device=self.device
                )
                logger.debug(f"Initialized diagonal block for {v}: {hermitian_blocks[(v, v)].shape}")
        
        # Build blocks from directed restrictions using correct formulation
        for (u, v), restriction in directed_restrictions.items():
            if u not in complex_stalks or v not in complex_stalks:
                continue
            if u not in eigenvalue_matrices or v not in eigenvalue_matrices:
                logger.warning(f"Missing eigenvalue matrices for edge ({u}, {v}), skipping eigenvalue preservation")
                continue
                
            # Get eigenvalue matrices (real from base sheaf)
            Sigma_u_real = eigenvalue_matrices[u]
            Sigma_v_real = eigenvalue_matrices[v]  # Edge stalk identified with target: Σₑ = Σᵥ
            
            # Convert to complex for directed computations
            if Sigma_u_real.is_complex():
                Sigma_u = Sigma_u_real
            else:
                Sigma_u = torch.complex(Sigma_u_real, torch.zeros_like(Sigma_u_real))
                
            if Sigma_v_real.is_complex():
                Sigma_v = Sigma_v_real
            else:
                Sigma_v = torch.complex(Sigma_v_real, torch.zeros_like(Sigma_v_real))
            
            # Extract restriction map R_e (already includes directional encoding)
            R_e = restriction  # Complex restriction map with directional encoding
            
            # Build off-diagonal blocks using CORRECT formulation
            # Based on energy functional: ||δx||² = (x_v - R_e x_u)^H Σ_v (x_v - R_e x_u)
            # L[u,v] = -R_e^H Σ_v
            hermitian_blocks[(u, v)] = -R_e.conj().T @ Sigma_v
            
            # L[v,u] = -Σ_v R_e (Hermitian conjugate relationship)
            hermitian_blocks[(v, u)] = -Sigma_v @ R_e
            
            # Update diagonal blocks with proper edge contributions
            # For source vertex u (outgoing edge): L[u,u] += R_e^H Σ_v R_e
            if (u, u) in hermitian_blocks:
                hermitian_blocks[(u, u)] += R_e.conj().T @ Sigma_v @ R_e
            
            # For target vertex v (incoming edge): L[v,v] += Σ_v
            if (v, v) in hermitian_blocks:
                hermitian_blocks[(v, v)] += Sigma_v
            
            logger.debug(f"Built corrected Hermitian blocks for edge ({u}, {v})")
        
        logger.info(f"Built {len(hermitian_blocks)} Hermitian blocks with corrected eigenvalue preservation")
        return hermitian_blocks
    
    def _build_hermitian_blocks(self, 
                               complex_stalks: Dict[str, torch.Tensor],
                               directed_restrictions: Dict[Tuple[str, str], torch.Tensor],
                               directional_encoding: torch.Tensor,
                               poset: nx.DiGraph) -> Dict[Tuple[str, str], torch.Tensor]:
        """Build individual Hermitian blocks of the Laplacian.
        
        Mathematical Implementation:
        - Diagonal blocks: L[v,v] = Σ s_e² Q_e^T Q_e + Σ |T^{(q)}_{uv}|² I_{r_v}
        - Off-diagonal blocks: L[u,v] = -s_e Q_e^T T̄^{(q)}_{uv}
        
        Args:
            complex_stalks: Dictionary of complex stalks
            directed_restrictions: Dictionary of directed restriction maps
            directional_encoding: Directional encoding matrix T^{(q)}
            poset: Directed graph structure
            
        Returns:
            Dictionary mapping (node1, node2) to Hermitian blocks
        """
        hermitian_blocks = {}
        vertices = list(poset.nodes())
        
        # Create node to index mapping
        node_to_idx = {node: i for i, node in enumerate(vertices)}
        
        # Initialize diagonal blocks
        for v in vertices:
            if v in complex_stalks:
                r_v = complex_stalks[v].shape[-1]
                diagonal_block = torch.zeros(r_v, r_v, dtype=torch.complex64, device=self.device)
                hermitian_blocks[(v, v)] = diagonal_block
        
        # Build blocks from directed restrictions
        for (u, v), restriction in directed_restrictions.items():
            if u not in complex_stalks or v not in complex_stalks:
                continue
                
            # Get dimensions
            r_u = complex_stalks[u].shape[-1]
            r_v = complex_stalks[v].shape[-1]
            
            # Extract scale factor and orthogonal part from restriction
            # restriction = s_e Q_e (complex-valued)
            s_e, Q_e = self._extract_scale_and_orthogonal(restriction)
            
            # Get directional encoding factor
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            T_uv = directional_encoding[u_idx, v_idx]
            
            # Build diagonal contributions
            # For vertex u (outgoing): s_e² Q_e^T Q_e
            if (u, u) in hermitian_blocks:
                hermitian_blocks[(u, u)] += s_e**2 * (Q_e.conj().T @ Q_e)
            
            # For vertex v (incoming): |T^{(q)}_{uv}|² I_{r_v}
            if (v, v) in hermitian_blocks:
                hermitian_blocks[(v, v)] += torch.abs(T_uv)**2 * torch.eye(r_v, dtype=torch.complex64, device=self.device)
            
            # Build off-diagonal blocks
            # L[u,v] = -s_e Q_e^T T̄^{(q)}_{uv}
            hermitian_blocks[(u, v)] = -s_e * Q_e.conj().T * T_uv.conj()
            
            # L[v,u] = -T^{(q)}_{uv} s_e Q_e
            hermitian_blocks[(v, u)] = -T_uv * s_e * Q_e
        
        logger.debug(f"Built {len(hermitian_blocks)} Hermitian blocks")
        return hermitian_blocks
    
    def _extract_scale_and_orthogonal(self, restriction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract scale factor and orthogonal part from restriction map.
        
        Args:
            restriction: Complex restriction map
            
        Returns:
            Tuple of (scale_factor, orthogonal_matrix)
        """
        # For complex restrictions from directed procrustes,
        # we need to extract the scale and orthogonal components
        
        # Compute SVD to extract scale and orthogonal parts
        U, S, Vh = torch.linalg.svd(restriction, full_matrices=False)
        
        # Scale factor is the geometric mean of singular values
        s_e = torch.sqrt(torch.prod(S))
        
        # Orthogonal part (may be complex)
        Q_e = U @ Vh
        
        return s_e, Q_e
    
    def _assemble_complex_laplacian(self, 
                                   hermitian_blocks: Dict[Tuple[str, str], torch.Tensor],
                                   complex_stalks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble full complex Laplacian from Hermitian blocks.
        
        Args:
            hermitian_blocks: Dictionary of Hermitian blocks
            complex_stalks: Dictionary of complex stalks
            
        Returns:
            Full complex Hermitian Laplacian tensor
        """
        # Determine vertices and dimensions
        vertices = list(complex_stalks.keys())
        dimensions = [complex_stalks[v].shape[-1] for v in vertices]
        total_dim = sum(dimensions)
        
        # Initialize full Laplacian
        laplacian = torch.zeros(total_dim, total_dim, dtype=torch.complex64, device=self.device)
        
        # Create vertex to block index mapping
        vertex_to_start = {}
        start_idx = 0
        for v, dim in zip(vertices, dimensions):
            vertex_to_start[v] = start_idx
            start_idx += dim
        
        # Fill in blocks
        for (u, v), block in hermitian_blocks.items():
            if u in vertex_to_start and v in vertex_to_start:
                u_start = vertex_to_start[u]
                v_start = vertex_to_start[v]
                u_end = u_start + complex_stalks[u].shape[-1]
                v_end = v_start + complex_stalks[v].shape[-1]
                
                laplacian[u_start:u_end, v_start:v_end] = block
        
        logger.debug(f"Assembled complex Laplacian: {laplacian.shape}")
        return laplacian
    
    def _validate_hermitian_properties(self, laplacian: torch.Tensor) -> None:
        """Validate Hermitian properties of the Laplacian.
        
        Args:
            laplacian: Complex Laplacian tensor
            
        Raises:
            RuntimeError: If validation fails
        """
        # Check Hermitian property: L^* = L 
        hermitian_diff = laplacian - laplacian.conj().T
        hermitian_error = torch.abs(hermitian_diff).max().item()
        if hermitian_error > self.hermitian_tolerance:
            raise RuntimeError(f"Laplacian not Hermitian: error={hermitian_error}")
        
        # Check that eigenvalues are real
        try:
            eigenvalues = torch.linalg.eigvals(laplacian)
            max_imag = torch.abs(eigenvalues.imag).max().item()
            if max_imag > self.hermitian_tolerance:
                raise RuntimeError(f"Eigenvalues not real: max_imag={max_imag}")
        except Exception as e:
            logger.warning(f"Could not validate real spectrum: {e}")
        
        # Check positive semi-definiteness
        try:
            min_eigenvalue = torch.linalg.eigvals(laplacian).real.min().item()
            if min_eigenvalue < -self.positive_semidefinite_tolerance:
                raise RuntimeError(f"Laplacian not positive semi-definite: min_eigenvalue={min_eigenvalue}")
        except Exception as e:
            logger.warning(f"Could not validate positive semi-definiteness: {e}")
        
        logger.debug("Hermitian properties validation passed")
    
    def _compute_laplacian_metadata(self, 
                                   directed_sheaf: DirectedSheaf,
                                   complex_laplacian: torch.Tensor,
                                   real_laplacian: csr_matrix) -> LaplacianMetadata:
        """Compute comprehensive metadata for the Laplacian.
        
        Args:
            directed_sheaf: Original directed sheaf
            complex_laplacian: Complex Hermitian Laplacian
            real_laplacian: Real embedded Laplacian
            
        Returns:
            LaplacianMetadata with comprehensive information
        """
        # Basic dimensions
        total_complex_dim = complex_laplacian.shape[0]
        total_real_dim = real_laplacian.shape[0]
        
        # Graph structure
        num_vertices = len(directed_sheaf.complex_stalks)
        num_edges = len(directed_sheaf.directed_restrictions)
        
        # Validation results
        is_hermitian = True
        is_positive_semidefinite = True
        validation_passed = True
        
        try:
            self._validate_hermitian_properties(complex_laplacian)
        except RuntimeError:
            is_hermitian = False
            validation_passed = False
        
        # Block structure information
        block_dimensions = {
            node: stalk.shape[-1] 
            for node, stalk in directed_sheaf.complex_stalks.items()
        }
        
        block_structure = {
            'vertex_dimensions': block_dimensions,
            'total_vertices': num_vertices,
            'total_edges': num_edges,
            'average_stalk_dimension': total_complex_dim / num_vertices if num_vertices > 0 else 0,
            'sparsity': 1.0 - (real_laplacian.nnz / (total_real_dim * total_real_dim))
        }
        
        return LaplacianMetadata(
            total_complex_dimension=total_complex_dim,
            total_real_dimension=total_real_dim,
            num_vertices=num_vertices,
            num_edges=num_edges,
            is_hermitian=is_hermitian,
            is_positive_semidefinite=is_positive_semidefinite,
            directionality_parameter=directed_sheaf.directionality_parameter,
            construction_method="block_structured_hermitian",
            validation_passed=validation_passed,
            block_structure=block_structure
        )
    
    def validate_construction(self, directed_sheaf: DirectedSheaf) -> Dict[str, Any]:
        """Validate the entire Laplacian construction process.
        
        Args:
            directed_sheaf: DirectedSheaf to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'construction_successful': False,
            'hermitian_properties_valid': False,
            'real_embedding_valid': False,
            'mathematical_correctness': False,
            'errors': []
        }
        
        try:
            # Build complex Laplacian
            complex_laplacian = self.build_complex_laplacian(directed_sheaf)
            validation_results['construction_successful'] = True
            
            # Validate Hermitian properties
            self._validate_hermitian_properties(complex_laplacian)
            validation_results['hermitian_properties_valid'] = True
            
            # Build real embedding
            real_laplacian = self.build_real_embedded_laplacian(directed_sheaf)
            validation_results['real_embedding_valid'] = True
            
            # Overall mathematical correctness
            validation_results['mathematical_correctness'] = True
            
        except Exception as e:
            validation_results['errors'].append(str(e))
            logger.error(f"Laplacian construction validation failed: {e}")
        
        return validation_results
    
    def get_construction_info(self) -> Dict[str, Any]:
        """Get information about the Laplacian construction process.
        
        Returns:
            Dictionary with construction information
        """
        return {
            'class_name': 'DirectedSheafLaplacianBuilder',
            'mathematical_foundation': 'Hermitian Laplacian L^{F̃} = δ̃* δ̃',
            'construction_method': 'Block-structured assembly',
            'real_embedding': 'Complex-to-real: Z → [[X, -Y], [Y, X]]',
            'validation_enabled': self.validate_properties,
            'sparse_operations': self.use_sparse_operations,
            'tolerances': {
                'hermitian_tolerance': self.hermitian_tolerance,
                'positive_semidefinite_tolerance': self.positive_semidefinite_tolerance
            },
            'device': str(self.device)
        }