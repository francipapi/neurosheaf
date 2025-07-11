"""Sparse sheaf Laplacian construction for neural network analysis.

This module implements efficient sparse Laplacian assembly from whitened sheaf
data, optimized for GPU operations and memory efficiency. The Laplacian Δ = δ^T δ
encodes the topological structure of the neural network with exact mathematical
properties in whitened coordinates.

Mathematical Foundation:
- Stalks: ℝ^{r_v} with identity inner product (whitened coordinates)  
- Restrictions: R̃_e exactly orthogonal in whitened space
- Laplacian: Δ = δ^T δ where δ is the coboundary operator
- Structure: Symmetric positive semi-definite sparse matrix
"""

import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix, block_diag
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from ..utils.logging import setup_logger
from ..utils.exceptions import ComputationError
from .construction import Sheaf

logger = setup_logger(__name__)


@dataclass
class LaplacianMetadata:
    """Metadata for constructed sheaf Laplacian.
    
    Attributes:
        total_dimension: Total dimension of the Laplacian matrix
        stalk_dimensions: Dictionary mapping node names to their whitened dimensions
        stalk_offsets: Starting indices for each stalk in the global matrix
        edge_positions: Mapping from edges to their matrix positions
        sparsity_ratio: Fraction of non-zero entries
        condition_number: Estimated condition number
        construction_time: Time taken to build the Laplacian
        memory_usage: Peak memory usage during construction
    """
    total_dimension: int = 0
    stalk_dimensions: Dict[str, int] = None
    stalk_offsets: Dict[str, int] = None
    edge_positions: Dict[Tuple[str, str], List[Tuple[int, int]]] = None
    sparsity_ratio: float = 0.0
    condition_number: float = 0.0
    construction_time: float = 0.0
    memory_usage: float = 0.0
    
    def __post_init__(self):
        """Initialize empty dictionaries if not provided."""
        if self.stalk_dimensions is None:
            self.stalk_dimensions = {}
        if self.stalk_offsets is None:
            self.stalk_offsets = {}
        if self.edge_positions is None:
            self.edge_positions = {}


class SheafLaplacianBuilder:
    """Efficient sparse Laplacian construction from whitened sheaf data.
    
    This class implements the core Laplacian assembly Δ = δ^T δ where δ is
    the coboundary operator. Key optimizations:
    - Works with whitened coordinates (reduced dimensions)
    - Sparse matrix assembly using COO → CSR conversion
    - GPU-compatible tensor formats
    - Memory-efficient block construction
    
    The Laplacian has the block structure:
    - Diagonal blocks: Δ_vv = Σ R̃_e^T R̃_e + Σ R̃_e R̃_e^T
    - Off-diagonal blocks: Δ_vw = -R̃_e for edge e = (v,w)
    
    Attributes:
        enable_gpu: Whether to create GPU-compatible sparse tensors
        memory_efficient: Whether to use memory-efficient assembly
        validate_properties: Whether to validate mathematical properties
    """
    
    def __init__(self, enable_gpu: bool = True, memory_efficient: bool = True,
                 validate_properties: bool = True):
        """Initialize Laplacian builder.
        
        Args:
            enable_gpu: Whether to support GPU sparse tensor operations
            memory_efficient: Whether to use memory-efficient assembly strategies
            validate_properties: Whether to validate Laplacian mathematical properties
        """
        self.enable_gpu = enable_gpu
        self.memory_efficient = memory_efficient
        self.validate_properties = validate_properties
        
        # Check GPU availability
        if self.enable_gpu and not torch.cuda.is_available():
            logger.warning("GPU not available, falling back to CPU operations")
            self.enable_gpu = False
    
    def build_laplacian(self, sheaf: Sheaf, edge_weights: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple[csr_matrix, LaplacianMetadata]:
        """Build sparse sheaf Laplacian from whitened sheaf data.
        
        Constructs Δ = δ^T δ where δ is the coboundary operator using
        whitened restriction maps R̃_e that are exactly orthogonal.
        
        Args:
            sheaf: Sheaf object with whitened stalks and restrictions
            edge_weights: Optional edge weights (default: use scale factors from restriction maps)
            
        Returns:
            Tuple of (sparse_laplacian, metadata):
            - sparse_laplacian: scipy.sparse.csr_matrix of the Laplacian
            - metadata: LaplacianMetadata with construction details
            
        Raises:
            ComputationError: If Laplacian construction fails
        """
        import time
        import psutil
        
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**3  # GB
        
        logger.info(f"Building sheaf Laplacian with {len(sheaf.stalks)} stalks, {len(sheaf.restrictions)} restrictions")
        
        try:
            # Build metadata structure
            metadata = self._initialize_metadata(sheaf)
            
            # Extract edge weights from restriction map info or use provided weights
            if edge_weights is None:
                edge_weights = self._extract_edge_weights(sheaf)
            
            # Build Laplacian using block assembly
            if self.memory_efficient:
                laplacian = self._build_laplacian_memory_efficient(sheaf, edge_weights, metadata)
            else:
                laplacian = self._build_laplacian_standard(sheaf, edge_weights, metadata)
            
            # Validate mathematical properties if requested
            if self.validate_properties:
                self._validate_laplacian_properties(laplacian, metadata)
            
            # Update metadata with final statistics
            end_time = time.time()
            peak_memory = process.memory_info().rss / 1024**3  # GB
            
            metadata.construction_time = end_time - start_time
            metadata.memory_usage = peak_memory - initial_memory
            metadata.sparsity_ratio = 1.0 - (laplacian.nnz / (laplacian.shape[0] * laplacian.shape[1]))
            
            logger.info(f"Laplacian constructed: {laplacian.shape[0]}×{laplacian.shape[1]}, "
                       f"{laplacian.nnz} non-zeros ({metadata.sparsity_ratio:.1%} sparse), "
                       f"{metadata.construction_time:.2f}s, {metadata.memory_usage:.2f}GB")
            
            return laplacian, metadata
            
        except Exception as e:
            raise ComputationError(f"Laplacian construction failed: {e}", operation="build_laplacian")
    
    def _initialize_metadata(self, sheaf: Sheaf) -> LaplacianMetadata:
        """Initialize metadata structure from sheaf topology."""
        metadata = LaplacianMetadata()
        
        # Get stalk dimensions (from whitened coordinates)
        for node, stalk_data in sheaf.stalks.items():
            if isinstance(stalk_data, torch.Tensor):
                # For whitened Gram matrices, dimension is the matrix size
                metadata.stalk_dimensions[node] = stalk_data.shape[0]
            else:
                logger.warning(f"Unknown stalk data type for {node}: {type(stalk_data)}")
                metadata.stalk_dimensions[node] = 1  # Fallback
        
        # Compute stalk offsets for global matrix indexing
        offset = 0
        for node in sheaf.poset.nodes():
            if node in metadata.stalk_dimensions:
                metadata.stalk_offsets[node] = offset
                offset += metadata.stalk_dimensions[node]
        
        metadata.total_dimension = offset
        
        logger.debug(f"Stalk dimensions: {metadata.stalk_dimensions}")
        logger.debug(f"Total Laplacian dimension: {metadata.total_dimension}")
        
        return metadata
    
    def _extract_edge_weights(self, sheaf: Sheaf) -> Dict[Tuple[str, str], float]:
        """Extract edge weights from restriction map metadata."""
        edge_weights = {}
        
        for edge, restriction in sheaf.restrictions.items():
            # Try to get scale factor from sheaf metadata
            if hasattr(restriction, 'scale_factor'):
                edge_weights[edge] = float(restriction.scale_factor)
            elif 'scale' in sheaf.metadata.get('restriction_info', {}).get(edge, {}):
                edge_weights[edge] = float(sheaf.metadata['restriction_info'][edge]['scale'])
            else:
                # Default weight: use Frobenius norm as proxy for edge strength
                edge_weights[edge] = torch.norm(restriction, p='fro').item()
        
        logger.debug(f"Edge weights: {edge_weights}")
        return edge_weights
    
    def _build_laplacian_memory_efficient(self, sheaf: Sheaf, edge_weights: Dict[Tuple[str, str], float],
                                         metadata: LaplacianMetadata) -> csr_matrix:
        """Build Laplacian using memory-efficient block assembly."""
        n = metadata.total_dimension
        
        # Pre-allocate coordinate lists for COO matrix
        rows, cols, data = [], [], []
        
        # Track edge positions for masking support
        edge_positions = {}
        
        # Process each edge to build coboundary operator δ
        for edge, restriction in sheaf.restrictions.items():
            source, target = edge
            weight = edge_weights.get(edge, 1.0)
            
            if source not in metadata.stalk_offsets or target not in metadata.stalk_offsets:
                logger.warning(f"Edge {edge} references unknown nodes, skipping")
                continue
            
            # Get global matrix indices
            source_start = metadata.stalk_offsets[source]
            target_start = metadata.stalk_offsets[target]
            source_dim = metadata.stalk_dimensions[source]
            target_dim = metadata.stalk_dimensions[target]
            
            # Convert restriction to numpy for compatibility
            R = restriction.detach().cpu().numpy() * weight
            
            # Ensure dimensions match (handle potential dimension mismatches)
            if R.shape != (target_dim, source_dim):
                logger.warning(f"Restriction {edge} shape {R.shape} doesn't match expected ({target_dim}, {source_dim})")
                # Truncate or pad as needed
                R_adjusted = np.zeros((target_dim, source_dim))
                min_rows = min(R.shape[0], target_dim)
                min_cols = min(R.shape[1], source_dim)
                R_adjusted[:min_rows, :min_cols] = R[:min_rows, :min_cols]
                R = R_adjusted
            
            positions = []
            
            # Add restriction map R_e to off-diagonal block: Δ[target, source] = -R_e
            for i in range(target_dim):
                for j in range(source_dim):
                    if abs(R[i, j]) > 1e-12:  # Skip near-zero entries
                        global_i = target_start + i
                        global_j = source_start + j
                        rows.append(global_i)
                        cols.append(global_j)
                        data.append(-R[i, j])
                        positions.append((global_i, global_j))
            
            # Add transpose R_e^T to off-diagonal block: Δ[source, target] = -R_e^T
            for i in range(source_dim):
                for j in range(target_dim):
                    if abs(R[j, i]) > 1e-12:  # Skip near-zero entries
                        global_i = source_start + i
                        global_j = target_start + j
                        rows.append(global_i)
                        cols.append(global_j)
                        data.append(-R[j, i])
                        positions.append((global_i, global_j))
            
            # Store edge positions for masking
            edge_positions[edge] = positions
        
        # Build diagonal blocks: Δ_vv = Σ R_e^T R_e (outgoing) + Σ R_e R_e^T (incoming)
        for node in sheaf.poset.nodes():
            if node not in metadata.stalk_offsets:
                continue
                
            node_start = metadata.stalk_offsets[node]
            node_dim = metadata.stalk_dimensions[node]
            
            # Accumulate diagonal contribution
            diagonal_contrib = np.zeros((node_dim, node_dim))
            
            # Outgoing edges: R_e^T R_e where e = (node, target)
            for target in sheaf.poset.successors(node):
                edge = (node, target)
                if edge in sheaf.restrictions:
                    R = sheaf.restrictions[edge].detach().cpu().numpy()
                    weight = edge_weights.get(edge, 1.0)
                    R = R * weight
                    
                    # Ensure proper dimension alignment
                    if R.shape[1] == node_dim:
                        diagonal_contrib += R.T @ R
                    else:
                        logger.warning(f"Dimension mismatch for outgoing edge {edge}")
            
            # Incoming edges: R_e R_e^T where e = (source, node)  
            for source in sheaf.poset.predecessors(node):
                edge = (source, node)
                if edge in sheaf.restrictions:
                    R = sheaf.restrictions[edge].detach().cpu().numpy()
                    weight = edge_weights.get(edge, 1.0)
                    R = R * weight
                    
                    # Ensure proper dimension alignment
                    if R.shape[0] == node_dim:
                        diagonal_contrib += R @ R.T
                    else:
                        logger.warning(f"Dimension mismatch for incoming edge {edge}")
            
            # Add diagonal block to sparse matrix
            for i in range(node_dim):
                for j in range(node_dim):
                    if abs(diagonal_contrib[i, j]) > 1e-12:
                        global_i = node_start + i
                        global_j = node_start + j
                        rows.append(global_i)
                        cols.append(global_j)
                        data.append(diagonal_contrib[i, j])
        
        # Store edge positions in metadata
        metadata.edge_positions = edge_positions
        
        # Construct sparse matrix
        laplacian_coo = coo_matrix((data, (rows, cols)), shape=(n, n))
        laplacian_csr = laplacian_coo.tocsr()
        
        # Ensure symmetry (should be exact for valid sheaf)
        laplacian_symmetric = (laplacian_csr + laplacian_csr.T) / 2
        
        return laplacian_symmetric
    
    def _build_laplacian_standard(self, sheaf: Sheaf, edge_weights: Dict[Tuple[str, str], float],
                                 metadata: LaplacianMetadata) -> csr_matrix:
        """Build Laplacian using standard block matrix assembly."""
        n = metadata.total_dimension
        
        # Initialize dense matrix (converted to sparse at the end)
        laplacian_dense = np.zeros((n, n))
        
        # Build off-diagonal blocks
        for edge, restriction in sheaf.restrictions.items():
            source, target = edge
            weight = edge_weights.get(edge, 1.0)
            
            if source not in metadata.stalk_offsets or target not in metadata.stalk_offsets:
                continue
                
            source_start = metadata.stalk_offsets[source]
            target_start = metadata.stalk_offsets[target]
            source_dim = metadata.stalk_dimensions[source]
            target_dim = metadata.stalk_dimensions[target]
            
            R = restriction.detach().cpu().numpy() * weight
            
            # Ensure dimension compatibility
            if R.shape != (target_dim, source_dim):
                R_adjusted = np.zeros((target_dim, source_dim))
                min_rows = min(R.shape[0], target_dim)
                min_cols = min(R.shape[1], source_dim)
                R_adjusted[:min_rows, :min_cols] = R[:min_rows, :min_cols]
                R = R_adjusted
            
            # Set off-diagonal blocks: Δ[target, source] = -R, Δ[source, target] = -R^T
            laplacian_dense[target_start:target_start+target_dim, 
                           source_start:source_start+source_dim] = -R
            laplacian_dense[source_start:source_start+source_dim,
                           target_start:target_start+target_dim] = -R.T
        
        # Build diagonal blocks
        for node in sheaf.poset.nodes():
            if node not in metadata.stalk_offsets:
                continue
                
            node_start = metadata.stalk_offsets[node]
            node_dim = metadata.stalk_dimensions[node]
            
            # Compute diagonal contribution
            diagonal_contrib = np.zeros((node_dim, node_dim))
            
            # Sum over all adjacent edges
            for neighbor in list(sheaf.poset.successors(node)) + list(sheaf.poset.predecessors(node)):
                edge = (node, neighbor) if sheaf.poset.has_edge(node, neighbor) else (neighbor, node)
                if edge in sheaf.restrictions:
                    R = sheaf.restrictions[edge].detach().cpu().numpy()
                    weight = edge_weights.get(edge, 1.0)
                    R = R * weight
                    
                    if edge[0] == node:  # Outgoing edge
                        if R.shape[1] == node_dim:
                            diagonal_contrib += R.T @ R
                    else:  # Incoming edge
                        if R.shape[0] == node_dim:
                            diagonal_contrib += R @ R.T
            
            # Set diagonal block
            laplacian_dense[node_start:node_start+node_dim,
                           node_start:node_start+node_dim] = diagonal_contrib
        
        # Convert to sparse format
        laplacian_csr = csr_matrix(laplacian_dense)
        
        return laplacian_csr
    
    def _validate_laplacian_properties(self, laplacian: csr_matrix, metadata: LaplacianMetadata):
        """Validate mathematical properties of the constructed Laplacian."""
        try:
            # Check symmetry
            laplacian_T = laplacian.T
            symmetry_error = (laplacian - laplacian_T).max()
            
            if symmetry_error > 1e-10:
                logger.warning(f"Laplacian not symmetric: max error = {symmetry_error:.2e}")
            else:
                logger.debug("Laplacian symmetry verified")
            
            # Check positive semi-definite property (via smallest eigenvalues)
            try:
                from scipy.sparse.linalg import eigsh
                min_eigenvals = eigsh(laplacian, k=min(10, laplacian.shape[0]-1), 
                                     which='SA', return_eigenvectors=False)
                min_eigenval = min_eigenvals[0]
                
                if min_eigenval < -1e-10:
                    logger.warning(f"Laplacian not positive semi-definite: min eigenvalue = {min_eigenval:.2e}")
                else:
                    logger.debug(f"Laplacian positive semi-definite verified: min eigenvalue = {min_eigenval:.2e}")
                
                # Estimate condition number
                max_eigenvals = eigsh(laplacian, k=min(5, laplacian.shape[0]-1),
                                     which='LA', return_eigenvectors=False)
                max_eigenval = max_eigenvals[-1]
                
                if min_eigenval > 1e-12:
                    metadata.condition_number = max_eigenval / min_eigenval
                    logger.info(f"Laplacian condition number: {metadata.condition_number:.2e}")
                
            except Exception as e:
                logger.warning(f"Could not compute eigenvalues for validation: {e}")
                
        except Exception as e:
            logger.warning(f"Laplacian property validation failed: {e}")
    
    def to_torch_sparse(self, laplacian: csr_matrix) -> torch.sparse.FloatTensor:
        """Convert scipy sparse matrix to torch sparse tensor.
        
        Args:
            laplacian: scipy.sparse.csr_matrix to convert
            
        Returns:
            torch.sparse.FloatTensor compatible with GPU operations
        """
        # Convert to COO format for torch compatibility
        laplacian_coo = laplacian.tocoo()
        
        # Create torch sparse tensor
        indices = torch.stack([
            torch.from_numpy(laplacian_coo.row).long(),
            torch.from_numpy(laplacian_coo.col).long()
        ])
        values = torch.from_numpy(laplacian_coo.data).float()
        shape = laplacian_coo.shape
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
        
        # Move to GPU if enabled
        if self.enable_gpu and torch.cuda.is_available():
            sparse_tensor = sparse_tensor.cuda()
        
        return sparse_tensor.coalesce()  # Optimize sparse representation


def build_sheaf_laplacian(sheaf: Sheaf, enable_gpu: bool = True, 
                         memory_efficient: bool = True) -> Tuple[csr_matrix, LaplacianMetadata]:
    """Convenience function to build sheaf Laplacian.
    
    Args:
        sheaf: Sheaf object with whitened stalks and restrictions
        enable_gpu: Whether to enable GPU-compatible operations
        memory_efficient: Whether to use memory-efficient assembly
        
    Returns:
        Tuple of (sparse_laplacian, metadata)
    """
    builder = SheafLaplacianBuilder(enable_gpu=enable_gpu, memory_efficient=memory_efficient)
    return builder.build_laplacian(sheaf)