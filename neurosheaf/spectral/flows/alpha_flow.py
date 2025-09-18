"""α-Flow Implementation for Non-trivial Order-Free Analysis.

This module implements the α-flow method for comparing different GW sheaves using
baseline and residual Laplacian decomposition. The method splits within-network 
restriction energy into confident (base) and uncertain (residual) components:

L(α) = L_base + α * L_resid,  α ∈ ℝ≥0

Key Features:
- Edge partitioning strategies (quantile, topk, by_tag)
- Guard rails to prevent empty splits
- Fixed mass matrix mode for cross-architecture comparability
- Memory-efficient LinearOperator implementation
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Literal

__all__ = ["AlphaGroupingPolicy", "AlphaFlowBuild", "AlphaFlowBuilder"]
import logging

from ...utils.logging import setup_logger
from ...sheaf.data_structures import Sheaf

try:
    logger = setup_logger(__name__)
except ImportError:
    # Fallback to standard logging if setup_logger is unavailable
    import logging
    logger = logging.getLogger(__name__)


def _ensure_spd(D: sp.spmatrix, eps: float = 1e-12) -> sp.spmatrix:
    """Ensure mass matrix D is CSR format and SPD.
    
    This function converts the mass matrix to CSR format and adds a small
    ridge regularization to guarantee strict positive definiteness, which
    is required for stable eigenvalue computation with scipy.sparse.linalg.eigsh.
    
    Args:
        D: Mass matrix (must be scipy.sparse matrix)
        eps: Ridge regularization parameter
        
    Returns:
        CSR sparse matrix that is guaranteed to be SPD
        
    Raises:
        TypeError: If D is not a scipy.sparse matrix
    """
    if not isinstance(D, sp.spmatrix):
        raise TypeError("Mass matrix D must be a scipy.sparse matrix (CSR preferred).")
    
    # Convert to CSR format
    D = D.tocsr()
    
    # Optional: symmetrize just in case
    if (D - D.T).nnz:
        D = 0.5 * (D + D.T)
    
    # Add tiny ridge for strict positive definiteness
    return D + eps * sp.eye(D.shape[0], format='csr')


@dataclass(frozen=True)
class AlphaGroupingPolicy:
    """Policy for partitioning edges into base and residual sets.
    
    The α-flow method requires partitioning edges into two groups:
    - Base edges: "confident" edges with low cost or high similarity
    - Residual edges: "uncertain" edges with high cost or low similarity
    
    Edge Partitioning Strategies:
    - quantile: base = edges with score ≤ quantile_q; residual = rest
    - topk: base = k% best-scoring edges; residual = rest  
    - by_tag: base = edges with specific tags; residual = rest
    
    Semantics:
    - 'cost': Lower values = more confident (costs, dissimilarities)
    - 'similarity': Higher values = more confident (similarities, affinities)
    """
    kind: Literal['quantile', 'topk', 'by_tag'] = 'quantile'
    param: float = 0.5  # quantile threshold (0-1) or topk fraction (0-1)
    tags_base: Optional[Sequence[str]] = None  # for 'by_tag' mode only
    semantics: Literal['cost', 'similarity'] = 'cost'  # How to interpret edge values
    
    def __post_init__(self):
        """Validate parameters."""
        if self.kind in ['quantile', 'topk']:
            if not 0.0 < self.param < 1.0:
                raise ValueError(f"param must be in (0, 1) for {self.kind}, got {self.param}")
        elif self.kind == 'by_tag':
            if self.tags_base is None:
                raise ValueError("tags_base must be provided for by_tag mode")


@dataclass 
class AlphaFlowBuild:
    """Result from α-flow Laplacian construction.
    
    Contains the two component operators and mass matrix needed for
    α-flow analysis: L(α) = L_base + α*L_resid with mass matrix D.
    
    Attributes:
        L_base: Baseline Laplacian from "confident" edges
        L_resid: Residual Laplacian from "uncertain" edges  
        D: Mass matrix (CSR sparse matrix format for eigsh compatibility)
        meta: Metadata including edge counts, indices, and statistics
    """
    L_base: LinearOperator    # Exact matvec for B^T W_base B
    L_resid: LinearOperator   # Exact matvec for B^T W_resid B  
    D: sp.spmatrix           # Mass matrix in CSR format, SPD guaranteed
    meta: Dict  # Contains: nnz_base, nnz_resid, |base|, |resid|, base_indices, resid_indices


class AlphaFlowBuilder:
    """Builder for α-flow Laplacian decomposition.
    
    This class handles the partition of edges into base/residual sets and
    constructs the corresponding Laplacian operators for α-flow analysis.
    
    The builder integrates with the existing GW sheaf infrastructure to:
    1. Extract edge costs from GW metadata
    2. Partition edges according to grouping policy
    3. Apply guard rails to prevent empty splits
    4. Construct L_base and L_resid operators
    5. Build consistent mass matrix D
    """
    
    def __init__(self, sheaf: Sheaf, gw_laplacian_builder, use_normalized_laplacian: Union[str, bool] = False):
        """Initialize α-flow builder.
        
        Args:
            sheaf: GW sheaf containing edge costs and restrictions
            gw_laplacian_builder: GWLaplacianBuilder instance for matrix assembly
            use_normalized_laplacian: Normalization mode:
                                    - False/'none': No normalization (generalized eigenproblem)
                                    - True/'sym': Symmetric normalization
                                    - 'rw': Random walk normalization
        """
        self.sheaf = sheaf
        self.gw_builder = gw_laplacian_builder
        self.normalization_mode = ('sym' if use_normalized_laplacian is True else
                                 'none' if use_normalized_laplacian in (False, 'none', None) else
                                 use_normalized_laplacian)
        self._validate_sheaf()
    
    def _validate_sheaf(self):
        """Validate that sheaf is suitable for α-flow analysis."""
        if not self.sheaf.is_gw_sheaf():
            raise ValueError("α-flow requires GW sheaf with cost metadata")
        
        gw_costs = self.sheaf.metadata.get('gw_costs', {})
        if not gw_costs:
            logger.warning("No GW costs found in metadata, will use fallback weights")
    
    def _extract_edge_costs(self, active_edges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """Extract costs for active edges from sheaf metadata.
        
        Uses GW costs if available, otherwise falls back to restriction norms.
        
        Args:
            active_edges: List of edges to extract costs for
            
        Returns:
            Dictionary mapping edges to costs (lower = more confident)
        """
        gw_costs = self.sheaf.metadata.get('gw_costs', {})
        edge_costs = {}
        missing_edges = []
        
        # Extract available GW costs
        for edge in active_edges:
            if edge in gw_costs:
                edge_costs[edge] = gw_costs[edge]
            else:
                missing_edges.append(edge)
        
        # Handle missing edges with fallback
        if missing_edges:
            logger.warning(f"Missing GW costs for {len(missing_edges)} edges, using restriction norms")
            for edge in missing_edges:
                if edge in self.sheaf.restrictions:
                    restriction = self.sheaf.restrictions[edge]
                    arr = restriction.numpy() if hasattr(restriction, "numpy") else restriction
                    try:
                        if sp.issparse(arr):
                            cost = float(sp.linalg.norm(arr))  # Frobenius for sparse
                        else:
                            a = np.asarray(arr)
                            if a.size == 0:
                                cost = 0.0  # Guard for 0-size arrays
                            else:
                                cost = float(np.linalg.norm(a))    # Frobenius for 2D, L2 for 1D
                    except Exception:
                        logger.warning("Could not compute restriction norm; falling back to cost=1.0")
                        cost = 1.0
                    edge_costs[edge] = cost
                else:
                    logger.error(f"Edge {edge} missing from both gw_costs and restrictions")
                    
        return edge_costs
    
    def _ensure_edge_tuples(self, edges: List) -> List[Tuple]:
        """Ensure edges are properly formatted as 2-tuples.
        
        Args:
            edges: List of edges (may contain non-tuple formats)
            
        Returns:
            List of properly formatted edge tuples (preserving original key types)
        """
        formatted_edges = []
        for edge in edges:
            if isinstance(edge, (tuple, list)) and len(edge) == 2:
                # Keep original key types - no str() coercion
                formatted_edges.append((edge[0], edge[1]))
            else:
                logger.warning(f"Invalid edge format: {edge}, skipping")
        return formatted_edges
    
    def _partition_edges(self, edge_costs: Dict[Tuple[str, str], float], 
                        grouping: AlphaGroupingPolicy) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Partition edges into base and residual sets.
        
        Args:
            edge_costs: Dictionary mapping edges to costs
            grouping: Grouping policy for partitioning
            
        Returns:
            Tuple of (base_edges, residual_edges)
            
        Raises:
            ValueError: If partitioning results in empty sets after guard rails
        """
        if not edge_costs:
            raise ValueError("Cannot partition empty edge set")
        
        edges = list(edge_costs.keys())
        edge_to_idx = {e: i for i, e in enumerate(edges)}  # Avoid O(n²) index lookups
        values = np.array([edge_costs[edge] for edge in edges], dtype=float)
        
        # Convert values to scores where lower score = more confident
        if grouping.semantics == 'similarity':
            scores = -values  # Flip similarity to cost-like semantics
        else:
            scores = values   # Cost semantics: lower = better
        
        if grouping.kind == 'quantile':
            # Split at quantile of score distribution (lower scores = more confident)
            threshold = np.quantile(scores, grouping.param)
            base_mask = scores <= threshold
            
        elif grouping.kind == 'topk':
            # Select top fraction of best-scoring (lowest score) edges
            n_base = max(1, int(len(edges) * grouping.param))
            sorted_indices = np.argsort(scores)
            base_mask = np.zeros(len(edges), dtype=bool)
            base_mask[sorted_indices[:n_base]] = True
            
        elif grouping.kind == 'by_tag':
            # Use edge metadata tags if available
            edge_tags = self.sheaf.metadata.get('edge_tags', {})  # Dict[edge, str]
            if edge_tags:
                tagset = set(grouping.tags_base or [])
                base_mask = np.array([edge_tags.get(e) in tagset for e in edges], dtype=bool)
                
                # Log if tags_base selects 0 edges (guard rail will handle this)
                if np.sum(base_mask) == 0:
                    logger.warning(f"by_tag selection with tags {list(tagset)} matched 0 edges, guard rails will adjust")
            else:
                # Extract available edge tags for error message
                available_tags = set()
                if hasattr(self.sheaf, 'edge_metadata') and self.sheaf.edge_metadata:
                    for edge_meta in self.sheaf.edge_metadata.values():
                        if 'tags' in edge_meta:
                            available_tags.update(edge_meta['tags'])
                
                available_tags_msg = f"Available edge tags: {sorted(available_tags)}" if available_tags else "No edge tags found in sheaf metadata"
                raise NotImplementedError(f"by_tag partitioning not yet implemented. {available_tags_msg}")
        
        # Apply partitioning
        base_edges = [edges[i] for i in range(len(edges)) if base_mask[i]]
        resid_edges = [edges[i] for i in range(len(edges)) if not base_mask[i]]
        
        # Guard rails: ensure both sets are non-empty (deterministic)
        if len(base_edges) == 0:
            logger.warning("Empty base partition detected; moving lowest-score edge to base")
            # Move global minimum score into base (most confident)
            idx = int(np.argmin(scores))
            e = edges[idx]
            base_edges.append(e)
            if e in resid_edges:
                resid_edges.remove(e)

        if len(resid_edges) == 0:
            logger.warning("Empty residual partition detected; moving highest-score edge to residual")
            # Move global maximum score from base to residual (least confident)
            base_scores = np.array([scores[edge_to_idx[e]] for e in base_edges])
            j = int(np.argmax(base_scores))
            e = base_edges.pop(j)
            resid_edges.append(e)
        
        # Final validation
        if len(base_edges) == 0 or len(resid_edges) == 0:
            raise ValueError(f"Guard rails failed: base={len(base_edges)}, resid={len(resid_edges)}")
        
        # Calculate split statistics for logging
        total_edges = len(edges)
        base_fraction = len(base_edges) / total_edges if total_edges > 0 else 0.0
        resid_fraction = len(resid_edges) / total_edges if total_edges > 0 else 0.0
        
        logger.info(
            f"α-flow edge partitioning: base={len(base_edges)} ({base_fraction:.1%}), "
            f"resid={len(resid_edges)} ({resid_fraction:.1%}) "
            f"[{grouping.kind}:{grouping.param}, semantics:{grouping.semantics}]"
        )
        return base_edges, resid_edges
    
    def build(self, *, 
              grouping: AlphaGroupingPolicy,
              mass_mode: Literal['fixed', 'adaptive'] = 'fixed',
              as_linear_operator: bool = True) -> AlphaFlowBuild:
        """Build α-flow Laplacian decomposition.
        
        Args:
            grouping: Policy for partitioning edges into base/residual
            mass_mode: Mass matrix mode ('fixed' for consistency, 'adaptive' for accuracy)
            as_linear_operator: Whether to return LinearOperator (True) or sparse matrix (False)
            
        Returns:
            AlphaFlowBuild containing L_base, L_resid, D, and metadata
        """
        # Extract edge costs
        active_edges = list(self.sheaf.restrictions.keys())
        active_edges = self._ensure_edge_tuples(active_edges)  # Ensure proper formatting
        edge_costs = self._extract_edge_costs(active_edges)
        
        # Partition edges
        base_edges, resid_edges = self._partition_edges(edge_costs, grouping)
        
        # Build component Laplacians using GW builder with normalization
        # Note: grouping parameter is passed but builder should use explicit base_edges/resid_edges
        L_base, L_resid, D, metadata = self.gw_builder.build_laplacian_grouped(
            sheaf=self.sheaf,  # Pass the sheaf explicitly
            grouping=grouping,  # For metadata only - builder should use explicit edge lists
            base_edges=base_edges,
            resid_edges=resid_edges,
            mass_mode=mass_mode,
            as_linear_operator=as_linear_operator,
            normalize=self.normalization_mode  # Pass normalization mode
        )
        
        # Ensure mass matrix is CSR and SPD for eigsh compatibility (fail fast if not sparse)
        D = _ensure_spd(D)  # D must be sparse; raise TypeError otherwise
        
        # Log diagnostics for dtype and sparsity
        L_base_nnz = getattr(L_base, 'nnz', 'N/A')
        L_resid_nnz = getattr(L_resid, 'nnz', 'N/A') 
        D_nnz = getattr(D, 'nnz', 'N/A')
        
        logger.info(f"α-flow build diagnostics: "
                   f"L_base(dtype={L_base.dtype}, nnz={L_base_nnz}), "
                   f"L_resid(dtype={L_resid.dtype}, nnz={L_resid_nnz}), "
                   f"D(dtype={D.dtype}, nnz={D_nnz})")
        
        # Compile metadata
        meta = {
            'n_base_edges': len(base_edges),
            'n_resid_edges': len(resid_edges), 
            'base_edges': base_edges,
            'resid_edges': resid_edges,
            'grouping_policy': grouping,
            'mass_mode': mass_mode,
            'normalization_mode': self.normalization_mode,
            'L_base_dtype': str(L_base.dtype),
            'L_resid_dtype': str(L_resid.dtype),
            'D_dtype': str(D.dtype),
            'L_base_nnz': L_base_nnz,
            'L_resid_nnz': L_resid_nnz,
            'D_nnz': D_nnz,
            **metadata  # Include GW builder metadata
        }
        
        # Quick size assert on return
        assert L_base.shape == L_resid.shape == (D.shape[0], D.shape[0]), \
            f"Shape mismatch: L_base={L_base.shape}, L_resid={L_resid.shape}, D={D.shape}"

        return AlphaFlowBuild(
            L_base=L_base,
            L_resid=L_resid,
            D=D,
            meta=meta
        )
    
    def get_csr_matrices(self, *, 
                        grouping: AlphaGroupingPolicy,
                        mass_mode: Literal['fixed', 'adaptive'] = 'fixed') -> AlphaFlowBuild:
        """Build α-flow Laplacian decomposition with explicit CSR matrices.
        
        This is optimized for eigenvalue computation where CSR matrices
        can be more efficient for shift-invert solvers than LinearOperator.
        
        Args:
            grouping: Policy for partitioning edges into base/residual
            mass_mode: Mass matrix mode ('fixed' for consistency, 'adaptive' for accuracy)
            
        Returns:
            AlphaFlowBuild containing L_base, L_resid as CSR matrices, D, and metadata
        """
        return self.build(
            grouping=grouping,
            mass_mode=mass_mode,
            as_linear_operator=False  # Return CSR matrices
        )
    
    def as_csr_combined(self, build: AlphaFlowBuild, alpha: float):
        """Create explicit CSR matrix for L(α) = L_base + α*L_resid.
        
        This is more efficient for shift-invert eigenvalue computation
        than the LinearOperator version.
        
        Args:
            build: AlphaFlowBuild from get_csr_matrices() method (CSR format)
            alpha: Flow parameter (α ≥ 0)
            
        Returns:
            Explicit CSR matrix implementing L(α) = L_base + α*L_resid
        """
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")
        
        L_base = build.L_base
        L_resid = build.L_resid
        
        # Ensure we have CSR matrices, not LinearOperator
        if not hasattr(L_base, 'tocsr') or not hasattr(L_resid, 'tocsr'):
            raise TypeError("as_csr_combined requires CSR matrices. Use get_csr_matrices() to build.")
        
        # Convert to CSR if needed
        L_base_csr = L_base.tocsr() if hasattr(L_base, 'tocsr') else L_base
        L_resid_csr = L_resid.tocsr() if hasattr(L_resid, 'tocsr') else L_resid
        
        # Form explicit L(α) = L_base + α*L_resid
        L_alpha_csr = L_base_csr + alpha * L_resid_csr
        
        return L_alpha_csr
    
    def as_operator(self, build: AlphaFlowBuild, alpha: float) -> LinearOperator:
        """Create LinearOperator for L(α) = L_base + α*L_resid.
        
        Args:
            build: AlphaFlowBuild from build() method
            alpha: Flow parameter (α ≥ 0)
            
        Returns:
            LinearOperator implementing L(α) = L_base + α*L_resid
        """
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")
        
        L_base = build.L_base
        L_resid = build.L_resid
        
        # Shape and dtype assertions for component compatibility
        assert L_base.shape == L_resid.shape, f"L_base and L_resid must have same shape: {L_base.shape} vs {L_resid.shape}"
        assert L_base.dtype == L_resid.dtype, f"L_base and L_resid must have same dtype: {L_base.dtype} vs {L_resid.dtype}"
        assert L_base.shape[0] == L_base.shape[1], f"L_base must be square: {L_base.shape}"
        assert L_resid.shape[0] == L_resid.shape[1], f"L_resid must be square: {L_resid.shape}"
        assert build.D.shape[0] == L_base.shape[0], f"Mass matrix D must match Laplacian size: {build.D.shape[0]} vs {L_base.shape[0]}"
        
        def matvec(x):
            return L_base @ x + alpha * (L_resid @ x)
        
        def rmatvec(x):
            # For symmetric operators, rmatvec = matvec
            return matvec(x)
        
        # Guard dtype - some LinearOperators may have dtype=None
        dtype = L_base.dtype or np.float64
        
        return LinearOperator(
            shape=L_base.shape,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=dtype
        )
    
    def sanity_check_monotonicity(self, build: AlphaFlowBuild, 
                                 alpha_values: Tuple[float, float] = (0.0, 1.0),
                                 n_probes: int = 8) -> Dict[str, float]:
        """Sanity check monotonicity: trace(L(α)) should be non-decreasing in α.
        
        Uses Hutchinson trace estimation to quickly verify that the α-flow
        construction is mathematically sound.
        
        Args:
            build: AlphaFlowBuild from build() method
            alpha_values: Two α values to test (default: 0.0, 1.0)
            n_probes: Number of Hutchinson probes for trace estimation
            
        Returns:
            Dictionary with trace estimates and monotonicity check result
        """
        alpha1, alpha2 = alpha_values
        if alpha1 >= alpha2:
            raise ValueError(f"Need alpha1 < alpha2, got {alpha1} >= {alpha2}")
        
        n = build.L_base.shape[0]
        
        # Generate Rademacher probes (vectorized, deterministic)
        rng = np.random.default_rng(0)
        probes = rng.integers(0, 2, size=(n, n_probes))*2 - 1  # Rademacher in {-1,1}
        
        # Estimate trace for α₁
        L1 = self.as_operator(build, alpha1)
        trace1_samples = np.array([probes[:, i].T @ (L1 @ probes[:, i]) for i in range(n_probes)])
        trace1_est = np.mean(trace1_samples)
        
        # Estimate trace for α₂  
        L2 = self.as_operator(build, alpha2)
        trace2_samples = np.array([probes[:, i].T @ (L2 @ probes[:, i]) for i in range(n_probes)])
        trace2_est = np.mean(trace2_samples)
        
        is_monotonic = trace2_est >= trace1_est
        
        logger.debug(f"Monotonicity check: tr(L({alpha1})) ≈ {trace1_est:.6e}, "
                    f"tr(L({alpha2})) ≈ {trace2_est:.6e}, "
                    f"monotonic: {is_monotonic}")
        
        return {
            'alpha_values': alpha_values,
            'trace_estimates': [trace1_est, trace2_est],
            'is_monotonic': is_monotonic,
            'n_probes': n_probes
        }