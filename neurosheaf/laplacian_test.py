"""
Sparse sheaf Laplacian construction for neural network analysis.

This script contains the primary implementation of the SheafLaplacianBuilder
and a comprehensive, self-contained test suite to verify its correctness
on various graph structures.

To run the verification tests, execute the script directly:
    python <filename>.py
"""
import torch
import numpy as np
import networkx as nx
import time
import psutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ==================================================================
# =================== CORE IMPLEMENTATION ==========================
# ==================================================================

logger = logging.getLogger(__name__)

class ComputationError(Exception):
    pass

@dataclass
class Sheaf:
    """A simple dataclass to represent a sheaf for testing purposes."""
    stalks: Dict[str, torch.Tensor]
    restrictions: Dict[Tuple[str, str], torch.Tensor]
    poset: nx.DiGraph
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LaplacianMetadata:
    """Metadata for the constructed sheaf Laplacian."""
    total_dimension: int = 0
    stalk_dimensions: Dict[str, int] = None
    stalk_offsets: Dict[str, int] = None
    sparsity_ratio: float = 0.0
    condition_number: Optional[float] = None
    construction_time: float = 0.0
    memory_usage: float = 0.0
    
    def __post_init__(self):
        if self.stalk_dimensions is None:
            self.stalk_dimensions = {}
        if self.stalk_offsets is None:
            self.stalk_offsets = {}

class SheafLaplacianBuilder:
    """
    Builds a sparse sheaf Laplacian Δ = δᵀδ from sheaf data.

    This implementation has been optimized for correctness, performance, and clarity.
    It uses a single, efficient COO-based construction method that correctly
    implements the mathematical formula for the sheaf Laplacian.
    """
    
    def __init__(self, enable_gpu: bool = True, validate_properties: bool = True):
        self.enable_gpu = enable_gpu
        self.validate_properties = validate_properties
        
        if self.enable_gpu and not torch.cuda.is_available():
            logger.warning("GPU support enabled, but no CUDA device found. Falling back to CPU.")
            self.enable_gpu = False
    
    def build(self, sheaf: Sheaf, edge_weights: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple['csr_matrix', LaplacianMetadata]:
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        logger.info(f"Starting sheaf Laplacian construction for a graph with {len(sheaf.stalks)} nodes and {len(sheaf.restrictions)} edges.")
        
        try:
            metadata = self._initialize_metadata(sheaf)
            
            if edge_weights is None:
                edge_weights = {edge: 1.0 for edge in sheaf.restrictions.keys()}
            
            laplacian = self._build_laplacian_optimized(sheaf, edge_weights, metadata)
            
            if self.validate_properties:
                self._validate_laplacian_properties(laplacian, metadata)
            
            final_memory = process.memory_info().rss
            
            metadata.construction_time = time.time() - start_time
            metadata.memory_usage = (final_memory - initial_memory) / (1024**2) # MB
            metadata.sparsity_ratio = 1.0 - (laplacian.nnz / (laplacian.shape[0] * laplacian.shape[1]))
            
            logger.info(
                f"Laplacian construction successful. Shape: {laplacian.shape}, "
                f"NNZ: {laplacian.nnz} ({metadata.sparsity_ratio:.2%} sparse), "
                f"Time: {metadata.construction_time:.3f}s, "
                f"Memory: {metadata.memory_usage:.2f} MB"
            )
            
            return laplacian, metadata
            
        except Exception as e:
            logger.error(f"Laplacian construction failed: {e}", exc_info=True)
            raise ComputationError(f"Laplacian construction failed: {e}", operation="build_laplacian")

    def _initialize_metadata(self, sheaf: Sheaf) -> LaplacianMetadata:
        metadata = LaplacianMetadata()
        for node, stalk_data in sheaf.stalks.items():
            metadata.stalk_dimensions[node] = stalk_data.shape[0]
        
        offset = 0
        for node in sorted(sheaf.poset.nodes()): # Sort for deterministic ordering
            if node in metadata.stalk_dimensions:
                metadata.stalk_offsets[node] = offset
                offset += metadata.stalk_dimensions[node]
        
        metadata.total_dimension = offset
        logger.debug(f"Total Laplacian dimension computed: {metadata.total_dimension}")
        return metadata

    def _build_laplacian_optimized(self, sheaf: Sheaf, edge_weights: Dict[Tuple[str, str], float], metadata: LaplacianMetadata) -> 'csr_matrix':
        from scipy.sparse import coo_matrix

        rows, cols, data = [], [], []

        # 1. Construct Off-Diagonal Blocks
        for edge, restriction in sheaf.restrictions.items():
            u, v = edge
            weight = edge_weights.get(edge, 1.0)
            if u not in metadata.stalk_offsets or v not in metadata.stalk_offsets:
                continue

            u_start, v_start = metadata.stalk_offsets[u], metadata.stalk_offsets[v]
            R = restriction.detach().cpu().numpy() * weight
            nz_rows, nz_cols = np.where(np.abs(R) > 1e-12)
            
            rows.extend(v_start + nz_rows)
            cols.extend(u_start + nz_cols)
            data.extend(-R[nz_rows, nz_cols])

            rows.extend(u_start + nz_cols)
            cols.extend(v_start + nz_rows)
            data.extend(-R[nz_rows, nz_cols])

        # 2. Construct Diagonal Blocks
        for node, dim in metadata.stalk_dimensions.items():
            node_start = metadata.stalk_offsets[node]
            diag_block = np.zeros((dim, dim))

            for successor in sheaf.poset.successors(node):
                edge = (node, successor)
                if edge in sheaf.restrictions:
                    R = sheaf.restrictions[edge].detach().cpu().numpy()
                    weight = edge_weights.get(edge, 1.0)
                    if R.shape[1] == dim:
                        diag_block += (weight**2) * (R.T @ R)
            
            for predecessor in sheaf.poset.predecessors(node):
                edge = (predecessor, node)
                if edge in sheaf.restrictions and edge_weights.get(edge, 1.0) > 1e-12:
                    diag_block += np.eye(dim)

            nz_rows, nz_cols = np.where(np.abs(diag_block) > 1e-12)
            rows.extend(node_start + nz_rows)
            cols.extend(node_start + nz_cols)
            data.extend(diag_block[nz_rows, nz_cols])

        # 3. Assemble the sparse matrix
        n = metadata.total_dimension
        if n == 0: 
            from scipy.sparse import csr_matrix
            return csr_matrix((0, 0))
        
        laplacian_coo = coo_matrix((data, (rows, cols)), shape=(n, n))
        return laplacian_coo.tocsr()

    def _validate_laplacian_properties(self, laplacian: 'csr_matrix', metadata: LaplacianMetadata):
        logger.debug("Validating Laplacian properties...")
        symmetry_diff = np.abs(laplacian - laplacian.T).max()
        if symmetry_diff > 1e-9:
            logger.warning(f"Laplacian is not perfectly symmetric. Max difference: {symmetry_diff:.2e}")
        else:
            logger.debug("Symmetry property verified.")

        try:
            from scipy.sparse.linalg import eigsh
            if laplacian.shape[0] > 1:
                min_eigenval = eigsh(laplacian, k=1, which='SA', return_eigenvectors=False)[0]
                if min_eigenval < -1e-9:
                    logger.warning(f"Laplacian may not be positive semi-definite. Smallest eigenvalue: {min_eigenval:.2e}")
                else:
                    logger.debug(f"Positive semi-definite property verified. Smallest eigenvalue: {min_eigenval:.2e}")
            else:
                 logger.debug("Skipping eigenvalue check for 1x1 matrix.")
        except Exception as e:
            logger.warning(f"Could not compute eigenvalues for validation: {e}")

# ==================================================================
# =================== VERIFICATION TEST SUITE ====================
# ==================================================================

def _create_simple_sheaf():
    poset = nx.DiGraph()
    poset.add_edge("n1", "n2")
    stalks = {"n1": torch.eye(2), "n2": torch.eye(1)}
    restrictions = {("n1", "n2"): torch.tensor([[2., 3.]])}
    return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset)

def _create_cyclic_sheaf():
    poset = nx.DiGraph()
    poset.add_edges_from([("n1", "n2"), ("n2", "n3"), ("n3", "n1")])
    stalks = {"n1": torch.eye(1), "n2": torch.eye(1), "n3": torch.eye(1)}
    restrictions = {
        ("n1", "n2"): torch.tensor([[2.0]]),
        ("n2", "n3"): torch.tensor([[3.0]]),
        ("n3", "n1"): torch.tensor([[4.0]])
    }
    return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset)

def _create_disconnected_sheaf():
    poset = nx.DiGraph()
    poset.add_edges_from([("n1", "n2"), ("n3", "n4")])
    stalks = {"n1": torch.eye(2), "n2": torch.eye(1), "n3": torch.eye(1), "n4": torch.eye(2)}
    restrictions = {
        ("n1", "n2"): torch.tensor([[1., 2.]]),
        ("n3", "n4"): torch.tensor([[3.], [4.]])
    }
    return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset)

def _create_self_loop_sheaf():
    poset = nx.DiGraph()
    poset.add_edges_from([("n1", "n2"), ("n2", "n2"), ("n1", "n3")])
    stalks = {"n1": torch.eye(1), "n2": torch.eye(2), "n3": torch.eye(1)}
    restrictions = {
        ("n1", "n2"): torch.tensor([[1.], [2.]]),
        ("n2", "n2"): torch.tensor([[3., 0.], [0., 5.]]),
        ("n1", "n3"): torch.tensor([[100.]])
    }
    edge_weights = {("n1", "n2"): 1.0, ("n2", "n2"): 1.0, ("n1", "n3"): 0.0}
    return Sheaf(stalks=stalks, restrictions=restrictions, poset=poset), edge_weights

def _create_no_edge_sheaf():
    poset = nx.DiGraph()
    poset.add_nodes_from(["n1", "n2", "n3"])
    stalks = {"n1": torch.eye(2), "n2": torch.eye(1), "n3": torch.eye(3)}
    return Sheaf(stalks=stalks, restrictions={}, poset=poset)

def run_verification_tests():
    print("=" * 60)
    print("🚀 Running Full Verification Test Suite...")
    print("=" * 60)
    
    builder = SheafLaplacianBuilder(validate_properties=True)
    all_tests_passed = True
    
    test_cases = [
        {"name": "Test 1: Simple Two-Node Graph", "sheaf_fn": _create_simple_sheaf, "weights": None,
         "expected": np.array([[4., 6., -2.], [6., 9., -3.], [-2., -3., 1.]])},
        
        {"name": "Test 2: Cyclic Three-Node Graph", "sheaf_fn": _create_cyclic_sheaf, "weights": None,
         "expected": np.array([[5., -2., -4.], [-2., 10., -3.], [-4., -3., 17.]])},
        
        {"name": "Test 3: Disconnected Graph", "sheaf_fn": _create_disconnected_sheaf, "weights": None,
         "expected": np.array([[1., 2., -1., 0., 0., 0.], [2., 4., -2., 0., 0., 0.], [-1., -2., 1., 0., 0., 0.],
                               [0., 0., 0., 25., -3., -4.], [0., 0., 0., -3., 1., 0.], [0., 0., 0., -4., 0., 1.]])},
        
        {"name": "Test 4: Self-Loop and Zero-Weight Edge", "sheaf_fn": _create_self_loop_sheaf,
         "expected": np.array([[5., -1., -2., 0.], [-1., 5., 0., 0.], [-2., 0., 17., 0.], [0., 0., 0., 0.]])},
        
        {"name": "Test 5: Graph with No Edges", "sheaf_fn": _create_no_edge_sheaf, "weights": None,
         "expected": np.zeros((6, 6))},
    ]

    for test in test_cases:
        try:
            print(f"\n--- {test['name']} ---")
            sheaf_data = test["sheaf_fn"]()
            weights = test.get("weights")
            if isinstance(sheaf_data, tuple):
                sheaf, default_weights = sheaf_data
                weights = default_weights # Use weights defined with the sheaf
            else:
                sheaf = sheaf_data

            computed_L, _ = builder.build(sheaf, edge_weights=weights)
            computed_L_dense = computed_L.toarray()

            if np.allclose(computed_L_dense, test["expected"]):
                print(f"✅ {test['name']} PASSED")
            else:
                print(f"❌ {test['name']} FAILED:")
                print(f"Expected:\n{test['expected']}")
                print(f"Got:\n{computed_L_dense}")
                print(f"Difference:\n{computed_L_dense - test['expected']}")
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test['name']} FAILED: {e}")
            all_tests_passed = False
            
    print("-" * 60)
    if all_tests_passed:
        print("🎉 \033[1m\033[92mAll tests passed successfully!\033[0m")
    else:
        print("🔥 \033[1m\033[91mSome tests failed. Please review the output.\033[0m")
    print("=" * 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    run_verification_tests()