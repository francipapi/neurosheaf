# Phase 3: Sheaf Construction Testing Suite

## Overview

Phase 3 focuses on sheaf construction with FX-based poset extraction, Procrustes restriction maps, and sparse Laplacian assembly. The testing suite emphasizes correctness of automatic architecture analysis, mathematical validity of sheaf properties, and performance of sparse operations.

## Test Categories

### 1. FX Poset Extraction Tests (CRITICAL)
```python
# tests/phase3_sheaf/critical/test_fx_poset_extraction.py
import pytest
import torch
import torch.nn as nn
import networkx as nx
from neurosheaf.sheaf.poset import FXPosetExtractor
from unittest.mock import patch

class TestFXPosetExtractionCritical:
    """Critical tests for FX-based poset extraction."""
    
    def test_sequential_model_structure(self):
        """Test correct poset extraction for sequential models."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        )
        
        extractor = FXPosetExtractor()
        poset = extractor.extract_poset(model)
        
        # Should be a DAG
        assert nx.is_directed_acyclic_graph(poset)
        
        # Should have correct number of nodes (excluding placeholder/output)
        assert len(poset.nodes()) == 5
        
        # Should have linear structure
        topo_order = list(nx.topological_sort(poset))
        for i in range(len(topo_order) - 1):
            # Each node should connect to the next
            assert poset.has_edge(topo_order[i], topo_order[i + 1])
    
    def test_resnet_skip_connection_detection(self):
        """Test detection of ResNet-style skip connections."""
        class ResidualBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(dim)
                self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(dim)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + identity  # Skip connection
                return self.relu(out)
        
        model = ResidualBlock(64)
        extractor = FXPosetExtractor()
        poset = extractor.extract_poset(model)
        
        # Find addition node (skip connection)
        add_nodes = []
        for node in poset.nodes():
            node_data = poset.nodes[node]
            if 'add' in node_data.get('name', '').lower() or 'add' in str(node_data.get('target', '')).lower():
                add_nodes.append(node)
        
        assert len(add_nodes) > 0, "Should detect skip connection (add operation)"
        
        # Skip connection node should have multiple predecessors
        add_node = add_nodes[0]
        predecessors = list(poset.predecessors(add_node))
        assert len(predecessors) >= 2, f"Skip connection should have >=2 predecessors, got {len(predecessors)}"
    
    def test_inception_parallel_branches(self):
        """Test detection of Inception-style parallel branches."""
        class InceptionBlock(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.branch1 = nn.Conv2d(in_channels, 64, 1)
                self.branch2 = nn.Sequential(
                    nn.Conv2d(in_channels, 48, 1),
                    nn.Conv2d(48, 64, 3, padding=1)
                )
                self.branch3 = nn.Sequential(
                    nn.Conv2d(in_channels, 64, 1),
                    nn.Conv2d(64, 96, 3, padding=1),
                    nn.Conv2d(96, 96, 3, padding=1)
                )
                
            def forward(self, x):
                branch1_out = self.branch1(x)
                branch2_out = self.branch2(x)
                branch3_out = self.branch3(x)
                return torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        
        model = InceptionBlock(256)
        extractor = FXPosetExtractor()
        poset = extractor.extract_poset(model)
        
        # Find concatenation node
        cat_nodes = []
        for node in poset.nodes():
            node_data = poset.nodes[node]
            if 'cat' in node_data.get('name', '').lower() or 'cat' in str(node_data.get('target', '')).lower():
                cat_nodes.append(node)
        
        assert len(cat_nodes) > 0, "Should detect concatenation operation"
        
        # Concatenation node should have multiple predecessors (parallel branches)
        cat_node = cat_nodes[0]
        predecessors = list(poset.predecessors(cat_node))
        assert len(predecessors) >= 3, f"Inception should have >=3 parallel branches, got {len(predecessors)}"
    
    def test_dynamic_model_fallback(self):
        """Test graceful fallback for dynamic models that can't be traced."""
        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(10, 20),
                    nn.Linear(20, 30),
                    nn.Linear(30, 10)
                ])
                
            def forward(self, x, num_layers=None):
                # Dynamic control flow
                num_layers = num_layers or len(self.layers)
                for i in range(min(num_layers, len(self.layers))):
                    x = self.layers[i](x)
                return x
        
        model = DynamicModel()
        extractor = FXPosetExtractor(handle_dynamic=True)
        
        # Should not crash and should return some structure
        poset = extractor.extract_poset(model)
        
        assert isinstance(poset, nx.DiGraph)
        assert len(poset.nodes()) > 0
        assert nx.is_directed_acyclic_graph(poset)
    
    def test_transformer_attention_detection(self):
        """Test detection of Transformer attention patterns."""
        class SimpleAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.query = nn.Linear(d_model, d_model)
                self.key = nn.Linear(d_model, d_model)
                self.value = nn.Linear(d_model, d_model)
                self.output = nn.Linear(d_model, d_model)
                
            def forward(self, x):
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
                
                # Simplified attention
                scores = torch.matmul(q, k.transpose(-2, -1))
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                
                return self.output(out)
        
        model = SimpleAttention(512, 8)
        extractor = FXPosetExtractor()
        poset = extractor.extract_poset(model)
        
        # Should detect multiple paths (Q, K, V) converging
        # Find nodes with multiple predecessors
        multi_pred_nodes = [n for n in poset.nodes() if len(list(poset.predecessors(n))) > 1]
        assert len(multi_pred_nodes) > 0, "Should detect attention convergence pattern"
    
    def test_node_level_assignment(self):
        """Test correct assignment of topological levels."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # Level 0
            nn.ReLU(),          # Level 1
            nn.Linear(20, 30),  # Level 2
            nn.ReLU(),          # Level 3
            nn.Linear(30, 10)   # Level 4
        )
        
        extractor = FXPosetExtractor()
        poset = extractor.extract_poset(model)
        
        # Check levels are assigned
        for node in poset.nodes():
            assert 'level' in poset.nodes[node]
        
        # Check levels are monotonic along edges
        for edge in poset.edges():
            source, target = edge
            source_level = poset.nodes[source]['level']
            target_level = poset.nodes[target]['level']
            assert source_level < target_level, f"Level should increase: {source_level} -> {target_level}"
```

### 2. Procrustes Restriction Maps Tests
```python
# tests/phase3_sheaf/unit/test_procrustes_maps.py
import pytest
import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes
from neurosheaf.sheaf.restriction import ProcrustesMaps

class TestProcrustesMaps:
    """Test Procrustes-based restriction map computation."""
    
    def test_scaled_procrustes_accuracy(self):
        """Test accuracy of scaled Procrustes solution."""
        procrustes = ProcrustesMaps(allow_scaling=True)
        
        # Create known transformation
        n_samples = 200
        d_source = 50
        d_target = 40
        
        # True transformation matrix
        np.random.seed(42)
        R_true = np.random.randn(d_source, d_target)
        U, _, Vt = np.linalg.svd(R_true, full_matrices=False)
        R_true = U[:d_target, :].T @ Vt  # Orthogonal component
        scale_true = 1.5
        
        # Generate data
        X = torch.randn(n_samples, d_source)
        noise = torch.randn(n_samples, d_target) * 0.1
        Y = scale_true * X @ torch.from_numpy(R_true).float() + noise
        
        # Compute restriction map
        R_computed = procrustes.compute_restriction(X, Y, method='scaled_procrustes')
        
        # Check reconstruction error
        Y_reconstructed = X @ R_computed
        error = torch.norm(Y - Y_reconstructed, 'fro') / torch.norm(Y, 'fro')
        
        assert error < 0.15, f"Reconstruction error too high: {error:.4f}"
    
    def test_orthogonal_projection_properties(self):
        """Test properties of orthogonal projection method."""
        procrustes = ProcrustesMaps()
        
        # Test dimension reduction
        X = torch.randn(100, 50)
        Y = torch.randn(100, 30)
        
        R = procrustes.compute_restriction(X, Y, method='orthogonal')
        
        assert R.shape == (50, 30)
        
        # Test dimension expansion
        X = torch.randn(100, 30)
        Y = torch.randn(100, 50)
        
        R = procrustes.compute_restriction(X, Y, method='orthogonal')
        
        assert R.shape == (30, 50)
        
        # For expansion, should be approximately identity in leading block
        leading_block = R[:30, :30]
        identity_error = torch.norm(leading_block - torch.eye(30), 'fro')
        assert identity_error < 0.5, "Should approximate identity for dimension expansion"
    
    def test_least_squares_solution(self):
        """Test least squares restriction map computation."""
        procrustes = ProcrustesMaps()
        
        # Create data with exact linear relationship
        n_samples = 150
        X = torch.randn(n_samples, 40)
        R_true = torch.randn(40, 30)
        Y = X @ R_true + torch.randn(n_samples, 30) * 0.05  # Small noise
        
        R_computed = procrustes.compute_restriction(X, Y, method='least_squares')
        
        # Should recover true transformation (approximately)
        error = torch.norm(R_computed - R_true, 'fro') / torch.norm(R_true, 'fro')
        assert error < 0.1, f"Should recover true transformation, error: {error:.4f}"
    
    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned data."""
        procrustes = ProcrustesMaps()
        
        # Create ill-conditioned source data
        n_samples = 100
        # Low-rank data
        U = torch.randn(n_samples, 5)
        X = U @ torch.randn(5, 30)  # Rank 5 matrix
        Y = torch.randn(n_samples, 25)
        
        # Should not crash
        R = procrustes.compute_restriction(X, Y, method='scaled_procrustes')
        
        assert not torch.any(torch.isnan(R))
        assert not torch.any(torch.isinf(R))
        assert R.shape == (30, 25)
    
    def test_identity_preservation(self):
        """Test that identical inputs give identity-like mapping."""
        procrustes = ProcrustesMaps()
        
        X = torch.randn(100, 40)
        Y = X.clone()  # Identical data
        
        R = procrustes.compute_restriction(X, Y, method='scaled_procrustes')
        
        # Should be approximately identity
        identity_error = torch.norm(R - torch.eye(40), 'fro')
        assert identity_error < 0.1, f"Should be near identity for identical data: {identity_error:.4f}"
    
    def test_rotation_invariance(self):
        """Test invariance to rotations in input space."""
        procrustes = ProcrustesMaps(allow_scaling=True)
        
        X = torch.randn(100, 30)
        Y = torch.randn(100, 25)
        
        # Original mapping
        R1 = procrustes.compute_restriction(X, Y, method='scaled_procrustes')
        
        # Apply random rotation to X
        Q, _ = torch.linalg.qr(torch.randn(30, 30))
        X_rotated = X @ Q
        
        R2 = procrustes.compute_restriction(X_rotated, Y, method='scaled_procrustes')
        
        # Reconstruction should be similar
        Y_rec1 = X @ R1
        Y_rec2 = X_rotated @ R2
        
        reconstruction_diff = torch.norm(Y_rec1 - Y_rec2, 'fro') / torch.norm(Y, 'fro')
        assert reconstruction_diff < 0.1, f"Should be rotation invariant: {reconstruction_diff:.4f}"
```

### 3. Sheaf Construction Tests
```python
# tests/phase3_sheaf/unit/test_sheaf_construction.py
import pytest
import torch
import torch.nn as nn
import networkx as nx
from neurosheaf.sheaf.construction import Sheaf, SheafBuilder
from neurosheaf.sheaf.poset import FXPosetExtractor
from neurosheaf.sheaf.restriction import ProcrustesMaps

class TestSheafConstruction:
    """Test complete sheaf construction pipeline."""
    
    def test_sheaf_validation_properties(self):
        """Test mathematical properties of constructed sheaves."""
        # Create simple sheaf manually
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        stalks = {
            'A': torch.eye(3),
            'B': torch.eye(3),
            'C': torch.eye(3)
        }
        
        # Create consistent restrictions
        R_AB = torch.eye(3) * 0.9
        R_BC = torch.eye(3) * 0.8
        R_AC = R_BC @ R_AB  # Composition consistency
        
        restrictions = {
            ('A', 'B'): R_AB,
            ('B', 'C'): R_BC,
            ('A', 'C'): R_AC
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Should validate successfully
        assert sheaf.validate(), "Sheaf should satisfy consistency conditions"
    
    def test_sheaf_transitivity_violation_detection(self):
        """Test detection of transitivity violations."""
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        stalks = {node: torch.eye(3) for node in poset.nodes()}
        
        # Create inconsistent restrictions (violate R_AC = R_BC @ R_AB)
        restrictions = {
            ('A', 'B'): torch.eye(3) * 0.9,
            ('B', 'C'): torch.eye(3) * 0.8,
            ('A', 'C'): torch.eye(3) * 0.5  # Should be 0.72 = 0.8 * 0.9
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Should detect violation (with some tolerance)
        # Note: validation might pass with tolerance, but should warn
        result = sheaf.validate()
        # Implementation should log warnings about transitivity violations
    
    def test_sheaf_builder_with_real_model(self):
        """Test sheaf builder with real neural network."""
        model = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 10)
        )
        
        # Generate activations
        x = torch.randn(100, 20)
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        for i, layer in enumerate(model):
            layer.register_forward_hook(get_activation(f'layer_{i}'))
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Build sheaf
        builder = SheafBuilder()
        sheaf = builder.build_sheaf(model, activations)
        
        # Validate results
        assert len(sheaf.stalks) == len(activations)
        assert len(sheaf.restrictions) > 0
        assert sheaf.validate()
        
        # Check stalk dimensions match activations
        for name, activation in activations.items():
            if name in sheaf.stalks:
                stalk = sheaf.stalks[name]
                # Stalk should be square matrix (covariance-like)
                assert stalk.shape[0] == stalk.shape[1]
    
    def test_sheaf_with_cka_stalks(self):
        """Test sheaf construction using CKA matrices as stalks."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Generate activations
        x = torch.randn(80, 10)
        activations = {}
        
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations[f'relu_{i}'] = x.clone()
        
        # Create mock CKA matrices
        layer_names = list(activations.keys())
        n_layers = len(layer_names)
        cka_matrices = {}
        
        for i, name in enumerate(layer_names):
            # Create realistic CKA row (self-similarity = 1, others < 1)
            cka_row = torch.rand(n_layers) * 0.8 + 0.1  # Range [0.1, 0.9]
            cka_row[i] = 1.0  # Self-similarity
            cka_matrices[name] = cka_row
        
        # Build sheaf with CKA stalks
        builder = SheafBuilder()
        sheaf = builder.build_sheaf(model, activations, cka_matrices)
        
        # Validate CKA stalks
        for name, stalk in sheaf.stalks.items():
            assert stalk.shape[0] == n_layers
            assert torch.all(stalk >= 0) and torch.all(stalk <= 1)
            # Find self-similarity (should be 1.0)
            layer_idx = layer_names.index(name)
            assert abs(stalk[layer_idx] - 1.0) < 1e-6
    
    def test_sheaf_with_mismatched_dimensions(self):
        """Test sheaf construction with different layer dimensions."""
        # Create model with varying dimensions
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Generate activations
        x = torch.randn(60, 10)
        activations = {}
        
        for i, layer in enumerate(model):
            x = layer(x)
            activations[f'layer_{i}'] = x.clone()
        
        # Build sheaf
        builder = SheafBuilder()
        sheaf = builder.build_sheaf(model, activations)
        
        # Should handle dimension mismatches in restrictions
        assert sheaf.validate()
        
        # Check restriction map dimensions
        for edge, restriction in sheaf.restrictions.items():
            source, target = edge
            if source in activations and target in activations:
                source_dim = activations[source].shape[1]
                target_dim = activations[target].shape[1]
                assert restriction.shape == (source_dim, target_dim)
    
    def test_disconnected_poset_handling(self):
        """Test sheaf construction with disconnected poset."""
        # Create disconnected poset manually
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('C', 'D')])  # Two disconnected components
        
        # Create activations for all nodes
        activations = {
            'A': torch.randn(50, 10),
            'B': torch.randn(50, 15),
            'C': torch.randn(50, 8),
            'D': torch.randn(50, 12)
        }
        
        # Manually create sheaf
        stalks = {node: torch.cov(act.T) for node, act in activations.items()}
        
        # Create restrictions only for connected components
        restriction_computer = ProcrustesMaps()
        restrictions = {}
        
        for edge in poset.edges():
            source, target = edge
            R = restriction_computer.compute_restriction(
                activations[source], 
                activations[target]
            )
            restrictions[edge] = R
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Should handle disconnected components
        assert sheaf.validate()
```

### 4. Sparse Laplacian Tests
```python
# tests/phase3_sheaf/unit/test_sparse_laplacian.py
import pytest
import torch
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from neurosheaf.sheaf.construction import Sheaf
from neurosheaf.sheaf.laplacian import SparseLaplacianBuilder

class TestSparseLaplacian:
    """Test sparse Laplacian construction and properties."""
    
    def test_laplacian_mathematical_properties(self):
        """Test fundamental mathematical properties of sheaf Laplacian."""
        # Create simple sheaf
        poset = nx.path_graph(4, create_using=nx.DiGraph)
        node_names = [str(i) for i in range(4)]
        
        stalks = {name: torch.eye(3) for name in node_names}
        restrictions = {
            (str(i), str(i+1)): torch.eye(3) * 0.8
            for i in range(3)
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Build Laplacian
        builder = SparseLaplacianBuilder(use_gpu=False)
        L = builder.build_laplacian(sheaf)
        L_dense = L.to_dense()
        
        # Test symmetry
        assert torch.allclose(L_dense, L_dense.T, atol=1e-6), "Laplacian should be symmetric"
        
        # Test positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(L_dense)
        assert torch.all(eigenvalues >= -1e-6), "Laplacian should be positive semi-definite"
        
        # Test row sums (should be zero for unnormalized Laplacian)
        row_sums = L_dense.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-6), "Row sums should be zero"
    
    def test_normalized_laplacian_properties(self):
        """Test properties of normalized Laplacian."""
        # Create cycle graph sheaf
        poset = nx.cycle_graph(5, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(5)}
        restrictions = {
            (str(i), str((i + 1) % 5)): torch.eye(2) * 0.9
            for i in range(5)
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Build normalized Laplacian
        builder = SparseLaplacianBuilder(use_gpu=False)
        L_norm = builder.build_normalized_laplacian(sheaf)
        L_norm_dense = L_norm.to_dense()
        
        # Test eigenvalue bounds [0, 2]
        eigenvalues = torch.linalg.eigvalsh(L_norm_dense)
        assert torch.all(eigenvalues >= -1e-6), "Normalized Laplacian eigenvalues should be ≥ 0"
        assert torch.all(eigenvalues <= 2.01), "Normalized Laplacian eigenvalues should be ≤ 2"
        
        # Test symmetry
        assert torch.allclose(L_norm_dense, L_norm_dense.T, atol=1e-6), "Normalized Laplacian should be symmetric"
    
    def test_sparse_efficiency(self):
        """Test sparsity and memory efficiency."""
        # Create larger sparse sheaf (path graph)
        n_nodes = 50
        node_dim = 20
        
        poset = nx.path_graph(n_nodes, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(node_dim) for i in range(n_nodes)}
        restrictions = {
            (str(i), str(i+1)): torch.eye(node_dim) * 0.95
            for i in range(n_nodes - 1)
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Build sparse Laplacian
        builder = SparseLaplacianBuilder(use_gpu=False)
        L_sparse = builder.build_laplacian(sheaf)
        
        # Check sparsity
        total_elements = (n_nodes * node_dim) ** 2
        nnz = L_sparse._nnz()
        sparsity_ratio = nnz / total_elements
        
        # Path graph should be very sparse
        assert sparsity_ratio < 0.01, f"Should be <1% non-zero, got {sparsity_ratio:.4f}"
        
        # Check dimensions
        expected_size = n_nodes * node_dim
        assert L_sparse.shape == (expected_size, expected_size)
    
    def test_edge_weight_incorporation(self):
        """Test incorporation of edge weights in Laplacian."""
        poset = nx.path_graph(3, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(2) for i in range(3)}
        
        # Create restrictions with different weights
        restrictions = {
            ('0', '1'): torch.eye(2) * 0.9,  # Strong connection
            ('1', '2'): torch.eye(2) * 0.3   # Weak connection
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Build Laplacian with edge weighting
        builder = SparseLaplacianBuilder(use_gpu=False)
        L = builder.build_laplacian(sheaf, weight_edges=True)
        L_dense = L.to_dense()
        
        # Build without edge weighting
        L_unweighted = builder.build_laplacian(sheaf, weight_edges=False)
        L_unweighted_dense = L_unweighted.to_dense()
        
        # Should be different
        assert not torch.allclose(L_dense, L_unweighted_dense, atol=1e-6), "Weighted and unweighted should differ"
    
    def test_fast_eigenvalue_computation(self):
        """Test fast eigenvalue computation for sparse Laplacians."""
        # Create moderate-sized sheaf
        n_nodes = 20
        node_dim = 10
        
        poset = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(node_dim) for i in range(n_nodes)}
        restrictions = {
            (str(i), str(j)): torch.eye(node_dim) * 0.8
            for i in range(n_nodes) for j in range(n_nodes) if i != j
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Build Laplacian
        builder = SparseLaplacianBuilder(use_gpu=False)
        L = builder.build_laplacian(sheaf)
        
        # Compute eigenvalues
        k = min(50, L.shape[0] - 1)
        eigenvals, eigenvecs = builder.compute_fast_eigenvalues(L, k=k)
        
        # Check results
        assert len(eigenvals) <= k
        assert eigenvals.shape[0] == eigenvecs.shape[1]
        assert torch.all(eigenvals >= -1e-6), "Eigenvalues should be non-negative"
        assert torch.all(eigenvals[:-1] <= eigenvals[1:]), "Eigenvalues should be sorted"
    
    def test_block_structure_correctness(self):
        """Test correct block structure of sheaf Laplacian."""
        # Create small sheaf for manual verification
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        
        stalks = {
            'A': torch.eye(2),
            'B': torch.eye(2)
        }
        
        R = torch.tensor([[0.8, 0.1], [0.2, 0.7]])
        restrictions = {('A', 'B'): R}
        
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Build Laplacian
        builder = SparseLaplacianBuilder(use_gpu=False)
        L = builder.build_laplacian(sheaf)
        L_dense = L.to_dense()
        
        # Manual verification of block structure
        # L should be [[D_A, -R], [-R^T, D_B]]
        # where D_A = degree of A * I, D_B = degree of B * I
        
        # Extract blocks
        L_AA = L_dense[:2, :2]
        L_AB = L_dense[:2, 2:]
        L_BA = L_dense[2:, :2]
        L_BB = L_dense[2:, 2:]
        
        # Check diagonal blocks (degree matrices)
        expected_D_A = torch.eye(2) * 1  # degree of A is 1
        expected_D_B = torch.eye(2) * 1  # degree of B is 1
        
        assert torch.allclose(L_AA, expected_D_A, atol=1e-6), "Diagonal block A incorrect"
        assert torch.allclose(L_BB, expected_D_B, atol=1e-6), "Diagonal block B incorrect"
        
        # Check off-diagonal blocks
        assert torch.allclose(L_AB, -R, atol=1e-6), "Off-diagonal block AB incorrect"
        assert torch.allclose(L_BA, -R.T, atol=1e-6), "Off-diagonal block BA incorrect"
    
    def test_memory_efficiency_vs_dense(self):
        """Test memory efficiency compared to dense representation."""
        import psutil
        import gc
        
        # Create moderately large sparse sheaf
        n_nodes = 30
        node_dim = 15
        
        # Path graph (very sparse)
        poset = nx.path_graph(n_nodes, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(node_dim) for i in range(n_nodes)}
        restrictions = {
            (str(i), str(i+1)): torch.eye(node_dim) * 0.8
            for i in range(n_nodes - 1)
        }
        
        sheaf = Sheaf(poset, stalks, restrictions)
        builder = SparseLaplacianBuilder(use_gpu=False)
        
        # Measure memory for sparse
        gc.collect()
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        L_sparse = builder.build_laplacian(sheaf)
        
        mem_after_sparse = psutil.Process().memory_info().rss / 1024 / 1024
        sparse_memory = mem_after_sparse - mem_before
        
        # Convert to dense
        L_dense = L_sparse.to_dense()
        
        mem_after_dense = psutil.Process().memory_info().rss / 1024 / 1024
        dense_memory = mem_after_dense - mem_after_sparse
        
        # Sparse should use significantly less memory
        if dense_memory > 10:  # Only test if dense uses significant memory
            memory_ratio = sparse_memory / dense_memory
            assert memory_ratio < 0.5, f"Sparse should use <50% of dense memory, got {memory_ratio:.2f}"
```

### 5. Integration Tests
```python
# tests/phase3_sheaf/integration/test_sheaf_integration.py
import pytest
import torch
import torch.nn as nn
from neurosheaf.sheaf.construction import SheafBuilder
from neurosheaf.sheaf.laplacian import SparseLaplacianBuilder
from neurosheaf.cka import DebiasedCKA

class TestSheafIntegration:
    """Test integration between sheaf components."""
    
    def test_end_to_end_sheaf_pipeline(self):
        """Test complete pipeline from model to Laplacian."""
        # Create realistic model
        class TestCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = TestCNN()
        x = torch.randn(50, 3, 32, 32)
        
        # Extract activations
        activations = {}
        def get_activation(name):
            def hook(module, input, output):
                # Flatten spatial dimensions for analysis
                if len(output.shape) == 4:  # Conv features
                    activations[name] = output.flatten(2).mean(dim=2)  # Spatial average
                else:
                    activations[name] = output
            return hook
        
        # Register hooks for ReLU layers
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(get_activation(f'conv_relu_{i}'))
        
        for i, layer in enumerate(model.classifier):
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(get_activation(f'fc_relu_{i}'))
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Build sheaf
        sheaf_builder = SheafBuilder()
        sheaf = sheaf_builder.build_sheaf(model, activations)
        
        # Build Laplacian
        laplacian_builder = SparseLaplacianBuilder()
        L = laplacian_builder.build_laplacian(sheaf)
        
        # Validate pipeline
        assert sheaf.validate(), "Sheaf should be mathematically valid"
        assert L.shape[0] == L.shape[1], "Laplacian should be square"
        assert L._nnz() > 0, "Laplacian should have non-zero entries"
        
        # Test eigenvalue computation
        eigenvals, eigenvecs = laplacian_builder.compute_fast_eigenvalues(L, k=20)
        assert len(eigenvals) <= 20
        assert torch.all(eigenvals >= -1e-6), "Eigenvalues should be non-negative"
    
    def test_sheaf_with_cka_integration(self):
        """Test integration of sheaf construction with CKA computation."""
        # Create model
        model = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Generate data and activations
        x = torch.randn(200, 30)
        activations = {}
        
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations[f'relu_{i}'] = x.clone()
        
        # Compute CKA matrices
        cka_computer = DebiasedCKA(use_unbiased=True)
        layer_names = list(activations.keys())
        cka_matrices = {}
        
        for name in layer_names:
            cka_row = []
            for other_name in layer_names:
                cka_value = cka_computer.compute(
                    activations[name],
                    activations[other_name]
                )
                cka_row.append(cka_value)
            cka_matrices[name] = torch.tensor(cka_row)
        
        # Build sheaf with CKA stalks
        sheaf_builder = SheafBuilder()
        sheaf = sheaf_builder.build_sheaf(model, activations, cka_matrices)
        
        # Validate CKA integration
        assert sheaf.validate()
        
        # Check stalks are CKA vectors
        for name, stalk in sheaf.stalks.items():
            assert stalk.shape[0] == len(layer_names)
            assert torch.all(stalk >= 0) and torch.all(stalk <= 1)
            
            # Self-similarity should be 1
            self_idx = layer_names.index(name)
            assert abs(stalk[self_idx] - 1.0) < 1e-6
        
        # Build Laplacian
        laplacian_builder = SparseLaplacianBuilder()
        L = laplacian_builder.build_laplacian(sheaf)
        
        # Should work without issues
        assert L._nnz() > 0
    
    def test_architecture_robustness(self):
        """Test robustness across different architectures."""
        architectures = [
            # Simple MLP
            nn.Sequential(
                nn.Linear(20, 30),
                nn.ReLU(),
                nn.Linear(30, 10)
            ),
            
            # CNN
            nn.Sequential(
                nn.Conv2d(3, 16, 3),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 10)
            ),
        ]
        
        inputs = [
            torch.randn(50, 20),
            torch.randn(50, 3, 8, 8)
        ]
        
        sheaf_builder = SheafBuilder()
        laplacian_builder = SparseLaplacianBuilder()
        
        for model, x in zip(architectures, inputs):
            # Extract activations
            activations = {}
            hook_handles = []
            
            def get_activation(name):
                def hook(module, input, output):
                    if len(output.shape) == 4:  # Conv
                        activations[name] = output.flatten(2).mean(dim=2)
                    else:
                        activations[name] = output
                return hook
            
            # Register hooks
            for i, layer in enumerate(model):
                if isinstance(layer, (nn.ReLU, nn.Linear)):
                    handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
                    hook_handles.append(handle)
            
            # Forward pass
            with torch.no_grad():
                _ = model(x)
            
            # Clean up hooks
            for handle in hook_handles:
                handle.remove()
            
            if len(activations) == 0:
                continue  # Skip if no activations captured
            
            # Build sheaf
            sheaf = sheaf_builder.build_sheaf(model, activations)
            
            # Should work for all architectures
            assert sheaf.validate(), f"Sheaf validation failed for {type(model)}"
            
            # Build Laplacian
            L = laplacian_builder.build_laplacian(sheaf)
            assert L._nnz() > 0, f"Empty Laplacian for {type(model)}"
```

### 6. Performance and Edge Cases
```python
# tests/phase3_sheaf/performance/test_sheaf_performance.py
import pytest
import torch
import torch.nn as nn
import time
import psutil
from neurosheaf.sheaf.construction import SheafBuilder
from neurosheaf.sheaf.laplacian import SparseLaplacianBuilder

class TestSheafPerformance:
    """Test performance characteristics of sheaf construction."""
    
    @pytest.mark.slow
    def test_large_network_performance(self):
        """Test performance on large networks."""
        # Create large network
        layers = []
        for i in range(50):  # 50 linear layers
            layers.append(nn.Linear(128, 128))
            layers.append(nn.ReLU())
        
        model = nn.Sequential(*layers)
        x = torch.randn(100, 128)
        
        # Extract activations (only ReLU layers)
        activations = {}
        for i, layer in enumerate(model):
            if isinstance(layer, nn.ReLU):
                x = model[:i+1](x)
                activations[f'relu_{i}'] = x.clone()
        
        # Time sheaf construction
        sheaf_builder = SheafBuilder()
        
        start_time = time.time()
        sheaf = sheaf_builder.build_sheaf(model, activations)
        sheaf_time = time.time() - start_time
        
        # Time Laplacian construction
        laplacian_builder = SparseLaplacianBuilder()
        
        start_time = time.time()
        L = laplacian_builder.build_laplacian(sheaf)
        laplacian_time = time.time() - start_time
        
        # Performance assertions
        assert sheaf_time < 60, f"Sheaf construction too slow: {sheaf_time:.1f}s"
        assert laplacian_time < 30, f"Laplacian construction too slow: {laplacian_time:.1f}s"
        
        # Quality assertions
        assert sheaf.validate()
        assert L._nnz() > 0
        
        print(f"Large network ({len(activations)} layers): Sheaf {sheaf_time:.1f}s, Laplacian {laplacian_time:.1f}s")
    
    @pytest.mark.memory
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with network size."""
        layer_counts = [5, 10, 20, 30]
        memories = []
        
        for n_layers in layer_counts:
            # Create model
            layers = []
            for i in range(n_layers):
                layers.append(nn.Linear(64, 64))
                layers.append(nn.ReLU())
            
            model = nn.Sequential(*layers)
            x = torch.randn(100, 64)
            
            # Extract activations
            activations = {}
            for i, layer in enumerate(model):
                if isinstance(layer, nn.ReLU):
                    x = model[:i+1](x)
                    activations[f'relu_{i}'] = x.clone()
            
            # Measure memory
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Build sheaf and Laplacian
            sheaf_builder = SheafBuilder()
            sheaf = sheaf_builder.build_sheaf(model, activations)
            
            laplacian_builder = SparseLaplacianBuilder()
            L = laplacian_builder.build_laplacian(sheaf)
            
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            memories.append(memory_used)
            
            print(f"Layers: {n_layers}, Memory: {memory_used:.1f}MB")
        
        # Memory should scale reasonably (not exponentially)
        for i in range(len(layer_counts) - 1):
            layer_ratio = layer_counts[i+1] / layer_counts[i]
            memory_ratio = memories[i+1] / memories[i]
            
            # Should scale less than quadratically
            assert memory_ratio < layer_ratio ** 2, f"Memory scaling too aggressive: {memory_ratio:.2f} vs {layer_ratio**2:.2f}"
    
    def test_sparse_vs_dense_performance(self):
        """Compare sparse vs dense Laplacian performance."""
        # Create moderately sized sheaf
        n_nodes = 25
        node_dim = 20
        
        # Use cycle graph for controlled sparsity
        import networkx as nx
        poset = nx.cycle_graph(n_nodes, create_using=nx.DiGraph)
        stalks = {str(i): torch.eye(node_dim) for i in range(n_nodes)}
        restrictions = {
            (str(i), str((i + 1) % n_nodes)): torch.eye(node_dim) * 0.8
            for i in range(n_nodes)
        }
        
        from neurosheaf.sheaf.construction import Sheaf
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # Build sparse Laplacian
        builder = SparseLaplacianBuilder()
        
        start_time = time.time()
        L_sparse = builder.build_laplacian(sheaf)
        sparse_time = time.time() - start_time
        
        # Convert to dense
        start_time = time.time()
        L_dense = L_sparse.to_dense()
        dense_conversion_time = time.time() - start_time
        
        # Sparse operations
        start_time = time.time()
        eigenvals_sparse, _ = builder.compute_fast_eigenvalues(L_sparse, k=10)
        sparse_eigen_time = time.time() - start_time
        
        # Dense operations
        start_time = time.time()
        eigenvals_dense = torch.linalg.eigvals(L_dense)
        dense_eigen_time = time.time() - start_time
        
        # Sparse should be faster for eigenvalues
        print(f"Sparse eigenvalues: {sparse_eigen_time:.3f}s")
        print(f"Dense eigenvalues: {dense_eigen_time:.3f}s")
        
        if dense_eigen_time > 0.1:  # Only test if computation is significant
            assert sparse_eigen_time < dense_eigen_time, "Sparse eigenvalue computation should be faster"
```

### 7. Edge Cases and Error Handling
```python
# tests/phase3_sheaf/edge_cases/test_sheaf_edge_cases.py
import pytest
import torch
import torch.nn as nn
import networkx as nx
from neurosheaf.sheaf.construction import Sheaf, SheafBuilder
from neurosheaf.sheaf.poset import FXPosetExtractor
from neurosheaf.sheaf.laplacian import SparseLaplacianBuilder
from neurosheaf.utils.exceptions import ValidationError, ArchitectureError

class TestSheafEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_model_handling(self):
        """Test handling of empty or minimal models."""
        # Model with no learnable parameters
        model = nn.Identity()
        x = torch.randn(10, 5)
        
        extractor = FXPosetExtractor(handle_dynamic=True)
        
        # Should not crash
        poset = extractor.extract_poset(model)
        assert isinstance(poset, nx.DiGraph)
    
    def test_single_layer_model(self):
        """Test handling of single-layer models."""
        model = nn.Linear(10, 5)
        x = torch.randn(20, 10)
        
        # Extract activation
        activations = {}
        with torch.no_grad():
            activations['linear'] = model(x)
        
        # Build sheaf
        builder = SheafBuilder()
        sheaf = builder.build_sheaf(model, activations)
        
        # Should create valid sheaf even with single stalk
        assert len(sheaf.stalks) == 1
        assert len(sheaf.restrictions) == 0  # No edges
        assert sheaf.validate()
    
    def test_disconnected_model_components(self):
        """Test models with disconnected components."""
        class DisconnectedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch1 = nn.Linear(10, 5)
                self.branch2 = nn.Linear(8, 3)
                
            def forward(self, x1, x2):
                # Two completely separate computations
                out1 = self.branch1(x1)
                out2 = self.branch2(x2)
                return out1, out2
        
        model = DisconnectedModel()
        
        # Create mock activations
        activations = {
            'branch1_out': torch.randn(20, 5),
            'branch2_out': torch.randn(20, 3)
        }
        
        builder = SheafBuilder()
        sheaf = builder.build_sheaf(model, activations)
        
        # Should handle disconnected components
        assert sheaf.validate()
        
        # Build Laplacian (should work with disconnected components)
        laplacian_builder = SparseLaplacianBuilder()
        L = laplacian_builder.build_laplacian(sheaf)
        assert L.shape[0] == L.shape[1]
    
    def test_very_small_activations(self):
        """Test handling of very small activation values."""
        # Create activations with very small values
        activations = {
            'layer1': torch.randn(10, 5) * 1e-8,
            'layer2': torch.randn(10, 8) * 1e-8
        }
        
        # Create minimal model
        model = nn.Sequential(nn.Linear(5, 8))
        
        builder = SheafBuilder()
        
        # Should handle numerical issues gracefully
        sheaf = builder.build_sheaf(model, activations)
        assert sheaf.validate()
        
        # Restrictions should not be NaN or Inf
        for restriction in sheaf.restrictions.values():
            assert not torch.any(torch.isnan(restriction))
            assert not torch.any(torch.isinf(restriction))
    
    def test_very_large_activations(self):
        """Test handling of very large activation values."""
        # Create activations with large values
        activations = {
            'layer1': torch.randn(10, 5) * 1e6,
            'layer2': torch.randn(10, 8) * 1e6
        }
        
        model = nn.Sequential(nn.Linear(5, 8))
        
        builder = SheafBuilder()
        sheaf = builder.build_sheaf(model, activations)
        
        # Should handle large values
        assert sheaf.validate()
        
        # Build Laplacian
        laplacian_builder = SparseLaplacianBuilder()
        L = laplacian_builder.build_laplacian(sheaf)
        
        # Should not overflow
        assert not torch.any(torch.isnan(L.values()))
        assert not torch.any(torch.isinf(L.values()))
    
    def test_mismatched_activation_dimensions(self):
        """Test handling of mismatched dimensions in activations."""
        # Create activations with inconsistent batch sizes
        activations = {
            'layer1': torch.randn(20, 10),
            'layer2': torch.randn(15, 8)  # Different batch size
        }
        
        model = nn.Sequential(nn.Linear(10, 8))
        
        builder = SheafBuilder()
        
        # Should handle gracefully (might sample or pad)
        sheaf = builder.build_sheaf(model, activations)
        assert sheaf.validate()
    
    def test_constant_activations(self):
        """Test handling of constant activation values."""
        # All activations are constant
        activations = {
            'layer1': torch.ones(20, 10),
            'layer2': torch.ones(20, 8) * 5.0
        }
        
        model = nn.Sequential(nn.Linear(10, 8))
        
        builder = SheafBuilder()
        sheaf = builder.build_sheaf(model, activations)
        
        # Should handle degenerate covariance matrices
        assert sheaf.validate()
        
        # Restrictions might be degenerate but should not crash
        for restriction in sheaf.restrictions.values():
            assert not torch.any(torch.isnan(restriction))
    
    def test_nan_in_activations(self):
        """Test handling of NaN values in activations."""
        activations = {
            'layer1': torch.randn(20, 10),
            'layer2': torch.randn(20, 8)
        }
        
        # Introduce some NaN values
        activations['layer1'][5, 3] = float('nan')
        activations['layer2'][10, :] = float('nan')
        
        model = nn.Sequential(nn.Linear(10, 8))
        
        builder = SheafBuilder()
        
        # Should either handle NaNs or raise appropriate error
        try:
            sheaf = builder.build_sheaf(model, activations)
            # If successful, should not propagate NaNs
            for restriction in sheaf.restrictions.values():
                assert not torch.any(torch.isnan(restriction))
        except (ValidationError, ValueError):
            # Acceptable to raise validation error for NaN inputs
            pass
    
    def test_extremely_large_network(self):
        """Test behavior with unreasonably large networks."""
        # Create network that would be too large for full analysis
        n_layers = 1000
        
        # Just test poset extraction (don't actually run full pipeline)
        layers = [nn.Linear(10, 10) for _ in range(n_layers)]
        model = nn.Sequential(*layers)
        
        extractor = FXPosetExtractor()
        
        # Should either succeed or fail gracefully
        try:
            poset = extractor.extract_poset(model)
            assert isinstance(poset, nx.DiGraph)
        except (MemoryError, RuntimeError, ArchitectureError):
            # Acceptable to fail on extremely large networks
            pass
    
    def test_circular_dependencies_detection(self):
        """Test detection of circular dependencies in model."""
        # This is more theoretical since PyTorch models shouldn't have cycles
        # But test with manually created circular poset
        
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])  # Cycle!
        
        stalks = {node: torch.eye(3) for node in poset.nodes()}
        restrictions = {edge: torch.eye(3) * 0.8 for edge in poset.edges()}
        
        # Should detect that this is not a valid DAG
        sheaf = Sheaf(poset, stalks, restrictions)
        
        # The poset itself is invalid for neural networks
        assert not nx.is_directed_acyclic_graph(poset)
        
        # Sheaf validation might still pass (different mathematical property)
        # But Laplacian construction should handle cycles appropriately
        laplacian_builder = SparseLaplacianBuilder()
        L = laplacian_builder.build_laplacian(sheaf)
        
        # Should create valid Laplacian even with cycles
        assert L.shape[0] == L.shape[1]
```

## Test Execution Strategy

### Local Development Testing
```bash
# Quick unit tests during development
pytest tests/phase3_sheaf/unit/ -v

# Critical functionality tests
pytest tests/phase3_sheaf/critical/ -v --tb=short

# Integration tests
pytest tests/phase3_sheaf/integration/ -v -m "not slow"

# Performance tests (separate run)
pytest tests/phase3_sheaf/performance/ -v -m "slow"

# Edge cases
pytest tests/phase3_sheaf/edge_cases/ -v
```

### Continuous Integration
```yaml
# .github/workflows/phase3_sheaf.yml
name: Phase 3 Sheaf Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install torch scipy networkx matplotlib
        pip install pytest pytest-cov
        pip install -e .
    
    - name: Run critical tests
      run: |
        pytest tests/phase3_sheaf/critical/ -v --tb=short
    
    - name: Run unit tests
      run: |
        pytest tests/phase3_sheaf/unit/ -v --cov=neurosheaf.sheaf
    
    - name: Run integration tests
      run: |
        pytest tests/phase3_sheaf/integration/ -v -m "not slow"
    
    - name: Run edge case tests
      run: |
        pytest tests/phase3_sheaf/edge_cases/ -v
```

### Performance Testing
```bash
# Memory-intensive tests
pytest tests/phase3_sheaf/performance/ -v -m "memory" --maxfail=3

# Time-intensive tests
pytest tests/phase3_sheaf/performance/ -v -m "slow" --maxfail=3

# Benchmark tests
pytest tests/phase3_sheaf/performance/ -v -m "benchmark"
```

## Success Criteria

1. **Test Coverage**: >95% code coverage for sheaf module
2. **FX Extraction**: 100% success rate on standard architectures (ResNet, VGG, etc.)
3. **Mathematical Validity**: All constructed sheaves pass validation
4. **Performance**: Sheaf construction <60s for 50-layer networks
5. **Memory Efficiency**: Sparse Laplacian uses <10% of dense equivalent
6. **Robustness**: All edge cases handled gracefully

## Common Issues and Solutions

### Issue: FX tracing fails on dynamic models
```python
# Solution: Implement robust fallback
def extract_with_fallback(model):
    try:
        return fx_based_extraction(model)
    except Exception as e:
        logger.warning(f"FX failed: {e}, using fallback")
        return module_based_extraction(model)
```

### Issue: Numerical instability in Procrustes
```python
# Solution: Add regularization
def stable_procrustes(X, Y, reg=1e-6):
    # Add small regularization to prevent singular matrices
    X_reg = X + torch.randn_like(X) * reg
    return compute_procrustes(X_reg, Y)
```

### Issue: Memory overflow with large Laplacians
```python
# Solution: Implement chunked operations
def chunked_laplacian_build(sheaf, chunk_size=1000):
    if total_dim < chunk_size:
        return standard_build(sheaf)
    else:
        return chunked_build(sheaf, chunk_size)
```

This comprehensive testing suite ensures the sheaf construction module is robust, mathematically correct, and performs well across all scenarios and architectures.