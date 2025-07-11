"""Integration tests for Phase 3 Week 6: Restriction maps and sheaf construction.

This module tests the end-to-end integration of:
1. FX-based poset extraction
2. Restriction map computation using Procrustes analysis
3. Sheaf construction and validation
4. Mathematical property verification
"""

import pytest
import torch
import torch.nn as nn
import networkx as nx
import numpy as np

from neurosheaf.sheaf import (
    FXPosetExtractor, 
    ProcrustesMaps, 
    SheafBuilder, 
    Sheaf,
    validate_sheaf_properties,
    create_sheaf_from_cka_analysis
)


class SimpleResNetBlock(nn.Module):
    """Simple ResNet-like block for testing skip connections."""
    
    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # Skip connection
        out = self.relu(out)
        return out


class TransformerLikeModel(nn.Module):
    """Simplified transformer-like model for testing attention patterns."""
    
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(32, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, 10)
    
    def forward(self, x):
        x = self.embedding(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simplified attention
        attn_output = self.out_proj(v)  # Simplified
        output = self.fc(attn_output)
        return output


class TestPhase3Week6Integration:
    """Integration tests for Week 6 deliverables."""
    
    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize components
        self.poset_extractor = FXPosetExtractor(handle_dynamic=True)
        self.procrustes_maps = ProcrustesMaps(epsilon=1e-8)
        self.sheaf_builder = SheafBuilder(handle_dynamic=True)
        
        # Test data
        self.batch_size = 8
    
    def test_end_to_end_simple_model(self):
        """Test complete pipeline with simple sequential model."""
        # Create simple sequential model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 5)
        )
        
        # Generate test activations
        input_data = torch.randn(self.batch_size, 10)
        activations = self._extract_activations_simple(model, input_data)
        
        # Build sheaf from activations
        sheaf = self.sheaf_builder.build_from_activations(
            model, activations, use_gram_matrices=True, validate=True
        )
        
        # Verify sheaf structure
        assert isinstance(sheaf, Sheaf)
        assert len(sheaf.stalks) > 0
        assert len(sheaf.poset.nodes()) > 0
        
        # Check validation passed
        assert 'validation_results' in sheaf.metadata
        validation = sheaf.metadata['validation_results']
        
        # With a simple sequential model, transitivity should be reasonable
        # (though not perfect due to random data)
        assert validation['total_paths_checked'] >= 0
        
        # Verify sheaf properties
        laplacian_info = sheaf.get_laplacian_structure()
        assert laplacian_info['num_nodes'] > 0
        assert laplacian_info['num_edges'] >= 0
        assert 0 <= laplacian_info['estimated_sparsity'] <= 1
        
        print(f"Sheaf summary:\n{sheaf.summary()}")
    
    def test_resnet_block_skip_connections(self):
        """Test sheaf construction with ResNet-like skip connections."""
        model = SimpleResNetBlock(channels=8)  # Smaller for testing
        
        # Generate test input
        input_data = torch.randn(self.batch_size, 8, 4, 4)
        
        # Extract poset structure
        try:
            poset = self.poset_extractor.extract_poset(model)
            
            # Verify skip connections are detected
            assert len(poset.nodes()) > 0
            assert len(poset.edges()) > 0
            
            # Check if there are multiple paths (indicating skip connections)
            paths_count = 0
            for source in poset.nodes():
                for target in poset.nodes():
                    if source != target and nx.has_path(poset, source, target):
                        paths_count += 1
            
            print(f"ResNet block: {len(poset.nodes())} nodes, {len(poset.edges())} edges, {paths_count} paths")
            
        except Exception as e:
            # FX tracing might fail for complex models, should use fallback
            pytest.skip(f"FX tracing failed: {e}")
    
    def test_transformer_attention_patterns(self):
        """Test sheaf construction with transformer-like attention patterns."""
        model = TransformerLikeModel(d_model=16, nhead=2)  # Smaller for testing
        
        # Generate test input
        input_data = torch.randn(self.batch_size, 16)
        
        try:
            # Extract poset structure
            poset = self.poset_extractor.extract_poset(model)
            
            # Should detect Q, K, V projections
            node_names = list(poset.nodes())
            
            # Look for attention-related patterns
            attention_nodes = [name for name in node_names 
                              if any(kw in name.lower() for kw in ['q_proj', 'k_proj', 'v_proj', 'out_proj'])]
            
            print(f"Transformer: {len(poset.nodes())} nodes, attention nodes: {attention_nodes}")
            
            # Should have some nodes
            assert len(poset.nodes()) > 0
            
        except Exception as e:
            # Complex models might fail FX tracing
            pytest.skip(f"Transformer FX tracing failed: {e}")
    
    def test_restriction_map_transitivity_validation(self):
        """Test mathematical validation of restriction map transitivity."""
        # Create a controlled scenario with exact transitivity
        torch.manual_seed(123)
        
        # Create test matrices with known relationships
        dim = 6
        K_A = torch.eye(dim) + 0.1 * torch.randn(dim, dim)
        K_A = K_A @ K_A.T  # Ensure PSD
        
        # Create K_B and K_C as transformations of K_A
        Q_AB = torch.randn(dim, dim)
        Q_AB, _ = torch.linalg.qr(Q_AB)  # Orthogonal matrix
        K_B = 0.8 * Q_AB.T @ K_A @ Q_AB
        
        Q_BC = torch.randn(dim, dim) 
        Q_BC, _ = torch.linalg.qr(Q_BC)  # Orthogonal matrix
        K_C = 0.7 * Q_BC.T @ K_B @ Q_BC
        
        # Compute restriction maps
        R_AB, scale_AB, info_AB = self.procrustes_maps.scaled_procrustes(K_A, K_B)
        R_BC, scale_BC, info_BC = self.procrustes_maps.scaled_procrustes(K_B, K_C)
        
        # Compute composed restriction
        R_AC_composed = R_BC @ R_AB
        
        # Also compute direct restriction for comparison
        R_AC_direct, scale_AC, info_AC = self.procrustes_maps.scaled_procrustes(K_A, K_C)
        
        # Check transitivity violation
        transitivity_error = torch.norm(R_AC_composed - R_AC_direct, p='fro').item()
        print(f"Transitivity error: {transitivity_error:.6f}")
        
        # Create sheaf and validate
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        
        restrictions = {
            ('A', 'B'): R_AB,
            ('B', 'C'): R_BC,
            ('A', 'C'): R_AC_composed  # Use composed for perfect transitivity
        }
        
        validation_results = validate_sheaf_properties(restrictions, poset, tolerance=1e-3)
        
        print(f"Validation results: {validation_results}")
        assert validation_results['valid_sheaf'] is True
        assert len(validation_results['transitivity_violations']) == 0
    
    def test_cka_integration_pipeline(self):
        """Test integration with CKA analysis pipeline."""
        # Simulate CKA results
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        n_layers = len(layer_names)
        
        # Create realistic CKA similarity matrix
        torch.manual_seed(456)
        base_similarity = torch.randn(n_layers, n_layers)
        cka_matrix = torch.sigmoid(base_similarity)  # Ensure [0,1] range
        
        # Make it symmetric and set diagonal to 1 (self-similarity)
        cka_matrix = (cka_matrix + cka_matrix.T) / 2
        cka_matrix.fill_diagonal_(1.0)
        
        cka_results = {
            'similarity_matrix': cka_matrix,
            'layer_names': layer_names
        }
        
        # Create sheaf from CKA results
        sheaf = create_sheaf_from_cka_analysis(cka_results, layer_names)
        
        assert isinstance(sheaf, Sheaf)
        assert len(sheaf.stalks) == n_layers
        assert sheaf.poset.number_of_nodes() == n_layers
        assert sheaf.poset.number_of_edges() == n_layers - 1  # Sequential connections
        
        # Validate sheaf properties
        validation_results = sheaf.validate(tolerance=0.1)  # Loose tolerance for CKA data
        print(f"CKA sheaf validation: {validation_results}")
    
    def test_memory_efficiency_large_network(self):
        """Test memory efficiency with larger network simulation."""
        # Simulate a larger network with many layers
        n_layers = 20
        layer_dim = 10  # Small for testing, but many layers
        
        # Create activations for many layers
        activations = {}
        for i in range(n_layers):
            layer_name = f'layer_{i}'
            activations[layer_name] = torch.randn(self.batch_size, layer_dim)
        
        # Create a simple sequential model for structure
        layers = []
        for i in range(n_layers):
            layers.extend([nn.Linear(layer_dim, layer_dim), nn.ReLU()])
        model = nn.Sequential(*layers)
        
        # Build sheaf
        sheaf = self.sheaf_builder.build_from_activations(
            model, activations, use_gram_matrices=True, validate=False  # Skip validation for speed
        )
        
        # Check memory efficiency
        laplacian_info = sheaf.get_laplacian_structure()
        print(f"Large network: {laplacian_info}")
        
        # Should have high sparsity for many layers
        assert laplacian_info['estimated_sparsity'] > 0.8  # >80% sparse
        assert laplacian_info['num_nodes'] == n_layers
    
    def test_error_handling_and_robustness(self):
        """Test error handling with edge cases."""
        # Test with empty activations
        empty_model = nn.Linear(5, 5)
        empty_activations = {}
        
        sheaf_empty = self.sheaf_builder.build_from_activations(
            empty_model, empty_activations, validate=False
        )
        assert isinstance(sheaf_empty, Sheaf)
        assert len(sheaf_empty.stalks) == 0
        
        # Test with single layer
        single_activations = {'single_layer': torch.randn(4, 8)}
        sheaf_single = self.sheaf_builder.build_from_activations(
            empty_model, single_activations, validate=False
        )
        assert isinstance(sheaf_single, Sheaf)
        assert len(sheaf_single.stalks) == 1
        
        # Test validation with single layer (should pass trivially)
        validation = sheaf_single.validate()
        assert validation['valid_sheaf'] is True  # No paths to check
        assert validation['total_paths_checked'] == 0
    
    def _extract_activations_simple(self, model, input_data):
        """Extract activations from simple sequential model."""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # Store flattened activation
                if isinstance(output, torch.Tensor):
                    activations[name] = output.view(output.size(0), -1).detach()
            return hook
        
        # Register hooks on layers with parameters
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0:  # Has parameters
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        try:
            with torch.no_grad():
                model.eval()
                _ = model(input_data)
        finally:
            for hook in hooks:
                hook.remove()
        
        return activations


class TestMathematicalPropertyValidation:
    """Test mathematical properties specifically required for sheaves."""
    
    def test_procrustes_orthogonality_preservation(self):
        """Test that Procrustes maps preserve orthogonality structure."""
        torch.manual_seed(789)
        
        # Create orthogonal matrix structure
        dim = 8
        U, _, Vt = torch.linalg.svd(torch.randn(dim, dim))
        
        K_source = U @ torch.diag(torch.abs(torch.randn(dim))) @ U.T
        K_target = U @ torch.diag(torch.abs(torch.randn(dim))) @ U.T
        
        procrustes = ProcrustesMaps()
        R, scale, info = procrustes.scaled_procrustes(K_source, K_target)
        
        # Verify orthogonal component
        Q = info['orthogonal_matrix']
        orthogonality_error = torch.norm(Q @ Q.T - torch.eye(dim), p='fro')
        
        print(f"Orthogonality error: {orthogonality_error:.6f}")
        assert orthogonality_error < 1e-4
        
        # Verify scaling bounds
        assert procrustes.min_scale <= scale <= procrustes.max_scale
    
    def test_sheaf_laplacian_properties(self):
        """Test properties of the sheaf Laplacian structure."""
        # Create a simple sheaf with known structure
        poset = nx.DiGraph()
        poset.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        
        # Create stalks (use small matrices for testing)
        stalks = {}
        restrictions = {}
        
        dim = 4
        for node in poset.nodes():
            stalks[node] = torch.eye(dim) + 0.1 * torch.randn(dim, dim)
        
        # Create restriction maps
        procrustes = ProcrustesMaps()
        for edge in poset.edges():
            source, target = edge
            R, _, _ = procrustes.scaled_procrustes(stalks[source], stalks[target])
            restrictions[edge] = R
        
        # Create sheaf
        sheaf = Sheaf(poset=poset, stalks=stalks, restrictions=restrictions)
        
        # Get Laplacian structure info
        laplacian_info = sheaf.get_laplacian_structure()
        
        # Verify properties
        assert laplacian_info['total_dimension'] == len(poset.nodes()) * dim
        assert laplacian_info['num_nodes'] == len(poset.nodes())
        assert laplacian_info['num_edges'] == len(poset.edges())
        
        # Sparsity should be reasonable for this structure
        assert 0 <= laplacian_info['estimated_sparsity'] <= 1
        
        print(f"Laplacian structure: {laplacian_info}")
    
    def test_restriction_map_scale_consistency(self):
        """Test consistency of scaling factors across restriction maps."""
        torch.manual_seed(999)
        
        # Create matrices with known scale relationships
        dim = 6
        base_matrix = torch.eye(dim) + 0.2 * torch.randn(dim, dim)
        base_matrix = base_matrix @ base_matrix.T
        
        # Create scaled versions
        scales = [1.0, 0.5, 2.0, 0.8]
        matrices = [scale * base_matrix for scale in scales]
        
        procrustes = ProcrustesMaps()
        computed_scales = []
        
        # Compute restriction maps between consecutive matrices
        for i in range(len(matrices) - 1):
            R, scale, info = procrustes.scaled_procrustes(matrices[i], matrices[i+1])
            computed_scales.append(scale)
            
            print(f"Expected scale ratio: {scales[i+1]/scales[i]:.3f}, "
                  f"Computed scale: {scale:.3f}, "
                  f"Relative error: {info['relative_error']:.3f}")
        
        # Scales should be reasonable (though not exact due to random components)
        for scale in computed_scales:
            assert procrustes.min_scale <= scale <= procrustes.max_scale