"""Comprehensive mathematical validation tests for GW sheaf construction.

This module tests the mathematical correctness of the GW implementation,
focusing on theoretical properties that must hold for valid sheaf construction:
- Quasi-sheaf functoriality approximation
- Laplacian positive semi-definiteness
- Spectral stability and continuity
- Numerical conditioning

Test Categories:
1. Quasi-sheaf properties (ε-sheaf validation)
2. Laplacian mathematical properties
3. Spectral stability under perturbations
4. Numerical conditioning analysis
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from unittest.mock import patch

from neurosheaf.sheaf.assembly import SheafBuilder
from neurosheaf.sheaf.assembly.laplacian import SheafLaplacianBuilder
from neurosheaf.sheaf.core import GWConfig
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.utils.logging import setup_logger

logger = setup_logger(__name__)


class TestQuasiSheafProperties:
    """Validate ε-sheaf approximation quality and functoriality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.epsilon_values = [0.01, 0.05, 0.1, 0.2]
        self.tolerance_threshold = 0.1  # Maximum acceptable violation
        
    def _create_test_network(self, layer_sizes: List[int]) -> nn.Module:
        """Create a simple feedforward network for testing."""
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def _extract_activations(self, model: nn.Module, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract layer activations from model."""
        activations = {}
        x = data
        
        for i, layer in enumerate(model):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                layer_name = f"layer_{i}"
                activations[layer_name] = x.detach().clone()
                
        return activations
    
    def test_approximate_functoriality(self):
        """Test ||ρ_{k→i} - ρ_{j→i} ∘ ρ_{k→j}||_F ≤ η for all paths."""
        # Create a 3-layer network for testing transitivity
        model = self._create_test_network([10, 8, 6, 4])
        data = torch.randn(50, 10)
        
        for epsilon in self.epsilon_values:
            config = GWConfig(epsilon=epsilon, max_iter=1000)
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            
            # Extract activations and build sheaf
            activations = self._extract_activations(model, data)
            sheaf = builder.build_from_activations(
                activations, 
                model,
                gw_config=config
            )
            
            # Check all 3-node paths for functoriality
            violations = self._compute_functoriality_violations(sheaf)
            
            # Log results
            max_violation = max(violations) if violations else 0.0
            avg_violation = np.mean(violations) if violations else 0.0
            
            logger.info(f"Epsilon={epsilon}: max_violation={max_violation:.6f}, "
                       f"avg_violation={avg_violation:.6f}")
            
            # Assert violations are within tolerance
            assert max_violation <= self.tolerance_threshold, \
                f"Functoriality violation {max_violation:.6f} exceeds threshold {self.tolerance_threshold}"
            
            # Store in metadata for validation
            assert 'quasi_sheaf_tolerance' in sheaf.metadata
            assert sheaf.metadata['quasi_sheaf_tolerance'] <= self.tolerance_threshold
    
    def _compute_functoriality_violations(self, sheaf: Sheaf) -> List[float]:
        """Compute functoriality violations for all 3-node paths."""
        violations = []
        nodes = list(sheaf.poset.nodes())
        
        # Find all 3-node paths
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                for k in range(len(nodes)):
                    if i < j < k:  # Ensure ordering
                        node_i, node_j, node_k = nodes[i], nodes[j], nodes[k]
                        
                        # Check if path exists
                        if (sheaf.poset.has_edge(node_i, node_j) and 
                            sheaf.poset.has_edge(node_j, node_k) and
                            sheaf.poset.has_edge(node_i, node_k)):
                            
                            # Get restrictions
                            rho_ji = sheaf.restrictions[(node_i, node_j)]
                            rho_kj = sheaf.restrictions[(node_j, node_k)]
                            rho_ki = sheaf.restrictions[(node_i, node_k)]
                            
                            # Compute composition
                            rho_composed = rho_kj @ rho_ji
                            
                            # Compute violation
                            violation = torch.norm(rho_ki - rho_composed, 'fro').item()
                            violations.append(violation)
                            
        return violations
    
    def test_different_architectures(self):
        """Test quasi-sheaf properties across different network architectures."""
        architectures = [
            [10, 8, 6, 4],        # Decreasing width
            [10, 15, 10, 5],      # Bottleneck
            [10, 20, 30, 5],      # Expanding then contracting
            [10, 10, 10, 10]      # Constant width
        ]
        
        data = torch.randn(50, 10)
        config = GWConfig(epsilon=0.1, max_iter=1000)
        
        for arch in architectures:
            model = self._create_test_network(arch)
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            
            activations = self._extract_activations(model, data)
            sheaf = builder.build_from_activations(
                activations,
                model, 
                gw_config=config
            )
            
            violations = self._compute_functoriality_violations(sheaf)
            max_violation = max(violations) if violations else 0.0
            
            logger.info(f"Architecture {arch}: max_violation={max_violation:.6f}")
            assert max_violation <= self.tolerance_threshold
    
    def test_violation_vs_epsilon(self):
        """Test that smaller epsilon generally yields better functoriality."""
        model = self._create_test_network([10, 8, 6, 4])
        data = torch.randn(50, 10)
        
        violations_by_epsilon = {}
        
        for epsilon in self.epsilon_values:
            config = GWConfig(epsilon=epsilon, max_iter=2000, tolerance=1e-10)
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            
            activations = self._extract_activations(model, data)
            sheaf = builder.build_from_activations(
                activations,
                model,
                gw_config=config
            )
            
            violations = self._compute_functoriality_violations(sheaf)
            violations_by_epsilon[epsilon] = {
                'max': max(violations) if violations else 0.0,
                'avg': np.mean(violations) if violations else 0.0,
                'violations': violations
            }
        
        # Generally expect trend: smaller epsilon -> smaller violations
        # But not strictly monotonic due to optimization
        epsilons_sorted = sorted(self.epsilon_values)
        avg_violations = [violations_by_epsilon[e]['avg'] for e in epsilons_sorted]
        
        # Check overall trend (allow some non-monotonicity)
        decreasing_pairs = sum(1 for i in range(len(avg_violations)-1) 
                              if avg_violations[i] <= avg_violations[i+1])
        assert decreasing_pairs >= len(avg_violations) - 2, \
            "Violations should generally decrease with epsilon"


class TestLaplacianProperties:
    """Validate Laplacian mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tolerance = 1e-6
        self.gw_config = GWConfig(epsilon=0.1, max_iter=1000)
        
    def _build_test_sheaf(self, layer_sizes: List[int], 
                         data_size: int = 50) -> Tuple[Sheaf, torch.Tensor]:
        """Build a test GW sheaf and return it with test data."""
        model = nn.Sequential(*[
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
        
        data = torch.randn(data_size, layer_sizes[0])
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        
        sheaf = builder.build_from_activations(
            self._extract_activations_simple(model, data),
            model,
            gw_config=self.gw_config
        )
        
        return sheaf, data
    
    def _extract_activations_simple(self, model: nn.Module, 
                                   data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simple activation extraction."""
        activations = {}
        x = data
        for i, layer in enumerate(model):
            x = layer(x)
            activations[f"layer_{i}"] = x.detach().clone()
        return activations
    
    def test_laplacian_positive_semidefinite(self):
        """Verify L ⪰ 0 numerically."""
        test_architectures = [
            [10, 8, 6],
            [10, 15, 10, 5],
            [8, 8, 8, 8]
        ]
        
        laplacian_builder = SheafLaplacianBuilder(method='gromov_wasserstein')
        
        for arch in test_architectures:
            sheaf, _ = self._build_test_sheaf(arch)
            
            # Build Laplacian
            result = laplacian_builder.build_laplacian(sheaf, sparse=False)
            L = result.laplacian
            
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvalsh(L)
            min_eigenvalue = eigenvalues.min().item()
            
            logger.info(f"Architecture {arch}: min_eigenvalue={min_eigenvalue:.8f}")
            
            # Check PSD (allowing small negative values due to numerical errors)
            assert min_eigenvalue >= -self.tolerance, \
                f"Laplacian not PSD: min eigenvalue {min_eigenvalue}"
            
            # Check that we have expected number of near-zero eigenvalues
            # (at least one per connected component)
            num_small = torch.sum(eigenvalues.abs() < 1e-6).item()
            assert num_small >= 1, "Should have at least one near-zero eigenvalue"
    
    def test_laplacian_block_structure(self):
        """Verify GW Laplacian follows equations (5.1) and (5.2) from plan."""
        sheaf, _ = self._build_test_sheaf([10, 8, 6, 4])
        
        # Build Laplacian
        laplacian_builder = SheafLaplacianBuilder(method='gromov_wasserstein')
        result = laplacian_builder.build_laplacian(sheaf, sparse=False)
        L = result.laplacian
        
        # Extract block structure
        block_info = self._extract_block_structure(sheaf, L)
        
        # Verify diagonal blocks follow formula:
        # L_{ii} = (∑_{e∈in(i)} I_{n_i}) + ∑_{e=(i→j)∈out(i)} ρ_{j→i}^T ρ_{j→i}
        for node in sheaf.poset.nodes():
            diagonal_block = block_info['diagonal_blocks'][node]
            expected_block = self._compute_expected_diagonal_block(sheaf, node)
            
            block_diff = torch.norm(diagonal_block - expected_block, 'fro').item()
            assert block_diff < self.tolerance, \
                f"Diagonal block mismatch for node {node}: diff={block_diff}"
        
        # Verify off-diagonal blocks
        for edge in sheaf.restrictions:
            source, target = edge
            off_diag_block = block_info['off_diagonal_blocks'].get((source, target))
            
            if off_diag_block is not None:
                # Should be -ρ_{j→i}^T for edge (i→j)
                restriction = sheaf.restrictions[edge]
                expected = -restriction.T
                
                block_diff = torch.norm(off_diag_block - expected, 'fro').item()
                assert block_diff < self.tolerance, \
                    f"Off-diagonal block mismatch for edge {edge}: diff={block_diff}"
    
    def _extract_block_structure(self, sheaf: Sheaf, 
                                L: torch.Tensor) -> Dict[str, Dict]:
        """Extract block structure from Laplacian."""
        # Get node ordering and dimensions
        nodes = list(sheaf.poset.nodes())
        node_dims = {node: sheaf.stalks[node].shape[0] for node in nodes}
        
        # Compute block offsets
        offsets = {}
        current_offset = 0
        for node in nodes:
            offsets[node] = current_offset
            current_offset += node_dims[node]
        
        # Extract blocks
        diagonal_blocks = {}
        off_diagonal_blocks = {}
        
        for i, node_i in enumerate(nodes):
            start_i = offsets[node_i]
            end_i = start_i + node_dims[node_i]
            
            # Diagonal block
            diagonal_blocks[node_i] = L[start_i:end_i, start_i:end_i]
            
            # Off-diagonal blocks
            for j, node_j in enumerate(nodes):
                if i != j:
                    start_j = offsets[node_j]
                    end_j = start_j + node_dims[node_j]
                    
                    block = L[start_i:end_i, start_j:end_j]
                    if torch.any(block.abs() > 1e-10):
                        off_diagonal_blocks[(node_i, node_j)] = block
        
        return {
            'diagonal_blocks': diagonal_blocks,
            'off_diagonal_blocks': off_diagonal_blocks,
            'offsets': offsets,
            'node_dims': node_dims
        }
    
    def _compute_expected_diagonal_block(self, sheaf: Sheaf, node: str) -> torch.Tensor:
        """Compute expected diagonal block according to formula."""
        dim = sheaf.stalks[node].shape[0]
        
        # Start with zero
        block = torch.zeros((dim, dim))
        
        # Add identity for each incoming edge
        in_edges = list(sheaf.poset.in_edges(node))
        block += len(in_edges) * torch.eye(dim)
        
        # Add ρ^T ρ for each outgoing edge
        out_edges = list(sheaf.poset.out_edges(node))
        for edge in out_edges:
            source, target = edge
            if edge in sheaf.restrictions:
                rho = sheaf.restrictions[edge]
                block += rho.T @ rho
        
        return block
    
    def test_laplacian_row_col_sums(self):
        """Test that Laplacian rows/columns sum appropriately."""
        sheaf, _ = self._build_test_sheaf([8, 6, 4])
        
        laplacian_builder = SheafLaplacianBuilder(method='gromov_wasserstein')
        result = laplacian_builder.build_laplacian(sheaf, sparse=False)
        L = result.laplacian
        
        # For uniform measures, check properties
        if sheaf.metadata.get('measure_type') == 'uniform':
            # Row sums should be zero for connected components
            row_sums = L.sum(dim=1)
            max_row_sum = row_sums.abs().max().item()
            
            logger.info(f"Max row sum: {max_row_sum}")
            # Allow some numerical error
            assert max_row_sum < 1e-5, f"Row sums not near zero: {max_row_sum}"


class TestSpectralStability:
    """Test numerical stability of spectral properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.perturbation_scales = [1e-6, 1e-5, 1e-4, 1e-3]
        self.gw_config = GWConfig(epsilon=0.1, max_iter=1000)
        
    def test_eigenvalue_continuity(self):
        """Test eigenvalue continuity under small perturbations."""
        # Build reference sheaf
        model = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 6),
            nn.Linear(6, 4)
        )
        data = torch.randn(50, 10)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        laplacian_builder = SheafLaplacianBuilder(method='gromov_wasserstein')
        
        # Get reference eigenvalues
        activations = self._get_activations(model, data)
        ref_sheaf = builder.build_from_activations(
            activations, model, gw_config=self.gw_config
        )
        ref_result = laplacian_builder.build_laplacian(ref_sheaf, sparse=False)
        ref_eigenvalues = torch.linalg.eigvalsh(ref_result.laplacian)
        
        # Test perturbations
        for scale in self.perturbation_scales:
            # Perturb data
            perturbed_data = data + scale * torch.randn_like(data)
            
            # Build perturbed sheaf
            perturbed_activations = self._get_activations(model, perturbed_data)
            perturbed_sheaf = builder.build_from_activations(
                perturbed_activations, model, gw_config=self.gw_config
            )
            perturbed_result = laplacian_builder.build_laplacian(perturbed_sheaf, sparse=False)
            perturbed_eigenvalues = torch.linalg.eigvalsh(perturbed_result.laplacian)
            
            # Check continuity
            eigenvalue_diff = self._compute_eigenvalue_distance(
                ref_eigenvalues, perturbed_eigenvalues
            )
            
            logger.info(f"Perturbation scale {scale}: eigenvalue_diff={eigenvalue_diff:.6f}")
            
            # Eigenvalue difference should scale with perturbation
            # Allow factor of 100 for amplification through network
            assert eigenvalue_diff < 100 * scale, \
                f"Eigenvalues not continuous: diff={eigenvalue_diff} for scale={scale}"
    
    def _get_activations(self, model: nn.Module, 
                        data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from model."""
        activations = {}
        x = data
        for i, layer in enumerate(model):
            x = layer(x)
            activations[f"layer_{i}"] = x.detach().clone()
        return activations
    
    def _compute_eigenvalue_distance(self, eig1: torch.Tensor, 
                                   eig2: torch.Tensor) -> float:
        """Compute distance between eigenvalue sets (sorted)."""
        # Sort eigenvalues
        eig1_sorted = torch.sort(eig1)[0]
        eig2_sorted = torch.sort(eig2)[0]
        
        # Pad if different sizes
        if len(eig1_sorted) != len(eig2_sorted):
            max_len = max(len(eig1_sorted), len(eig2_sorted))
            if len(eig1_sorted) < max_len:
                eig1_sorted = torch.cat([
                    eig1_sorted,
                    torch.zeros(max_len - len(eig1_sorted))
                ])
            else:
                eig2_sorted = torch.cat([
                    eig2_sorted,
                    torch.zeros(max_len - len(eig2_sorted))
                ])
        
        # Compute L2 distance
        return torch.norm(eig1_sorted - eig2_sorted).item()
    
    def test_subspace_stability(self):
        """Test stability of eigenspaces under perturbations."""
        from scipy.linalg import subspace_angles
        
        # Build reference
        model = nn.Sequential(
            nn.Linear(8, 6),
            nn.Linear(6, 4)
        )
        data = torch.randn(40, 8)
        
        builder = SheafBuilder(restriction_method='gromov_wasserstein')
        laplacian_builder = SheafLaplacianBuilder(method='gromov_wasserstein')
        
        # Get reference eigenspaces
        activations = self._get_activations(model, data)
        ref_sheaf = builder.build_from_activations(
            activations, model, gw_config=self.gw_config
        )
        ref_result = laplacian_builder.build_laplacian(ref_sheaf, sparse=False)
        ref_L = ref_result.laplacian
        ref_eigvals, ref_eigvecs = torch.linalg.eigh(ref_L)
        
        # Select subspace of smallest k eigenvalues
        k = 3
        ref_subspace = ref_eigvecs[:, :k].numpy()
        
        # Test stability
        for scale in self.perturbation_scales[:2]:  # Test smaller perturbations
            perturbed_data = data + scale * torch.randn_like(data)
            
            perturbed_activations = self._get_activations(model, perturbed_data)
            perturbed_sheaf = builder.build_from_activations(
                perturbed_activations, model, gw_config=self.gw_config
            )
            perturbed_result = laplacian_builder.build_laplacian(perturbed_sheaf, sparse=False)
            perturbed_L = perturbed_result.laplacian
            perturbed_eigvals, perturbed_eigvecs = torch.linalg.eigh(perturbed_L)
            
            perturbed_subspace = perturbed_eigvecs[:, :k].numpy()
            
            # Compute principal angles
            angles = subspace_angles(ref_subspace, perturbed_subspace)
            max_angle = np.max(angles)
            
            logger.info(f"Perturbation {scale}: max principal angle = {max_angle:.6f}")
            
            # Angles should be small for small perturbations
            assert max_angle < 0.1, f"Subspace not stable: angle={max_angle}"
    
    def test_condition_number_analysis(self):
        """Test conditioning of GW Laplacians."""
        architectures = [
            [10, 8, 6, 4],      # Well-conditioned
            [10, 2, 10],        # Potential bottleneck
            [10, 10, 10, 10]    # Uniform
        ]
        
        for arch in architectures:
            model = nn.Sequential(*[
                nn.Linear(arch[i], arch[i+1])
                for i in range(len(arch) - 1)
            ])
            data = torch.randn(50, arch[0])
            
            builder = SheafBuilder(restriction_method='gromov_wasserstein')
            laplacian_builder = SheafLaplacianBuilder(method='gromov_wasserstein')
            
            activations = self._get_activations(model, data)
            sheaf = builder.build_from_activations(
                activations, model, gw_config=self.gw_config
            )
            result = laplacian_builder.build_laplacian(sheaf, sparse=False)
            L = result.laplacian
            
            # Compute condition number (using non-zero eigenvalues)
            eigenvalues = torch.linalg.eigvalsh(L)
            nonzero_eigs = eigenvalues[eigenvalues.abs() > 1e-10]
            
            if len(nonzero_eigs) > 0:
                condition_number = (nonzero_eigs.max() / nonzero_eigs.min()).item()
                logger.info(f"Architecture {arch}: condition number = {condition_number:.2f}")
                
                # GW Laplacians should be reasonably well-conditioned
                assert condition_number < 1e6, f"Poor conditioning: {condition_number}"


class TestNumericalProperties:
    """Test numerical properties specific to GW construction."""
    
    def test_coupling_marginal_constraints(self):
        """Verify GW couplings satisfy marginal constraints."""
        from neurosheaf.sheaf.core import GromovWassersteinComputer
        
        config = GWConfig(epsilon=0.1, validate_couplings=True)
        gw_computer = GromovWassersteinComputer(config)
        
        # Test different sizes
        test_sizes = [(10, 10), (10, 15), (20, 10)]
        
        for n_source, n_target in test_sizes:
            # Create test data
            X_source = torch.randn(n_source, 8)
            X_target = torch.randn(n_target, 8)
            
            # Compute cost matrices
            C_source = gw_computer.compute_cosine_cost_matrix(X_source)
            C_target = gw_computer.compute_cosine_cost_matrix(X_target)
            
            # Compute GW coupling
            result = gw_computer.compute_gw_coupling(C_source, C_target)
            coupling = result.coupling
            
            # Check marginal constraints
            # For uniform measures: π1 = 1/n_target, π^T 1 = 1/n_source
            row_sums = coupling.sum(dim=1)
            col_sums = coupling.sum(dim=0)
            
            expected_row = torch.ones(n_source) / n_target
            expected_col = torch.ones(n_target) / n_source
            
            row_error = torch.norm(row_sums - expected_row).item()
            col_error = torch.norm(col_sums - expected_col).item()
            
            logger.info(f"Size ({n_source}, {n_target}): "
                       f"row_error={row_error:.8f}, col_error={col_error:.8f}")
            
            assert row_error < 1e-5, f"Row marginal violated: {row_error}"
            assert col_error < 1e-5, f"Column marginal violated: {col_error}"
    
    def test_cost_matrix_properties(self):
        """Test properties of cosine cost matrices."""
        from neurosheaf.sheaf.core import GromovWassersteinComputer
        
        config = GWConfig()
        gw_computer = GromovWassersteinComputer(config)
        
        # Test data
        X = torch.randn(20, 10)
        X[5] = X[10]  # Duplicate for testing
        
        C = gw_computer.compute_cosine_cost_matrix(X)
        
        # Check symmetry
        symmetry_error = torch.norm(C - C.T).item()
        assert symmetry_error < 1e-10, f"Cost matrix not symmetric: {symmetry_error}"
        
        # Check non-negativity
        assert torch.all(C >= -1e-10), "Cost matrix has negative values"
        
        # Check diagonal is zero
        diag_error = torch.norm(torch.diag(C)).item()
        assert diag_error < 1e-10, f"Diagonal not zero: {diag_error}"
        
        # Check range [0, 2] for cosine distance
        assert C.max().item() <= 2.0 + 1e-10, "Cost exceeds maximum"
        
        # Check duplicates have zero cost
        assert C[5, 10].item() < 1e-10, "Duplicate vectors should have zero cost"
    
    def test_zero_vector_handling(self):
        """Test handling of zero activation vectors."""
        from neurosheaf.sheaf.core import GromovWassersteinComputer
        
        config = GWConfig()
        gw_computer = GromovWassersteinComputer(config)
        
        # Create data with zero vectors
        X = torch.randn(10, 8)
        X[3] = 0  # Zero vector
        X[7] = 0  # Another zero vector
        
        # Should handle gracefully
        C = gw_computer.compute_cosine_cost_matrix(X)
        
        # Zero vectors should have well-defined costs
        assert not torch.any(torch.isnan(C)), "NaN in cost matrix"
        assert not torch.any(torch.isinf(C)), "Inf in cost matrix"
        
        # Cost between zero vectors should be 0 (or small)
        assert C[3, 7].item() < 0.1, "Zero vectors should have small cost"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])