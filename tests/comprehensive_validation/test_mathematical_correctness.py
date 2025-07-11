"""Comprehensive mathematical correctness testing for Week 7 Laplacian assembly.

This module implements rigorous validation of mathematical properties that
must hold for a valid cellular sheaf and its associated Laplacian operator.
Tests go beyond basic functionality to ensure mathematical exactness.

Mathematical Properties Tested:
1. Sheaf Theory Compliance
   - Transitivity: R_AC = R_BC @ R_AB
   - Exact orthogonality in whitened coordinates
   - Metric compatibility: R_e^T K_w R_e = K_v

2. Laplacian Properties  
   - Exact symmetry: Δ = Δ^T
   - Positive semi-definite: all eigenvalues ≥ 0
   - Correct block structure from whitened maps
   - Spectral properties and rank analysis

3. Filtration Integrity
   - Monotonic sparsity with thresholds
   - Preserved mathematical properties through filtration
   - Topological consistency
"""

import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import subspace_angles
import time
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# Add neurosheaf to path
sys.path.append('/Users/francescopapini/GitRepo/neurosheaf')

from neurosheaf.sheaf import (
    Sheaf, SheafBuilder, SheafLaplacianBuilder, LaplacianMetadata,
    ProcrustesMaps, WhiteningProcessor
)
from neurosheaf.spectral import StaticMaskedLaplacian, create_static_masked_laplacian
from tests.test_data_generators import NeuralNetworkDataGenerator


class MathematicalCorrectnessValidator:
    """Comprehensive validator for mathematical properties of sheaves and Laplacians."""
    
    def __init__(self, tolerance_exact: float = 1e-12, tolerance_approximate: float = 1e-6):
        """Initialize validator with specified tolerances.
        
        Args:
            tolerance_exact: Tolerance for properties that should be exact (e.g., symmetry)
            tolerance_approximate: Tolerance for approximate properties (e.g., eigenvalues)
        """
        self.tolerance_exact = tolerance_exact
        self.tolerance_approximate = tolerance_approximate
        
    def validate_sheaf_mathematical_properties(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Comprehensive validation of sheaf mathematical properties.
        
        Args:
            sheaf: Sheaf object to validate
            
        Returns:
            Dictionary with detailed validation results
        """
        results = {
            'sheaf_axioms': self._validate_sheaf_axioms(sheaf),
            'transitivity': self._validate_transitivity_exact(sheaf),
            'whitened_exactness': self._validate_whitened_exactness(sheaf),
            'metric_compatibility': self._validate_metric_compatibility(sheaf),
            'restriction_orthogonality': self._validate_restriction_orthogonality(sheaf),
            'stalk_consistency': self._validate_stalk_consistency(sheaf)
        }
        
        # Overall validation
        results['all_exact'] = all(
            result.get('exact', False) for result in results.values() 
            if isinstance(result, dict)
        )
        
        return results
    
    def _validate_sheaf_axioms(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Validate fundamental cellular sheaf axioms."""
        # Axiom 1: Poset structure is acyclic
        is_dag = nx.is_directed_acyclic_graph(sheaf.poset)
        
        # Axiom 2: Each node has exactly one stalk
        stalks_complete = all(node in sheaf.stalks for node in sheaf.poset.nodes())
        
        # Axiom 3: Each edge has exactly one restriction map
        restrictions_complete = all(
            edge in sheaf.restrictions for edge in sheaf.poset.edges()
        )
        
        # Axiom 4: Restriction maps have correct dimensions
        dimension_consistency = True
        dimension_errors = []
        
        for edge, restriction in sheaf.restrictions.items():
            source, target = edge
            if source in sheaf.stalks and target in sheaf.stalks:
                source_stalk = sheaf.stalks[source]
                target_stalk = sheaf.stalks[target]
                
                expected_shape = (target_stalk.shape[0], source_stalk.shape[0])
                if restriction.shape != expected_shape:
                    dimension_consistency = False
                    dimension_errors.append({
                        'edge': edge,
                        'expected': expected_shape,
                        'actual': restriction.shape
                    })
        
        return {
            'is_dag': is_dag,
            'stalks_complete': stalks_complete,
            'restrictions_complete': restrictions_complete,
            'dimension_consistency': dimension_consistency,
            'dimension_errors': dimension_errors,
            'exact': is_dag and stalks_complete and restrictions_complete and dimension_consistency
        }
    
    def _validate_transitivity_exact(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Validate exact transitivity: R_AC = R_BC @ R_AB."""
        transitivity_violations = []
        max_violation = 0.0
        total_paths = 0
        
        # Find all paths of length 2 for transitivity testing
        for node_a in sheaf.poset.nodes():
            for node_b in sheaf.poset.successors(node_a):
                for node_c in sheaf.poset.successors(node_b):
                    if sheaf.poset.has_edge(node_a, node_c):  # Direct edge exists
                        total_paths += 1
                        
                        # Get restriction maps
                        R_AB = sheaf.restrictions[(node_a, node_b)]
                        R_BC = sheaf.restrictions[(node_b, node_c)]
                        R_AC = sheaf.restrictions[(node_a, node_c)]
                        
                        # Compute composed restriction
                        R_AC_computed = R_BC @ R_AB
                        
                        # Check exact equality
                        violation = torch.norm(R_AC - R_AC_computed, p='fro').item()
                        max_violation = max(max_violation, violation)
                        
                        if violation > self.tolerance_exact:
                            transitivity_violations.append({
                                'path': (node_a, node_b, node_c),
                                'violation': violation,
                                'direct_norm': torch.norm(R_AC, p='fro').item(),
                                'composed_norm': torch.norm(R_AC_computed, p='fro').item()
                            })
        
        return {
            'total_paths_tested': total_paths,
            'violations': len(transitivity_violations),
            'max_violation': max_violation,
            'violation_details': transitivity_violations,
            'exact': max_violation < self.tolerance_exact
        }
    
    def _validate_whitened_exactness(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Validate exactness properties in whitened coordinates."""
        whitened_results = {}
        
        # Check if stalks are exactly whitened (identity Gram matrices)
        stalk_whitening_errors = {}
        for node, stalk in sheaf.stalks.items():
            stalk_np = stalk.detach().cpu().numpy()
            identity = np.eye(stalk.shape[0])
            whitening_error = np.linalg.norm(stalk_np - identity, 'fro')
            stalk_whitening_errors[node] = whitening_error
        
        max_stalk_error = max(stalk_whitening_errors.values())
        
        # Check restriction map orthogonality in whitened space
        restriction_orthogonality_errors = {}
        for edge, restriction in sheaf.restrictions.items():
            R = restriction.detach().cpu().numpy()
            # For orthogonal matrix: R @ R.T should be identity (if square) or R.T @ R should be identity
            if R.shape[0] <= R.shape[1]:  # More columns than rows
                orthogonality_check = R @ R.T
                identity_target = np.eye(R.shape[0])
            else:  # More rows than columns
                orthogonality_check = R.T @ R
                identity_target = np.eye(R.shape[1])
            
            orthogonality_error = np.linalg.norm(orthogonality_check - identity_target, 'fro')
            restriction_orthogonality_errors[edge] = orthogonality_error
        
        max_restriction_error = max(restriction_orthogonality_errors.values()) if restriction_orthogonality_errors else 0.0
        
        return {
            'stalk_whitening_errors': stalk_whitening_errors,
            'max_stalk_whitening_error': max_stalk_error,
            'restriction_orthogonality_errors': restriction_orthogonality_errors,
            'max_restriction_orthogonality_error': max_restriction_error,
            'stalks_exactly_whitened': max_stalk_error < self.tolerance_exact,
            'restrictions_exactly_orthogonal': max_restriction_error < self.tolerance_exact,
            'exact': max_stalk_error < self.tolerance_exact and max_restriction_error < self.tolerance_exact
        }
    
    def _validate_metric_compatibility(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Validate metric compatibility: R_e^T K_w R_e = K_v in whitened coordinates."""
        compatibility_violations = []
        max_violation = 0.0
        
        for edge, restriction in sheaf.restrictions.items():
            source, target = edge
            K_source = sheaf.stalks[source]  # Whitened Gram matrix
            K_target = sheaf.stalks[target]  # Whitened Gram matrix
            
            # In whitened coordinates: R_e^T @ I @ R_e should equal I
            R = restriction.detach().cpu()
            computed_gram = R.T @ K_target @ R
            
            # For exact metric compatibility, this should equal K_source
            violation = torch.norm(computed_gram - K_source, p='fro').item()
            max_violation = max(max_violation, violation)
            
            if violation > self.tolerance_exact:
                compatibility_violations.append({
                    'edge': edge,
                    'violation': violation,
                    'source_norm': torch.norm(K_source, p='fro').item(),
                    'computed_norm': torch.norm(computed_gram, p='fro').item()
                })
        
        return {
            'violations': len(compatibility_violations),
            'max_violation': max_violation,
            'violation_details': compatibility_violations,
            'exact': max_violation < self.tolerance_exact
        }
    
    def _validate_restriction_orthogonality(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Validate orthogonality properties of restriction maps."""
        orthogonality_results = {}
        
        for edge, restriction in sheaf.restrictions.items():
            R = restriction.detach().cpu().numpy()
            
            # Compute condition number
            try:
                singular_values = np.linalg.svd(R, compute_uv=False)
                condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 1e-16 else np.inf
            except:
                condition_number = np.inf
            
            # Check orthogonality via SVD (U and V should be orthogonal)
            try:
                U, s, Vt = np.linalg.svd(R, full_matrices=False)
                U_orthogonality = np.linalg.norm(U.T @ U - np.eye(U.shape[1]), 'fro')
                V_orthogonality = np.linalg.norm(Vt @ Vt.T - np.eye(Vt.shape[0]), 'fro')
                max_orthogonality_error = max(U_orthogonality, V_orthogonality)
            except:
                max_orthogonality_error = np.inf
            
            orthogonality_results[edge] = {
                'condition_number': condition_number,
                'orthogonality_error': max_orthogonality_error,
                'is_well_conditioned': condition_number < 1e12,
                'is_orthogonal': max_orthogonality_error < self.tolerance_exact
            }
        
        all_well_conditioned = all(result['is_well_conditioned'] for result in orthogonality_results.values())
        all_orthogonal = all(result['is_orthogonal'] for result in orthogonality_results.values())
        
        return {
            'edge_results': orthogonality_results,
            'all_well_conditioned': all_well_conditioned,
            'all_orthogonal': all_orthogonal,
            'exact': all_well_conditioned and all_orthogonal
        }
    
    def _validate_stalk_consistency(self, sheaf: Sheaf) -> Dict[str, Any]:
        """Validate consistency of stalk data across the sheaf."""
        consistency_results = {}
        
        # Check stalk dimensions are reasonable
        stalk_dimensions = {node: stalk.shape[0] for node, stalk in sheaf.stalks.items()}
        dimension_variance = np.var(list(stalk_dimensions.values())) if len(stalk_dimensions) > 1 else 0
        
        # Check stalk condition numbers
        stalk_condition_numbers = {}
        for node, stalk in sheaf.stalks.items():
            stalk_np = stalk.detach().cpu().numpy()
            eigenvals = np.linalg.eigvals(stalk_np)
            eigenvals = eigenvals[eigenvals > 1e-16]  # Filter out numerical zeros
            if len(eigenvals) > 0:
                condition_number = np.max(eigenvals) / np.min(eigenvals)
            else:
                condition_number = np.inf
            stalk_condition_numbers[node] = condition_number
        
        # Check positive semi-definite property (Gram matrices should be PSD)
        stalk_psd_status = {}
        for node, stalk in sheaf.stalks.items():
            eigenvals = torch.linalg.eigvals(stalk).real
            min_eigenval = torch.min(eigenvals).item()
            stalk_psd_status[node] = min_eigenval >= -self.tolerance_exact
        
        all_psd = all(stalk_psd_status.values())
        
        return {
            'stalk_dimensions': stalk_dimensions,
            'dimension_variance': dimension_variance,
            'condition_numbers': stalk_condition_numbers,
            'psd_status': stalk_psd_status,
            'all_positive_semidefinite': all_psd,
            'consistent_dimensions': dimension_variance < 1000,  # Reasonable variance
            'exact': all_psd and dimension_variance < 1000
        }
    
    def validate_laplacian_mathematical_properties(self, laplacian: csr_matrix, 
                                                 metadata: LaplacianMetadata, 
                                                 sheaf: Sheaf) -> Dict[str, Any]:
        """Comprehensive validation of Laplacian mathematical properties."""
        results = {
            'symmetry': self._validate_laplacian_symmetry(laplacian),
            'positive_semidefinite': self._validate_laplacian_psd(laplacian),
            'block_structure': self._validate_laplacian_block_structure(laplacian, metadata, sheaf),
            'spectral_properties': self._validate_laplacian_spectral_properties(laplacian),
            'rank_analysis': self._validate_laplacian_rank_properties(laplacian),
            'numerical_conditioning': self._validate_laplacian_conditioning(laplacian)
        }
        
        # Overall validation
        results['mathematically_valid'] = all(
            result.get('valid', False) for result in results.values()
            if isinstance(result, dict)
        )
        
        return results
    
    def _validate_laplacian_symmetry(self, laplacian: csr_matrix) -> Dict[str, Any]:
        """Validate exact symmetry of Laplacian matrix."""
        laplacian_T = laplacian.T
        symmetry_error = (laplacian - laplacian_T).max()
        
        # Additional symmetry checks
        frobenius_symmetry_error = np.sqrt(((laplacian - laplacian_T).data ** 2).sum())
        
        return {
            'max_symmetry_error': float(symmetry_error),
            'frobenius_symmetry_error': float(frobenius_symmetry_error),
            'exactly_symmetric': symmetry_error < self.tolerance_exact,
            'valid': symmetry_error < self.tolerance_exact
        }
    
    def _validate_laplacian_psd(self, laplacian: csr_matrix) -> Dict[str, Any]:
        """Validate positive semi-definite property via eigenvalue analysis."""
        try:
            # Compute smallest eigenvalues
            k = min(20, laplacian.shape[0] - 1)
            if k > 0:
                eigenvals = eigsh(laplacian, k=k, which='SA', return_eigenvectors=False)
                min_eigenval = float(eigenvals[0])
                
                # Count negative eigenvalues
                negative_eigenvals = eigenvals[eigenvals < -self.tolerance_approximate]
                num_negative = len(negative_eigenvals)
                
                # Estimate rank via eigenvalue count
                positive_eigenvals = eigenvals[eigenvals > self.tolerance_approximate]
                estimated_rank = len(positive_eigenvals)
            else:
                min_eigenval = 0.0
                num_negative = 0
                estimated_rank = 0
                eigenvals = np.array([])
            
            is_psd = min_eigenval >= -self.tolerance_exact
            
            return {
                'min_eigenvalue': min_eigenval,
                'num_negative_eigenvals': int(num_negative),
                'estimated_rank': int(estimated_rank),
                'smallest_eigenvals': eigenvals.tolist(),
                'positive_semidefinite': is_psd,
                'valid': is_psd
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'positive_semidefinite': False,
                'valid': False
            }
    
    def _validate_laplacian_block_structure(self, laplacian: csr_matrix, 
                                          metadata: LaplacianMetadata, 
                                          sheaf: Sheaf) -> Dict[str, Any]:
        """Validate correct block structure assembly from restriction maps."""
        block_validation_results = {}
        
        # Check diagonal blocks
        for node, offset in metadata.stalk_offsets.items():
            if node in metadata.stalk_dimensions:
                dim = metadata.stalk_dimensions[node]
                
                # Extract diagonal block
                diagonal_block = laplacian[offset:offset+dim, offset:offset+dim].toarray()
                
                # Diagonal block should be symmetric and PSD
                block_symmetry_error = np.max(np.abs(diagonal_block - diagonal_block.T))
                
                try:
                    block_eigenvals = np.linalg.eigvals(diagonal_block)
                    min_block_eigenval = np.min(block_eigenvals.real)
                    block_is_psd = min_block_eigenval >= -self.tolerance_exact
                except:
                    min_block_eigenval = np.nan
                    block_is_psd = False
                
                block_validation_results[node] = {
                    'symmetry_error': float(block_symmetry_error),
                    'min_eigenvalue': float(min_block_eigenval),
                    'positive_semidefinite': block_is_psd,
                    'frobenius_norm': float(np.linalg.norm(diagonal_block, 'fro'))
                }
        
        # Check off-diagonal blocks match restriction maps
        off_diagonal_errors = {}
        for edge, restriction in sheaf.restrictions.items():
            source, target = edge
            if source in metadata.stalk_offsets and target in metadata.stalk_offsets:
                source_offset = metadata.stalk_offsets[source]
                target_offset = metadata.stalk_offsets[target]
                source_dim = metadata.stalk_dimensions[source]
                target_dim = metadata.stalk_dimensions[target]
                
                # Extract off-diagonal block
                off_diag_block = laplacian[target_offset:target_offset+target_dim,
                                         source_offset:source_offset+source_dim].toarray()
                
                # Should equal -R_e
                R = restriction.detach().cpu().numpy()
                expected_block = -R
                
                if off_diag_block.shape == expected_block.shape:
                    block_error = np.linalg.norm(off_diag_block - expected_block, 'fro')
                    off_diagonal_errors[edge] = float(block_error)
                else:
                    off_diagonal_errors[edge] = np.inf
        
        max_off_diagonal_error = max(off_diagonal_errors.values()) if off_diagonal_errors else 0.0
        all_blocks_valid = all(result['positive_semidefinite'] for result in block_validation_results.values())
        
        return {
            'diagonal_blocks': block_validation_results,
            'off_diagonal_errors': off_diagonal_errors,
            'max_off_diagonal_error': max_off_diagonal_error,
            'all_diagonal_blocks_valid': all_blocks_valid,
            'block_structure_correct': max_off_diagonal_error < self.tolerance_exact,
            'valid': all_blocks_valid and max_off_diagonal_error < self.tolerance_exact
        }
    
    def _validate_laplacian_spectral_properties(self, laplacian: csr_matrix) -> Dict[str, Any]:
        """Validate spectral properties including eigenvalue distribution."""
        try:
            # Compute eigenvalue spectrum
            n = laplacian.shape[0]
            k_small = min(10, n - 1)
            k_large = min(10, n - 1)
            
            small_eigenvals = eigsh(laplacian, k=k_small, which='SA', return_eigenvectors=False) if k_small > 0 else []
            large_eigenvals = eigsh(laplacian, k=k_large, which='LA', return_eigenvectors=False) if k_large > 0 else []
            
            # Spectral gap analysis
            if len(small_eigenvals) > 1:
                spectral_gap = float(small_eigenvals[1] - small_eigenvals[0])
            else:
                spectral_gap = 0.0
            
            # Condition number estimate
            if len(small_eigenvals) > 0 and len(large_eigenvals) > 0:
                max_eigenval = float(large_eigenvals[-1])
                min_positive_eigenval = float(small_eigenvals[small_eigenvals > self.tolerance_approximate][0]) if any(small_eigenvals > self.tolerance_approximate) else 1.0
                condition_number = max_eigenval / min_positive_eigenval
            else:
                condition_number = np.inf
            
            # Eigenvalue decay analysis
            eigenval_ratios = []
            if len(small_eigenvals) > 1:
                for i in range(1, len(small_eigenvals)):
                    if small_eigenvals[i-1] > self.tolerance_approximate:
                        ratio = small_eigenvals[i] / small_eigenvals[i-1]
                        eigenval_ratios.append(float(ratio))
            
            return {
                'smallest_eigenvals': small_eigenvals.tolist(),
                'largest_eigenvals': large_eigenvals.tolist(),
                'spectral_gap': spectral_gap,
                'condition_number': float(condition_number),
                'eigenvalue_ratios': eigenval_ratios,
                'well_conditioned': condition_number < 1e12,
                'valid': condition_number < 1e15
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'valid': False
            }
    
    def _validate_laplacian_rank_properties(self, laplacian: csr_matrix) -> Dict[str, Any]:
        """Validate rank properties and nullspace analysis."""
        try:
            # Estimate rank via SVD of dense submatrix (for small matrices)
            if laplacian.shape[0] <= 100:
                dense_laplacian = laplacian.toarray()
                singular_values = np.linalg.svd(dense_laplacian, compute_uv=False)
                numerical_rank = np.sum(singular_values > self.tolerance_approximate)
                
                # Nullspace dimension
                nullspace_dim = laplacian.shape[0] - numerical_rank
            else:
                # For large matrices, estimate via eigenvalues
                k = min(50, laplacian.shape[0] - 1)
                if k > 0:
                    eigenvals = eigsh(laplacian, k=k, which='SA', return_eigenvectors=False)
                    numerical_rank = np.sum(eigenvals > self.tolerance_approximate)
                    nullspace_dim = laplacian.shape[0] - numerical_rank
                    singular_values = eigenvals  # Approximate
                else:
                    numerical_rank = 0
                    nullspace_dim = laplacian.shape[0]
                    singular_values = np.array([])
            
            # Check if rank is consistent with graph structure
            # For connected graph, nullspace should be 1-dimensional
            expected_nullspace_dim = 1  # For connected Laplacian
            rank_consistent = abs(nullspace_dim - expected_nullspace_dim) <= 1
            
            return {
                'numerical_rank': int(numerical_rank),
                'nullspace_dimension': int(nullspace_dim),
                'matrix_size': laplacian.shape[0],
                'rank_deficiency': laplacian.shape[0] - numerical_rank,
                'singular_values': singular_values.tolist() if len(singular_values) <= 20 else singular_values[:20].tolist(),
                'rank_consistent_with_graph': rank_consistent,
                'valid': rank_consistent
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'valid': False
            }
    
    def _validate_laplacian_conditioning(self, laplacian: csr_matrix) -> Dict[str, Any]:
        """Validate numerical conditioning of the Laplacian."""
        try:
            # Estimate condition number via eigenvalues
            k = min(10, laplacian.shape[0] - 1)
            if k > 0:
                small_eigenvals = eigsh(laplacian, k=k, which='SA', return_eigenvectors=False)
                large_eigenvals = eigsh(laplacian, k=k, which='LA', return_eigenvectors=False)
                
                # Effective condition number (excluding near-zero eigenvalues)
                positive_eigenvals = small_eigenvals[small_eigenvals > self.tolerance_approximate]
                if len(positive_eigenvals) > 0:
                    min_positive = float(positive_eigenvals[0])
                    max_eigenval = float(large_eigenvals[-1])
                    effective_condition_number = max_eigenval / min_positive
                else:
                    effective_condition_number = np.inf
                
                # Clustering analysis
                eigenval_gaps = []
                if len(small_eigenvals) > 1:
                    eigenval_gaps = [float(small_eigenvals[i+1] - small_eigenvals[i]) 
                                   for i in range(len(small_eigenvals)-1)]
                
                return {
                    'effective_condition_number': effective_condition_number,
                    'eigenvalue_gaps': eigenval_gaps,
                    'min_positive_eigenvalue': float(positive_eigenvals[0]) if len(positive_eigenvals) > 0 else 0.0,
                    'max_eigenvalue': float(large_eigenvals[-1]) if len(large_eigenvals) > 0 else 0.0,
                    'well_conditioned': effective_condition_number < 1e12,
                    'valid': effective_condition_number < 1e15
                }
            else:
                return {
                    'effective_condition_number': 1.0,
                    'well_conditioned': True,
                    'valid': True
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'valid': False
            }


class TestMathematicalCorrectness:
    """Comprehensive test suite for mathematical correctness validation."""
    
    def test_sheaf_mathematical_properties_small_network(self):
        """Test sheaf mathematical properties on small, controllable network."""
        validator = MathematicalCorrectnessValidator()
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Generate high-quality test data
        activations = generator.generate_linear_transformation_sequence(
            num_layers=4, input_dim=12, batch_size=10,
            transformation_strength=0.8, noise_level=0.01
        )
        
        # Create simple chain poset
        layer_names = list(activations.keys())
        poset = nx.DiGraph()
        for name in layer_names:
            poset.add_node(name)
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        
        # Build sheaf with whitening
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        # Comprehensive validation
        results = validator.validate_sheaf_mathematical_properties(sheaf)
        
        # Assert mathematical correctness
        assert results['sheaf_axioms']['exact'], f"Sheaf axioms violated: {results['sheaf_axioms']}"
        assert results['transitivity']['exact'], f"Transitivity violated: max error = {results['transitivity']['max_violation']}"
        assert results['whitened_exactness']['exact'], f"Whitening not exact: {results['whitened_exactness']}"
        assert results['metric_compatibility']['exact'], f"Metric compatibility violated: {results['metric_compatibility']}"
        assert results['restriction_orthogonality']['exact'], f"Restriction orthogonality violated: {results['restriction_orthogonality']}"
        assert results['stalk_consistency']['exact'], f"Stalk consistency violated: {results['stalk_consistency']}"
        
        assert results['all_exact'], "Not all sheaf properties are exact"
        
        print(f"✅ Sheaf mathematical properties validated with exact precision")
        print(f"   Transitivity: {results['transitivity']['total_paths_tested']} paths, max error = {results['transitivity']['max_violation']:.2e}")
        print(f"   Whitening: max stalk error = {results['whitened_exactness']['max_stalk_whitening_error']:.2e}")
        print(f"   Orthogonality: max restriction error = {results['whitened_exactness']['max_restriction_orthogonality_error']:.2e}")
    
    def test_laplacian_mathematical_properties_comprehensive(self):
        """Test comprehensive Laplacian mathematical properties."""
        validator = MathematicalCorrectnessValidator()
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Generate structured test data
        activations = generator.generate_linear_transformation_sequence(
            num_layers=5, input_dim=16, batch_size=12,
            transformation_strength=0.6, noise_level=0.02
        )
        
        # Create branching poset
        layer_names = list(activations.keys())
        poset = nx.DiGraph()
        for name in layer_names:
            poset.add_node(name)
        for i in range(len(layer_names) - 2):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        poset.add_edge(layer_names[1], layer_names[-1])  # Branch
        
        # Build complete pipeline
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        laplacian, metadata = builder.build_laplacian(sheaf)
        
        # Comprehensive Laplacian validation
        results = validator.validate_laplacian_mathematical_properties(laplacian, metadata, sheaf)
        
        # Assert mathematical properties
        assert results['symmetry']['valid'], f"Laplacian not symmetric: error = {results['symmetry']['max_symmetry_error']:.2e}"
        assert results['positive_semidefinite']['valid'], f"Laplacian not PSD: min eigenvalue = {results['positive_semidefinite']['min_eigenvalue']:.2e}"
        assert results['block_structure']['valid'], f"Block structure invalid: {results['block_structure']}"
        assert results['spectral_properties']['valid'], f"Spectral properties invalid: {results['spectral_properties']}"
        assert results['rank_analysis']['valid'], f"Rank analysis failed: {results['rank_analysis']}"
        assert results['numerical_conditioning']['valid'], f"Numerical conditioning poor: {results['numerical_conditioning']}"
        
        assert results['mathematically_valid'], "Laplacian does not satisfy all mathematical properties"
        
        print(f"✅ Laplacian mathematical properties validated")
        print(f"   Symmetry error: {results['symmetry']['max_symmetry_error']:.2e}")
        print(f"   Min eigenvalue: {results['positive_semidefinite']['min_eigenvalue']:.2e}")
        print(f"   Condition number: {results['spectral_properties']['condition_number']:.2e}")
        print(f"   Numerical rank: {results['rank_analysis']['numerical_rank']}")
    
    def test_filtration_mathematical_integrity(self):
        """Test mathematical integrity through filtration sequence."""
        validator = MathematicalCorrectnessValidator()
        generator = NeuralNetworkDataGenerator(seed=42)
        
        # Generate test data with varied edge weights
        activations = generator.generate_linear_transformation_sequence(
            num_layers=6, input_dim=14, batch_size=16,
            transformation_strength=0.5, noise_level=0.05
        )
        
        # Create chain poset
        layer_names = list(activations.keys())
        poset = nx.DiGraph()
        for name in layer_names:
            poset.add_node(name)
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
        
        # Build sheaf and static Laplacian
        builder = SheafBuilder(use_whitening=True, enable_edge_filtering=False)
        gram_matrices = generator.generate_gram_matrices_from_activations(activations)
        sheaf = builder.build_from_cka_matrices(poset, gram_matrices, validate=True)
        
        static_laplacian = builder.build_static_masked_laplacian(sheaf)
        
        # Generate filtration sequence
        thresholds = static_laplacian.suggest_thresholds(8, 'quantile')
        sequence = static_laplacian.compute_filtration_sequence(thresholds)
        
        # Validate each filtered Laplacian
        filtration_results = []
        for i, (threshold, filtered_laplacian) in enumerate(zip(thresholds, sequence)):
            # Validate mathematical properties
            symmetry_error = float((filtered_laplacian - filtered_laplacian.T).max())
            
            # Check PSD property
            try:
                if filtered_laplacian.shape[0] > 1 and filtered_laplacian.nnz > 0:
                    min_eigenval = eigsh(filtered_laplacian, k=1, which='SA', return_eigenvectors=False)[0]
                    is_psd = min_eigenval >= -validator.tolerance_exact
                else:
                    is_psd = True
                    min_eigenval = 0.0
            except:
                is_psd = False
                min_eigenval = np.nan
            
            result = {
                'threshold': threshold,
                'nnz': filtered_laplacian.nnz,
                'symmetry_error': symmetry_error,
                'min_eigenvalue': float(min_eigenval),
                'symmetric': symmetry_error < validator.tolerance_exact,
                'positive_semidefinite': is_psd
            }
            filtration_results.append(result)
            
            # Assert mathematical properties preserved
            assert result['symmetric'], f"Symmetry lost at threshold {i}: error = {symmetry_error:.2e}"
            assert result['positive_semidefinite'], f"PSD property lost at threshold {i}: min eigenvalue = {min_eigenval:.2e}"
        
        # Validate monotonicity
        for i in range(len(sequence) - 1):
            assert sequence[i+1].nnz <= sequence[i].nnz, f"Non-monotonic sparsity at threshold {i}"
        
        print(f"✅ Filtration mathematical integrity validated")
        print(f"   {len(thresholds)} thresholds tested")
        print(f"   Monotonic sparsity: {[r['nnz'] for r in filtration_results]}")
        print(f"   All symmetric and PSD: {all(r['symmetric'] and r['positive_semidefinite'] for r in filtration_results)}")


if __name__ == "__main__":
    # Run comprehensive mathematical correctness tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])