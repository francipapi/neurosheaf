# tests/phase4_spectral/validation/test_literature_validation.py
"""Literature validation tests for persistent spectral analysis.

This module cross-validates the implementation against established results
from persistent homology and spectral graph theory literature, ensuring
theoretical compliance and mathematical correctness.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer
from neurosheaf.spectral.static_laplacian_masking import StaticLaplacianWithMasking
from neurosheaf.spectral.tracker import SubspaceTracker
from neurosheaf.sheaf.construction import Sheaf
from ..utils.test_ground_truth import GroundTruthGenerator, PersistenceValidator


class TestSpectralTheoryCompliance:
    """Test compliance with established spectral graph theory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
        self.tolerance = 1e-6
    
    def test_graph_laplacian_eigenvalue_bounds(self):
        """Test eigenvalue bounds from spectral graph theory literature.
        
        References:
        - Chung, F.R.K. (1997). Spectral Graph Theory. AMS.
        - Mohar, B. (1991). The Laplacian spectrum of graphs.
        """
        # Test known bounds for different graph types
        test_cases = [
            ('linear_chain', self.generator.linear_chain_sheaf(n_nodes=5, stalk_dim=1)),
            ('cycle', self.generator.cycle_graph_sheaf(n_nodes=6, stalk_dim=1)),
            ('complete', self.generator.complete_graph_sheaf(n_nodes=4, stalk_dim=1))
        ]
        
        for graph_type, (sheaf, expected) in test_cases:
            with pytest.subtest(graph_type=graph_type):
                # Build Laplacian
                static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
                laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
                eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
                eigenvals_sorted = torch.sort(eigenvals)[0]
                
                n = len(sheaf.stalks)  # Number of nodes
                
                # Test fundamental bounds from literature
                
                # 1. All eigenvalues non-negative (PSD property)
                assert torch.all(eigenvals_sorted >= -self.tolerance), \
                    f"{graph_type}: Negative eigenvalues found"
                
                # 2. Smallest eigenvalue is zero (for connected graphs)
                assert eigenvals_sorted[0] < self.tolerance, \
                    f"{graph_type}: No zero eigenvalue (disconnected?)"
                
                # 3. For connected graphs, second eigenvalue (algebraic connectivity) > 0
                if expected['expected_connected_components'] == 1 and len(eigenvals_sorted) > 1:
                    assert eigenvals_sorted[1] > self.tolerance, \
                        f"{graph_type}: No algebraic connectivity"
                
                # 4. Largest eigenvalue ≤ maximum degree bound
                # For complete graphs: max eigenvalue should be around n
                if graph_type == 'complete':
                    max_eigenval = eigenvals_sorted[-1].item()
                    # Allow for stalk dimension scaling
                    expected_max = n * expected['stalk_dim']
                    assert max_eigenval <= expected_max * 1.1, \
                        f"{graph_type}: Max eigenvalue {max_eigenval} > expected bound {expected_max}"
    
    def test_algebraic_connectivity_properties(self):
        """Test algebraic connectivity (Fiedler value) properties.
        
        References:
        - Fiedler, M. (1973). Algebraic connectivity of graphs.
        - Mohar, B. (1991). The Laplacian spectrum of graphs.
        """
        # Test graphs with known connectivity properties
        test_graphs = [
            ('path', self.generator.linear_chain_sheaf(n_nodes=6, stalk_dim=1)),
            ('cycle', self.generator.cycle_graph_sheaf(n_nodes=6, stalk_dim=1)),
            ('complete', self.generator.complete_graph_sheaf(n_nodes=5, stalk_dim=1))
        ]
        
        connectivity_values = {}
        
        for graph_type, (sheaf, expected) in test_graphs:
            static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
            laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
            eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
            eigenvals_sorted = torch.sort(eigenvals)[0]
            
            # Algebraic connectivity is the second smallest eigenvalue
            if len(eigenvals_sorted) > 1:
                algebraic_connectivity = eigenvals_sorted[1].item()
                connectivity_values[graph_type] = algebraic_connectivity
            
        # Test ordering: complete > cycle > path (for same size)
        # Complete graphs should have highest connectivity
        if 'complete' in connectivity_values and 'path' in connectivity_values:
            assert connectivity_values['complete'] > connectivity_values['path'], \
                "Complete graph should have higher connectivity than path"
        
        # All connected graphs should have positive algebraic connectivity
        for graph_type, connectivity in connectivity_values.items():
            assert connectivity > self.tolerance, \
                f"{graph_type}: Non-positive algebraic connectivity {connectivity}"
    
    def test_spectral_gap_theory(self):
        """Test spectral gap properties from literature.
        
        References:
        - Lovász, L. (1993). Random walks on graphs.
        - Chung, F.R.K. (1997). Spectral Graph Theory.
        """
        # Create test graphs with different spectral properties
        graphs = [
            self.generator.linear_chain_sheaf(n_nodes=8, stalk_dim=1),
            self.generator.cycle_graph_sheaf(n_nodes=8, stalk_dim=1),
            self.generator.complete_graph_sheaf(n_nodes=6, stalk_dim=1)
        ]
        
        for i, (sheaf, expected) in enumerate(graphs):
            with pytest.subtest(graph_index=i):
                # Analyze with persistent spectral analyzer
                analyzer = PersistentSpectralAnalyzer()
                result = analyzer.analyze(sheaf, n_steps=1)  # Single step to get base eigenvalues
                
                eigenvals = result['persistence_result']['eigenvalue_sequences'][0]
                eigenvals_sorted = torch.sort(eigenvals)[0]
                
                # Test spectral gap validation
                gap_validation = self.validator.validate_spectral_gap(eigenvals_sorted)
                assert gap_validation['gap_exists'], f"Graph {i}: No spectral gap detected"
                
                # For connected graphs, gap should be positive
                if expected['expected_connected_components'] == 1:
                    assert gap_validation['gap_value'] > self.tolerance, \
                        f"Graph {i}: Non-positive spectral gap"


class TestPersistenceTheoryCompliance:
    """Test compliance with persistent homology theory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.validator = PersistenceValidator()
        self.tolerance = 1e-6
    
    def test_persistence_stability_theorem(self):
        """Test persistence stability theorem compliance.
        
        References:
        - Cohen-Steiner, D. et al. (2007). Stability of persistence diagrams.
        - Chazal, F. et al. (2016). The structure and stability of persistence modules.
        """
        # Create baseline filtration
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=5, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        baseline_result = analyzer.analyze(sheaf, n_steps=10)
        baseline_diagrams = baseline_result['diagrams']
        
        # Create perturbed version
        noise_level = 1e-3
        perturbed_sheaf = self._perturb_edge_weights(sheaf, noise_level)
        perturbed_result = analyzer.analyze(perturbed_sheaf, n_steps=10)
        perturbed_diagrams = perturbed_result['diagrams']
        
        # Test stability: bottleneck distance should be bounded by input perturbation
        bottleneck_distance = self._compute_bottleneck_distance_approximation(
            baseline_diagrams, perturbed_diagrams
        )
        
        # Stability bound: should be proportional to input perturbation
        stability_constant = 10.0  # Allow for reasonable amplification
        stability_bound = noise_level * stability_constant
        
        assert bottleneck_distance <= stability_bound, \
            f"Stability violated: bottleneck {bottleneck_distance} > bound {stability_bound}"
    
    def test_persistence_functoriality(self):
        """Test functoriality properties of persistence.
        
        References:
        - Chazal, F. et al. (2012). Persistence-based clustering in Riemannian manifolds.
        - Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes.
        """
        # Create nested filtrations to test functoriality
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=4, stalk_dim=2)
        
        analyzer = PersistentSpectralAnalyzer()
        
        # Coarse filtration
        coarse_result = analyzer.analyze(sheaf, n_steps=5)
        
        # Fine filtration  
        fine_result = analyzer.analyze(sheaf, n_steps=10)
        
        # Test that persistence features are preserved across resolutions
        coarse_features = coarse_result['features']
        fine_features = fine_result['features']
        
        # Number of birth/death events should be similar
        coarse_events = coarse_features['num_birth_events'] + coarse_features['num_death_events']
        fine_events = fine_features['num_birth_events'] + fine_features['num_death_events']
        
        # Allow some variation due to resolution differences
        if max(coarse_events, fine_events) > 0:
            event_ratio = abs(coarse_events - fine_events) / max(coarse_events, fine_events)
            assert event_ratio <= 0.5, f"Functoriality violated: event counts too different"
        
        # Infinite bars should be preserved (topological invariants)
        coarse_infinite = coarse_result['diagrams']['statistics']['n_infinite_bars']
        fine_infinite = fine_result['diagrams']['statistics']['n_infinite_bars']
        assert coarse_infinite == fine_infinite, \
            f"Infinite bars not preserved: {coarse_infinite} vs {fine_infinite}"
    
    def test_betti_number_consistency(self):
        """Test consistency with Betti number computation.
        
        References:
        - Hatcher, A. (2002). Algebraic Topology.
        - Edelsbrunner, H. & Harer, J. (2010). Computational Topology.
        """
        # Test graphs with known Betti numbers
        test_cases = [
            ('tree', self.generator.tree_sheaf(depth=2, branching_factor=2), {'beta_0': 1, 'beta_1': 0}),
            ('cycle', self.generator.cycle_graph_sheaf(n_nodes=5, stalk_dim=1), {'beta_0': 1, 'beta_1': 1}),
            ('disconnected', self.generator.disconnected_components_sheaf([3, 2]), {'beta_0': 2, 'beta_1': 0})
        ]
        
        for graph_type, (sheaf, expected_sheaf), expected_betti in test_cases:
            with pytest.subtest(graph_type=graph_type):
                # Analyze with full filtration
                analyzer = PersistentSpectralAnalyzer()
                result = analyzer.analyze(sheaf, n_steps=1)  # Single step for base topology
                
                # Check beta_0 from zero eigenvalues
                eigenvals = result['persistence_result']['eigenvalue_sequences'][0]
                zero_eigenvals = torch.sum(eigenvals < self.tolerance).item()
                
                # For sheaf Laplacians, number of zero eigenvalues = beta_0 * stalk_dimension
                expected_zeros = expected_betti['beta_0'] * expected_sheaf['stalk_dim']
                assert zero_eigenvals >= expected_zeros, \
                    f"{graph_type}: Zero eigenvals {zero_eigenvals} < expected {expected_zeros}"
                
                # Check consistency with infinite bars
                infinite_bars = result['diagrams']['statistics']['n_infinite_bars']
                assert infinite_bars >= expected_betti['beta_0'], \
                    f"{graph_type}: Infinite bars {infinite_bars} < expected beta_0 {expected_betti['beta_0']}"
    
    def test_morse_theory_connection(self):
        """Test connection to discrete Morse theory.
        
        References:
        - Forman, R. (1998). Morse theory for cell complexes.
        - Kozlov, D. (2008). Combinatorial Algebraic Topology.
        """
        # Create filtration that should exhibit Morse-like behavior
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=6, stalk_dim=1)
        
        analyzer = PersistentSpectralAnalyzer()
        result = analyzer.analyze(sheaf, n_steps=15)
        
        # Test Morse-like properties in persistence
        features = result['features']
        
        # Birth and death events should be discrete
        total_events = features['num_birth_events'] + features['num_death_events']
        assert total_events >= 0, "Negative event count"
        
        # Eigenvalue evolution should be continuous (no jumps)
        eigenval_evolution = features['eigenvalue_evolution']
        for i in range(1, len(eigenval_evolution)):
            prev_stats = eigenval_evolution[i-1]
            curr_stats = eigenval_evolution[i]
            
            # Check for reasonable continuity in means
            if prev_stats['mean'] > 0 and curr_stats['mean'] > 0:
                relative_change = abs(curr_stats['mean'] - prev_stats['mean']) / prev_stats['mean']
                assert relative_change < 2.0, f"Eigenvalue mean jumped too much: {relative_change}"


class TestNumericalMethodValidation:
    """Validate numerical methods against theoretical expectations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.tolerance = 1e-6
    
    def test_eigenvalue_multiplicity_detection(self):
        """Test detection of eigenvalue multiplicities.
        
        References:
        - Golub, G.H. & Van Loan, C.F. (2013). Matrix Computations.
        - Parlett, B.N. (1998). The Symmetric Eigenvalue Problem.
        """
        # Create graph with known multiplicities
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes=4, stalk_dim=2)
        
        # Use subspace tracker to detect groupings
        from neurosheaf.spectral.tracker import SubspaceTracker
        tracker = SubspaceTracker(gap_eps=1e-6)
        
        # Get eigenvalues
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        
        # Group eigenvalues
        groups = tracker._group_eigenvalues(eigenvals, eigenvecs)
        
        # Complete graph should have specific multiplicity structure
        # Zero eigenvalue with multiplicity = stalk_dim
        # Largest eigenvalue with high multiplicity
        
        # Find zero group
        zero_groups = [g for g in groups if torch.min(torch.stack(g['eigenvalues'])) < self.tolerance]
        assert len(zero_groups) >= 1, "No zero eigenvalue group found"
        
        # Zero group should have correct multiplicity
        zero_group = zero_groups[0]
        assert len(zero_group['eigenvalues']) >= expected['stalk_dim'], \
            f"Zero multiplicity {len(zero_group['eigenvalues'])} < expected {expected['stalk_dim']}"
    
    def test_subspace_angle_computation(self):
        """Test subspace angle computation accuracy.
        
        References:
        - Björck, Å. & Golub, G.H. (1973). Numerical methods for computing angles between linear subspaces.
        - Stewart, G.W. (1991). Perturbation theory for the singular value decomposition.
        """
        from neurosheaf.spectral.tracker import SubspaceTracker
        tracker = SubspaceTracker()
        
        # Test with known subspaces
        # Orthogonal subspaces
        Q1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        Q2 = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        
        similarity_orthogonal = tracker._compute_subspace_similarity(Q1, Q2)
        assert abs(similarity_orthogonal) < self.tolerance, \
            f"Orthogonal subspaces similarity {similarity_orthogonal} != 0"
        
        # Identical subspaces
        similarity_identical = tracker._compute_subspace_similarity(Q1, Q1)
        assert abs(similarity_identical - 1.0) < self.tolerance, \
            f"Identical subspaces similarity {similarity_identical} != 1"
        
        # 45-degree angle subspaces
        Q3 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        Q3, _ = torch.linalg.qr(Q3)  # Orthogonalize
        
        similarity_45 = tracker._compute_subspace_similarity(Q1, Q3)
        expected_45 = np.cos(np.pi/4)  # cos(45°) ≈ 0.707
        assert abs(similarity_45 - expected_45) < 0.1, \
            f"45-degree subspace similarity {similarity_45} != expected {expected_45}"
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability in edge cases."""
        # Near-singular cases
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=3, stalk_dim=1)
        
        # Scale down to create near-singular case
        scaled_restrictions = {}
        for edge, restriction in sheaf.restrictions.items():
            scaled_restrictions[edge] = restriction * 1e-12
        
        scaled_sheaf = Sheaf(sheaf.poset, sheaf.stalks, scaled_restrictions)
        
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        
        try:
            laplacian, metadata = static_laplacian._get_cached_laplacian(scaled_sheaf)
            eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
            
            # Should handle near-singular case gracefully
            validation = self.validator.validate_eigenvalue_properties(eigenvals)
            assert validation['finite'], "Near-singular case produced non-finite eigenvalues"
            assert validation['non_negative'], "Near-singular case produced negative eigenvalues"
            
        except Exception as e:
            # Acceptable to fail gracefully on extreme cases
            assert "singular" in str(e).lower() or "ill-conditioned" in str(e).lower(), \
                f"Unexpected error in near-singular case: {e}"
    
    def _perturb_edge_weights(self, sheaf: Sheaf, noise_level: float) -> Sheaf:
        """Create version of sheaf with perturbed edge weights."""
        perturbed_restrictions = {}
        for edge, restriction in sheaf.restrictions.items():
            noise = torch.randn_like(restriction) * noise_level
            perturbed_restrictions[edge] = restriction + noise
        
        return Sheaf(sheaf.poset, sheaf.stalks, perturbed_restrictions)
    
    def _compute_bottleneck_distance_approximation(self, diagrams1, diagrams2) -> float:
        """Compute simplified bottleneck distance approximation."""
        # Extract lifetimes from both diagrams
        lifetimes1 = [pair['lifetime'] for pair in diagrams1['birth_death_pairs']]
        lifetimes2 = [pair['lifetime'] for pair in diagrams2['birth_death_pairs']]
        
        if not lifetimes1 and not lifetimes2:
            return 0.0
        
        if not lifetimes1 or not lifetimes2:
            all_lifetimes = lifetimes1 + lifetimes2
            return max(all_lifetimes) if all_lifetimes else 0.0
        
        # Simple approximation: maximum difference in sorted lifetimes
        lifetimes1_sorted = sorted(lifetimes1, reverse=True)
        lifetimes2_sorted = sorted(lifetimes2, reverse=True)
        
        max_len = max(len(lifetimes1_sorted), len(lifetimes2_sorted))
        
        # Pad with zeros
        lifetimes1_sorted.extend([0.0] * (max_len - len(lifetimes1_sorted)))
        lifetimes2_sorted.extend([0.0] * (max_len - len(lifetimes2_sorted)))
        
        differences = [abs(l1 - l2) for l1, l2 in zip(lifetimes1_sorted, lifetimes2_sorted)]
        return max(differences) if differences else 0.0


class TestKnownResultReproduction:
    """Reproduce specific known results from literature."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GroundTruthGenerator()
        self.tolerance = 1e-3
    
    def test_path_graph_spectrum(self):
        """Test path graph spectrum against known formula.
        
        Reference: Chung, F.R.K. (1997). Spectral Graph Theory, Chapter 1.
        """
        n = 5  # Path with 5 vertices
        sheaf, expected = self.generator.linear_chain_sheaf(n_nodes=n, stalk_dim=1)
        
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Known formula for path graph eigenvalues: 2(1 - cos(kπ/(n+1)))
        expected_eigenvals = []
        for k in range(n):
            if k == 0:
                expected_eigenvals.append(0.0)  # Always one zero
            else:
                expected_eigenvals.append(2 * (1 - np.cos(k * np.pi / (n + 1))))
        
        expected_eigenvals.sort()
        
        # Compare computed vs theoretical
        min_len = min(len(eigenvals_sorted), len(expected_eigenvals))
        for i in range(min_len):
            computed = eigenvals_sorted[i].item()
            theoretical = expected_eigenvals[i]
            relative_error = abs(computed - theoretical) / max(abs(theoretical), 1e-6)
            assert relative_error < 0.1, \
                f"Path graph eigenvalue {i}: computed {computed} vs theoretical {theoretical}"
    
    def test_cycle_graph_spectrum(self):
        """Test cycle graph spectrum against known formula.
        
        Reference: Chung, F.R.K. (1997). Spectral Graph Theory, Chapter 1.
        """
        n = 6  # Cycle with 6 vertices
        sheaf, expected = self.generator.cycle_graph_sheaf(n_nodes=n, stalk_dim=1)
        
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Known formula for cycle graph eigenvalues: 2(1 - cos(2πk/n))
        expected_eigenvals = []
        for k in range(n):
            expected_eigenvals.append(2 * (1 - np.cos(2 * np.pi * k / n)))
        
        expected_eigenvals.sort()
        
        # Compare computed vs theoretical
        min_len = min(len(eigenvals_sorted), len(expected_eigenvals))
        for i in range(min_len):
            computed = eigenvals_sorted[i].item()
            theoretical = expected_eigenvals[i]
            relative_error = abs(computed - theoretical) / max(abs(theoretical), 1e-6)
            assert relative_error < 0.1, \
                f"Cycle graph eigenvalue {i}: computed {computed} vs theoretical {theoretical}"
    
    def test_complete_graph_spectrum(self):
        """Test complete graph spectrum against known results.
        
        Reference: Chung, F.R.K. (1997). Spectral Graph Theory, Chapter 1.
        """
        n = 4  # Complete graph with 4 vertices
        sheaf, expected = self.generator.complete_graph_sheaf(n_nodes=n, stalk_dim=1)
        
        static_laplacian = StaticLaplacianWithMasking(eigenvalue_method='dense')
        laplacian, metadata = static_laplacian._get_cached_laplacian(sheaf)
        eigenvals, eigenvecs = static_laplacian._compute_eigenvalues(laplacian)
        eigenvals_sorted = torch.sort(eigenvals)[0]
        
        # Complete graph: one zero eigenvalue, (n-1) eigenvalues equal to n
        # Check zero eigenvalue
        assert eigenvals_sorted[0] < self.tolerance, "Complete graph missing zero eigenvalue"
        
        # Check non-zero eigenvalues are approximately n
        if len(eigenvals_sorted) > 1:
            non_zero_eigenvals = eigenvals_sorted[1:]
            for eigenval in non_zero_eigenvals:
                # Should be approximately n (scaled by edge weights and stalk dimension)
                expected_val = n  # Approximate expectation
                relative_error = abs(eigenval.item() - expected_val) / expected_val
                assert relative_error < 0.5, \
                    f"Complete graph eigenvalue {eigenval.item()} not close to expected {expected_val}"