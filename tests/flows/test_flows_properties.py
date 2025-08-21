"""Cross-flow property tests using Hypothesis and randomized testing.

This module tests invariance properties and mathematical relationships that
should hold across both α-flow and t-flow implementations.

Tests cover:
- Permutation invariance of flow fingerprints
- Stability under small perturbations  
- Monotonicity properties across parameter grids
- Functional similarity detection across different topologies
"""

import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from typing import Dict, List, Tuple
import warnings
import torch
from hypothesis import given, strategies as st, settings, assume

from neurosheaf.spectral.flows.alpha_flow import (
    AlphaGroupingPolicy, AlphaFlowBuilder
)
from neurosheaf.spectral.flows.diffusion_flow import (
    DiffusionSpec, DiffusionFlowAnalyzer
)
from neurosheaf.sheaf.data_structures import Sheaf

from .fixtures import (
    small_path_sheaf, small_star_sheaf, FakeGWLaplacianBuilder,
    fallback_path_sheaf, fallback_star_sheaf,
    permute_sheaf, random_gw_costs, trace_hutchinson
)


class TestPermutationInvariance:
    """Test that flow fingerprints are invariant under node permutation."""
    
    def test_alpha_flow_permutation_invariance(self, fallback_path_sheaf):
        """Test α-flow fingerprints stay identical under node permutation."""
        # Create permutation mapping for fallback fixture
        permutation = {'layer1': 'node_X', 'layer2': 'node_Y', 'layer3': 'node_Z'}
        sheaf_permuted = permute_sheaf(fallback_path_sheaf, permutation)
        
        # Build α-flow for both sheaves
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        
        builder_orig = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        builder_perm = AlphaFlowBuilder(sheaf_permuted, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        build_orig = builder_orig.build(grouping=grouping)
        build_perm = builder_perm.build(grouping=grouping)
        
        # Test α-flow fingerprints (trace moments)
        alpha_grid = [0.0, 0.5, 1.0, 2.0]
        moments = [1, 2, 3]  # Different matrix powers
        
        for alpha in alpha_grid:
            L_orig = builder_orig.as_operator(build_orig, alpha)
            L_perm = builder_perm.as_operator(build_perm, alpha)
            
            for moment in moments:
                # Compute trace moments
                trace_orig = trace_hutchinson(L_orig, power=moment, probes=64, seed=42)
                trace_perm = trace_hutchinson(L_perm, power=moment, probes=64, seed=42)
                
                # Should be identical within numerical tolerance
                rel_error = abs(trace_orig - trace_perm) / (abs(trace_orig) + 1e-15)
                assert rel_error < 1e-10, \
                    f"α={alpha}, moment={moment}: permutation broke invariance, " \
                    f"rel_error={rel_error:.2e}"
    
    def test_t_flow_permutation_invariance(self, small_star_sheaf):
        """Test t-flow heat traces are invariant under node permutation."""
        # Create permutation for star sheaf
        permutation = {'A': 'P', 'B': 'Q', 'C': 'CENTER', 'D': 'R', 'E': 'S'}
        sheaf_permuted = permute_sheaf(small_star_sheaf, permutation)
        
        # Build t-flow for both sheaves
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        
        analyzer_orig = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        analyzer_perm = DiffusionFlowAnalyzer(sheaf_permuted, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.1, 0.5, 1.0],
            k_small=3,
            probes=64,
            slq_iters=20
        )
        
        result_orig = analyzer_orig.analyze(spec)
        result_perm = analyzer_perm.analyze(spec)
        
        # Heat traces should be identical (within SLQ noise)
        for i, t in enumerate(spec.t_grid):
            rel_error = abs(result_orig.heat_trace[i] - result_perm.heat_trace[i]) / \
                       (abs(result_orig.heat_trace[i]) + 1e-15)
            assert rel_error < 1e-6, \
                f"t={t}: permutation broke heat trace invariance, rel_error={rel_error:.2e}"
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=5, deadline=5000)
    def test_alpha_flow_random_permutation_hypothesis(self, perm_seed):
        """Hypothesis-based test with random permutations."""
        import networkx as nx
        import torch
        
        # Create deterministic test sheaf using proper constructor
        nodes = ['A', 'B', 'C']
        
        # Create poset
        poset = nx.DiGraph()
        poset.add_edge('A', 'B')
        poset.add_edge('B', 'C')
        
        # Create stalks
        stalks = {
            node: torch.randn(8, 2)  # 8 samples, 2 dimensions each
            for node in nodes
        }
        
        # Create restrictions
        restrictions = {
            ('A', 'B'): torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64),
            ('B', 'C'): torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
        }
        
        # Create metadata
        metadata = {
            'construction_method': 'gromov_wasserstein',
            'is_gw_sheaf': True,
            'gw_costs': {('A', 'B'): 0.3, ('B', 'C'): 0.7}
        }
        
        # Construct sheaf
        sheaf = Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata
        )
        
        # Generate random permutation
        rng = np.random.default_rng(perm_seed)
        perm_nodes = nodes.copy()
        rng.shuffle(perm_nodes)
        permutation = {orig: new for orig, new in zip(nodes, perm_nodes)}
        
        # Skip identity permutation (not interesting)
        assume(permutation != {n: n for n in nodes})
        
        sheaf_permuted = permute_sheaf(sheaf, permutation)
        
        # Test invariance
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        
        builder_orig = AlphaFlowBuilder(sheaf, fake_builder)
        builder_perm = AlphaFlowBuilder(sheaf_permuted, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        build_orig = builder_orig.build(grouping=grouping)
        build_perm = builder_perm.build(grouping=grouping)
        
        # Test trace at α=1.0
        L_orig = builder_orig.as_operator(build_orig, 1.0)
        L_perm = builder_perm.as_operator(build_perm, 1.0)
        
        trace_orig = trace_hutchinson(L_orig, power=1, probes=32, seed=42)
        trace_perm = trace_hutchinson(L_perm, power=1, probes=32, seed=42)
        
        rel_error = abs(trace_orig - trace_perm) / (abs(trace_orig) + 1e-15)
        assert rel_error < 1e-8, f"Random permutation broke invariance: rel_error={rel_error:.2e}"


class TestStabilityUnderPerturbations:
    """Test stability of flow fingerprints under small perturbations."""
    
    def test_alpha_flow_restriction_perturbation_stability(self, fallback_path_sheaf):
        """Test α-flow stability under small restriction perturbations."""
        import torch
        import copy
        
        # Create perturbed restrictions
        rng = np.random.default_rng(42)
        perturbation_scale = 1e-4
        
        perturbed_restrictions = {}
        for edge, restriction in fallback_path_sheaf.restrictions.items():
            R_orig = restriction.numpy()
            noise = rng.normal(0, perturbation_scale, size=R_orig.shape)
            R_perturbed = R_orig + noise
            
            # Maintain reasonable properties (optional clipping)
            R_perturbed = np.clip(R_perturbed, -2.0, 2.0)  # Reasonable bounds
            perturbed_restrictions[edge] = torch.tensor(R_perturbed, dtype=torch.float64)
        
        # Create perturbed sheaf using dataclass constructor
        perturbed_sheaf = Sheaf(
            poset=copy.deepcopy(fallback_path_sheaf.poset),
            stalks=copy.deepcopy(fallback_path_sheaf.stalks),
            restrictions=perturbed_restrictions,
            metadata=copy.deepcopy(fallback_path_sheaf.metadata)
        )
        
        # Build α-flow for both versions
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        
        builder_orig = AlphaFlowBuilder(fallback_path_sheaf, fake_builder)
        builder_pert = AlphaFlowBuilder(perturbed_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        
        build_orig = builder_orig.build(grouping=grouping)
        build_pert = builder_pert.build(grouping=grouping)
        
        # Test stability of trace moments
        alpha_values = [0.5, 1.0]
        
        for alpha in alpha_values:
            L_orig = builder_orig.as_operator(build_orig, alpha)
            L_pert = builder_pert.as_operator(build_pert, alpha)
            
            trace_orig = trace_hutchinson(L_orig, power=1, probes=64, seed=42)
            trace_pert = trace_hutchinson(L_pert, power=1, probes=64, seed=42)
            
            # Perturbation should cause small change
            rel_change = abs(trace_orig - trace_pert) / (abs(trace_orig) + 1e-15)
            
            # Should be stable but not identical (some sensitivity expected)
            assert rel_change < 0.1, \
                f"α={alpha}: too sensitive to perturbations, rel_change={rel_change:.3f}"
            assert rel_change > 1e-6, \
                f"α={alpha}: suspiciously insensitive, rel_change={rel_change:.2e}"
    
    def test_t_flow_cost_perturbation_stability(self, small_star_sheaf):
        """Test t-flow stability under small GW cost perturbations."""
        # Create sheaf with perturbed costs
        perturbed_sheaf = small_star_sheaf
        orig_costs = small_star_sheaf.metadata['gw_costs'].copy()
        
        # Add small random perturbations to costs
        rng = np.random.default_rng(123)
        perturbation_scale = 0.05  # 5% relative perturbation
        
        perturbed_costs = {}
        for edge, cost in orig_costs.items():
            noise = rng.normal(0, perturbation_scale * cost)
            perturbed_cost = max(0.001, cost + noise)  # Keep positive
            perturbed_costs[edge] = perturbed_cost
        
        perturbed_sheaf.metadata['gw_costs'] = perturbed_costs
        
        # Build t-flow for both versions
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        
        # Reset original costs for comparison
        orig_sheaf = small_star_sheaf
        orig_sheaf.metadata['gw_costs'] = orig_costs
        
        analyzer_orig = DiffusionFlowAnalyzer(orig_sheaf, fake_builder, random_seed=42)
        analyzer_pert = DiffusionFlowAnalyzer(perturbed_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.2, 1.0],
            probes=64,
            slq_iters=25
        )
        
        result_orig = analyzer_orig.analyze(spec)
        result_pert = analyzer_pert.analyze(spec)
        
        # Check stability of heat traces
        for i, t in enumerate(spec.t_grid):
            rel_change = abs(result_orig.heat_trace[i] - result_pert.heat_trace[i]) / \
                        (abs(result_orig.heat_trace[i]) + 1e-15)
            
            # Should be stable but show some sensitivity
            assert rel_change < 0.2, \
                f"t={t}: too sensitive to cost perturbations, rel_change={rel_change:.3f}"
    
    def test_variance_reduction_with_increased_probes(self, small_path_sheaf):
        """Test that variance decreases when doubling probes (for stochastic estimators)."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        L_alpha = builder.as_operator(build, 1.0)
        
        # Test variance scaling with number of probes
        n_trials = 10
        probe_counts = [32, 64]  # Double the probes
        
        variances = []
        for n_probes in probe_counts:
            estimates = []
            for trial in range(n_trials):
                # Use different seeds for each trial to get variance
                trace_est = trace_hutchinson(L_alpha, power=1, probes=n_probes, seed=trial)
                estimates.append(trace_est)
            
            variance = np.var(estimates)
            variances.append(variance)
        
        # Variance should decrease roughly as 1/n_probes (halve when doubling probes)
        variance_ratio = variances[1] / (variances[0] + 1e-15)
        
        # Should approximately halve (allow factor of 2 tolerance)
        assert 0.25 <= variance_ratio <= 1.0, \
            f"Variance reduction not as expected: ratio={variance_ratio:.3f}"


class TestMonotonicityConsistency:
    """Test monotonicity properties across parameter grids."""
    
    def test_alpha_flow_monotonicity_multiple_grids(self, small_path_sheaf):
        """Test trace monotonicity across different α grids."""
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        builder = AlphaFlowBuilder(small_path_sheaf, fake_builder)
        
        grouping = AlphaGroupingPolicy(kind='quantile', param=0.5)
        build = builder.build(grouping=grouping)
        
        # Test multiple α grids
        alpha_grids = [
            [0.0, 0.1, 0.5, 1.0],
            [0.0, 0.3, 1.0, 3.0],
            [0.0, 0.05, 0.2, 0.8, 2.0]
        ]
        
        for alpha_grid in alpha_grids:
            traces = []
            
            for alpha in alpha_grid:
                L_alpha = builder.as_operator(build, alpha)
                trace_est = trace_hutchinson(L_alpha, power=1, probes=64, seed=42)
                traces.append(trace_est)
            
            # Check monotonicity within this grid
            monotonicity_violations = 0
            for i in range(1, len(traces)):
                if traces[i] < traces[i-1] - 1e-8:  # Allow small numerical slack
                    monotonicity_violations += 1
            
            # Should have very few violations (≤5% of transitions)
            violation_rate = monotonicity_violations / max(1, len(traces) - 1)
            assert violation_rate <= 0.05, \
                f"Too many monotonicity violations in grid {alpha_grid}: rate={violation_rate:.2f}"
    
    def test_t_flow_monotonicity_different_ranges(self, small_star_sheaf):
        """Test heat trace monotonicity across different t ranges."""
        fake_builder = FakeGWLaplacianBuilder(default_size=6)
        analyzer = DiffusionFlowAnalyzer(small_star_sheaf, fake_builder, random_seed=42)
        
        # Test different t ranges
        t_grids = [
            [0.01, 0.1, 0.5, 1.0],
            [0.1, 0.5, 2.0, 5.0],
            [0.05, 0.2, 0.8, 3.0]
        ]
        
        for t_grid in t_grids:
            spec = DiffusionSpec(
                t_grid=t_grid,
                probes=64,
                slq_iters=20
            )
            
            result = analyzer.analyze(spec)
            heat_trace = result.heat_trace
            
            # Check monotonicity (should be decreasing)
            increasing_violations = 0
            for i in range(1, len(heat_trace)):
                if heat_trace[i] > heat_trace[i-1] + 1e-9:  # Small tolerance
                    increasing_violations += 1
            
            # Should be mostly decreasing (≤10% violations due to SLQ noise)
            violation_rate = increasing_violations / max(1, len(heat_trace) - 1)
            assert violation_rate <= 0.1, \
                f"Too many non-decreasing violations in t_grid {t_grid}: rate={violation_rate:.2f}"


class TestFunctionalSimilarityDetection:
    """Test that flows can distinguish functional vs topological differences."""
    
    def test_same_function_different_topology_similarity(self, small_path_sheaf, small_star_sheaf):
        """Test that functionally similar sheaves have similar fingerprints despite different topology."""
        # Modify both sheaves to have similar "functional behavior"
        # by making their composite restriction effects similar
        
        # For path: A -> B -> C -> D, composite is R_CD @ R_BC @ R_AB
        # For star: leaves -> center, all restrictions similar
        
        # Create modified versions with similar restriction magnitudes
        import torch
        
        # Modify path to have uniform-strength restrictions
        path_mod = Sheaf()
        for node, data in small_path_sheaf.nodes.items():
            path_mod.add_node(node, data=data.copy())
        
        # Use similar restriction patterns
        uniform_restriction_2x3 = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.3, 0.7]], dtype=torch.float64)
        uniform_restriction_3x2 = torch.tensor([[0.7, 0.3], [0.3, 0.7], [0.0, 0.0]], dtype=torch.float64)
        uniform_restriction_2x1 = torch.tensor([[0.8], [0.2]], dtype=torch.float64)
        
        path_mod.add_restriction(('A', 'B'), uniform_restriction_2x3)
        path_mod.add_restriction(('B', 'C'), uniform_restriction_3x2)
        path_mod.add_restriction(('C', 'D'), uniform_restriction_2x1)
        
        # Modify star to have similar restriction strengths
        star_mod = Sheaf()
        for node, data in small_star_sheaf.nodes.items():
            star_mod.add_node(node, data=data.copy())
        
        # Similar magnitude restrictions for star
        uniform_restriction_2x2 = torch.tensor([[0.7, 0.3], [0.3, 0.7]], dtype=torch.float64)
        
        for edge in small_star_sheaf.restrictions.keys():
            star_mod.add_restriction(edge, uniform_restriction_2x2)
        
        # Add similar GW costs
        similar_cost = 0.4
        path_mod.metadata = {
            'is_gw_sheaf': True,
            'gw_costs': {edge: similar_cost for edge in path_mod.restrictions.keys()}
        }
        star_mod.metadata = {
            'is_gw_sheaf': True,
            'gw_costs': {edge: similar_cost for edge in star_mod.restrictions.keys()}
        }
        
        # Compare t-flow fingerprints (more robust to topology)
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        
        analyzer_path = DiffusionFlowAnalyzer(path_mod, fake_builder, random_seed=42)
        analyzer_star = DiffusionFlowAnalyzer(star_mod, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.1, 0.5, 1.0],
            probes=128,  # More probes for better similarity detection
            slq_iters=30
        )
        
        result_path = analyzer_path.analyze(spec)
        result_star = analyzer_star.analyze(spec)
        
        # Compute L2 distance between heat trace vectors
        h_path = result_path.heat_trace
        h_star = result_star.heat_trace
        
        l2_distance = np.linalg.norm(h_path - h_star)
        cosine_similarity = np.dot(h_path, h_star) / (np.linalg.norm(h_path) * np.linalg.norm(h_star))
        
        # Should be reasonably similar (threshold τ₁)
        tau1 = 0.3  # Similarity threshold
        assert l2_distance < tau1, \
            f"Functionally similar sheaves too different: L2={l2_distance:.3f} >= {tau1}"
        assert cosine_similarity > 0.7, \
            f"Functionally similar sheaves too different: cosine={cosine_similarity:.3f} < 0.7"
    
    def test_different_function_increased_distance(self, small_path_sheaf):
        """Test that functionally different sheaves have larger distances."""
        # Create two versions: one with original restrictions, one with very different ones
        
        # Original sheaf
        orig_sheaf = small_path_sheaf
        
        # Modified sheaf with very different restriction behavior
        diff_sheaf = Sheaf()
        for node, data in orig_sheaf.nodes.items():
            diff_sheaf.add_node(node, data=data.copy())
        
        # Add very different restrictions (nearly opposite patterns)
        import torch
        R_AB_diff = torch.tensor([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]], dtype=torch.float64)
        R_BC_diff = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.0, 0.0]], dtype=torch.float64)
        R_CD_diff = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
        
        diff_sheaf.add_restriction(('A', 'B'), R_AB_diff)
        diff_sheaf.add_restriction(('B', 'C'), R_BC_diff)
        diff_sheaf.add_restriction(('C', 'D'), R_CD_diff)
        
        # Different costs too
        diff_costs = {edge: 0.9 for edge in diff_sheaf.restrictions.keys()}  # High cost = low confidence
        diff_sheaf.metadata = {
            'is_gw_sheaf': True,
            'gw_costs': diff_costs
        }
        
        # Compare t-flow fingerprints
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        
        analyzer_orig = DiffusionFlowAnalyzer(orig_sheaf, fake_builder, random_seed=42)
        analyzer_diff = DiffusionFlowAnalyzer(diff_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.1, 0.5, 1.0],
            probes=128,
            slq_iters=30
        )
        
        result_orig = analyzer_orig.analyze(spec)
        result_diff = analyzer_diff.analyze(spec)
        
        # Compute distance
        h_orig = result_orig.heat_trace
        h_diff = result_diff.heat_trace
        
        l2_distance = np.linalg.norm(h_orig - h_diff)
        cosine_similarity = np.dot(h_orig, h_diff) / (np.linalg.norm(h_orig) * np.linalg.norm(h_diff))
        
        # Should be more different (threshold τ₂ > τ₁)
        tau2 = 0.1  # Difference threshold (τ₂ < τ₁ from previous test)
        assert l2_distance > tau2, \
            f"Functionally different sheaves too similar: L2={l2_distance:.3f} <= {tau2}"
        assert cosine_similarity < 0.9, \
            f"Functionally different sheaves too similar: cosine={cosine_similarity:.3f} >= 0.9"
    
    def test_edge_subsampling_robustness(self, small_star_sheaf):
        """Test that fingerprints remain stable under edge subsampling."""
        # Create subsampled version by removing one edge
        orig_sheaf = small_star_sheaf
        edges = list(orig_sheaf.restrictions.keys())
        
        # Remove the last edge (keep most of the structure)
        edges_subset = edges[:-1]  # Remove last edge
        
        # Create subsampled sheaf
        sub_sheaf = Sheaf()
        for node, data in orig_sheaf.nodes.items():
            sub_sheaf.add_node(node, data=data.copy())
        
        for edge in edges_subset:
            sub_sheaf.add_restriction(edge, orig_sheaf.restrictions[edge])
        
        # Subset costs
        orig_costs = orig_sheaf.metadata['gw_costs']
        sub_costs = {edge: orig_costs[edge] for edge in edges_subset}
        sub_sheaf.metadata = {
            'is_gw_sheaf': True,
            'gw_costs': sub_costs
        }
        
        # Compare fingerprints
        fake_builder = FakeGWLaplacianBuilder(default_size=8)
        
        analyzer_orig = DiffusionFlowAnalyzer(orig_sheaf, fake_builder, random_seed=42)
        analyzer_sub = DiffusionFlowAnalyzer(sub_sheaf, fake_builder, random_seed=42)
        
        spec = DiffusionSpec(
            t_grid=[0.2, 1.0],
            probes=64,
            slq_iters=25
        )
        
        result_orig = analyzer_orig.analyze(spec)
        result_sub = analyzer_sub.analyze(spec)
        
        # Should be reasonably close (robust to edge removal)
        h_orig = result_orig.heat_trace
        h_sub = result_sub.heat_trace
        
        l2_distance = np.linalg.norm(h_orig - h_sub)
        rel_distance = l2_distance / (np.linalg.norm(h_orig) + 1e-15)
        
        # Should be robust (relative distance < threshold)
        robustness_threshold = 0.5
        assert rel_distance < robustness_threshold, \
            f"Edge subsampling not robust: rel_distance={rel_distance:.3f} >= {robustness_threshold}"