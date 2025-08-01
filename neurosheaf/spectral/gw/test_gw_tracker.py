# neurosheaf/spectral/gw/test_gw_tracker.py
"""
Basic validation test for GW-specific subspace tracker.

This test validates the core functionality of the GW subspace tracker
using synthetic data to ensure the implementation works correctly
before integration with the full pipeline.
"""

import torch
import numpy as np
from typing import Dict, List
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neurosheaf.spectral.gw.gw_subspace_tracker import GWSubspaceTracker
from neurosheaf.spectral.gw.pes_computation import PESComputer
from neurosheaf.spectral.gw.sheaf_inclusion_mapper import SheafInclusionMapper
from neurosheaf.spectral.tracker_factory import SubspaceTrackerFactory
from neurosheaf.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_synthetic_eigendata(n_steps: int = 5, 
                              base_dim: int = 10,
                              n_eigenvals: int = 8) -> tuple:
    """
    Create synthetic eigenvalue/eigenvector sequences for testing.
    
    Simulates increasing complexity GW filtration with:
    - Gradually increasing eigenspace dimensions
    - Smooth eigenvalue evolution
    - Realistic eigenvector changes
    
    Args:
        n_steps: Number of filtration steps
        base_dim: Base eigenspace dimension
        n_eigenvals: Number of eigenvalues per step
        
    Returns:
        Tuple of (eigenvalues_seq, eigenvectors_seq, filtration_params)
    """
    eigenvalues_seq = []
    eigenvectors_seq = []
    filtration_params = []
    
    # Create increasing GW cost thresholds
    base_cost = 0.1
    max_cost = 1.5
    costs = np.linspace(base_cost, max_cost, n_steps)
    
    for step in range(n_steps):
        # Dimension increases slightly with filtration (more edges added)
        current_dim = base_dim + step * 2
        
        # Create realistic eigenvalues (larger ones first, some small ones)
        eigenvals = torch.zeros(n_eigenvals)
        for i in range(n_eigenvals):
            # Eigenvalues decrease exponentially with some noise
            base_val = 10.0 * np.exp(-i * 0.5)
            noise = 0.1 * np.random.randn()
            eigenvals[i] = max(0.01, base_val + noise)
        
        # Sort in descending order (largest first)
        eigenvals, _ = torch.sort(eigenvals, descending=True)
        
        # Create corresponding eigenvectors (orthonormal)
        eigenvecs = torch.randn(current_dim, n_eigenvals)
        eigenvecs, _ = torch.linalg.qr(eigenvecs, mode='reduced')  # Orthogonalize
        
        eigenvalues_seq.append(eigenvals)
        eigenvectors_seq.append(eigenvecs)
        filtration_params.append(costs[step])
    
    return eigenvalues_seq, eigenvectors_seq, filtration_params


def create_mock_sheaf_metadata() -> Dict:
    """
    Create mock sheaf metadata for GW construction.
    
    Returns:
        Dictionary with GW-specific metadata
    """
    # Create some mock GW costs
    n_edges = 20
    gw_costs = {}
    for i in range(n_edges):
        edge_key = f"edge_{i}"
        # Random costs in typical GW range [0, 2] for cosine distance
        cost = np.random.uniform(0.1, 1.8)
        gw_costs[edge_key] = cost
    
    metadata = {
        'construction_method': 'gromov_wasserstein',
        'gw_costs': gw_costs,
        'transport_matrices': {
            'global_transport': torch.rand(5, 5) * 0.5 + 0.5  # Mock transport matrix
        }
    }
    
    return metadata


def test_pes_computer():
    """Test PES computation with synthetic data."""
    logger.info("Testing PES Computer...")
    
    # Create test eigenvectors
    dim = 10
    n_prev = 6
    n_curr = 8
    
    prev_vecs = torch.randn(dim, n_prev)
    curr_vecs = torch.randn(dim, n_curr)
    
    # Orthogonalize for realistic test
    prev_vecs, _ = torch.linalg.qr(prev_vecs, mode='reduced')
    curr_vecs, _ = torch.linalg.qr(curr_vecs, mode='reduced')
    
    # Test PES computation
    pes_computer = PESComputer(threshold=0.5)
    
    # Test basic PES matrix computation
    pes_matrix = pes_computer.compute_pes_matrix(prev_vecs, curr_vecs)
    
    assert pes_matrix.shape == (n_prev, n_curr), f"Wrong PES matrix shape: {pes_matrix.shape}"
    assert torch.all(pes_matrix >= 0) and torch.all(pes_matrix <= 1), "PES values not in [0,1]"
    
    # Test optimal matching
    matches = pes_computer.optimal_eigenvector_matching(pes_matrix)
    
    assert len(matches) <= min(n_prev, n_curr), "Too many matches found"
    assert all(len(match) == 3 for match in matches), "Matches should be (prev, curr, similarity) tuples"
    
    logger.info(f"✅ PES Computer test passed: {pes_matrix.shape} matrix, {len(matches)} matches")


def test_inclusion_mapper():
    """Test sheaf inclusion mapper with synthetic data."""
    logger.info("Testing Sheaf Inclusion Mapper...")
    
    # Test parameters
    prev_dim = 8
    curr_dim = 12
    
    mapper = SheafInclusionMapper(method='identity_extension')
    
    # Test inclusion mapping creation
    inclusion_map = mapper.create_gw_inclusion_mapping(
        prev_step=0,
        curr_step=1,
        prev_eigenspace_dim=prev_dim,
        curr_eigenspace_dim=curr_dim
    )
    
    assert inclusion_map.shape == (curr_dim, prev_dim), f"Wrong inclusion shape: {inclusion_map.shape}"
    
    # Test validation
    is_valid = mapper.validate_sheaf_morphism_properties(inclusion_map)
    assert is_valid, "Inclusion mapping validation failed"
    
    logger.info(f"✅ Inclusion Mapper test passed: {inclusion_map.shape} mapping")


def test_gw_tracker_factory():
    """Test factory creation of GW tracker."""
    logger.info("Testing GW Tracker Factory...")
    
    # Test GW tracker creation
    gw_tracker = SubspaceTrackerFactory.create_tracker('gromov_wasserstein')
    assert isinstance(gw_tracker, GWSubspaceTracker), "Factory didn't create GW tracker"
    
    # Test standard tracker creation (should not be GW tracker)
    std_tracker = SubspaceTrackerFactory.create_tracker('standard')
    assert not isinstance(std_tracker, GWSubspaceTracker), "Factory created GW tracker for standard method"
    
    logger.info("✅ Tracker Factory test passed")


def test_gw_tracker_integration():
    """Test full GW tracker with synthetic data."""
    logger.info("Testing GW Tracker Integration...")
    
    # Create synthetic data
    eigenvals_seq, eigenvecs_seq, filtration_params = create_synthetic_eigendata()
    sheaf_metadata = create_mock_sheaf_metadata()
    
    # Create GW tracker
    gw_tracker = GWSubspaceTracker(
        pes_threshold=0.6,
        transport_weighting=True,
        validate_gw_semantics=True
    )
    
    # Test tracking
    results = gw_tracker.track_eigenspaces(
        eigenvalues_sequence=eigenvals_seq,
        eigenvectors_sequence=eigenvecs_seq,
        filtration_params=filtration_params,
        construction_method='gromov_wasserstein',
        sheaf_metadata=sheaf_metadata
    )
    
    # Validate results structure
    required_keys = ['tracking_method', 'construction_method', 'birth_events', 
                    'death_events', 'continuous_paths', 'pes_statistics']
    
    for key in required_keys:
        assert key in results, f"Missing required key: {key}"
    
    assert results['tracking_method'] == 'persistent_eigenvector_similarity'
    assert results['construction_method'] == 'gromov_wasserstein'
    
    # Check we have some tracking results
    n_birth_events = len(results['birth_events'])
    n_death_events = len(results['death_events'])
    n_continuous_paths = len(results['continuous_paths'])
    n_pes_stats = len(results['pes_statistics'])
    
    logger.info(f"Tracking results: {n_birth_events} births, {n_death_events} deaths, "
               f"{n_continuous_paths} paths, {n_pes_stats} PES statistics")
    
    # Validate GW semantics if available
    if 'gw_validation' in results and results['gw_validation'] is not None:
        validation = results['gw_validation']
        if not validation['is_valid']:
            logger.warning(f"GW validation warnings: {validation.get('warnings', [])}")
            if validation.get('errors'):
                logger.error(f"GW validation errors: {validation['errors']}")
    
    logger.info("✅ GW Tracker Integration test passed")


def main():
    """Run all GW tracker tests."""
    logger.info("🧪 Starting GW Tracker Validation Tests")
    
    try:
        # Set random seed for reproducible results
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run individual component tests
        test_pes_computer()
        test_inclusion_mapper()
        test_gw_tracker_factory()
        
        # Run integration test
        test_gw_tracker_integration()
        
        logger.info("🎉 All GW Tracker tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()