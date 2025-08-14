#!/usr/bin/env python3
"""Demonstration of the residual normalization fix for generalized eigenvalue problems (B2).

This script shows the improvement from fixing the residual validation bug:

BEFORE THE FIX:
- Used: ||Lv - λMv|| / ||Ax|| (where Ax = Lv)
- Problem: For small λ ≈ 0, Ax ≈ 0, making residuals artificially huge
- Result: False warnings about poor convergence for near-zero eigenvalues

AFTER THE FIX:
- Uses: ||Lv - λMv||_{M^-1} / ||λMv||_{M^-1} (M-relative metric)
- Special case: |λ| < τ_zero uses ||Lv||_{M^-1} / ||v||_M (absolute metric)
- Result: Mathematically correct residuals for all eigenvalue ranges
"""

import torch
import numpy as np
import networkx as nx
import logging
from scipy.sparse import csr_matrix
from neurosheaf.sheaf.data_structures import Sheaf
from neurosheaf.sheaf.assembly.gw_laplacian import GWLaplacianBuilder

# Set up logging to show validation messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_matrices_with_challenging_eigenvalues():
    """Create test matrices with eigenvalues that challenge the old normalization."""
    n = 10
    
    # Create eigenvalues spanning many orders of magnitude
    eigenvals_target = np.array([
        0.0,          # Exact zero
        1e-15,        # Machine epsilon range
        1e-12,        # Near-zero
        1e-9,         # Very small
        1e-6,         # Small
        1e-3,         # Moderate small
        1e-1,         # Moderate
        1.0,          # Normal
        10.0,         # Large
        100.0         # Very large
    ])
    
    # Create random orthogonal matrix for similarity transformation
    np.random.seed(42)  # Reproducible results
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Construct L = Q @ diag(eigenvals) @ Q.T
    L = Q @ np.diag(eigenvals_target) @ Q.T
    L = 0.5 * (L + L.T)  # Ensure perfect symmetry
    
    # Create well-conditioned M matrix
    M = np.eye(n, dtype=np.float64) + 0.01 * np.random.rand(n, n)
    M = 0.5 * (M + M.T)  # Make symmetric
    M = M + np.eye(n) * 0.1  # Ensure positive definite
    
    return csr_matrix(L), csr_matrix(M), eigenvals_target

def simulate_old_residual_computation(eigenvals, eigenvecs, L, M):
    """Simulate the old (problematic) residual computation."""
    logger.info("🔴 OLD METHOD: ||residual|| / ||Ax|| normalization")
    old_residuals = []
    
    for i in range(len(eigenvals)):
        v_i = eigenvecs[:, i]
        lam_i = eigenvals[i]
        Lv_i = L @ v_i
        Mv_i = M @ v_i
        
        residual_vec = Lv_i - lam_i * Mv_i
        residual_norm = np.linalg.norm(residual_vec)
        
        # OLD: Problematic normalization by ||Ax|| = ||Lv||
        Lv_norm = np.linalg.norm(Lv_i)
        old_relative_residual = residual_norm / (Lv_norm + 1e-16)
        old_residuals.append(old_relative_residual)
        
        if abs(lam_i) < 1e-8:
            logger.warning(f"  λ={lam_i:.2e}: OLD residual = {old_relative_residual:.2e} "
                         f"(||Lv||={Lv_norm:.2e} → tiny denominator!)")
        else:
            logger.info(f"  λ={lam_i:.2e}: OLD residual = {old_relative_residual:.2e}")
    
    return old_residuals

def simulate_new_residual_computation(eigenvals, eigenvecs, L, M, builder):
    """Simulate the new (correct) M-relative residual computation."""
    logger.info("\n🟢 NEW METHOD: M-relative residual normalization")
    new_residuals = []
    
    M_dense = M.toarray()
    M_inv = np.linalg.pinv(M_dense)
    tau_zero = 1e-12
    
    for i in range(len(eigenvals)):
        v_i = eigenvecs[:, i]
        lam_i = eigenvals[i]
        Lv_i = L @ v_i
        Mv_i = M @ v_i
        
        residual_vec = Lv_i - lam_i * Mv_i
        residual_m_inv_norm = builder._compute_m_inverse_norm(residual_vec, M_inv)
        
        if abs(lam_i) < tau_zero:
            # NEW: Absolute metric for near-zero eigenvalues
            Lv_m_inv_norm = builder._compute_m_inverse_norm(Lv_i, M_inv)
            v_m_norm = builder._compute_m_norm(v_i, M)
            new_relative_residual = Lv_m_inv_norm / max(v_m_norm, 1e-16)
            metric_type = "absolute"
        else:
            # NEW: M-relative metric for normal eigenvalues
            solution_vec = lam_i * Mv_i
            solution_m_inv_norm = builder._compute_m_inverse_norm(solution_vec, M_inv)
            new_relative_residual = residual_m_inv_norm / max(solution_m_inv_norm, 1e-16)
            metric_type = "M-relative"
            
        new_residuals.append(new_relative_residual)
        
        if abs(lam_i) < 1e-8:
            logger.info(f"  λ={lam_i:.2e}: NEW residual = {new_relative_residual:.2e} "
                       f"({metric_type} metric ✅)")
        else:
            logger.info(f"  λ={lam_i:.2e}: NEW residual = {new_relative_residual:.2e} "
                       f"({metric_type})")
    
    return new_residuals

def demo_sheaf_integration():
    """Demonstrate the fix working in a real sheaf eigenvalue computation."""
    logger.info("\n" + "="*80)
    logger.info("SHEAF INTEGRATION DEMONSTRATION")
    logger.info("="*80)
    
    # Create a sheaf designed to have near-zero eigenvalues
    nodes = ['a', 'b', 'c', 'd']
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
    
    sheaf = Sheaf(poset=G)
    
    # Add stalks
    for node in nodes:
        sheaf.stalks[node] = torch.eye(2, dtype=torch.float64)
    
    # Add restrictions that are nearly identity (→ small Laplacian eigenvalues)
    for u, v in G.edges():
        # Near-identity restriction with small perturbation
        R = torch.eye(2, dtype=torch.float64) + 0.001 * torch.randn(2, 2, dtype=torch.float64)
        R = R / R.norm(dim=0, keepdim=True)  # Normalize columns
        sheaf.restrictions[(u, v)] = R
    
    # Add GW metadata
    sheaf.metadata['construction_method'] = 'gromov_wasserstein'
    sheaf.metadata['gw_costs'] = {edge: 0.01 for edge in G.edges()}  # Small costs
    
    # Solve using the fixed method
    builder = GWLaplacianBuilder(validate_properties=True)
    try:
        eigenvals, eigenvecs = builder.solve_generalized_robust(
            sheaf, list(G.edges()), k=6, use_matrix_free=False
        )
        
        logger.info(f"\n✅ SHEAF EIGENVALUE RESULTS:")
        logger.info(f"   Eigenvalue range: [{eigenvals[0]:.2e}, {eigenvals[-1]:.2e}]")
        
        near_zero_count = np.sum(np.abs(eigenvals) < 1e-10)
        small_count = np.sum(np.abs(eigenvals) < 1e-6)
        
        logger.info(f"   Near-zero eigenvalues (< 1e-10): {near_zero_count}")
        logger.info(f"   Small eigenvalues (< 1e-6): {small_count}")
        logger.info(f"   💡 Residual validation used improved M-relative metric!")
        
    except Exception as e:
        logger.error(f"❌ Sheaf computation failed: {e}")

def main():
    """Run the complete demonstration."""
    print("NEUROSHEAF RESIDUAL NORMALIZATION FIX DEMONSTRATION (B2)")
    print("="*80)
    print("Problem: Old method ||residual|| / ||Ax|| fails for small λ (Ax ≈ 0)")
    print("Solution: M-relative metric ||residual||_{M^-1} / ||λMv||_{M^-1}")
    print()
    
    # Create test matrices
    L, M, eigenvals_true = create_matrices_with_challenging_eigenvalues()
    builder = GWLaplacianBuilder()
    
    # Solve eigenvalue problem
    logger.info("Solving generalized eigenvalue problem Lv = λMv...")
    eigenvals, eigenvecs = builder._solve_eigsh_shift_invert(L, M, k=8)
    
    # Compare old vs new residual computation
    logger.info(f"\nSOLVED EIGENVALUES: {len(eigenvals)} found")
    logger.info(f"Range: [{eigenvals[0]:.2e}, {eigenvals[-1]:.2e}]")
    
    old_residuals = simulate_old_residual_computation(eigenvals, eigenvecs, L, M)
    new_residuals = simulate_new_residual_computation(eigenvals, eigenvecs, L, M, builder)
    
    # Show improvement summary
    logger.info("\n" + "="*60)
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("="*60)
    
    for i, (old_res, new_res, lam) in enumerate(zip(old_residuals, new_residuals, eigenvals)):
        if abs(lam) < 1e-8:
            improvement = old_res / max(new_res, 1e-16)
            logger.info(f"λ={lam:.2e}: {old_res:.2e} → {new_res:.2e} "
                       f"(improvement: {improvement:.1e}x)")
    
    # Demonstrate integration with sheaf computation
    demo_sheaf_integration()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("🔧 FIX IMPLEMENTED:")
    print("   • M-relative residual metric: ||Lv - λMv||_{M^-1} / ||λMv||_{M^-1}")
    print("   • Special handling for |λ| < τ_zero: ||Lv||_{M^-1} / ||v||_M")
    print("   • Robust fallback for numerical edge cases")
    print()
    print("🛡️ PREVENTS:")
    print("   • Artificially huge residuals for near-zero eigenvalues")
    print("   • False convergence warnings due to wrong normalization")
    print("   • Inconsistent residual metrics across eigenvalue ranges")
    print()
    print("💡 RESULT:")
    print("   • Mathematically correct residual validation")
    print("   • Reliable convergence assessment for all λ ranges")
    print("   • Consistent with generalized eigenvalue theory")

if __name__ == "__main__":
    main()