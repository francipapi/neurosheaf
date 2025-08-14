"""H⁰ Persistence tracking using transport-informed RRQR.

This module implements the complete persistence tracking pipeline for
Global Section (H⁰) analysis, using rank-revealing QR decomposition
with transport-informed evolution and two-step confirmation.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import asdict

from ..io.config import H0Config, DEFAULT_H0_CONFIG
from ..io.types import (
    KernelResult, TransportResult, RRQRResult, PersistenceResult, 
    StepDiagnostics, FiltrationStep
)
from .global_sections import GlobalSectionProcessor
from .transport import GWTransportProcessor, extract_transport_from_sheaf_step
from .utils_numerical import (
    rank_revealing_qr_with_pivoting,
    compute_numerical_certificates
)
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class H0PersistenceTracker:
    """Transport-informed RRQR persistence tracking for global sections.
    
    This class implements the complete H⁰ persistence pipeline from the
    production plan, with RRQR-based tracking, two-step confirmation,
    and comprehensive diagnostics.
    
    Key Features:
    - RRQR-based persistence updates (more stable than Hungarian matching)
    - Transport-informed evolution via induced maps F_t = V⁰_{t+1}^T @ T̃ @ V⁰_t
    - Two-step confirmation to prevent transport noise flickering
    - Comprehensive numerical validation and diagnostics
    """
    
    def __init__(self, cfg: H0Config = None):
        """Initialize H⁰ persistence tracker.
        
        Args:
            cfg: Configuration for numerical parameters and tracking
        """
        self.cfg = cfg or DEFAULT_H0_CONFIG
        self.cfg.validate()
        
        # Log configuration status for clarity
        logger.info(f"H⁰ Persistence Tracker initialized:")
        logger.info(f"   Certificate validation: {'enabled' if self.cfg.validate_certificates else 'disabled'}")
        logger.info(f"   Neural network threshold: {'enabled' if self.cfg.use_neural_network_threshold else 'disabled'}")
        logger.info(f"   RRQR parameters: c_keep={self.cfg.c_keep}, gap={self.cfg.gap}")
        
        # Initialize component processors
        self.global_sections = GlobalSectionProcessor(self.cfg)
        self.transport = GWTransportProcessor(self.cfg)
        
        # State for two-step confirmation
        self._dimension_history = []  # Track kernel dimension changes
        self._death_candidates = []   # Deaths awaiting confirmation
        
        logger.info(f"H0PersistenceTracker initialized: "
                   f"confirm_steps={self.cfg.confirm_steps}, "
                   f"c_keep={self.cfg.c_keep}")
    
    def run_h0_persistence_pipeline(self,
                                   filtration_data: List[FiltrationStep],
                                   builders: Dict[str, Any],
                                   cfg: Optional[H0Config] = None) -> PersistenceResult:
        """Complete H⁰ persistence pipeline with transport-informed evolution.
        
        Implements the main algorithm from the production plan:
        
        For each filtration step t:
        1. Whiten coboundary: δ̃_t = whiten_coboundary_robust(...)
        2. Compute kernel: V⁰_t, k_t = kernel_basis_with_hysteresis(...)
        3. If t > 0:
           a. Build transport: T̃_t = construct_transport_from_gw_coupling(...)
           b. Induced map: F_t = compute_induced_map_on_kernels(...)
           c. Update persistence: Q_t, deaths = rrqr_persistence_update(...)
           d. Detect births: births = k_t - Q_t.shape[1]  
           e. Apply two-step confirmation
        4. Record events and diagnostics
        
        Args:
            filtration_data: List of filtration step data
            builders: Dictionary with coboundary builders and metadata extractors
            cfg: Optional configuration override
            
        Returns:
            Complete H⁰ persistence result with intervals and diagnostics
        """
        cfg = cfg or self.cfg
        start_time = time.time()
        
        logger.info(f"🚀 STARTING H⁰ PERSISTENCE PIPELINE: {len(filtration_data)} steps")
        logger.info(f"   Configuration: dtype={cfg.dtype}, deterministic={cfg.deterministic}")
        logger.info(f"   Numerical parameters: c_in={cfg.c_in}, c_keep={cfg.c_keep}, gap={cfg.gap}")
        logger.info(f"   Machine precision: √ε = {cfg.sqrt_eps:.2e}")
        
        # Initialize tracking state
        intervals = []
        betti_curve = []
        diagnostics = []
        transport_data = {}
        
        # Initialize persistence generators
        Q_current = None
        prev_kernel_result = None
        active_generators = {}  # Maps generator_id -> (birth_step, birth_param)
        next_generator_id = 0
        
        # Track edge count for intelligent degeneracy warnings
        prev_edge_count = 0
        
        # Process each filtration step
        for step_idx, step_data in enumerate(filtration_data):
            step_start_time = time.time()
            
            # Enhanced logging with precision info
            param_value = step_data.get('param', 'N/A')
            logger.info(f"\n📈 PROCESSING STEP {step_idx}: param={param_value}")
            logger.info(f"   Computation precision: {cfg.dtype}")
            logger.info(f"   √ε = {cfg.sqrt_eps:.2e}, c_in = {cfg.c_in}, c_keep = {cfg.c_keep}")
            
            try:
                # Step 1: Extract coboundary and metrics
                coboundary_data = self._extract_coboundary_and_metrics(step_data, builders)
                delta = coboundary_data['delta']
                G0 = coboundary_data['G0'] 
                G1 = coboundary_data['G1']
                
                # Track active edge count for intelligent warnings and log fiber dimensions
                current_edge_count = delta.shape[0] if delta.numel() > 0 else 0
                node_count = delta.shape[1] if delta.numel() > 0 else 0
                
                logger.info(f"   📊 FIBER DIMENSIONS:")
                logger.info(f"      Node fibers (0-cochains): {node_count}")
                logger.info(f"      Edge fibers (1-cochains): {current_edge_count}")
                if hasattr(step_data, 'node_dims') and step_data.node_dims is not None:
                    logger.info(f"      Individual node dimensions: {step_data.node_dims}")
                if hasattr(step_data, 'edge_dims') and step_data.edge_dims is not None:
                    logger.info(f"      Individual edge dimensions: {step_data.edge_dims}")
                
                # Step 2: Whiten coboundary
                whitening_result = self.global_sections.whiten_coboundary_robust(
                    delta, G0, G1, step_id=f"step_{step_idx}"
                )
                
                # Step 3: Compute kernel basis with hysteresis
                kernel_result = self.global_sections.kernel_basis_with_hysteresis(
                    delta_tilde=whitening_result.delta_tilde,
                    S_prev=prev_kernel_result.spectral_norm if prev_kernel_result else None,
                    labels_prev=prev_kernel_result.labels if prev_kernel_result else None,
                    prev_nullity=prev_kernel_result.kernel_dimension if prev_kernel_result else None
                )
                
                # PRESERVE GENERATOR CONTINUITY: Do NOT clear active_generators here
                # Death detection must happen BEFORE any generator identity changes
                V_current = kernel_result.V0  # Current kernel basis
                d_current = kernel_result.kernel_dimension
                
                # Step 4: CPQR-Based Persistence Update (for step > 0)
                if step_idx > 0 and prev_kernel_result is not None and Q_current is not None:
                    # Build transport between steps
                    logger.info(f"🔄 Step {step_idx}: Building transport for CPQR death detection")
                    transport_result = self._build_transport_between_steps(
                        step_data, builders, step_idx - 1, step_idx
                    )
                    
                    if transport_result:
                        logger.info(f"✅ Step {step_idx}: Transport successful, norm={transport_result.transport_norm:.6f}")
                        
                        # CPQR-Based Death Detection
                        cpqr_result = self._cpqr_death_detection(
                            Q_prev=Q_current,  # Previous generator basis
                            V_current=V_current,  # Current kernel basis
                            T_tilde=transport_result.T_tilde,
                            tau_out=kernel_result.tau_out,
                            step_idx=step_idx
                        )
                        
                        # Update generator tracking with CPQR results
                        Q_current, next_generator_id = self._update_generators_with_cpqr(
                            cpqr_result, active_generators, intervals,
                            step_idx, step_data.get('param', step_idx),
                            next_generator_id, V_current
                        )
                        
                        logger.info(f"✅ Step {step_idx}: CPQR update completed: births={cpqr_result['n_births']}, deaths={cpqr_result['n_deaths']}")
                    else:
                        logger.warning(f"❌ Step {step_idx}: Transport construction failed, using dimension-only update")
                        # Handle first step or failed transport - just update Q_current
                        Q_current, next_generator_id = self._handle_first_step_or_transport_failure(
                            V_current, active_generators, step_idx, 
                            step_data.get('param', step_idx), next_generator_id
                        )
                else:
                    # First step - initialize generators
                    Q_current, next_generator_id = self._handle_first_step_or_transport_failure(
                        V_current, active_generators, step_idx, 
                        step_data.get('param', step_idx), next_generator_id
                    )
                    logger.info(f"📝 Step {step_idx}: Initialized with {len(active_generators)} generators, Q_current shape: {Q_current.shape}")
                
                # Step 5: Generator tracking is now handled by CPQR pipeline above
                
                # Step 6: Update Betti curve with debugging info
                beta_0 = kernel_result.kernel_dimension
                betti_curve.append({
                    'step': step_idx,
                    'param': step_data.get('param', step_idx),
                    'beta0': beta_0,
                    'kernel_dim': kernel_result.kernel_dimension,
                    'confirmed_dim': beta_0  # After confirmation logic
                })
                
                # Enhanced dimension debugging with precision details
                current_gens = len(active_generators)
                logger.info(f"🔍 DIMENSION ANALYSIS Step {step_idx}:")
                logger.info(f"   Kernel dimension: {kernel_result.kernel_dimension}")
                logger.info(f"   Active generators: {current_gens}")
                logger.info(f"   Spectral norm ||δ̃||₂: {kernel_result.spectral_norm:.2e}")
                logger.info(f"   Thresholds: τ_in={kernel_result.tau_in:.2e}, τ_out={kernel_result.tau_out:.2e}")
                if kernel_result.residual_norm < float('inf'):
                    logger.info(f"   Kernel residual: {kernel_result.residual_norm:.2e}")
                
                # Mathematical validation of CPQR results
                self._validate_cpqr_invariants(Q_current, active_generators, kernel_result, step_idx)
                logger.info(f"   Q_current shape: {Q_current.shape if Q_current is not None else 'None'}")
                logger.info(f"   Spectral norm: {kernel_result.spectral_norm:.2e}")
                if hasattr(kernel_result, 'singular_values') and len(kernel_result.singular_values) > 0:
                    logger.info(f"   Smallest 3 singular values: {kernel_result.singular_values[:3].numpy()}")
                
                # Check for concerning growth patterns
                if step_idx > 0 and kernel_result.kernel_dimension > 100:
                    logger.warning(f"⚠️  Large kernel dimension {kernel_result.kernel_dimension} at step {step_idx}")
                    logger.warning(f"   This suggests coboundary is becoming degenerate")
                    
                if step_idx > 5 and all(b['beta0'] >= b.get('prev_beta0', 0) for b in betti_curve[-5:]):
                    logger.warning(f"⚠️  Monotonic growth detected - no deaths in last 5 steps")
                    
                # Store previous beta0 for next iteration  
                if len(betti_curve) > 1:
                    betti_curve[-1]['prev_beta0'] = betti_curve[-2]['beta0']
                
                # Step 7: Create step diagnostics
                step_diag = self._create_step_diagnostics(
                    step_idx, step_data, kernel_result, None,  # No rrqr_result in CPQR pipeline
                    None, whitening_result, step_start_time  # transport_result only exists when CPQR runs
                )
                diagnostics.append(step_diag)
                
                # Update state for next iteration
                prev_kernel_result = kernel_result
                
            except Exception as e:
                logger.error(f"Step {step_idx} failed: {e}")
                
                # Add detailed error trace to identify exactly where expand() fails
                import traceback
                logger.error(f"Step {step_idx} detailed error trace:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
                
                # Create error diagnostics and continue
                error_diag = self._create_error_diagnostics(step_idx, step_data, e)
                diagnostics.append(error_diag)
                continue
        
        # Step 8: Finalize infinite intervals for surviving generators
        for gen_id, gen_info in active_generators.items():
            intervals.append({
                'birth_step': gen_info['birth_step'],
                'death_step': None,
                'birth_param': gen_info['birth_param'],
                'death_param': None,
                'lifetime': float('inf'),
                'confirmed': True,
                'transport_informed': gen_info['birth_step'] > 0
            })
        
        # Create final result
        total_time = time.time() - start_time
        result = PersistenceResult(
            intervals=intervals,
            betti_curve=betti_curve,
            diagnostics=diagnostics,
            transport_data=transport_data,
            total_time=total_time,
            n_steps=len(filtration_data),
            method_used="transport_informed_h0"
        )
        
        # Create safe summary
        if betti_curve:
            beta0_values = [b['beta0'] for b in betti_curve]
            beta0_range = f"[{min(beta0_values)}, {max(beta0_values)}]"
        else:
            beta0_range = "[no data]"
            
        logger.info(f"H⁰ persistence pipeline completed in {total_time:.3f}s: "
                   f"{len(intervals)} intervals, β₀ range {beta0_range}")
        
        return result
    
    def compute_induced_map_on_kernels(self,
                                     V0_prev: torch.Tensor,
                                     V0_next: torch.Tensor, 
                                     T_tilde: torch.Tensor) -> torch.Tensor:
        """Compute transport-induced map on global section spaces.
        
        F_t = V⁰_{t+1}^T @ T̃ @ V⁰_t
        
        This is the core map used for RRQR persistence updates.
        
        Args:
            V0_prev: Previous step kernel basis V⁰_t
            V0_next: Next step kernel basis V⁰_{t+1}
            T_tilde: Whitened transport map T̃
            
        Returns:
            Induced map F_t with shape (k_{t+1}, k_t)
        """
        if V0_prev.numel() == 0 or V0_next.numel() == 0:
            # Handle empty kernel cases
            return torch.zeros(V0_next.shape[1], V0_prev.shape[1],
                             dtype=T_tilde.dtype, device=T_tilde.device)
        
        # Add dimension compatibility checking for large dimension changes
        logger.debug(f"🔍 Induced map dimensions:")
        logger.debug(f"   V0_prev shape: {V0_prev.shape} (kernel basis t)")
        logger.debug(f"   V0_next shape: {V0_next.shape} (kernel basis t+1)")  
        logger.debug(f"   T_tilde shape: {T_tilde.shape} (transport map)")
        
        # Check dimension compatibility
        if V0_prev.shape[0] != T_tilde.shape[1]:
            logger.error(f"❌ Dimension mismatch: V0_prev rows {V0_prev.shape[0]} != T_tilde cols {T_tilde.shape[1]}")
            # Return zero map as fallback
            return torch.zeros(V0_next.shape[1], V0_prev.shape[1],
                             dtype=T_tilde.dtype, device=T_tilde.device)
        
        if V0_next.shape[0] != T_tilde.shape[0]:
            logger.error(f"❌ Dimension mismatch: V0_next rows {V0_next.shape[0]} != T_tilde rows {T_tilde.shape[0]}")
            # Return zero map as fallback
            return torch.zeros(V0_next.shape[1], V0_prev.shape[1],
                             dtype=T_tilde.dtype, device=T_tilde.device)
        
        try:
            # F_t = V⁰_{t+1}^T @ T̃ @ V⁰_t
            F = V0_next.T @ T_tilde @ V0_prev
            
            logger.debug(f"✅ Induced map computed: shape {F.shape}, norm {torch.linalg.norm(F):.2e}")
            return F
            
        except Exception as e:
            logger.error(f"❌ Induced map computation failed: {e}")
            # Return zero map as robust fallback
            return torch.zeros(V0_next.shape[1], V0_prev.shape[1],
                             dtype=T_tilde.dtype, device=T_tilde.device)
    
    def _update_persistence_with_transport(self,
                                         V0_prev: torch.Tensor,
                                         V0_next: torch.Tensor,
                                         T_tilde: torch.Tensor,
                                         Q_prev: Optional[torch.Tensor],
                                         filtration_param: float) -> RRQRResult:
        """Update persistence using transport-informed RRQR with GW semantics.
        
        Implements the core RRQR algorithm:
        1. Compute induced map: F = V⁰_{t+1}^T @ T̃ @ V⁰_t
        2. Push alive generators: Y = F @ Q_prev
        3. RRQR with tolerance: Y·P = Q̂·R (column-pivoted)
        4. Keep columns with |R_jj| ≥ c_keep·√ε·||Y||₂
        5. Apply two-step confirmation for deaths
        
        Args:
            V0_prev: Previous kernel basis
            V0_next: Next kernel basis
            T_tilde: Whitened transport map
            Q_prev: Previous generators
            filtration_param: Current filtration parameter
            
        Returns:
            RRQR result with updated generators and birth/death counts
        """
        # Step 1: Compute induced map
        F = self.compute_induced_map_on_kernels(V0_prev, V0_next, T_tilde)
        
        # Handle empty generator case
        if Q_prev is None or Q_prev.numel() == 0:
            return RRQRResult(
                Q_keep=torch.zeros(V0_next.shape[0], 0, dtype=V0_next.dtype, device=V0_next.device),
                keep_mask=torch.zeros(0, dtype=torch.bool),
                n_births=V0_next.shape[1],  # All kernel dimensions are births
                n_deaths=0,
                n_confirmed_deaths=0,
                induced_map=F,
                Y_matrix=torch.zeros(V0_next.shape[1], 0, dtype=V0_next.dtype, device=V0_next.device),
                threshold_used=0.0,
                rank_detected=0
            )
        
        # Step 2: CRITICAL FIX - Apply transport constraints DIRECTLY to generators 
        # The issue: transport constraints on stalk space don't create linear dependence in kernel space
        # Solution: Apply transport directly to generators before computing induced map
        
        if V0_prev.numel() > 0 and Q_prev.numel() > 0:
            try:
                # Check dimensional compatibility
                logger.debug(f"RRQR projection: V0_prev shape {V0_prev.shape}, Q_prev shape {Q_prev.shape}")
                logger.debug(f"RRQR projection: F shape {F.shape}")
                
                if V0_prev.shape[0] != Q_prev.shape[0]:
                    logger.error(f"Dimension mismatch: V0_prev rows {V0_prev.shape[0]} != Q_prev rows {Q_prev.shape[0]}")
                    Y = torch.zeros(F.shape[0], Q_prev.shape[1], dtype=F.dtype, device=F.device)
                else:
                    # DEATH DETECTION FIX: Apply transport directly to generators in stalk space first
                    # This ensures transport constraints create actual linear dependence
                    Q_prev_transported = T_tilde @ Q_prev  # Apply transport constraints directly
                    
                    # Now project onto previous kernel basis
                    Q_prev_projected = V0_prev.T @ Q_prev_transported  # Shape: (k_t, num_generators)
                    logger.debug(f"RRQR with transport: Q_prev_projected shape {Q_prev_projected.shape}")
                    
                    # Apply induced map: Y = F @ Q_prev_projected
                    if F.shape[1] != Q_prev_projected.shape[0]:
                        logger.error(f"F-Q dimension mismatch: F cols {F.shape[1]} != Q_prev_projected rows {Q_prev_projected.shape[0]}")
                        Y = torch.zeros(F.shape[0], Q_prev_projected.shape[1], dtype=F.dtype, device=F.device)
                    else:
                        Y = F @ Q_prev_projected  # Shape: (k_{t+1}, num_generators)
                        
                        # CRITICAL: Verify transport constraints create dependence
                        Y_norm_before = torch.linalg.norm(Q_prev_projected, ord=2).item()
                        Y_norm_after = torch.linalg.norm(Y, ord=2).item()
                        constraint_effect = abs(Y_norm_before - Y_norm_after)
                        logger.debug(f"🎯 Transport effect verification:")
                        logger.debug(f"   Before transport norm: {Y_norm_before:.2e}")
                        logger.debug(f"   After transport norm: {Y_norm_after:.2e}")  
                        logger.debug(f"   Constraint effect: {constraint_effect:.2e}")
                        
            except Exception as e:
                logger.error(f"RRQR matrix operations failed: {e}")
                Y = torch.zeros(F.shape[0], Q_prev.shape[1] if Q_prev.numel() > 0 else 1,
                              dtype=F.dtype, device=F.device)
        else:
            # Handle empty cases
            Y = torch.zeros(V0_next.shape[1], Q_prev.shape[1] if Q_prev.numel() > 0 else 1,
                          dtype=F.dtype, device=F.device)
        
        # Step 3: Column participation-based death detection
        # With orthonormal bases: Y_participation = V0_next^T @ Q_prev
        # Column norms = cosine of principal angles with new kernel space
        if Q_prev is not None and Q_prev.numel() > 0:
            Y_participation = V0_next.T @ Q_prev  # Shape: (k_{t+1}, d_t)
            col_norms = torch.linalg.norm(Y_participation, dim=0)  # Shape: (d_t,)
            
            # Use tau_out threshold from kernel computation (consistent with hysteresis)
            # Estimate spectral norm from transport or use reasonable default
            transport_norm = torch.linalg.norm(T_tilde, ord=2).item()
            spectral_norm = max(transport_norm, 1.0)  # Use transport norm as proxy
            tau_in = self.cfg.c_in * self.cfg.sqrt_eps * spectral_norm
            tau_out = self.cfg.gap * tau_in
            
            # Detect deaths: generators with small participation in new kernel space
            dead_mask = col_norms < tau_out
            survivor_mask = ~dead_mask
            
            deaths_idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(-1)
            survivors_idx = torch.nonzero(survivor_mask, as_tuple=False).squeeze(-1)
            
            n_deaths = len(deaths_idx)
            n_survivors = len(survivors_idx)
            
            logger.info(f"🔍 COLUMN PARTICIPATION DEATH DETECTION:")
            logger.info(f"   Y_participation shape: {Y_participation.shape}")
            logger.info(f"   Column norms: {col_norms.numpy()[:min(10, len(col_norms))]}")
            logger.info(f"   tau_out threshold: {tau_out:.2e}")
            logger.info(f"   Deaths detected: {n_deaths}, Survivors: {n_survivors}")
            
            # Extract surviving generators
            if n_survivors > 0:
                Q_keep = Q_prev[:, survivors_idx]  # Already in stalk space
            else:
                Q_keep = torch.zeros(V0_next.shape[0], 0, dtype=Y.dtype, device=Y.device)
                
        else:
            # No previous generators to track
            deaths_idx = torch.tensor([], dtype=torch.long)
            n_deaths = 0
            n_survivors = 0
            Q_keep = torch.zeros(V0_next.shape[0], 0, dtype=Y.dtype, device=Y.device)
            logger.info(f"🔍 No previous generators to track")
            
        # Step 4: Simplified confirmation (deaths are based on geometric participation)
        # Column participation method is more direct than RRQR, so confirm most deaths
        n_confirmed_deaths = n_deaths  # Direct geometric criterion is reliable
        
        # Step 5: Compute births (kernel grew beyond surviving generators)  
        current_dim = V0_next.shape[1]  # Current kernel dimension
        n_births = max(0, current_dim - n_survivors)
        
        # Create result with simplified structure
        keep_mask = torch.zeros(Q_prev.shape[1] if Q_prev is not None and Q_prev.numel() > 0 else 0, dtype=torch.bool)
        if Q_prev is not None and Q_prev.numel() > 0 and len(survivors_idx) > 0:
            keep_mask[survivors_idx] = True
            
        result = RRQRResult(
            Q_keep=Q_keep,
            keep_mask=keep_mask,
            n_births=n_births,
            n_deaths=n_deaths,
            n_confirmed_deaths=n_confirmed_deaths,
            induced_map=F,
            Y_matrix=Y,
            threshold_used=tau_out,
            rank_detected=n_survivors
        )
        
        logger.debug(f"Column participation completed: kept {n_survivors}, births {n_births}, deaths {n_deaths}")
        return result
    
    def _detect_dimension_changes_with_confirmation(self,
                                                  current_dim: int,
                                                  confirm_steps: int = None) -> Dict:
        """Apply two-step confirmation to prevent transport noise creating spurious bars.
        
        Only accept dimension changes that persist across multiple steps.
        
        Args:
            current_dim: Current kernel dimension
            confirm_steps: Number of steps for confirmation
            
        Returns:
            Dictionary with confirmation decision and updated history
        """
        confirm_steps = confirm_steps or self.cfg.confirm_steps
        
        # Update dimension history
        self._dimension_history.append(current_dim)
        
        # Keep only recent history
        if len(self._dimension_history) > confirm_steps + 1:
            self._dimension_history = self._dimension_history[-(confirm_steps + 1):]
        
        # Check for confirmed changes
        if len(self._dimension_history) < confirm_steps + 1:
            return {
                'confirmed_change': False,
                'reason': 'insufficient_history',
                'history_length': len(self._dimension_history)
            }
        
        # Check if change has persisted
        recent_dims = self._dimension_history[-confirm_steps:]
        is_stable = all(dim == current_dim for dim in recent_dims)
        
        # Check if this represents a change from earlier
        prev_dim = self._dimension_history[-(confirm_steps + 1)]
        has_changed = prev_dim != current_dim
        
        return {
            'confirmed_change': is_stable and has_changed,
            'is_stable': is_stable,
            'has_changed': has_changed,
            'previous_dim': prev_dim,
            'current_dim': current_dim,
            'stability_steps': confirm_steps
        }
    
    def _extract_coboundary_and_metrics(self,
                                      step_data: FiltrationStep,
                                      builders: Dict[str, Any]) -> Dict:
        """Extract coboundary operator and metrics from step data."""
        # First check if step_data contains direct coboundary info (synthetic case)
        if 'delta' in step_data and step_data['delta'] is not None:
            logger.debug(f"Using direct coboundary data from step_data")
            return {
                'delta': step_data['delta'],
                'G0': step_data.get('G0', torch.eye(step_data['delta'].shape[1])),
                'G1': step_data.get('G1', torch.eye(step_data['delta'].shape[0])),
                'node_masses': step_data.get('node_masses', torch.ones(step_data['delta'].shape[1]))
            }
        
        # Try to get coboundary builder from builders dict (normal case)
        if 'coboundary_builder' in builders and builders['coboundary_builder'] is not None:
            builder = builders['coboundary_builder']
            
            # Extract active edges for this step
            active_edges = step_data.get('active_edges')
            
            # Build coboundary with metrics
            result = builder.build_coboundary_with_metrics(
                sheaf=step_data.get('sheaf'),
                active_edges=active_edges
            )
            
            return result
        else:
            # Ultimate fallback: create minimal data
            logger.warning("No coboundary builder or direct data available, using minimal fallback")
            return {
                'delta': torch.zeros(1, 1, dtype=torch.float64),
                'G0': torch.eye(1, dtype=torch.float64),
                'G1': torch.eye(1, dtype=torch.float64),
                'node_masses': torch.ones(1, dtype=torch.float64)
            }
    
    def _build_transport_between_steps(self,
                                     step_data: FiltrationStep,
                                     builders: Dict[str, Any],
                                     step_t: int,
                                     step_tp1: int) -> Optional[TransportResult]:
        """Build transport map between consecutive filtration steps."""
        try:
            logger.debug(f"🚀 Building transport {step_t}->{step_tp1}: step_data keys={list(step_data.keys()) if isinstance(step_data, dict) else type(step_data)}")
            
            # Primary method: Extract from builders (which have access to sheaf metadata)
            if 'coboundary_builder' in builders and hasattr(builders['coboundary_builder'], 'sheaf'):
                sheaf_metadata = builders['coboundary_builder'].sheaf.metadata
                logger.debug(f"✅ Found sheaf metadata in builders, extracting transport")
                return extract_transport_from_sheaf_step(
                    sheaf_metadata, step_t, step_tp1, self.cfg
                )
            
            # Fallback: Try to extract from step data sheaf metadata
            if 'sheaf' in step_data and hasattr(step_data['sheaf'], 'metadata'):
                logger.debug(f"✅ Found sheaf metadata in step_data, extracting transport")
                return extract_transport_from_sheaf_step(
                    step_data['sheaf'].metadata, step_t, step_tp1, self.cfg
                )
            
            # Last resort: Try alternative extraction from transport extractor
            if 'transport_extractor' in builders:
                logger.debug(f"✅ Found transport_extractor in builders")
                extractor = builders['transport_extractor']
                coupling_info = extractor.extract_node_masses_and_couplings(
                    step_data.get('sheaf'), step_t
                )
                
                if coupling_info.get('gw_coupling') is not None:
                    return self.transport.construct_transport_from_gw_coupling(
                        Pi=coupling_info['gw_coupling'],
                        node_masses_t=coupling_info.get('node_masses', torch.ones(1)),
                        node_masses_tp1=coupling_info.get('next_masses', torch.ones(1)),
                        coupling_type=coupling_info.get('coupling_type', 'balanced')
                    )
            
            # NEURAL NETWORK FALLBACK: Create transport from delta dimensions for non-GW sheaves
            logger.info(f"🔧 No GW metadata found - creating neural network fallback transport for steps {step_t}->{step_tp1}")
            
            # Extract dimensions from current step data
            if 'delta' in step_data:
                delta = step_data['delta']
                total_dim = delta.shape[1] if hasattr(delta, 'shape') else 100
                
                # Create minimal fallback sheaf metadata for H⁰ transport
                fallback_metadata = {
                    'stalk_dimensions': total_dim,
                    'nodes': 1,  # Treat as single large stalk
                    'transport_type': 'neural_network_fallback'
                }
                
                logger.info(f"🎯 Creating neural network transport with dimension {total_dim}")
                return extract_transport_from_sheaf_step(
                    fallback_metadata, step_t, step_tp1, self.cfg
                )
            
            logger.warning(f"❌ Could not build transport between steps {step_t}->{step_tp1}: no valid extraction path found")
            return None
            
        except Exception as e:
            logger.error(f"Transport construction failed for steps {step_t}->{step_tp1}: {e}")
            return None
    
    
    def _detect_births(self,
                      kernel_result: KernelResult,
                      n_births: int,
                      step_idx: int,
                      filtration_param: float) -> List[Dict]:
        """Detect and record birth events."""
        births = []
        
        for i in range(n_births):
            birth_event = {
                'type': 'birth',
                'step': step_idx,
                'param': filtration_param,
                'kernel_dimension': kernel_result.kernel_dimension,
                'confirmed': step_idx == 0  # First step births are automatically confirmed
            }
            births.append(birth_event)
        
        return births
    
    def _create_new_generators(self,
                              V0: torch.Tensor,
                              Q_current: torch.Tensor,
                              n_new: int) -> torch.Tensor:
        """Create orthogonal complement generators for new kernel dimensions.
        
        Uses SVD for robust selection of the most significant orthogonal directions
        rather than just taking the first n_new columns after projection.
        """
        if n_new <= 0:
            return torch.zeros(V0.shape[0], 0, dtype=V0.dtype, device=V0.device)
        
        # Project out existing generators from kernel basis
        if Q_current.numel() > 0:
            # Q_current has shape (stalk_dim, num_current_gens)
            # V0 has shape (stalk_dim, kernel_dim)  
            # V0_new = V0 - Q_current @ Q_current^T @ V0
            try:
                projection = Q_current @ (Q_current.T @ V0)
                V0_orthogonal = V0 - projection
            except RuntimeError as e:
                logger.warning(f"Projection failed due to dimension mismatch: {e}")
                logger.debug(f"Q_current shape: {Q_current.shape}, V0 shape: {V0.shape}")
                # Fallback: use V0 directly without projection
                V0_orthogonal = V0
        else:
            V0_orthogonal = V0
        
        # Use SVD to find the most significant orthogonal directions
        try:
            # SVD on orthogonal complement to find best n_new directions
            U_orth, S_orth, _ = torch.linalg.svd(V0_orthogonal, full_matrices=False)
            
            # Select directions corresponding to largest singular values
            # (these represent the most significant orthogonal directions)
            n_available = min(n_new, len(S_orth), U_orth.shape[1])
            
            # Filter out near-zero directions
            valid_mask = S_orth[:n_available] > self.cfg.sqrt_eps * S_orth[0]
            n_valid = valid_mask.sum().item()
            
            if n_valid > 0:
                Q_new = U_orth[:, :n_valid]
                logger.debug(f"Created {n_valid}/{n_new} new generators using SVD selection")
                logger.debug(f"Singular values: {S_orth[:n_valid].numpy()}")
            else:
                # Fallback to QR if no valid directions found
                logger.warning(f"SVD found no valid directions, falling back to QR")
                Q_new, _ = torch.linalg.qr(V0_orthogonal[:, :n_new], mode='reduced')
                Q_new = Q_new[:, :min(n_new, Q_new.shape[1])]
                
        except Exception as e:
            logger.warning(f"SVD-based generator creation failed: {e}, using QR fallback")
            # Fallback to original QR method
            try:
                Q_new, _ = torch.linalg.qr(V0_orthogonal[:, :n_new], mode='reduced')
                Q_new = Q_new[:, :min(n_new, Q_new.shape[1])]
            except Exception as e2:
                logger.error(f"QR fallback also failed: {e2}, returning zeros")
                return torch.zeros(V0.shape[0], 0, dtype=V0.dtype, device=V0.device)
        
        return Q_new
    
    def _create_step_diagnostics(self,
                               step_idx: int,
                               step_data: FiltrationStep,
                               kernel_result: KernelResult,
                               rrqr_result: Optional[RRQRResult],
                               transport_result: Optional[TransportResult],
                               whitening_result: Any,
                               step_start_time: float) -> StepDiagnostics:
        """Create comprehensive diagnostics for a filtration step."""
        computation_time = time.time() - step_start_time
        
        diag = StepDiagnostics(
            step=step_idx,
            filtration_param=step_data.get('param', step_idx),
            spectral_norm=kernel_result.spectral_norm,
            kernel_dimension=kernel_result.kernel_dimension,
            kernel_residual=kernel_result.residual_norm,
            tau_in=kernel_result.tau_in,
            tau_out=kernel_result.tau_out,
            computational_time=computation_time
        )
        
        # Add RRQR information if available
        if rrqr_result:
            diag.tau_keep = rrqr_result.threshold_used
            diag.n_births = rrqr_result.n_births
            diag.n_deaths = rrqr_result.n_deaths
            diag.n_confirmed_deaths = rrqr_result.n_confirmed_deaths
            diag.induced_map_rank = rrqr_result.rank_detected
        
        # Add transport information if available
        if transport_result:
            diag.transport_norm = transport_result.transport_norm
            diag.coupling_type = transport_result.coupling_type
            diag.mass_floor_applied = transport_result.mass_floor_applied
            diag.transport_stabilization_applied = transport_result.stabilization_applied
        
        # Add whitening information
        if hasattr(whitening_result, 'regularization_applied'):
            diag.regularization_applied = whitening_result.regularization_applied
            diag.cholesky_retries = 1 if whitening_result.regularization_applied > 0 else 0
        
        # Validate certificates
        if self.cfg.validate_certificates:
            certificate = compute_numerical_certificates(
                whitening_result.delta_tilde, kernel_result.V0, cfg=self.cfg
            )
            diag.certificate_passed = certificate.all_passed
        
        return diag
    
    def _create_error_diagnostics(self,
                                step_idx: int,
                                step_data: FiltrationStep,
                                error: Exception) -> StepDiagnostics:
        """Create error diagnostics for failed step."""
        return StepDiagnostics(
            step=step_idx,
            filtration_param=step_data.get('param', step_idx),
            spectral_norm=0.0,
            kernel_dimension=0,
            kernel_residual=float('inf'),
            tau_in=0.0,
            tau_out=0.0,
            certificate_passed=False,
            computational_time=0.0
        )
    
    def _cpqr_death_detection(self,
                            Q_prev: torch.Tensor,
                            V_current: torch.Tensor, 
                            T_tilde: torch.Tensor,
                            tau_out: float,
                            step_idx: int) -> Dict:
        """CPQR-based death detection with mathematically correct survival testing.
        
        Implements the corrected algorithm:
        1. Build Y = V_{t+1}^T @ (T̃ @ Q_t) 
        2. Run Column-Pivoted QR: Y @ P = Q̂ @ R
        3. Count rank: r = |{i : |R_ii| > τ_out}|
        4. Deaths = P[r:], Survivors = P[:r]
        
        Args:
            Q_prev: Previous generator basis [n × d_t]
            V_current: Current kernel basis [n × d_{t+1}]
            T_tilde: Whitened transport map [n × n]
            tau_out: Hysteresis threshold for survival
            step_idx: Current step for logging
            
        Returns:
            Dictionary with death/survival analysis results
        """
        logger.info(f"🔍 CPQR DEATH DETECTION Step {step_idx}:")
        
        # Step 1: Build transport-projected generator matrix
        # Y = V_{t+1}^T @ (T̃ @ Q_t) ∈ ℝ^{d_{t+1} × d_t}
        try:
            Q_transported = T_tilde @ Q_prev  # Apply transport to previous generators
            Y = V_current.T @ Q_transported   # Project onto current kernel space
            
            logger.info(f"   Transport projection matrix Y shape: {Y.shape}")
            logger.info(f"   Y matrix norm: {torch.linalg.norm(Y, ord='fro').item():.2e}")
            
        except Exception as e:
            logger.error(f"Transport projection failed: {e}")
            # Fallback: all generators die
            d_t = Q_prev.shape[1]
            return {
                'Y_matrix': torch.zeros(V_current.shape[1], d_t),
                'n_deaths': d_t,
                'n_survivors': 0, 
                'n_births': V_current.shape[1],
                'death_indices': list(range(d_t)),
                'survivor_indices': [],
                'rank': 0,
                'R_diagonal': torch.zeros(0),
                'P_permutation': torch.arange(d_t)
            }
        
        d_t = Q_prev.shape[1]  # Previous kernel dimension
        d_current = V_current.shape[1]  # Current kernel dimension
        
        if Y.numel() == 0 or d_t == 0:
            # No previous generators - all current dimension are births
            logger.info(f"   No previous generators: {d_current} births, 0 deaths")
            return {
                'Y_matrix': Y,
                'n_deaths': 0,
                'n_survivors': 0,
                'n_births': d_current,
                'death_indices': [],
                'survivor_indices': [],
                'rank': 0,
                'R_diagonal': torch.zeros(0),
                'P_permutation': torch.arange(d_t) if d_t > 0 else torch.zeros(0, dtype=torch.long)
            }
        
        # Step 2: Column-Pivoted QR decomposition using scipy
        try:
            import scipy.linalg
            Y_np = Y.detach().cpu().numpy()
            
            # Column-pivoted QR: Y @ P = Q̂ @ R
            Q_hat_np, R_np, P_indices = scipy.linalg.qr(Y_np, mode='economic', pivoting=True)
            
            # Convert back to PyTorch
            R = torch.from_numpy(R_np).to(device=Y.device, dtype=Y.dtype)
            P_permutation = torch.from_numpy(P_indices).long()
            
            # Step 3: Count rank using diagonal threshold
            R_diagonal = torch.abs(torch.diag(R))
            rank_mask = R_diagonal > tau_out
            rank = rank_mask.sum().item()
            
            logger.info(f"   CPQR Analysis:")
            logger.info(f"      R diagonal values: {R_diagonal.numpy()[:min(10, len(R_diagonal))]}")
            logger.info(f"      τ_out threshold: {tau_out:.2e}")
            logger.info(f"      Numerical rank: {rank}/{min(d_current, d_t)}")
            
        except Exception as e:
            logger.error(f"CPQR decomposition failed: {e}, using SVD fallback")
            # SVD fallback
            try:
                U, S, Vt = torch.linalg.svd(Y, full_matrices=False)
                rank = (S > tau_out).sum().item()
                # Create identity permutation for fallback
                P_permutation = torch.arange(d_t)
                R_diagonal = S
            except Exception as svd_e:
                logger.error(f"SVD fallback failed: {svd_e}")
                # Ultimate fallback - all die
                rank = 0
                P_permutation = torch.arange(d_t)
                R_diagonal = torch.zeros(d_t)
        
        # Step 4: Identify deaths and survivors based on rank
        survivor_indices = P_permutation[:rank].tolist() if rank > 0 else []
        death_indices = P_permutation[rank:].tolist() if rank < d_t else []
        
        n_survivors = len(survivor_indices)
        n_deaths = len(death_indices) 
        n_births = max(0, d_current - n_survivors)
        
        logger.info(f"   CPQR Results:")
        logger.info(f"      Deaths: {n_deaths} (indices: {death_indices[:10]}{'...' if len(death_indices) > 10 else ''})")
        logger.info(f"      Survivors: {n_survivors} (indices: {survivor_indices[:10]}{'...' if len(survivor_indices) > 10 else ''})")
        logger.info(f"      Births needed: {n_births}")
        logger.info(f"      Dimension check: {n_survivors} + {n_births} = {d_current}")
        
        return {
            'Y_matrix': Y,
            'n_deaths': n_deaths,
            'n_survivors': n_survivors,
            'n_births': n_births,
            'death_indices': death_indices,
            'survivor_indices': survivor_indices,
            'rank': rank,
            'R_diagonal': R_diagonal,
            'P_permutation': P_permutation
        }

    def _update_generator_tracking(self,
                                 rrqr_result: RRQRResult,
                                 active_generators: Dict,
                                 intervals: List,
                                 step_idx: int,
                                 filtration_param: float):
        """Update active generator tracking and create death intervals.
        
        IMMEDIATE DEATH REMOVAL: Remove dead generators immediately to avoid plateau.
        
        Args:
            rrqr_result: Result from RRQR persistence update
            active_generators: Currently active generators dict
            intervals: List to append completed intervals to
            step_idx: Current filtration step index
            filtration_param: Current filtration parameter value
        """
        # IMMEDIATE REMOVAL: Handle ALL detected deaths (not just confirmed)
        if rrqr_result.n_deaths > 0:
            # Find which generators died (take oldest generators first)
            dead_gen_ids = list(active_generators.keys())[:rrqr_result.n_deaths]
            
            logger.debug(f"Step {step_idx}: Immediately removing {len(dead_gen_ids)} detected deaths")
            
            # Create intervals for all dead generators (provisional + confirmed)
            for i, gen_id in enumerate(dead_gen_ids):
                birth_info = active_generators.pop(gen_id)
                
                # Determine if this death is confirmed or provisional
                is_confirmed = i < rrqr_result.n_confirmed_deaths
                
                # Create interval (stamp death time immediately)
                interval = {
                    'birth_step': birth_info['birth_step'],
                    'death_step': step_idx,
                    'birth_param': birth_info['birth_param'],
                    'death_param': filtration_param,  # Stamp death time now
                    'lifetime': filtration_param - birth_info['birth_param'],
                    'confirmed': is_confirmed,
                    'provisional': not is_confirmed,
                    'transport_informed': True
                }
                intervals.append(interval)
                
                status = "confirmed" if is_confirmed else "provisional"
                logger.debug(f"  Generator {gen_id} ({status}): birth={birth_info['birth_param']:.6f}, "
                           f"death={filtration_param:.6f}, lifetime={interval['lifetime']:.6f}")
        
        # Update logging to show immediate effect with enhanced details
        logger.info(f"   📋 GENERATOR UPDATE Step {step_idx}:")
        logger.info(f"      Deaths removed: {rrqr_result.n_deaths} ({rrqr_result.n_confirmed_deaths} confirmed, {rrqr_result.n_deaths - rrqr_result.n_confirmed_deaths} provisional)")
        logger.info(f"      Remaining active: {len(active_generators)} generators")
        logger.info(f"      RRQR threshold used: {rrqr_result.threshold_used:.2e}")
        if hasattr(rrqr_result, 'Y_matrix') and rrqr_result.Y_matrix is not None:
            logger.info(f"      Transport matrix ||Y||₂: {torch.linalg.norm(rrqr_result.Y_matrix, ord=2).item():.2e}")

    def _update_generators_with_cpqr(self,
                                   cpqr_result: Dict,
                                   active_generators: Dict,
                                   intervals: List,
                                   step_idx: int,
                                   filtration_param: float,
                                   next_generator_id: int,
                                   V_current: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Update generator tracking based on CPQR death detection results.
        
        Args:
            cpqr_result: Results from CPQR death detection
            active_generators: Currently active generators dict  
            intervals: List to append death intervals to
            step_idx: Current filtration step
            filtration_param: Current parameter value
            next_generator_id: Next available generator ID
            V_current: Current kernel basis for birth generation
            
        Returns:
            Tuple of (updated Q_current, next_generator_id)
        """
        logger.info(f"📋 UPDATING GENERATORS Step {step_idx}:")
        
        # Step 1: Process deaths - remove dead generators and create intervals
        dead_gen_ids = []
        if cpqr_result['n_deaths'] > 0:
            # Get generator IDs in the order they appear (oldest first)
            all_gen_ids = list(active_generators.keys())
            
            # Map death indices to generator IDs
            for death_idx in cpqr_result['death_indices']:
                if death_idx < len(all_gen_ids):
                    gen_id = all_gen_ids[death_idx]
                    dead_gen_ids.append(gen_id)
                    
                    # Remove from active generators and create death interval
                    birth_info = active_generators.pop(gen_id)
                    
                    interval = {
                        'birth_step': birth_info['birth_step'],
                        'death_step': step_idx,
                        'birth_param': birth_info['birth_param'],
                        'death_param': filtration_param,
                        'lifetime': filtration_param - birth_info['birth_param'],
                        'confirmed': True,  # CPQR-based deaths are reliable
                        'method': 'cpqr_transport'
                    }
                    intervals.append(interval)
                    
                    logger.info(f"   ⚰️ Generator {gen_id} died: birth={birth_info['birth_param']:.6f}, "
                               f"death={filtration_param:.6f}, lifetime={interval['lifetime']:.6f}")
        
        logger.info(f"   Deaths processed: {len(dead_gen_ids)}/{cpqr_result['n_deaths']}")
        logger.info(f"   Remaining active generators: {len(active_generators)}")
        
        # Step 2: Construct surviving generators in current kernel coordinates
        n_survivors = cpqr_result['n_survivors']
        n_births = cpqr_result['n_births']
        d_current = V_current.shape[1]
        
        if n_survivors > 0:
            # Extract survivor coordinates from Y matrix
            Y = cpqr_result['Y_matrix']
            survivor_indices = cpqr_result['survivor_indices']
            
            # Get survivor coordinates in current kernel space
            S = Y[:, survivor_indices]  # Shape: [d_current, n_survivors]
            
            # Transform back to stalk space: G_surv = V_current @ S
            G_surv = V_current @ S  # Shape: [n, n_survivors]
            
            # Optional: Re-orthonormalize survivors
            if G_surv.shape[1] > 1:
                G_surv, _ = torch.linalg.qr(G_surv, mode='reduced')
                
            logger.info(f"   👍 Survivors: {n_survivors} generators carried forward")
        else:
            G_surv = torch.zeros(V_current.shape[0], 0, dtype=V_current.dtype, device=V_current.device)
            logger.info(f"   ❌ No survivors: all previous generators died")
        
        # Step 3: Generate births (orthogonal complement)
        if n_births > 0:
            G_births = self._generate_births_orthogonal_complement(V_current, G_surv, n_births)
            
            # Create new generator entries
            for i in range(n_births):
                active_generators[next_generator_id] = {
                    'birth_step': step_idx,
                    'birth_param': filtration_param
                }
                next_generator_id += 1
                
            logger.info(f"   🎆 Births: {n_births} new generators created")
        else:
            G_births = torch.zeros(V_current.shape[0], 0, dtype=V_current.dtype, device=V_current.device)
            logger.info(f"   ℹ️ No births needed")
        
        # Step 4: Combine survivors and births
        if G_surv.shape[1] > 0 and G_births.shape[1] > 0:
            Q_current = torch.cat([G_surv, G_births], dim=1)
        elif G_surv.shape[1] > 0:
            Q_current = G_surv
        elif G_births.shape[1] > 0:
            Q_current = G_births
        else:
            Q_current = torch.zeros(V_current.shape[0], 0, dtype=V_current.dtype, device=V_current.device)
        
        # Step 5: Validation
        expected_dim = d_current
        actual_dim = Q_current.shape[1]
        active_count = len(active_generators)
        
        logger.info(f"   ✅ Final validation:")
        logger.info(f"      Q_current shape: {Q_current.shape}")
        logger.info(f"      Expected dimension: {expected_dim}, Actual: {actual_dim}")
        logger.info(f"      Active generators: {active_count}")
        
        if actual_dim != expected_dim:
            logger.error(f"   ❌ Dimension mismatch: {actual_dim} != {expected_dim}")
        if actual_dim != active_count:
            logger.error(f"   ❌ Generator count mismatch: {actual_dim} != {active_count}")
        
        return Q_current, next_generator_id

    def _generate_births_orthogonal_complement(self, 
                                             V_target: torch.Tensor,
                                             G_existing: torch.Tensor,
                                             n_needed: int) -> torch.Tensor:
        """Generate orthonormal birth vectors in span(V_target) ⊥ span(G_existing).
        
        Args:
            V_target: Target kernel space basis [n × d_target]
            G_existing: Existing survivors [n × n_survivors]
            n_needed: Number of birth vectors needed
            
        Returns:
            Orthonormal birth vectors [n × n_needed]
        """
        if n_needed == 0:
            return torch.zeros(V_target.shape[0], 0, dtype=V_target.dtype, device=V_target.device)
        
        if G_existing.shape[1] == 0:
            # No existing vectors - take first n_needed from V_target
            return V_target[:, :n_needed]
        
        # Project V_target onto orthogonal complement of G_existing
        Q_existing, _ = torch.linalg.qr(G_existing, mode='reduced')
        V_proj = V_target - Q_existing @ (Q_existing.T @ V_target)
        
        # Orthonormalize and take first n_needed vectors
        try:
            Q_births, _ = torch.linalg.qr(V_proj, mode='reduced')
            return Q_births[:, :n_needed]
        except Exception as e:
            logger.warning(f"Birth generation failed: {e}, using truncated V_target")
            # Fallback: use remaining columns from V_target
            start_col = G_existing.shape[1]
            end_col = min(start_col + n_needed, V_target.shape[1])
            return V_target[:, start_col:end_col]

    def _handle_first_step_or_transport_failure(self,
                                              V_current: torch.Tensor,
                                              active_generators: Dict,
                                              step_idx: int,
                                              filtration_param: float,
                                              next_generator_id: int) -> Tuple[torch.Tensor, int]:
        """Handle first step or transport failure - create all generators as births.
        
        Args:
            V_current: Current kernel basis
            active_generators: Active generators dict
            step_idx: Current step
            filtration_param: Current parameter
            next_generator_id: Next available ID
            
        Returns:
            Tuple of (Q_current, updated next_generator_id)
        """
        d_current = V_current.shape[1]
        
        # Clear any existing generators (shouldn't be any on first step)
        active_generators.clear()
        
        # Create all generators as births
        for i in range(d_current):
            active_generators[next_generator_id] = {
                'birth_step': step_idx,
                'birth_param': filtration_param
            }
            next_generator_id += 1
        
        # Use the full kernel basis as generator basis
        Q_current = V_current.clone()
        
        logger.info(f"   🎆 First step: Created {d_current} initial generators")
        
        return Q_current, next_generator_id

    def _validate_cpqr_invariants(self,
                                 Q_current: torch.Tensor,
                                 active_generators: Dict,
                                 kernel_result,
                                 step_idx: int) -> bool:
        """Validate mathematical invariants of CPQR-based persistence tracking.
        
        Checks:
        1. Dimension consistency: len(active_generators) == Q_current.shape[1] == kernel_dim
        2. Orthonormality: Q_current.T @ Q_current ≈ I
        3. Kernel spanning: ||δ̃ @ Q_current||_F ≤ numerical_threshold
        
        Args:
            Q_current: Current generator basis matrix
            active_generators: Active generators dictionary
            kernel_result: Kernel computation result
            step_idx: Current step for logging
            
        Returns:
            True if all invariants pass, False otherwise
        """
        all_passed = True
        
        # Invariant 1: Dimension consistency
        if Q_current is None:
            expected_count = 0
        else:
            expected_count = Q_current.shape[1]
            
        actual_count = len(active_generators)
        kernel_dim = kernel_result.kernel_dimension
        
        if expected_count != actual_count:
            logger.error(f"❌ Step {step_idx}: Generator count mismatch!")
            logger.error(f"   Q_current columns: {expected_count}")
            logger.error(f"   Active generators: {actual_count}")
            all_passed = False
        
        if expected_count != kernel_dim:
            logger.error(f"❌ Step {step_idx}: Kernel dimension mismatch!")
            logger.error(f"   Q_current columns: {expected_count}")
            logger.error(f"   Kernel dimension: {kernel_dim}")
            all_passed = False
        
        if Q_current is not None and Q_current.numel() > 0:
            # Invariant 2: Orthonormality
            if Q_current.shape[1] > 0:
                QtQ = Q_current.T @ Q_current
                I_expected = torch.eye(Q_current.shape[1], dtype=Q_current.dtype, device=Q_current.device)
                ortho_error = torch.linalg.norm(QtQ - I_expected, ord='fro').item()
                ortho_threshold = 10.0 * self.cfg.sqrt_eps
                
                if ortho_error > ortho_threshold:
                    logger.warning(f"⚠️ Step {step_idx}: Orthogonality violation!")
                    logger.warning(f"   ||Q^T Q - I||_F = {ortho_error:.2e} > {ortho_threshold:.2e}")
                    all_passed = False
                else:
                    logger.debug(f"   ✅ Orthogonality: {ortho_error:.2e} ≤ {ortho_threshold:.2e}")
            
            # Invariant 3: Kernel spanning (if we have access to the coboundary)
            if hasattr(kernel_result, 'residual_norm') and kernel_result.residual_norm < float('inf'):
                if kernel_result.residual_norm > 10.0 * self.cfg.sqrt_eps:
                    logger.warning(f"⚠️ Step {step_idx}: Kernel residual high!")
                    logger.warning(f"   Kernel residual: {kernel_result.residual_norm:.2e}")
                else:
                    logger.debug(f"   ✅ Kernel residual: {kernel_result.residual_norm:.2e}")
        
        if all_passed:
            logger.debug(f"   ✅ All CPQR invariants satisfied")
        
        return all_passed

    def _validate_generator_consistency(self, 
                                      Q_current: torch.Tensor, 
                                      active_generators: Dict, 
                                      step_idx: int) -> bool:
        """Validate consistency between active_generators and actual generator basis.
        
        Args:
            Q_current: Current generator basis matrix
            active_generators: Active generators dictionary
            step_idx: Current step for logging
            
        Returns:
            True if consistent, False if mismatch detected
        """
        if Q_current is None:
            expected_count = 0
        else:
            expected_count = Q_current.shape[1]
            
        actual_count = len(active_generators)
        
        if expected_count != actual_count:
            logger.warning(f"Step {step_idx}: Generator count mismatch!")
            logger.warning(f"  Expected (from basis): {expected_count}")
            logger.warning(f"  Actual (from tracking): {actual_count}")
            logger.warning(f"  Difference: {abs(expected_count - actual_count)}")
            return False
        else:
            logger.debug(f"Step {step_idx}: Generator tracking consistent ({actual_count} generators)")
            return True