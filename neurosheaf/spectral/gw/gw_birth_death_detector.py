# neurosheaf/spectral/gw/gw_birth_death_detector.py
"""
GW-aware birth-death event detection with proper semantics.

Implements birth-death event detection for Gromov-Wasserstein filtrations
with increasing complexity semantics. Handles the unique interpretation
of eigenvalue evolution in GW context where low cost thresholds correspond
to sparse structures and high cost thresholds to dense structures.

Mathematical Foundation:
- GW Increasing Filtration: parameter increases → more edges → higher connectivity
- Birth: eigenvalue appears as structure emerges (sparse → connected)
- Death: eigenvalue disappears due to structural merging (connected → over-connected)
- Lifetime: death_cost - birth_cost (always positive for valid pairs)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from ...utils.logging import setup_logger
from ...utils.exceptions import ComputationError

logger = setup_logger(__name__)


class GWBirthDeathDetector:
    """
    GW-aware birth-death event detection with proper semantics.
    
    Handles the increasing complexity semantics of GW filtrations where
    eigenvalue evolution follows different patterns than standard decreasing
    complexity filtrations.
    """
    
    def __init__(self, 
                 eigenvalue_threshold: float = 1e-10,
                 lifetime_threshold: float = 1e-8,
                 validate_semantics: bool = True):
        """
        Initialize GW birth-death detector.
        
        Args:
            eigenvalue_threshold: Threshold for considering eigenvalues as zero
            lifetime_threshold: Minimum lifetime for valid persistence pairs
            validate_semantics: Whether to validate GW-specific semantics
        """
        self.eigenvalue_threshold = eigenvalue_threshold
        self.lifetime_threshold = lifetime_threshold  
        self.validate_semantics = validate_semantics
        
        logger.info(f"GWBirthDeathDetector initialized: eigenval_thresh={eigenvalue_threshold}, "
                   f"lifetime_thresh={lifetime_threshold}, validate={validate_semantics}")
    
    def detect_gw_events(self, 
                        matches: List[Tuple[int, int, float]],
                        prev_eigenvals: torch.Tensor,
                        curr_eigenvals: torch.Tensor,
                        prev_param: float,
                        curr_param: float,
                        step: int) -> Dict[str, List[Dict]]:
        """
        Detect GW-specific birth-death events from eigenvalue matching.
        
        GW Semantics:
        - Birth: eigenvalue appears as connectivity increases (low → high cost)
        - Death: eigenvalue disappears due to over-connection (high cost)
        - Parameters always increase: prev_param < curr_param
        
        Args:
            matches: List of (prev_idx, curr_idx, similarity) matches
            prev_eigenvals: Previous step eigenvalues
            curr_eigenvals: Current step eigenvalues  
            prev_param: Previous filtration parameter (GW cost threshold)
            curr_param: Current filtration parameter (GW cost threshold)
            step: Current step index
            
        Returns:
            Dictionary with birth_events and death_events lists
            
        Raises:
            ComputationError: If parameter semantics are violated
        """
        # Validate GW parameter semantics
        if self.validate_semantics and curr_param <= prev_param:
            raise ComputationError(
                f"GW parameter semantics violation: {prev_param} → {curr_param} "
                f"(parameters must increase for GW filtration)",
                operation="detect_gw_events"
            )
        
        logger.debug(f"Detecting GW events at step {step}: {len(matches)} matches, "
                    f"param {prev_param:.6f} → {curr_param:.6f}")
        
        events = {
            'birth_events': [],
            'death_events': []
        }
        
        # Track which eigenvalues are matched
        matched_prev = set(match[0] for match in matches)
        matched_curr = set(match[1] for match in matches)
        
        # Detect birth events: current eigenvalues that weren't matched
        for curr_idx in range(len(curr_eigenvals)):
            if curr_idx not in matched_curr:
                eigenval = curr_eigenvals[curr_idx].item()
                
                # Only consider significant eigenvalues for birth
                if eigenval > self.eigenvalue_threshold:
                    birth_event = self._create_birth_event(
                        curr_idx, eigenval, curr_param, step
                    )
                    events['birth_events'].append(birth_event)
        
        # Detect death events: previous eigenvalues that weren't matched
        for prev_idx in range(len(prev_eigenvals)):
            if prev_idx not in matched_prev:
                eigenval = prev_eigenvals[prev_idx].item()
                
                # Consider death for any previously significant eigenvalue
                if eigenval > self.eigenvalue_threshold:
                    death_event = self._create_death_event(
                        prev_idx, eigenval, curr_param, step  # Death occurs at current param
                    )
                    events['death_events'].append(death_event)
        
        # Log event statistics
        logger.debug(f"GW events detected: {len(events['birth_events'])} births, "
                    f"{len(events['death_events'])} deaths")
        
        return events
    
    def _create_birth_event(self, 
                          eigenval_idx: int,
                          eigenvalue: float,
                          birth_param: float,
                          step: int) -> Dict[str, Any]:
        """
        Create birth event with GW-specific semantics.
        
        GW Birth Interpretation:
        - Eigenvalue appears as structure becomes more connected
        - Birth parameter is the GW cost threshold where feature emerges
        - Lower birth parameter = earlier emergence = more persistent feature
        
        Args:
            eigenval_idx: Index of eigenvalue in current step
            eigenvalue: Eigenvalue magnitude
            birth_param: GW cost threshold where eigenvalue appears
            step: Current step index
            
        Returns:
            Birth event dictionary
        """
        birth_event = {
            'type': 'birth',
            'eigenval_idx': eigenval_idx,
            'eigenvalue': eigenvalue,
            'birth_param': birth_param,
            'birth_step': step,
            'gw_semantics': 'connectivity_emergence',
            'interpretation': f'Feature emerges at GW cost threshold {birth_param:.6f}'
        }
        
        logger.debug(f"Birth event: eigenval={eigenvalue:.6f} at param={birth_param:.6f}")
        return birth_event
    
    def _create_death_event(self, 
                          eigenval_idx: int,
                          eigenvalue: float,
                          death_param: float,
                          step: int) -> Dict[str, Any]:
        """
        Create death event with GW-specific semantics.
        
        GW Death Interpretation:
        - Eigenvalue disappears as structure becomes over-connected
        - Death parameter is the GW cost threshold where feature vanishes
        - Higher death parameter = later disappearance = more persistent feature
        
        Args:
            eigenval_idx: Index of eigenvalue in previous step
            eigenvalue: Eigenvalue magnitude  
            death_param: GW cost threshold where eigenvalue disappears
            step: Current step index
            
        Returns:
            Death event dictionary
        """
        death_event = {
            'type': 'death',
            'eigenval_idx': eigenval_idx,
            'eigenvalue': eigenvalue,
            'death_param': death_param,
            'death_step': step,
            'gw_semantics': 'connectivity_saturation',
            'interpretation': f'Feature vanishes at GW cost threshold {death_param:.6f}'
        }
        
        logger.debug(f"Death event: eigenval={eigenvalue:.6f} at param={death_param:.6f}")
        return death_event
    
    def create_persistence_pairs(self, 
                                birth_events: List[Dict],
                                death_events: List[Dict],
                                filtration_params: List[float]) -> List[Dict[str, Any]]:
        """
        Create persistence pairs from birth and death events.
        
        Uses heuristic pairing strategies to match births with deaths
        while respecting GW filtration semantics (birth < death).
        
        Args:
            birth_events: List of birth event dictionaries
            death_events: List of death event dictionaries
            filtration_params: Full list of filtration parameters
            
        Returns:
            List of persistence pair dictionaries
        """
        pairs = []
        
        # Simple pairing strategy: match by eigenvalue magnitude proximity
        used_deaths = set()
        
        for birth in birth_events:
            best_death = None
            best_match_score = -1
            
            for i, death in enumerate(death_events):
                if i in used_deaths:
                    continue
                
                # Ensure proper temporal ordering
                if death['death_param'] <= birth['birth_param']:
                    continue  # Invalid pairing for GW semantics
                
                # Compute match score based on eigenvalue similarity
                eigenval_diff = abs(birth['eigenvalue'] - death['eigenvalue'])
                param_separation = death['death_param'] - birth['birth_param']
                
                # Prefer pairs with similar eigenvalues and reasonable lifetimes
                match_score = 1.0 / (1.0 + eigenval_diff) * min(param_separation, 1.0)
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_death = (i, death)
            
            # Create persistence pair if good match found
            if best_death is not None and best_match_score > 0.1:
                death_idx, death_event = best_death
                lifetime = death_event['death_param'] - birth['birth_param']
                
                # Only create pair if lifetime meets threshold
                if lifetime > self.lifetime_threshold:
                    pair = self._create_persistence_pair(birth, death_event, lifetime)
                    pairs.append(pair)
                    used_deaths.add(death_idx)
        
        # Create infinite bars for unmatched births
        for birth in birth_events:
            # Check if this birth was already paired
            birth_used = any(pair['birth_param'] == birth['birth_param'] and 
                           pair['birth_eigenvalue'] == birth['eigenvalue'] 
                           for pair in pairs)
            
            if not birth_used:
                infinite_pair = self._create_infinite_persistence_pair(birth)
                pairs.append(infinite_pair)
        
        logger.debug(f"Created {len(pairs)} persistence pairs from "
                    f"{len(birth_events)} births and {len(death_events)} deaths")
        
        return pairs
    
    def _create_persistence_pair(self, 
                               birth_event: Dict,
                               death_event: Dict,
                               lifetime: float) -> Dict[str, Any]:
        """
        Create finite persistence pair from birth and death events.
        
        Args:
            birth_event: Birth event dictionary
            death_event: Death event dictionary  
            lifetime: Persistence lifetime (death_param - birth_param)
            
        Returns:
            Persistence pair dictionary
        """
        pair = {
            'type': 'finite_pair',
            'birth_param': birth_event['birth_param'],
            'death_param': death_event['death_param'],
            'lifetime': lifetime,
            'birth_step': birth_event['birth_step'],
            'death_step': death_event['death_step'],
            'birth_eigenvalue': birth_event['eigenvalue'],
            'death_eigenvalue': death_event['eigenvalue'],
            'gw_interpretation': (
                f"Feature persists from GW cost {birth_event['birth_param']:.6f} "
                f"to {death_event['death_param']:.6f} (lifetime: {lifetime:.6f})"
            )
        }
        
        return pair
    
    def _create_infinite_persistence_pair(self, birth_event: Dict) -> Dict[str, Any]:
        """
        Create infinite persistence pair for unmatched birth.
        
        Args:
            birth_event: Birth event dictionary
            
        Returns:
            Infinite persistence pair dictionary
        """
        pair = {
            'type': 'infinite_pair',
            'birth_param': birth_event['birth_param'],
            'death_param': float('inf'),
            'lifetime': float('inf'),
            'birth_step': birth_event['birth_step'],
            'death_step': None,
            'birth_eigenvalue': birth_event['eigenvalue'],
            'death_eigenvalue': None,
            'gw_interpretation': (
                f"Feature emerges at GW cost {birth_event['birth_param']:.6f} "
                f"and persists indefinitely"
            )
        }
        
        return pair
    
    def validate_gw_semantics(self, 
                             persistence_pairs: List[Dict],
                             filtration_params: List[float]) -> Dict[str, Any]:
        """
        Validate persistence pairs against GW filtration semantics.
        
        Checks:
        1. Birth < Death for all finite pairs
        2. Parameters are within filtration range
        3. Lifetimes are positive and reasonable
        4. GW-specific interpretation consistency
        
        Args:
            persistence_pairs: List of persistence pair dictionaries
            filtration_params: Filtration parameter sequence
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            min_param = min(filtration_params)
            max_param = max(filtration_params)
            
            valid_pairs = 0
            invalid_pairs = 0
            infinite_pairs = 0
            negative_lifetimes = 0
            
            for pair in persistence_pairs:
                birth_param = pair['birth_param']
                death_param = pair['death_param']
                
                # Check parameter ranges
                if birth_param < min_param or birth_param > max_param:
                    validation['warnings'].append(
                        f"Birth parameter {birth_param:.6f} outside filtration range "
                        f"[{min_param:.6f}, {max_param:.6f}]"
                    )
                
                # Check finite pairs
                if pair['type'] == 'finite_pair':
                    if death_param <= birth_param:
                        validation['errors'].append(
                            f"Invalid GW semantics: birth {birth_param:.6f} >= death {death_param:.6f}"
                        )
                        validation['is_valid'] = False
                        negative_lifetimes += 1
                        invalid_pairs += 1
                    else:
                        valid_pairs += 1
                        
                    # Check reasonable lifetime
                    lifetime = pair['lifetime']
                    if lifetime < self.lifetime_threshold:
                        validation['warnings'].append(
                            f"Very short lifetime: {lifetime:.6f} < {self.lifetime_threshold}"
                        )
                
                elif pair['type'] == 'infinite_pair':
                    infinite_pairs += 1
            
            # Compile statistics
            validation['statistics'] = {
                'total_pairs': len(persistence_pairs),
                'valid_finite_pairs': valid_pairs,
                'invalid_pairs': invalid_pairs,
                'infinite_pairs': infinite_pairs,
                'negative_lifetimes': negative_lifetimes,
                'validation_rate': valid_pairs / max(len(persistence_pairs), 1)
            }
            
            if validation['is_valid']:
                logger.debug("GW semantics validation passed")
            else:
                logger.warning(f"GW semantics validation failed: {len(validation['errors'])} errors")
            
        except Exception as e:
            validation['errors'].append(f"Validation failed: {e}")
            validation['is_valid'] = False
            logger.error(f"GW semantics validation error: {e}")
        
        return validation
    
    def compute_gw_persistence_statistics(self, 
                                        persistence_pairs: List[Dict]) -> Dict[str, Union[float, int]]:
        """
        Compute statistics specific to GW persistence pairs.
        
        Args:
            persistence_pairs: List of persistence pair dictionaries
            
        Returns:
            Dictionary with GW-specific persistence statistics
        """
        if not persistence_pairs:
            return {'n_pairs': 0}
        
        finite_pairs = [p for p in persistence_pairs if p['type'] == 'finite_pair']
        infinite_pairs = [p for p in persistence_pairs if p['type'] == 'infinite_pair']
        
        stats = {
            'n_pairs': len(persistence_pairs),
            'n_finite_pairs': len(finite_pairs),
            'n_infinite_pairs': len(infinite_pairs)
        }
        
        if finite_pairs:
            lifetimes = [p['lifetime'] for p in finite_pairs]
            birth_params = [p['birth_param'] for p in finite_pairs]
            death_params = [p['death_param'] for p in finite_pairs]
            
            stats.update({
                'mean_lifetime': np.mean(lifetimes),
                'std_lifetime': np.std(lifetimes),
                'max_lifetime': max(lifetimes),
                'min_lifetime': min(lifetimes),
                'total_persistence': sum(lifetimes),
                'mean_birth_param': np.mean(birth_params),
                'mean_death_param': np.mean(death_params),
                'param_range': max(death_params) - min(birth_params)
            })
        
        return stats