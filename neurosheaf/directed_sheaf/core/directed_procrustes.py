"""Directed Procrustes computation for complex restriction maps.

This module implements the computation of directed restriction maps with
complex phase encoding as specified in the directed sheaf mathematical
formulation.

Mathematical Foundation:
- Base Real Maps: Computed via scaled Procrustes analysis
- Source Maps: Real valued s_e Q_e (from Procrustes)
- Target Maps: Complex valued T^{(q)}_{uv} I_{r_v} (phase encoded)
- Directional Encoding: T^{(q)} = exp(i 2π q (A - A^T))

Key Features:
- Integrates with existing scaled_procrustes_whitened function
- Applies complex phase encoding to create directed restrictions
- Maintains orthogonality properties where applicable
- Provides comprehensive validation

The directed restriction maps encode asymmetric network structure through
complex phases while preserving the underlying mathematical optimality
of the Procrustes solution.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import networkx as nx

# Import existing procrustes functionality
from ...sheaf.core.procrustes import scaled_procrustes_whitened
from ...sheaf.core.whitening import WhiteningProcessor
from ...sheaf.data_structures import Sheaf

# Import directed sheaf components
from .directional_encoding import DirectionalEncodingComputer

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


class DirectedProcrustesComputer:
    """Computes directed restriction maps with complex phase encoding.
    
    This class implements the computation of directed restriction maps by
    combining the existing scaled Procrustes analysis with complex phase
    encoding from the directional encoding matrix.
    
    Mathematical Formulation:
    For edge e = (u, v):
    - Source map: F_{u ⟵ e} = s_e Q_e (real, from Procrustes)
    - Target map: F_{v ⟵ e} = T^{(q)}_{uv} I_{r_v} (complex, phase encoded)
    
    Where:
    - (s_e, Q_e) comes from scaled Procrustes analysis
    - T^{(q)}_{uv} is the directional encoding matrix entry
    - I_{r_v} is the identity matrix of dimension r_v
    
    Key Properties:
    - Preserves optimality of Procrustes solution
    - Adds directional information through complex phases
    - Maintains orthogonality where mathematically appropriate
    - Reduces to undirected case when q = 0
    """
    
    def __init__(self, directionality_parameter: float = 0.25, 
                 validate_restrictions: bool = True, 
                 tolerance: float = 1e-12,
                 device: Optional[torch.device] = None):
        """Initialize the directed Procrustes computer.
        
        Args:
            directionality_parameter: q parameter for directional encoding
            validate_restrictions: Whether to validate restriction properties
            tolerance: Tolerance for numerical validation
            device: PyTorch device for computations
        """
        self.q = directionality_parameter
        self.validate_restrictions = validate_restrictions
        self.tolerance = tolerance
        self.device = device or torch.device('cpu')
        
        # Initialize directional encoding computer
        self.encoding_computer = DirectionalEncodingComputer(
            q=self.q, 
            validate_properties=validate_restrictions,
            tolerance=tolerance,
            device=device
        )
        
        logger.debug(f"DirectedProcrustesComputer initialized with q={self.q}")
    
    def compute_directed_restrictions(
        self, 
        base_restrictions: Dict[Tuple[str, str], torch.Tensor],
        encoding_matrix: torch.Tensor,
        node_mapping: Dict[str, int],
        stalk_dimensions: Optional[Dict[str, int]] = None
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute complex directed restrictions from real base restrictions.
        
        Applies directional encoding to transform real restriction maps into
        complex directed restriction maps. The source maps remain real while
        target maps acquire complex phase encoding.
        
        Args:
            base_restrictions: Real restriction maps from Procrustes analysis
            encoding_matrix: Directional encoding matrix T^{(q)}
            node_mapping: Mapping from node names to matrix indices
            stalk_dimensions: Optional dimensions for each stalk
            
        Returns:
            Dictionary mapping edges to complex directed restriction tensors
            
        Raises:
            ValueError: If inputs are incompatible
            RuntimeError: If computation fails
        """
        if not isinstance(base_restrictions, dict):
            raise ValueError("base_restrictions must be a dictionary")
        
        if not isinstance(encoding_matrix, torch.Tensor):
            raise ValueError("encoding_matrix must be a torch.Tensor")
        
        if not isinstance(node_mapping, dict):
            raise ValueError("node_mapping must be a dictionary")
        
        # Validate encoding matrix
        if not encoding_matrix.is_complex():
            raise ValueError("encoding_matrix must be complex")
        
        directed_restrictions = {}
        
        for edge, base_restriction in base_restrictions.items():
            try:
                # Apply directional encoding to create complex restriction
                directed_restriction = self._apply_directional_encoding(
                    edge, base_restriction, encoding_matrix, node_mapping, stalk_dimensions
                )
                directed_restrictions[edge] = directed_restriction
                
                logger.debug(f"Computed directed restriction for edge {edge}: {base_restriction.shape} → {directed_restriction.shape}")
                
            except Exception as e:
                logger.error(f"Failed to compute directed restriction for edge {edge}: {e}")
                raise RuntimeError(f"Directed restriction computation failed for edge {edge}: {e}")
        
        # Validate complete set of restrictions
        if self.validate_restrictions:
            self._validate_directed_restrictions(directed_restrictions, base_restrictions, encoding_matrix)
        
        logger.info(f"Computed {len(directed_restrictions)} directed restrictions")
        return directed_restrictions
    
    def compute_from_sheaf(
        self, 
        base_sheaf: Sheaf,
        directionality_parameter: Optional[float] = None
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute directed restrictions from existing sheaf.
        
        Convenience method that computes directed restrictions from an existing
        sheaf by applying directional encoding to the base restrictions.
        
        Args:
            base_sheaf: Existing sheaf with real restrictions
            directionality_parameter: Optional override for q parameter
            
        Returns:
            Dictionary mapping edges to complex directed restriction tensors
        """
        if directionality_parameter is not None:
            # Temporarily change q parameter
            old_q = self.q
            self.q = directionality_parameter
            self.encoding_computer.q = directionality_parameter
            self.encoding_computer.two_pi_q = 2 * torch.pi * directionality_parameter
        
        try:
            # Extract adjacency matrix and compute encoding
            encoding_matrix = self.encoding_computer.compute_from_poset(base_sheaf.poset)
            
            # Create node mapping
            node_mapping = self.encoding_computer.get_node_mapping(base_sheaf.poset)
            
            # Get stalk dimensions
            stalk_dimensions = {
                node: stalk.shape[0] for node, stalk in base_sheaf.stalks.items()
            }
            
            # Compute directed restrictions
            directed_restrictions = self.compute_directed_restrictions(
                base_sheaf.restrictions, encoding_matrix, node_mapping, stalk_dimensions
            )
            
            return directed_restrictions
            
        finally:
            # Restore original q parameter if it was changed
            if directionality_parameter is not None:
                self.q = old_q
                self.encoding_computer.q = old_q
                self.encoding_computer.two_pi_q = 2 * torch.pi * old_q
    
    def _apply_directional_encoding(
        self, 
        edge: Tuple[str, str],
        base_restriction: torch.Tensor,
        encoding_matrix: torch.Tensor,
        node_mapping: Dict[str, int],
        stalk_dimensions: Optional[Dict[str, int]] = None
    ) -> torch.Tensor:
        """Apply directional encoding to create complex restriction map.
        
        Mathematical Implementation:
        For edge e = (u, v):
        - Extract phase factor: T^{(q)}_{uv} from encoding matrix
        - Apply phase encoding: complex_restriction = T^{(q)}_{uv} * base_restriction
        
        Args:
            edge: Edge tuple (source, target)
            base_restriction: Real restriction map from Procrustes
            encoding_matrix: Directional encoding matrix
            node_mapping: Node name to matrix index mapping
            stalk_dimensions: Optional stalk dimensions
            
        Returns:
            Complex directed restriction map
        """
        source, target = edge
        
        # Get matrix indices
        if source not in node_mapping or target not in node_mapping:
            raise ValueError(f"Edge {edge} nodes not found in node_mapping")
        
        source_idx = node_mapping[source]
        target_idx = node_mapping[target]
        
        # Extract phase factor from encoding matrix
        phase_factor = encoding_matrix[source_idx, target_idx]
        
        # Convert base restriction to complex if needed
        if not base_restriction.is_complex():
            base_restriction = base_restriction.to(torch.complex64)
        
        # Apply phase encoding
        # For target map: T^{(q)}_{uv} * I_{r_v} * base_restriction
        # For source map: base_restriction remains real (but converted to complex)
        
        # Apply phase factor to create directed restriction
        directed_restriction = phase_factor * base_restriction
        
        return directed_restriction
    
    def _validate_directed_restrictions(
        self, 
        directed_restrictions: Dict[Tuple[str, str], torch.Tensor],
        base_restrictions: Dict[Tuple[str, str], torch.Tensor],
        encoding_matrix: torch.Tensor
    ) -> None:
        """Validate mathematical properties of directed restrictions.
        
        Args:
            directed_restrictions: Computed directed restrictions
            base_restrictions: Original real restrictions
            encoding_matrix: Directional encoding matrix
            
        Raises:
            RuntimeError: If validation fails
        """
        # Check same number of restrictions
        if len(directed_restrictions) != len(base_restrictions):
            raise RuntimeError("Number of directed restrictions doesn't match base restrictions")
        
        # Check same edges
        if set(directed_restrictions.keys()) != set(base_restrictions.keys()):
            raise RuntimeError("Edge sets don't match between directed and base restrictions")
        
        # Validate each restriction
        for edge in directed_restrictions.keys():
            directed_restriction = directed_restrictions[edge]
            base_restriction = base_restrictions[edge]
            
            # Check complex dtype
            if not directed_restriction.is_complex():
                raise RuntimeError(f"Directed restriction for edge {edge} is not complex")
            
            # Check shapes match
            if directed_restriction.shape != base_restriction.shape:
                raise RuntimeError(f"Shape mismatch for edge {edge}: {directed_restriction.shape} vs {base_restriction.shape}")
            
            # Check magnitude preservation (approximately)
            base_magnitude = torch.norm(base_restriction)
            directed_magnitude = torch.norm(directed_restriction)
            magnitude_ratio = directed_magnitude / (base_magnitude + 1e-12)
            
            if abs(magnitude_ratio - 1.0) > 1e-5:  # More relaxed tolerance for magnitude preservation
                raise RuntimeError(f"Magnitude not preserved for edge {edge}: ratio={magnitude_ratio}")
        
        # Check reduction to undirected case when q = 0
        if abs(self.q) < self.tolerance:
            for edge in directed_restrictions.keys():
                directed_restriction = directed_restrictions[edge]
                base_restriction = base_restrictions[edge]
                
                # Convert base to complex for comparison
                base_complex = base_restriction.to(torch.complex64)
                
                error = torch.abs(directed_restriction - base_complex).max()
                if error > self.tolerance:
                    raise RuntimeError(f"For q=0, directed restriction should equal base restriction for edge {edge}")
        
        logger.debug("Directed restrictions validation passed")
    
    def compute_with_metadata(
        self, 
        base_restrictions: Dict[Tuple[str, str], torch.Tensor],
        encoding_matrix: torch.Tensor,
        node_mapping: Dict[str, int],
        stalk_dimensions: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[Tuple[str, str], torch.Tensor], Dict[str, Any]]:
        """Compute directed restrictions with detailed metadata.
        
        Args:
            base_restrictions: Real restriction maps
            encoding_matrix: Directional encoding matrix
            node_mapping: Node mapping
            stalk_dimensions: Optional stalk dimensions
            
        Returns:
            Tuple of (directed_restrictions, metadata)
        """
        # Compute directed restrictions
        directed_restrictions = self.compute_directed_restrictions(
            base_restrictions, encoding_matrix, node_mapping, stalk_dimensions
        )
        
        # Compute metadata
        metadata = {
            'directionality_parameter': self.q,
            'num_restrictions': len(directed_restrictions),
            'num_base_restrictions': len(base_restrictions),
            'encoding_matrix_shape': encoding_matrix.shape,
            'restriction_analysis': self._analyze_restrictions(directed_restrictions, base_restrictions),
            'validation_passed': self.validate_restrictions,
            'tolerance': self.tolerance,
            'device': str(self.device)
        }
        
        return directed_restrictions, metadata
    
    def _analyze_restrictions(
        self, 
        directed_restrictions: Dict[Tuple[str, str], torch.Tensor],
        base_restrictions: Dict[Tuple[str, str], torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze properties of directed restrictions.
        
        Args:
            directed_restrictions: Complex directed restrictions
            base_restrictions: Real base restrictions
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'num_restrictions': len(directed_restrictions),
            'all_complex': all(r.is_complex() for r in directed_restrictions.values()),
            'shapes_preserved': True,
            'magnitude_ratios': [],
            'phase_statistics': {}
        }
        
        try:
            # Analyze each restriction
            for edge in directed_restrictions.keys():
                directed_restriction = directed_restrictions[edge]
                base_restriction = base_restrictions[edge]
                
                # Check shape preservation
                if directed_restriction.shape != base_restriction.shape:
                    analysis['shapes_preserved'] = False
                
                # Compute magnitude ratio
                base_magnitude = torch.norm(base_restriction)
                directed_magnitude = torch.norm(directed_restriction)
                if base_magnitude > 0:
                    ratio = (directed_magnitude / base_magnitude).item()
                    analysis['magnitude_ratios'].append(ratio)
                
                # Analyze phases
                phases = torch.angle(directed_restriction)
                analysis['phase_statistics'][edge] = {
                    'mean_phase': phases.mean().item(),
                    'std_phase': phases.std().item(),
                    'max_phase': phases.max().item(),
                    'min_phase': phases.min().item()
                }
            
            # Aggregate statistics
            if analysis['magnitude_ratios']:
                analysis['magnitude_ratio_mean'] = np.mean(analysis['magnitude_ratios'])
                analysis['magnitude_ratio_std'] = np.std(analysis['magnitude_ratios'])
            
        except Exception as e:
            analysis['analysis_error'] = str(e)
            logger.warning(f"Restriction analysis failed: {e}")
        
        return analysis
    
    def validate_restriction_orthogonality(
        self, 
        directed_restriction: torch.Tensor,
        check_type: str = 'column'
    ) -> Dict[str, Any]:
        """Validate orthogonality properties of directed restriction.
        
        Args:
            directed_restriction: Complex directed restriction map
            check_type: Type of orthogonality to check ('column' or 'row')
            
        Returns:
            Dictionary with orthogonality analysis
        """
        if not directed_restriction.is_complex():
            raise ValueError("Directed restriction must be complex")
        
        analysis = {
            'check_type': check_type,
            'is_orthogonal': False,
            'orthogonality_error': float('inf'),
            'shape': directed_restriction.shape
        }
        
        # Validate check_type first
        if check_type not in ['column', 'row']:
            raise ValueError(f"Invalid check_type: {check_type}")
        
        try:
            if check_type == 'column':
                # Check R^H R = I (column orthogonal)
                product = directed_restriction.conj().T @ directed_restriction
                identity = torch.eye(product.shape[0], dtype=product.dtype, device=product.device)
                
            elif check_type == 'row':
                # Check R R^H = I (row orthogonal)
                product = directed_restriction @ directed_restriction.conj().T
                identity = torch.eye(product.shape[0], dtype=product.dtype, device=product.device)
            
            # Compute orthogonality error
            error = torch.abs(product - identity).max().item()
            analysis['orthogonality_error'] = error
            analysis['is_orthogonal'] = error <= self.tolerance
            
        except Exception as e:
            analysis['error'] = str(e)
            logger.warning(f"Orthogonality validation failed: {e}")
        
        return analysis
    
    def get_restriction_summary(
        self, 
        directed_restrictions: Dict[Tuple[str, str], torch.Tensor]
    ) -> Dict[str, Any]:
        """Get summary of directed restrictions.
        
        Args:
            directed_restrictions: Complex directed restrictions
            
        Returns:
            Dictionary with restriction summary
        """
        summary = {
            'num_restrictions': len(directed_restrictions),
            'edges': list(directed_restrictions.keys()),
            'shapes': {edge: r.shape for edge, r in directed_restrictions.items()},
            'dtypes': {edge: r.dtype for edge, r in directed_restrictions.items()},
            'total_parameters': sum(r.numel() for r in directed_restrictions.values()),
            'memory_usage_mb': sum(r.numel() * r.element_size() for r in directed_restrictions.values()) / (1024**2),
            'directionality_parameter': self.q
        }
        
        return summary