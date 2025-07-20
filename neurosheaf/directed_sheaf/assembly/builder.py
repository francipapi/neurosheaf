"""Directed sheaf builder for constructing directed cellular sheaves from neural networks.

This module provides the DirectedSheafBuilder class that orchestrates the complete
directed sheaf construction pipeline, extending real sheaves to complex-valued
directed sheaves with Hermitian Laplacians.

Mathematical Foundation:
- Complex Extension: F(v) = R^{r_v} ⊗_R C
- Directional Encoding: T^{(q)} = exp(i 2π q (A - A^T))
- Directed Restrictions: Complex-valued maps with phase encoding
- Hermitian Laplacian: L^{F} = δ* δ

The builder follows the mathematical formulation in:
docs/DirectedSheaf_mathematicalFormulation.md Section 2-3
"""

import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

# Import base sheaf structures
from ...sheaf.data_structures import Sheaf, WhiteningInfo

# Import directed sheaf components
from ..data_structures import DirectedSheaf, DirectedSheafValidationResult
from ..core import (
    ComplexStalkExtender,
    DirectionalEncodingComputer,
    DirectedProcrustesComputer
)
from ..conversion import ComplexToRealEmbedding, RealToComplexReconstruction
from .laplacian import DirectedSheafLaplacianBuilder

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


@dataclass
class DirectedSheafMetadata:
    """Metadata for directed sheaf construction."""
    directionality_parameter: float
    construction_method: str
    num_vertices: int
    num_edges: int
    total_complex_dimension: int
    total_real_dimension: int
    base_sheaf_valid: bool
    extension_successful: bool
    encoding_computed: bool
    restrictions_computed: bool
    laplacian_hermitian: bool
    construction_time: float


class DirectedSheafBuilder:
    """Main builder for directed sheaf construction from neural networks.
    
    This class orchestrates the complete directed sheaf construction pipeline:
    1. Extends real whitened stalks to complex vector spaces
    2. Computes directional encoding matrix T^{(q)}
    3. Constructs directed restriction maps with phase encoding
    4. Validates mathematical properties
    5. Provides integration with existing pipeline
    
    The builder maintains full compatibility with the existing sheaf pipeline
    while providing new capabilities for asymmetric network analysis.
    
    Mathematical Properties:
    - Preserves whitened coordinate structure
    - Maintains stalk dimensions (r_v in both real and complex)
    - Ensures Hermitian Laplacian construction
    - Provides real embedding for computational efficiency
    """
    
    def __init__(self, 
                 directionality_parameter: float = 0.25,
                 validate_construction: bool = True,
                 device: Optional[torch.device] = None,
                 preserve_eigenvalues: bool = False):
        """Initialize the Directed Sheaf Builder.
        
        Args:
            directionality_parameter: q parameter controlling directional strength
            validate_construction: Whether to validate mathematical properties
            device: PyTorch device for computations
            preserve_eigenvalues: Whether to preserve eigenvalues in whitening (enables Hodge formulation)
        """
        self.q = directionality_parameter
        self.validate_construction = validate_construction
        self.device = device or torch.device('cpu')
        self.preserve_eigenvalues = preserve_eigenvalues
        
        # Initialize component modules
        self.complex_extender = ComplexStalkExtender(validate_extension=validate_construction)
        self.encoding_computer = DirectionalEncodingComputer(q=self.q)
        self.procrustes_computer = DirectedProcrustesComputer()
        self.laplacian_builder = DirectedSheafLaplacianBuilder(
            validate_properties=validate_construction,
            device=self.device
        )
        
        # Initialize conversion utilities
        self.complex_to_real = ComplexToRealEmbedding(
            validate_properties=validate_construction
        )
        
        logger.info(f"DirectedSheafBuilder initialized: q={self.q}, device={self.device}")
    
    def build_from_sheaf(self, base_sheaf: Sheaf) -> DirectedSheaf:
        """Build directed sheaf from existing real sheaf.
        
        This is the main entry point for converting existing real sheaves
        to directed sheaves. The conversion preserves all mathematical
        properties while adding directional information. If the base sheaf
        has eigenvalue preservation enabled, this information is propagated
        to the directed sheaf for proper Hermitian Laplacian construction.
        
        Args:
            base_sheaf: Real sheaf from existing pipeline
            
        Returns:
            DirectedSheaf with complex stalks and directed restrictions
            
        Raises:
            ValueError: If base sheaf is invalid
            RuntimeError: If construction fails
        """
        logger.info("Building directed sheaf from existing real sheaf")
        
        # Validate input sheaf
        if not isinstance(base_sheaf, Sheaf):
            raise ValueError("Input must be a Sheaf object")
        
        if not base_sheaf.stalks or not base_sheaf.restrictions:
            raise ValueError("Base sheaf must have non-empty stalks and restrictions")
        
        # Check for eigenvalue preservation in base sheaf
        has_eigenvalue_preservation = (
            hasattr(base_sheaf, 'eigenvalue_metadata') and 
            base_sheaf.eigenvalue_metadata is not None and
            base_sheaf.eigenvalue_metadata.preserve_eigenvalues
        )
        
        if has_eigenvalue_preservation:
            logger.info("Base sheaf has eigenvalue preservation enabled - propagating to directed sheaf")
        
        # Validate base sheaf if requested
        if self.validate_construction:
            if not base_sheaf.metadata.get('is_valid', True):
                logger.warning("Base sheaf validation failed, proceeding anyway")
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Extend real stalks to complex vector spaces
            logger.info("Step 1: Extending real stalks to complex spaces")
            complex_stalks = self._extend_complex_stalks(base_sheaf.stalks)
            
            # Step 2: Compute directional encoding matrix
            logger.info("Step 2: Computing directional encoding matrix")
            adjacency_matrix = self._extract_adjacency_matrix(base_sheaf.poset)
            directional_encoding = self.encoding_computer.compute_encoding_matrix(adjacency_matrix)
            
            # Step 3: Compute directed restriction maps
            logger.info("Step 3: Computing directed restriction maps")
            directed_restrictions = self._compute_directed_restrictions(
                base_sheaf.restrictions,
                directional_encoding,
                base_sheaf.poset
            )
            
            # Step 4: Create directed sheaf
            logger.info("Step 4: Creating directed sheaf data structure")
            directed_sheaf = DirectedSheaf(
                poset=base_sheaf.poset.copy(),
                complex_stalks=complex_stalks,
                directed_restrictions=directed_restrictions,
                directional_encoding=directional_encoding,
                directionality_parameter=self.q,
                base_sheaf=base_sheaf,
                metadata=self._create_metadata(
                    base_sheaf, 
                    complex_stalks, 
                    directed_restrictions,
                    time.time() - start_time,
                    has_eigenvalue_preservation
                )
            )
            
            # Step 5: Validate construction
            if self.validate_construction:
                logger.info("Step 5: Validating directed sheaf construction")
                validation_result = self._validate_construction(directed_sheaf)
                directed_sheaf.metadata.update(validation_result)
                
                if not validation_result.get('construction_successful', False):
                    logger.error("Directed sheaf construction validation failed")
                    raise RuntimeError("Construction validation failed")
            
            construction_time = time.time() - start_time
            logger.info(f"Directed sheaf construction completed in {construction_time:.3f}s")
            
            return directed_sheaf
            
        except Exception as e:
            logger.error(f"Directed sheaf construction failed: {e}", exc_info=True)
            raise RuntimeError(f"Directed sheaf construction failed: {e}")
    
    def build_from_activations(self, 
                             model: nn.Module, 
                             input_tensor: torch.Tensor,
                             validate: bool = True,
                             preserve_eigenvalues: Optional[bool] = None,
                             use_gram_regularization: bool = False,
                             regularization_config: Optional[Dict[str, Any]] = None) -> DirectedSheaf:
        """Build directed sheaf directly from a model and input tensor.
        
        This method mirrors the original SheafBuilder.build_from_activations
        but produces a DirectedSheaf with complex stalks and Hermitian Laplacians.
        When eigenvalue preservation is enabled, the resulting directed sheaf
        will use the Hermitian formulation with eigenvalue matrices.
        
        Args:
            model: The PyTorch model to analyze
            input_tensor: An example input tensor to run the forward pass
            validate: Whether to validate the final sheaf's properties
            preserve_eigenvalues: Optional override for eigenvalue preservation in base sheaf.
                If None, uses SheafBuilder default. If True/False, passed to base SheafBuilder.
            use_gram_regularization: Whether to apply Tikhonov regularization to Gram matrices
            regularization_config: Configuration for Tikhonov regularization (if None, uses defaults)
            
        Returns:
            DirectedSheaf constructed with complex stalks and directed restrictions
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If construction fails
        """
        logger.info("Starting directed sheaf construction from model and input tensor")
        
        try:
            # Step 1: Build base real sheaf using existing pipeline
            logger.info("Step 1: Building base real sheaf")
            from ...sheaf.assembly.builder import SheafBuilder
            
            # Create base builder with proper eigenvalue preservation setting
            # Use the runtime parameter, otherwise use builder's default, otherwise False
            eigenvalue_setting = (preserve_eigenvalues 
                                if preserve_eigenvalues is not None 
                                else self.preserve_eigenvalues)
            base_builder = SheafBuilder(preserve_eigenvalues=eigenvalue_setting)
            logger.info(f"Created base SheafBuilder with preserve_eigenvalues={eigenvalue_setting}")
            
            # Build base sheaf - now the builder is already configured correctly
            base_sheaf = base_builder.build_from_activations(
                model=model,
                input_tensor=input_tensor,
                validate=validate,
                preserve_eigenvalues=preserve_eigenvalues,  # Still pass as runtime override for consistency
                use_gram_regularization=use_gram_regularization,
                regularization_config=regularization_config
            )
            logger.info(f"Built base sheaf with eigenvalue preservation: {eigenvalue_setting}")
            
            # Step 2: Convert base sheaf to directed sheaf
            logger.info("Step 2: Converting base sheaf to directed sheaf")
            directed_sheaf = self.build_from_sheaf(base_sheaf)
            
            # Step 3: Update metadata to reflect direct construction
            logger.info("Step 3: Updating metadata for direct construction")
            directed_sheaf.metadata.update({
                'construction_method': 'directed_from_activations',
                'direct_construction': True,
                'base_construction_method': base_sheaf.metadata.get('construction_method', 'unknown'),
                'batch_size': input_tensor.shape[0],
                'gram_regularized': use_gram_regularization,
                'regularization_info': base_sheaf.metadata.get('regularization_info', None),
                'preserve_eigenvalues_override': preserve_eigenvalues,
                'preserve_eigenvalues_used': eigenvalue_setting,
                'base_preserve_eigenvalues': base_sheaf.metadata.get('preserve_eigenvalues', False)
            })
            
            logger.info("Directed sheaf construction from activations complete")
            return directed_sheaf
            
        except Exception as e:
            logger.error(f"Directed sheaf construction from activations failed: {e}", exc_info=True)
            raise RuntimeError(f"Directed sheaf building from activations failed: {e}")
    
    def _extend_complex_stalks(self, real_stalks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extend real stalks to complex vector spaces.
        
        Mathematical Implementation:
        F(v) = R^{r_v} ⊗_R C
        
        For whitened stalks (identity matrices), this creates complex
        identity matrices of the same dimension.
        
        Args:
            real_stalks: Dictionary of real stalk tensors
            
        Returns:
            Dictionary of complex stalk tensors
        """
        complex_stalks = {}
        
        for node_id, real_stalk in real_stalks.items():
            # Extend to complex space
            complex_stalk = self.complex_extender.extend_stalk(real_stalk)
            complex_stalks[node_id] = complex_stalk
            
            logger.debug(f"Extended stalk {node_id}: {real_stalk.shape} → {complex_stalk.shape}")
        
        logger.info(f"Extended {len(complex_stalks)} stalks to complex spaces")
        return complex_stalks
    
    def _extract_adjacency_matrix(self, poset: nx.DiGraph) -> torch.Tensor:
        """Extract adjacency matrix from poset.
        
        Args:
            poset: Directed graph poset
            
        Returns:
            Adjacency matrix tensor
        """
        # Create node ordering
        nodes = list(poset.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build adjacency matrix
        adjacency = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        
        for u, v in poset.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            adjacency[i, j] = 1.0
        
        logger.debug(f"Extracted adjacency matrix: {adjacency.shape}")
        return adjacency
    
    def _compute_directed_restrictions(self,
                                     base_restrictions: Dict[Tuple[str, str], torch.Tensor],
                                     directional_encoding: torch.Tensor,
                                     poset: nx.DiGraph) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute directed restriction maps with phase encoding.
        
        Mathematical Implementation:
        - Source restriction: s_e Q_e (real, from base sheaf)
        - Target restriction: T^{(q)}_{uv} I_{r_v} (complex, with phase)
        
        Args:
            base_restrictions: Real restriction maps from base sheaf
            directional_encoding: T^{(q)} matrix
            poset: Directed graph poset
            
        Returns:
            Dictionary of complex directed restriction maps
        """
        directed_restrictions = {}
        
        # Create node ordering for indexing
        nodes = list(poset.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for edge, base_restriction in base_restrictions.items():
            u, v = edge
            
            # Get directional encoding factor
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            T_uv = directional_encoding[u_idx, v_idx]
            
            # Compute directed restriction map by applying directional encoding
            # Convert real restriction to complex with phase encoding
            directed_restriction = torch.complex(base_restriction, torch.zeros_like(base_restriction))
            directed_restriction = directed_restriction * T_uv
            
            directed_restrictions[edge] = directed_restriction
            
            logger.debug(f"Computed directed restriction for edge {edge}: "
                        f"{base_restriction.shape} → {directed_restriction.shape}")
        
        logger.info(f"Computed {len(directed_restrictions)} directed restrictions")
        return directed_restrictions
    
    def _create_metadata(self,
                        base_sheaf: Sheaf,
                        complex_stalks: Dict[str, torch.Tensor],
                        directed_restrictions: Dict[Tuple[str, str], torch.Tensor],
                        construction_time: float,
                        has_eigenvalue_preservation: bool = False) -> Dict[str, Any]:
        """Create metadata for directed sheaf construction.
        
        Args:
            base_sheaf: Original real sheaf
            complex_stalks: Complex stalks dictionary
            directed_restrictions: Directed restrictions dictionary
            construction_time: Time taken for construction
            has_eigenvalue_preservation: Whether eigenvalue preservation is enabled
            
        Returns:
            Metadata dictionary
        """
        total_complex_dim = sum(stalk.shape[-1] for stalk in complex_stalks.values())
        total_real_dim = 2 * total_complex_dim
        
        metadata = {
            'construction_method': 'directed_sheaf_builder',
            'directionality_parameter': self.q,
            'num_vertices': len(complex_stalks),
            'num_edges': len(directed_restrictions),
            'total_complex_dimension': total_complex_dim,
            'total_real_dimension': total_real_dim,
            'base_sheaf_valid': base_sheaf.metadata.get('is_valid', True),
            'extension_successful': True,
            'encoding_computed': True,
            'restrictions_computed': True,
            'construction_time': construction_time,
            'device': str(self.device),
            'base_sheaf_metadata': base_sheaf.metadata.copy(),
            'preserve_eigenvalues': has_eigenvalue_preservation,
            'eigenvalue_hermitian_formulation': has_eigenvalue_preservation
        }
        
        return metadata
    
    def _validate_construction(self, directed_sheaf: DirectedSheaf) -> Dict[str, Any]:
        """Validate directed sheaf construction.
        
        Args:
            directed_sheaf: Constructed directed sheaf
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'construction_successful': False,
            'complex_structure_valid': False,
            'directional_encoding_valid': False,
            'hermitian_laplacian_valid': False,
            'real_embedding_valid': False,
            'validation_errors': []
        }
        
        try:
            # Validate complex structure
            complex_validation = directed_sheaf.validate_complex_structure()
            validation_results['complex_structure_valid'] = complex_validation['valid']
            
            if not complex_validation['valid']:
                validation_results['validation_errors'].extend(complex_validation['errors'])
            
            # Validate directional encoding
            if directed_sheaf.directional_encoding is not None:
                encoding_validation = self._validate_directional_encoding(
                    directed_sheaf.directional_encoding,
                    directed_sheaf.poset
                )
                validation_results['directional_encoding_valid'] = encoding_validation['valid']
                
                if not encoding_validation['valid']:
                    validation_results['validation_errors'].extend(encoding_validation['errors'])
            
            # Validate Hermitian Laplacian construction
            try:
                complex_laplacian = self.laplacian_builder.build_complex_laplacian(directed_sheaf)
                validation_results['hermitian_laplacian_valid'] = True
            except Exception as e:
                validation_results['validation_errors'].append(f"Laplacian construction failed: {e}")
            
            # Validate real embedding
            try:
                real_laplacian = self.laplacian_builder.build_real_embedded_laplacian(directed_sheaf)
                validation_results['real_embedding_valid'] = True
            except Exception as e:
                validation_results['validation_errors'].append(f"Real embedding failed: {e}")
            
            # Overall success
            validation_results['construction_successful'] = (
                validation_results['complex_structure_valid'] and
                validation_results['directional_encoding_valid'] and
                validation_results['hermitian_laplacian_valid'] and
                validation_results['real_embedding_valid']
            )
            
            
        except Exception as e:
            validation_results['validation_errors'].append(f"Validation failed: {e}")
            logger.error(f"Directed sheaf validation failed: {e}")
        
        return validation_results
    
    def _validate_directional_encoding(self,
                                     directional_encoding: torch.Tensor,
                                     poset: nx.DiGraph) -> Dict[str, Any]:
        """Validate directional encoding matrix.
        
        Args:
            directional_encoding: T^{(q)} matrix
            poset: Directed graph poset
            
        Returns:
            Validation results
        """
        validation_result = {
            'valid': True,
            'errors': []
        }
        
        # Check dimensions
        n_nodes = len(poset.nodes())
        if directional_encoding.shape != (n_nodes, n_nodes):
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Directional encoding has wrong shape: {directional_encoding.shape} "
                f"vs expected ({n_nodes}, {n_nodes})"
            )
        
        # Check that it's complex
        if not directional_encoding.is_complex():
            validation_result['valid'] = False
            validation_result['errors'].append("Directional encoding must be complex")
        
        # Check magnitude constraints (should be 1 for encoded entries)
        try:
            magnitudes = torch.abs(directional_encoding)
            # For edges, magnitude should be 1; for non-edges, should be 0
            edge_magnitudes = []
            for u, v in poset.edges():
                nodes = list(poset.nodes())
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                edge_magnitudes.append(magnitudes[u_idx, v_idx].item())
            
            # Check that edge magnitudes are close to 1
            if edge_magnitudes:
                max_magnitude_error = max(abs(1.0 - mag) for mag in edge_magnitudes)
                if max_magnitude_error > 1e-6:
                    validation_result['valid'] = False
                    validation_result['errors'].append(
                        f"Edge magnitudes not close to 1: max error = {max_magnitude_error}"
                    )
        except Exception as e:
            validation_result['errors'].append(f"Magnitude validation failed: {e}")
        
        return validation_result
    
    def build_laplacian(self, directed_sheaf: DirectedSheaf) -> torch.Tensor:
        """Build Hermitian Laplacian from directed sheaf.
        
        This is a convenience method that delegates to the LaplacianBuilder.
        
        Args:
            directed_sheaf: Directed sheaf to build Laplacian from
            
        Returns:
            Complex Hermitian Laplacian tensor
        """
        return self.laplacian_builder.build_complex_laplacian(directed_sheaf)
    
    def build_real_laplacian(self, directed_sheaf: DirectedSheaf):
        """Build real representation of Hermitian Laplacian.
        
        This is a convenience method that delegates to the LaplacianBuilder.
        
        Args:
            directed_sheaf: Directed sheaf to build Laplacian from
            
        Returns:
            Real sparse matrix representation
        """
        return self.laplacian_builder.build_real_embedded_laplacian(directed_sheaf)
    
    def get_construction_info(self) -> Dict[str, Any]:
        """Get information about the builder configuration.
        
        Returns:
            Dictionary with builder information
        """
        return {
            'class_name': 'DirectedSheafBuilder',
            'directionality_parameter': self.q,
            'validate_construction': self.validate_construction,
            'device': str(self.device),
            'mathematical_foundation': 'Complex stalks with Hermitian Laplacians',
            'construction_method': 'Real sheaf extension with directional encoding',
            'components': {
                'complex_extender': self.complex_extender.__class__.__name__,
                'encoding_computer': self.encoding_computer.__class__.__name__,
                'procrustes_computer': self.procrustes_computer.__class__.__name__,
                'laplacian_builder': self.laplacian_builder.__class__.__name__
            }
        }