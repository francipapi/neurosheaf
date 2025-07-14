"""Sheaf construction for neural network analysis.

This module implements the core sheaf data structure and construction process,
integrating FX-based poset extraction with CKA-based stalks and Procrustes
restriction maps to create mathematically valid cellular sheaves.

The sheaf construction satisfies the mathematical properties:
- Transitivity: R_AC = R_BC @ R_AB for restriction maps
- Consistency: All stalks and restrictions are computed coherently
- Sparse efficiency: Memory-efficient representation for large networks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

import networkx as nx
import numpy as np
import torch

from ..utils.exceptions import ArchitectureError, ComputationError
from ..utils.logging import setup_logger
from .name_mapper import FXToModuleNameMapper, create_unified_activation_dict
from .poset import FXPosetExtractor
from .restriction import ProcrustesMaps, validate_sheaf_properties

logger = setup_logger(__name__)


@dataclass
class Sheaf:
    """Cellular sheaf data structure for neural network analysis.
    
    A sheaf consists of:
    - poset: Directed acyclic graph representing network structure
    - stalks: Data attached to each node (CKA vectors, activations, or Gram matrices)
    - restrictions: Linear maps between connected nodes
    
    This structure enables spectral analysis of neural network similarity
    patterns using persistent sheaf Laplacians.
    
    Attributes:
        poset: NetworkX directed graph representing layer dependencies
        stalks: Dictionary mapping node names to tensor data (in whitened coordinates if use_whitening=True)
        restrictions: Dictionary mapping edges to restriction map tensors
        metadata: Additional information about construction and validation
        whitening_maps: Dictionary mapping node names to whitening transformations (if use_whitening=True)
    """
    poset: nx.DiGraph = field(default_factory=nx.DiGraph)
    stalks: Dict[str, torch.Tensor] = field(default_factory=dict)
    restrictions: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    whitening_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if not self.metadata:
            self.metadata = {
                'construction_method': 'unknown',
                'num_nodes': len(self.poset.nodes()),
                'num_edges': len(self.poset.edges()),
                'validation_passed': False,
                'memory_efficient': True
            }
    
    def validate(self, tolerance: float = 1e-2) -> Dict[str, Any]:
        """Validate sheaf mathematical properties.
        
        Args:
            tolerance: Tolerance for transitivity validation
            
        Returns:
            Dictionary with validation results
        """
        validation_results = validate_sheaf_properties(self.restrictions, self.poset, tolerance)
        self.metadata['validation_results'] = validation_results
        self.metadata['validation_passed'] = validation_results['valid_sheaf']
        return validation_results
    
    def get_laplacian_structure(self) -> Dict[str, Any]:
        """Get information about the sparse Laplacian structure.
        
        Returns:
            Dictionary with Laplacian structure information
        """
        nodes = list(self.poset.nodes())
        edges = list(self.poset.edges())
        
        # Compute total dimension
        total_dim = sum(stalk.shape[-1] if stalk.ndim > 1 else stalk.shape[0] 
                       for stalk in self.stalks.values())
        
        # Estimate sparsity
        max_entries = total_dim ** 2
        actual_entries = len(edges) * 2  # Each edge contributes to 2 off-diagonal blocks
        for node in nodes:
            if node in self.stalks:
                stalk_dim = (self.stalks[node].shape[-1] if self.stalks[node].ndim > 1 
                           else self.stalks[node].shape[0])
                actual_entries += stalk_dim ** 2  # Diagonal block
        
        sparsity = 1.0 - (actual_entries / max_entries) if max_entries > 0 else 0.0
        
        return {
            'total_dimension': total_dim,
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'estimated_sparsity': sparsity,
            'memory_savings': f"{sparsity * 100:.1f}%"
        }
    
    def summary(self) -> str:
        """Get a summary string of the sheaf structure."""
        laplacian_info = self.get_laplacian_structure()
        validation_status = "✓" if self.metadata.get('validation_passed', False) else "✗"
        
        return (f"Sheaf Summary:\n"
                f"  Nodes: {laplacian_info['num_nodes']}\n"
                f"  Edges: {laplacian_info['num_edges']}\n"
                f"  Total dimension: {laplacian_info['total_dimension']}\n"
                f"  Sparsity: {laplacian_info['memory_savings']}\n"
                f"  Validation: {validation_status}\n"
                f"  Method: {self.metadata.get('construction_method', 'unknown')}")


class SheafBuilder:
    """Build cellular sheaves from neural networks and activation data.
    
    This class orchestrates the sheaf construction process:
    1. Extract poset structure using FX tracing
    2. Assign stalk data from activations or precomputed CKA
    3. Compute restriction maps using Procrustes analysis
    4. Validate mathematical properties
    
    The builder supports multiple stalk data types:
    - Raw activations: X_v for each layer v
    - Gram matrices: K_v = X_v @ X_v.T
    - CKA similarity vectors: pre-computed pairwise similarities
    
    Attributes:
        poset_extractor: FX-based poset extraction
        procrustes_maps: Restriction map computation
        default_method: Default method for restriction map computation
    """
    
    def __init__(self, handle_dynamic: bool = True, procrustes_epsilon: float = 1e-8,
                 restriction_method: str = 'scaled_procrustes', use_whitening: bool = True,
                 residual_threshold: float = 0.05, enable_edge_filtering: bool = True):
        """Initialize sheaf builder.
        
        Args:
            handle_dynamic: Whether to handle dynamic models with fallback extraction
            procrustes_epsilon: Numerical stability parameter for Procrustes analysis
            restriction_method: Method for computing restriction maps
            use_whitening: Whether to use whitened coordinates by default (Patch P1)
            residual_threshold: Maximum acceptable relative residual for edge filtering (5% default)
            enable_edge_filtering: Whether to filter out high-residual edges
        """
        self.poset_extractor = FXPosetExtractor(handle_dynamic=handle_dynamic)
        self.procrustes_maps = ProcrustesMaps(epsilon=procrustes_epsilon, use_whitened_coordinates=use_whitening)
        self.default_method = restriction_method
        self.use_whitening = use_whitening
        self.residual_threshold = residual_threshold
        self.enable_edge_filtering = enable_edge_filtering
    
    def build_from_activations(self, model: torch.nn.Module, 
                              activations: Dict[str, torch.Tensor],
                              use_gram_matrices: bool = True,
                              validate: bool = True) -> Sheaf:
        """Build sheaf from model and activation data.
        
        Args:
            model: PyTorch model to analyze
            activations: Dictionary mapping layer names to activation tensors
            use_gram_matrices: Whether to use Gram matrices as stalks (recommended)
            validate: Whether to validate sheaf properties
            
        Returns:
            Constructed Sheaf object
            
        Raises:
            ArchitectureError: If model structure cannot be extracted
            ComputationError: If restriction map computation fails
        """
        logger.info("Building sheaf from model activations")
        
        # Translate activation keys to match FX node names first
        try:
            translated_activations, name_mapper = create_unified_activation_dict(model, activations)
            logger.info(f"Name mapping stats: {name_mapper.get_mapping_stats()}")
        except Exception as e:
            logger.warning(f"Name mapping failed: {e}. Using original activations.")
            translated_activations = activations
            name_mapper = None
        
        # Extract poset structure filtered to available activations
        try:
            available_activations = set(translated_activations.keys())
            poset = self.poset_extractor.extract_activation_filtered_poset(model, available_activations)
            logger.info(f"Using activation-filtered poset: {len(poset.nodes())} nodes, {len(poset.edges())} edges")
        except Exception as e:
            logger.warning(f"Filtered poset extraction failed: {e}. Using standard extraction.")
            try:
                poset = self.poset_extractor.extract_poset(model)
            except Exception as e2:
                raise ArchitectureError(f"Failed to extract poset: {e2}")
        
        # Assign stalks from translated activations
        stalks, whitening_maps = self._assign_stalks_from_activations(translated_activations, poset, use_gram_matrices)
        
        # Compute restriction maps
        restrictions = self._compute_all_restrictions(stalks, poset)
        
        # Create sheaf with filtering metadata
        metadata = {
            'construction_method': 'activations',
            'use_gram_matrices': use_gram_matrices,
            'restriction_method': self.default_method,
            'use_whitening': self.use_whitening,
            'edge_filtering_enabled': self.enable_edge_filtering,
            'residual_threshold': self.residual_threshold,
            'num_nodes': len(poset.nodes()),
            'num_edges': len(poset.edges()),
            'num_restrictions': len(restrictions),
            'activation_layers': list(activations.keys()),
            'translated_activation_layers': list(translated_activations.keys()),
            'name_mapping_success': name_mapper is not None,
            'name_mapping_stats': name_mapper.get_mapping_stats() if name_mapper else None
        }
        
        # Add filtering results if available
        if hasattr(self, '_last_filtering_results'):
            metadata['filtering_results'] = self._last_filtering_results
        
        sheaf = Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata,
            whitening_maps=whitening_maps if self.use_whitening else {}
        )
        
        # Validate if requested
        if validate:
            validation_results = sheaf.validate()
            logger.info(f"Sheaf validation: {validation_results['valid_sheaf']}")
        
        return sheaf
    
    def build_from_cka_matrices(self, poset: nx.DiGraph,
                               cka_matrices: Dict[str, torch.Tensor],
                               validate: bool = True) -> Sheaf:
        """Build sheaf from precomputed CKA matrices.
        
        Args:
            poset: Precomputed poset structure
            cka_matrices: Dictionary mapping layer names to CKA similarity matrices
            validate: Whether to validate sheaf properties
            
        Returns:
            Constructed Sheaf object
        """
        logger.info("Building sheaf from CKA matrices")
        
        # Use CKA matrices directly as stalks (apply whitening if enabled)
        stalks = {}
        whitening_maps = {}
        # Store original CKA matrices for restriction computation when using whitening
        self._original_gram_matrices = {} if self.use_whitening else None
        
        for node in poset.nodes():
            if node in cka_matrices:
                if self.use_whitening:
                    # Store original CKA matrix for restriction computation
                    self._original_gram_matrices[node] = cka_matrices[node]
                    
                    # Apply whitening transformation to CKA matrices
                    K_whitened, W, info = self.procrustes_maps.whitening_processor.whiten_gram_matrix(cka_matrices[node])
                    stalks[node] = K_whitened  # Store r×r identity matrix
                    whitening_maps[node] = W    # Store r×n whitening map
                    logger.debug(f"Assigned whitened CKA stalk for {node}: shape {stalks[node].shape} "
                               f"(rank {info['effective_rank']}/{cka_matrices[node].shape[0]})")
                else:
                    stalks[node] = cka_matrices[node]
            else:
                logger.warning(f"No CKA matrix for node {node}")
        
        # Compute restriction maps
        restrictions = self._compute_all_restrictions(stalks, poset)
        
        # Create sheaf with filtering metadata
        metadata = {
            'construction_method': 'cka_matrices',
            'restriction_method': self.default_method,
            'use_whitening': self.use_whitening,
            'edge_filtering_enabled': self.enable_edge_filtering,
            'residual_threshold': self.residual_threshold,
            'num_nodes': len(poset.nodes()),
            'num_edges': len(poset.edges()),
            'num_restrictions': len(restrictions),
            'cka_layers': list(cka_matrices.keys())
        }
        
        # Add filtering results if available
        if hasattr(self, '_last_filtering_results'):
            metadata['filtering_results'] = self._last_filtering_results
            
        sheaf = Sheaf(
            poset=poset,
            stalks=stalks,
            restrictions=restrictions,
            metadata=metadata,
            whitening_maps=whitening_maps if self.use_whitening else {}
        )
        
        # Validate if requested
        if validate:
            validation_results = sheaf.validate()
            logger.info(f"Sheaf validation: {validation_results['valid_sheaf']}")
        
        return sheaf
    
    def build_from_model_comparison(self, model1: torch.nn.Module, model2: torch.nn.Module,
                                   input_data: torch.Tensor, layers_to_compare: Optional[List[str]] = None,
                                   validate: bool = True) -> Sheaf:
        """Build sheaf for comparing two models.
        
        This method extracts activations from both models and creates a sheaf
        that enables comparison of their internal representations.
        
        Args:
            model1: First model to compare
            model2: Second model to compare
            input_data: Input tensor for activation extraction
            layers_to_compare: Specific layers to compare (None for all)
            validate: Whether to validate sheaf properties
            
        Returns:
            Sheaf with comparison structure
        """
        logger.info("Building sheaf for model comparison")
        
        # Extract activations from both models
        activations1 = self._extract_activations(model1, input_data, layers_to_compare)
        activations2 = self._extract_activations(model2, input_data, layers_to_compare)
        
        # Create combined poset (simplified: assume same structure)
        poset1 = self.poset_extractor.extract_poset(model1)
        
        # Combine activations for comparison
        combined_activations = {}
        for layer in activations1:
            if layer in activations2:
                # Stack activations for joint analysis
                combined_activations[layer] = torch.cat([activations1[layer], activations2[layer]], dim=0)
        
        return self.build_from_activations(model1, combined_activations, validate=validate)
    
    def _assign_stalks_from_activations(self, activations: Dict[str, torch.Tensor], 
                                       poset: nx.DiGraph, use_gram_matrices: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Assign stalk data from activation tensors.
        
        Args:
            activations: Dictionary of activation tensors
            poset: Poset structure
            use_gram_matrices: Whether to compute Gram matrices
            
        Returns:
            Tuple of (stalks, whitening_maps):
            - stalks: Dictionary mapping node names to stalk tensors (whitened if use_whitening=True)
            - whitening_maps: Dictionary mapping node names to whitening transformations
        """
        stalks = {}
        whitening_maps = {}
        # Store original Gram matrices for restriction computation when using whitening
        self._original_gram_matrices = {} if self.use_whitening else None
        
        for node in poset.nodes():
            if node in activations:
                activation = activations[node]
                
                # Ensure activation is 2D (samples x features)
                if activation.ndim > 2:
                    activation = activation.view(activation.shape[0], -1)
                
                if use_gram_matrices:
                    # Compute Gram matrix K = X @ X.T (raw activations, no centering)
                    gram_matrix = activation @ activation.T
                    
                    if self.use_whitening:
                        # Store original Gram matrix for restriction computation
                        self._original_gram_matrices[node] = gram_matrix
                        
                        # Apply whitening transformation
                        K_whitened, W, info = self.procrustes_maps.whitening_processor.whiten_gram_matrix(gram_matrix)
                        stalks[node] = K_whitened  # Store r×r identity matrix
                        whitening_maps[node] = W    # Store r×n whitening map
                        logger.debug(f"Assigned whitened stalk for {node}: shape {stalks[node].shape} "
                                   f"(rank {info['effective_rank']}/{gram_matrix.shape[0]})")
                    else:
                        stalks[node] = gram_matrix
                        logger.debug(f"Assigned stalk for {node}: shape {stalks[node].shape}")
                else:
                    # Use raw activations (no whitening for raw activations mode)
                    stalks[node] = activation
                    logger.debug(f"Assigned activation stalk for {node}: shape {stalks[node].shape}")
            else:
                logger.warning(f"No activation data for node {node}")
        
        return stalks, whitening_maps
    
    def _compute_all_restrictions(self, stalks: Dict[str, torch.Tensor], 
                                 poset: nx.DiGraph) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute restriction maps for all edges in the poset with quality filtering.
        
        Implements edge quality filtering as per pipeline report Section 2:
        - Track residual: ρ_e = ||E_e||_F / ||K_w||_F
        - Drop edges where ρ_e > threshold (default 5%)
        
        Args:
            stalks: Dictionary of stalk tensors
            poset: Poset structure
            
        Returns:
            Dictionary mapping edges to restriction map tensors (filtered for quality)
        """
        restrictions = {}
        failed_edges = []
        filtered_edges = []
        quality_metrics = []
        
        total_edges = len(poset.edges())
        logger.info(f"Computing restriction maps for {total_edges} edges "
                   f"(filtering={'enabled' if self.enable_edge_filtering else 'disabled'}, "
                   f"threshold={self.residual_threshold:.1%})")
        
        for edge in poset.edges():
            source, target = edge
            
            if source in stalks and target in stalks:
                try:
                    K_source = stalks[source]
                    K_target = stalks[target]
                    
                    if self.use_whitening:
                        # Use the new scaled_procrustes_whitened function with original Gram matrices
                        # This function handles the complete whitening process internally and 
                        # returns the optimal restriction map in whitened coordinates.
                        
                        if hasattr(self, '_original_gram_matrices') and self._original_gram_matrices:
                            # Use original Gram matrices for proper restriction computation
                            K_source_orig = self._original_gram_matrices.get(source)
                            K_target_orig = self._original_gram_matrices.get(target)
                            
                            if K_source_orig is not None and K_target_orig is not None:
                                R, scale, info = self.procrustes_maps.scaled_procrustes_whitened(
                                    K_source_orig, K_target_orig, validate=False
                                )
                            else:
                                # Fallback: use whitened stalks (identity matrices)
                                logger.warning(f"Missing original Gram matrices for edge {edge}, using identity fallback")
                                r_source = K_source.shape[0]
                                r_target = K_target.shape[0]
                                
                                if r_source <= r_target:
                                    R = torch.zeros(r_target, r_source)
                                    R[:r_source, :] = torch.eye(r_source)
                                else:
                                    R = torch.zeros(r_target, r_source)
                                    R[:, :r_target] = torch.eye(r_target)
                                
                                scale = 1.0
                                info = {'method': 'identity_fallback', 'scale': scale, 'relative_error': 0.0}
                        else:
                            # Legacy fallback for whitened identity matrices
                            logger.warning("Using legacy identity restriction for whitened stalks")
                            r_source = K_source.shape[0]
                            r_target = K_target.shape[0]
                            
                            if r_source <= r_target:
                                R = torch.zeros(r_target, r_source)
                                R[:r_source, :] = torch.eye(r_source)
                            else:
                                R = torch.zeros(r_target, r_source)
                                R[:, :r_target] = torch.eye(r_target)
                            
                            scale = 1.0
                            info = {'method': 'identity_legacy', 'scale': scale, 'relative_error': 0.0}
                    else:
                        # Compute restriction map using standard method
                        R, scale, info = self.procrustes_maps.compute_restriction_map(
                            K_source, K_target, method=self.default_method, 
                            validate=True, use_whitening=False
                        )
                    
                    # Extract quality metrics
                    rel_error = info.get('relative_error', float('inf'))
                    
                    # For whitened computations, also check whitened space quality
                    if 'whitened_validation' in info:
                        whitened_orthogonal = info['whitened_validation'].get('exact_orthogonal', False)
                        whitened_metric_compat = info['whitened_validation'].get('exact_metric_compatible', False)
                        logger.debug(f"Edge {edge} whitened quality: orthogonal={whitened_orthogonal}, "
                                   f"metric_compat={whitened_metric_compat}")
                    
                    # Edge quality filtering based on residual threshold
                    should_include = True
                    if self.enable_edge_filtering and rel_error > self.residual_threshold:
                        should_include = False
                        filtered_edges.append({
                            'edge': edge,
                            'relative_error': rel_error,
                            'threshold': self.residual_threshold,
                            'reason': 'high_residual'
                        })
                        logger.info(f"Filtered edge {edge}: residual {rel_error:.1%} > {self.residual_threshold:.1%}")
                    
                    if should_include:
                        restrictions[edge] = R
                        
                        # Store quality metrics for analysis
                        quality_metrics.append({
                            'edge': edge,
                            'scale': scale,
                            'relative_error': rel_error,
                            'method': info.get('method', self.default_method),
                            'whitened': 'whitened_validation' in info
                        })
                    
                    # Log quality metrics
                    if rel_error > 0.15:  # Still warn about high errors even if not filtering
                        logger.warning(f"High error for edge {edge}: {rel_error:.3f}")
                    
                    logger.debug(f"Computed restriction {edge}: scale={scale:.3f}, "
                                f"error={rel_error:.3f}, included={should_include}")
                    
                except Exception as e:
                    logger.error(f"Failed to compute restriction for edge {edge}: {e}")
                    failed_edges.append(edge)
            else:
                logger.warning(f"Missing stalks for edge {edge}")
                failed_edges.append(edge)
        
        # Log filtering results
        included_edges = len(restrictions)
        total_processed = total_edges - len(failed_edges)
        filtering_rate = len(filtered_edges) / total_processed if total_processed > 0 else 0.0
        
        logger.info(f"Edge filtering results: {included_edges}/{total_processed} edges included "
                   f"({filtering_rate:.1%} filtered, {len(failed_edges)} failed)")
        
        if filtered_edges:
            avg_filtered_error = np.mean([e['relative_error'] for e in filtered_edges])
            logger.info(f"Filtered edges had average residual: {avg_filtered_error:.1%}")
        
        if quality_metrics:
            avg_included_error = np.mean([m['relative_error'] for m in quality_metrics])
            logger.info(f"Included edges have average residual: {avg_included_error:.1%}")
        
        # Store filtering metadata for sheaf
        self._last_filtering_results = {
            'total_edges': total_edges,
            'included_edges': included_edges,
            'filtered_edges': len(filtered_edges),
            'failed_edges': len(failed_edges),
            'filtering_rate': filtering_rate,
            'threshold_used': self.residual_threshold,
            'whitening_used': self.use_whitening,
            'quality_metrics': quality_metrics,
            'filtered_edge_details': filtered_edges
        }
        
        return restrictions
    
    def _extract_activations(self, model: torch.nn.Module, input_data: torch.Tensor,
                           layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Extract activations from model for given input.
        
        This is a utility method for extracting activations. In practice,
        users would typically provide pre-extracted activations.
        
        Args:
            model: PyTorch model
            input_data: Input tensor
            layer_names: Specific layers to extract (None for all)
            
        Returns:
            Dictionary of activation tensors
        """
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    activations[name] = output[0].detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                if hasattr(module, 'weight') or hasattr(module, 'bias'):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        
        try:
            # Forward pass
            with torch.no_grad():
                model.eval()
                _ = model(input_data)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return activations
    
    def build_laplacian(self, sheaf: Sheaf, enable_gpu: bool = True, 
                       memory_efficient: bool = True) -> Tuple['csr_matrix', 'LaplacianMetadata']:
        """Build sparse sheaf Laplacian from constructed sheaf.
        
        This method creates the sparse Laplacian Δ = δ^T δ from the sheaf's
        whitened stalks and restriction maps, optimized for spectral analysis.
        
        Args:
            sheaf: Constructed Sheaf object with whitened stalks and restrictions
            enable_gpu: Whether to enable GPU-compatible operations
            memory_efficient: Whether to use memory-efficient assembly
            
        Returns:
            Tuple of (sparse_laplacian, metadata):
            - sparse_laplacian: scipy.sparse.csr_matrix of the Laplacian
            - metadata: LaplacianMetadata with construction details
            
        Raises:
            ComputationError: If Laplacian construction fails
        """
        from .laplacian import SheafLaplacianBuilder
        
        logger.info("Building sparse Laplacian from sheaf data")
        
        # Validate sheaf has required data
        if not sheaf.stalks:
            raise ComputationError("Cannot build Laplacian: sheaf has no stalks", 
                                  operation="build_laplacian")
        if not sheaf.restrictions:
            raise ComputationError("Cannot build Laplacian: sheaf has no restrictions",
                                  operation="build_laplacian")
        
        # Create Laplacian builder with same settings as sheaf construction
        builder = SheafLaplacianBuilder(
            enable_gpu=enable_gpu,
            memory_efficient=memory_efficient,
            validate_properties=True  # Always validate for production use
        )
        
        # Build Laplacian
        laplacian, metadata = builder.build_laplacian(sheaf)
        
        # Store Laplacian information in sheaf metadata
        if 'laplacian_info' not in sheaf.metadata:
            sheaf.metadata['laplacian_info'] = {}
        
        sheaf.metadata['laplacian_info'].update({
            'shape': laplacian.shape,
            'nnz': laplacian.nnz,
            'sparsity': metadata.sparsity_ratio,
            'construction_time': metadata.construction_time,
            'memory_usage_gb': metadata.memory_usage,
            'enable_gpu': enable_gpu,
            'memory_efficient': memory_efficient
        })
        
        logger.info(f"Laplacian built successfully: {laplacian.shape[0]}×{laplacian.shape[1]}, "
                   f"{laplacian.nnz} non-zeros, {metadata.construction_time:.2f}s")
        
        return laplacian, metadata
    
    def build_static_masked_laplacian(self, sheaf: Sheaf, enable_gpu: bool = True) -> 'StaticMaskedLaplacian':
        """Build StaticMaskedLaplacian for efficient filtration analysis.
        
        This method creates a StaticMaskedLaplacian that enables efficient
        threshold-based filtration without rebuilding the matrix.
        
        Args:
            sheaf: Constructed Sheaf object
            enable_gpu: Whether to enable GPU operations
            
        Returns:
            StaticMaskedLaplacian ready for persistent spectral analysis
        """
        from ..spectral.static_laplacian import create_static_masked_laplacian
        
        logger.info("Building static masked Laplacian for filtration analysis")
        
        # Create StaticMaskedLaplacian
        static_laplacian = create_static_masked_laplacian(sheaf, enable_gpu=enable_gpu)
        
        # Store filtration information in sheaf metadata
        memory_stats = static_laplacian.get_memory_usage()
        weight_range = static_laplacian.masking_metadata.weight_range
        
        if 'filtration_info' not in sheaf.metadata:
            sheaf.metadata['filtration_info'] = {}
        
        sheaf.metadata['filtration_info'].update({
            'edge_count': len(static_laplacian.masking_metadata.edge_weights),
            'weight_range': weight_range,
            'memory_usage': memory_stats,
            'enable_gpu': enable_gpu
        })
        
        logger.info(f"Static Laplacian ready: {len(static_laplacian.masking_metadata.edge_weights)} edges, "
                   f"weight range [{weight_range[0]:.4f}, {weight_range[1]:.4f}], "
                   f"{memory_stats['total_gb']:.2f}GB")
        
        return static_laplacian


def create_sheaf_from_cka_analysis(cka_results: Dict[str, Any], 
                                  layer_names: List[str],
                                  network_structure: Optional[nx.DiGraph] = None) -> Sheaf:
    """Create sheaf from CKA analysis results.
    
    This is a convenience function for creating sheaves from CKA analysis
    results, typically from the neurosheaf.cka module.
    
    Args:
        cka_results: Results from CKA analysis
        layer_names: List of layer names in order
        network_structure: Optional precomputed network structure
        
    Returns:
        Constructed Sheaf object
    """
    # Create simple sequential poset if none provided
    if network_structure is None:
        poset = nx.DiGraph()
        poset.add_nodes_from(layer_names)
        for i in range(len(layer_names) - 1):
            poset.add_edge(layer_names[i], layer_names[i + 1])
    else:
        poset = network_structure
    
    # Extract CKA matrices if available
    cka_matrices = {}
    if 'similarity_matrix' in cka_results:
        similarity_matrix = cka_results['similarity_matrix']
        for i, layer in enumerate(layer_names):
            if i < similarity_matrix.shape[0]:
                cka_matrices[layer] = similarity_matrix[i:i+1, :]  # Row vector
    
    # Build sheaf
    builder = SheafBuilder()
    return builder.build_from_cka_matrices(poset, cka_matrices)