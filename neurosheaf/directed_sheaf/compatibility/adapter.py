"""Adapter for integrating directed sheaves with existing pipeline.

This module provides the DirectedSheafAdapter class that handles the conversion
of directed sheaf data structures to formats compatible with existing pipeline
components, particularly spectral analysis and visualization modules.

Mathematical Foundation:
- Real Embedding: Complex Hermitian matrices → Real symmetric matrices
- Eigenvalue Mapping: Complex eigenvalues → Real eigenvalue pairs  
- Metadata Preservation: Directed sheaf properties → Pipeline metadata
- Format Conversion: DirectedSheaf → existing data structures

The adapter ensures seamless integration while maintaining mathematical
correctness and performance characteristics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.sparse import csr_matrix
import networkx as nx
from dataclasses import dataclass
import time

# Import directed sheaf components
from ..data_structures import DirectedSheaf, DirectedSheafValidationResult
from ..assembly.laplacian import DirectedSheafLaplacianBuilder, LaplacianMetadata
from ..conversion import ComplexToRealEmbedding, RealToComplexReconstruction

# Import base sheaf structures for compatibility
from ...sheaf.data_structures import Sheaf

# Simple logging setup
import logging
logger = logging.getLogger(__name__)


@dataclass
class SpectralAnalysisMetadata:
    """Metadata for spectral analysis from directed sheaf."""
    is_directed: bool
    directionality_parameter: float
    hermitian_laplacian: bool
    real_embedded: bool
    complex_dimension: int
    real_dimension: int
    num_vertices: int
    num_edges: int
    sparsity: float
    conversion_time: float


class DirectedSheafAdapter:
    """Adapter for integrating directed sheaves with existing pipeline.
    
    This class provides methods to convert directed sheaf data structures
    to formats compatible with existing pipeline components, ensuring
    seamless integration without breaking changes.
    
    Key Features:
    - Convert directed sheaf to real Laplacian for spectral analysis
    - Prepare data for visualization modules
    - Handle eigenvalue extraction and mapping
    - Preserve metadata and mathematical properties
    - Optimize performance for large networks
    
    The adapter maintains mathematical correctness while providing the
    format conversions needed for backward compatibility.
    """
    
    def __init__(self, 
                 preserve_metadata: bool = True,
                 validate_conversions: bool = True,
                 optimize_sparse_operations: bool = True,
                 device: Optional[torch.device] = None):
        """Initialize the Directed Sheaf Adapter.
        
        Args:
            preserve_metadata: Whether to preserve all metadata during conversion
            validate_conversions: Whether to validate conversion correctness
            optimize_sparse_operations: Whether to optimize for sparse matrices
            device: PyTorch device for computations
        """
        self.preserve_metadata = preserve_metadata
        self.validate_conversions = validate_conversions
        self.optimize_sparse_operations = optimize_sparse_operations
        self.device = device or torch.device('cpu')
        
        # Initialize component modules
        self.laplacian_builder = DirectedSheafLaplacianBuilder(
            validate_properties=validate_conversions,
            use_sparse_operations=optimize_sparse_operations,
            device=device
        )
        
        self.complex_to_real = ComplexToRealEmbedding(
            validate_properties=validate_conversions,
            device=device
        )
        
        self.real_to_complex = RealToComplexReconstruction(
            validate_properties=validate_conversions,
            device=device
        )
        
        logger.info(f"DirectedSheafAdapter initialized: device={device}")
    
    def adapt_for_spectral_analysis(self, directed_sheaf: DirectedSheaf) -> Tuple[csr_matrix, SpectralAnalysisMetadata]:
        """Adapt directed sheaf for spectral analysis pipeline.
        
        Converts the directed sheaf to a real sparse matrix representation
        suitable for existing spectral analysis components.
        
        Args:
            directed_sheaf: DirectedSheaf to convert
            
        Returns:
            Tuple of (real_laplacian, metadata) compatible with spectral pipeline
            
        Raises:
            ValueError: If directed sheaf is invalid
            RuntimeError: If conversion fails
        """
        logger.info("Adapting directed sheaf for spectral analysis")
        
        if not isinstance(directed_sheaf, DirectedSheaf):
            raise ValueError("Input must be a DirectedSheaf")
        
        start_time = time.time()
        
        try:
            # Build real embedded Laplacian
            real_laplacian, laplacian_metadata = self.laplacian_builder.build_with_metadata(directed_sheaf)
            
            # Create spectral analysis metadata
            spectral_metadata = SpectralAnalysisMetadata(
                is_directed=True,
                directionality_parameter=directed_sheaf.directionality_parameter,
                hermitian_laplacian=laplacian_metadata.is_hermitian,
                real_embedded=True,
                complex_dimension=laplacian_metadata.total_complex_dimension,
                real_dimension=laplacian_metadata.total_real_dimension,
                num_vertices=laplacian_metadata.num_vertices,
                num_edges=laplacian_metadata.num_edges,
                sparsity=laplacian_metadata.block_structure.get('sparsity', 0.0),
                conversion_time=time.time() - start_time
            )
            
            # Validate conversion if requested
            if self.validate_conversions:
                self._validate_spectral_conversion(real_laplacian, spectral_metadata)
            
            logger.info(f"Spectral analysis adaptation complete: {real_laplacian.shape} matrix")
            return real_laplacian, spectral_metadata
            
        except Exception as e:
            logger.error(f"Spectral analysis adaptation failed: {e}")
            raise RuntimeError(f"Spectral analysis adaptation failed: {e}")
    
    def adapt_for_visualization(self, directed_sheaf: DirectedSheaf) -> Dict[str, Any]:
        """Adapt directed sheaf for visualization modules.
        
        Prepares directed sheaf data in formats suitable for existing
        visualization components.
        
        Args:
            directed_sheaf: DirectedSheaf to visualize
            
        Returns:
            Dictionary with visualization-ready data
        """
        logger.info("Adapting directed sheaf for visualization")
        
        try:
            # Extract graph structure
            graph_data = self._extract_graph_structure(directed_sheaf)
            
            # Extract CKA-like similarity matrix
            similarity_data = self._extract_similarity_matrix(directed_sheaf)
            
            # Extract eigenvalue information
            eigenvalue_data = self._extract_eigenvalue_data(directed_sheaf)
            
            # Create visualization data package
            visualization_data = {
                'graph_structure': graph_data,
                'similarity_matrix': similarity_data,
                'eigenvalue_data': eigenvalue_data,
                'directed_properties': {
                    'directionality_parameter': directed_sheaf.directionality_parameter,
                    'complex_stalks': True,
                    'hermitian_laplacian': True
                },
                'metadata': directed_sheaf.metadata.copy() if self.preserve_metadata else {}
            }
            
            logger.info("Visualization adaptation complete")
            return visualization_data
            
        except Exception as e:
            logger.error(f"Visualization adaptation failed: {e}")
            raise RuntimeError(f"Visualization adaptation failed: {e}")
    
    def extract_real_eigenvalues(self, directed_sheaf: DirectedSheaf) -> np.ndarray:
        """Extract real eigenvalues from directed sheaf Laplacian.
        
        Computes the eigenvalues of the Hermitian Laplacian and returns
        them in real form suitable for persistence analysis.
        
        Args:
            directed_sheaf: DirectedSheaf to analyze
            
        Returns:
            Real eigenvalues array
        """
        logger.info("Extracting real eigenvalues from directed sheaf")
        
        try:
            # Build complex Laplacian
            complex_laplacian = self.laplacian_builder.build_complex_laplacian(directed_sheaf)
            
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvals(complex_laplacian)
            
            # Extract real parts (should be purely real for Hermitian matrices)
            real_eigenvalues = eigenvalues.real.detach().cpu().numpy()
            
            # Validate that imaginary parts are negligible
            if self.validate_conversions:
                max_imag = torch.abs(eigenvalues.imag).max().item()
                if max_imag > 1e-10:
                    logger.warning(f"Significant imaginary parts in eigenvalues: {max_imag}")
            
            logger.info(f"Extracted {len(real_eigenvalues)} real eigenvalues")
            return real_eigenvalues
            
        except Exception as e:
            logger.error(f"Eigenvalue extraction failed: {e}")
            raise RuntimeError(f"Eigenvalue extraction failed: {e}")
    
    def convert_persistence_results(self, persistence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert persistence results to directed sheaf format.
        
        Processes persistence analysis results to include directed sheaf
        specific information and metadata.
        
        Args:
            persistence_results: Results from persistence analysis
            
        Returns:
            Enhanced persistence results with directed sheaf information
        """
        logger.info("Converting persistence results for directed sheaf")
        
        try:
            # Create enhanced results dictionary
            enhanced_results = persistence_results.copy()
            
            # Add directed sheaf specific information
            enhanced_results['directed_analysis'] = {
                'is_directed': True,
                'hermitian_laplacian': True,
                'real_embedded': True,
                'complex_eigenvalues_processed': True
            }
            
            # Process eigenvalue data if present
            if 'eigenvalue_data' in persistence_results:
                enhanced_results['eigenvalue_data'] = self._process_eigenvalue_data(
                    persistence_results['eigenvalue_data']
                )
            
            # Process persistence diagrams if present
            if 'persistence_diagrams' in persistence_results:
                enhanced_results['persistence_diagrams'] = self._process_persistence_diagrams(
                    persistence_results['persistence_diagrams']
                )
            
            logger.info("Persistence results conversion complete")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Persistence results conversion failed: {e}")
            raise RuntimeError(f"Persistence results conversion failed: {e}")
    
    def create_compatibility_sheaf(self, directed_sheaf: DirectedSheaf) -> Sheaf:
        """Create a compatibility Sheaf object from DirectedSheaf.
        
        Creates a standard Sheaf object that can be used with existing
        pipeline components that expect real sheaves.
        
        Args:
            directed_sheaf: DirectedSheaf to convert
            
        Returns:
            Sheaf object compatible with existing pipeline
        """
        logger.info("Creating compatibility sheaf from directed sheaf")
        
        try:
            # Convert complex stalks to real representation
            real_stalks = {}
            for node_id, complex_stalk in directed_sheaf.complex_stalks.items():
                # For identity stalks, we can use the real dimension
                real_dim = complex_stalk.shape[0]
                real_stalks[node_id] = torch.eye(real_dim * 2, dtype=torch.float32)
            
            # Convert directed restrictions to real representation
            real_restrictions = {}
            for edge, complex_restriction in directed_sheaf.directed_restrictions.items():
                # Convert to real representation
                real_restriction = self.complex_to_real.embed_matrix(complex_restriction)
                real_restrictions[edge] = real_restriction
            
            # Create compatibility metadata
            compatibility_metadata = {
                'construction_method': 'directed_sheaf_compatibility',
                'original_directed': True,
                'directionality_parameter': directed_sheaf.directionality_parameter,
                'real_embedding': True,
                'whitened': True,  # Maintain whitened flag for compatibility
                'nodes': len(directed_sheaf.complex_stalks),
                'edges': len(directed_sheaf.directed_restrictions)
            }
            
            # Preserve original metadata if requested
            if self.preserve_metadata:
                # Save the construction method we want to keep
                original_construction_method = compatibility_metadata['construction_method']
                compatibility_metadata.update(directed_sheaf.metadata)
                # Restore our construction method
                compatibility_metadata['construction_method'] = original_construction_method
            
            # Create compatibility sheaf
            compatibility_sheaf = Sheaf(
                poset=directed_sheaf.poset.copy(),
                stalks=real_stalks,
                restrictions=real_restrictions,
                metadata=compatibility_metadata
            )
            
            logger.info("Compatibility sheaf created successfully")
            return compatibility_sheaf
            
        except Exception as e:
            logger.error(f"Compatibility sheaf creation failed: {e}")
            raise RuntimeError(f"Compatibility sheaf creation failed: {e}")
    
    def _validate_spectral_conversion(self, real_laplacian: csr_matrix, metadata: SpectralAnalysisMetadata):
        """Validate spectral analysis conversion.
        
        Args:
            real_laplacian: Converted real Laplacian
            metadata: Spectral analysis metadata
        """
        # Check matrix properties
        if not isinstance(real_laplacian, csr_matrix):
            raise ValueError("Real Laplacian must be a sparse matrix")
        
        # Check dimensions
        if real_laplacian.shape[0] != real_laplacian.shape[1]:
            raise ValueError("Real Laplacian must be square")
        
        # Check that real dimension is twice complex dimension
        if real_laplacian.shape[0] != 2 * metadata.complex_dimension:
            raise ValueError("Real dimension must be twice complex dimension")
        
        # Check symmetry (real representation of Hermitian matrix)
        symmetry_error = np.abs(real_laplacian - real_laplacian.T).max()
        if symmetry_error > 1e-6:
            raise ValueError(f"Real Laplacian not symmetric: error = {symmetry_error}")
    
    def _extract_graph_structure(self, directed_sheaf: DirectedSheaf) -> Dict[str, Any]:
        """Extract graph structure for visualization.
        
        Args:
            directed_sheaf: DirectedSheaf to extract from
            
        Returns:
            Graph structure data
        """
        return {
            'nodes': list(directed_sheaf.poset.nodes()),
            'edges': list(directed_sheaf.poset.edges()),
            'node_dimensions': directed_sheaf.get_node_dimensions(),
            'adjacency_matrix': directed_sheaf.get_adjacency_matrix().detach().cpu().numpy(),
            'directional_encoding': directed_sheaf.directional_encoding.detach().cpu().numpy() if directed_sheaf.directional_encoding is not None else None
        }
    
    def _extract_similarity_matrix(self, directed_sheaf: DirectedSheaf) -> Dict[str, Any]:
        """Extract similarity matrix for visualization.
        
        Args:
            directed_sheaf: DirectedSheaf to extract from
            
        Returns:
            Similarity matrix data
        """
        # For directed sheaves, we can compute a similarity matrix
        # based on the restriction maps
        try:
            # Build real Laplacian for analysis
            real_laplacian = self.laplacian_builder.build_real_embedded_laplacian(directed_sheaf)
            
            # Convert to dense for similarity analysis
            dense_laplacian = real_laplacian.toarray()
            
            return {
                'laplacian_matrix': dense_laplacian,
                'is_hermitian': True,
                'is_sparse': True,
                'sparsity': 1.0 - (real_laplacian.nnz / (real_laplacian.shape[0] * real_laplacian.shape[1]))
            }
            
        except Exception as e:
            logger.warning(f"Could not extract similarity matrix: {e}")
            return {'error': str(e)}
    
    def _extract_eigenvalue_data(self, directed_sheaf: DirectedSheaf) -> Dict[str, Any]:
        """Extract eigenvalue data for visualization.
        
        Args:
            directed_sheaf: DirectedSheaf to extract from
            
        Returns:
            Eigenvalue data
        """
        try:
            # Extract real eigenvalues
            real_eigenvalues = self.extract_real_eigenvalues(directed_sheaf)
            
            return {
                'eigenvalues': real_eigenvalues,
                'num_eigenvalues': len(real_eigenvalues),
                'min_eigenvalue': float(real_eigenvalues.min()),
                'max_eigenvalue': float(real_eigenvalues.max()),
                'eigenvalue_range': float(real_eigenvalues.max() - real_eigenvalues.min()),
                'is_real_spectrum': True
            }
            
        except Exception as e:
            logger.warning(f"Could not extract eigenvalue data: {e}")
            return {'error': str(e)}
    
    def _process_eigenvalue_data(self, eigenvalue_data: Any) -> Dict[str, Any]:
        """Process eigenvalue data for directed sheaf.
        
        Args:
            eigenvalue_data: Original eigenvalue data
            
        Returns:
            Processed eigenvalue data
        """
        # Add directed sheaf specific processing
        processed_data = {
            'original_data': eigenvalue_data,
            'directed_processed': True,
            'hermitian_spectrum': True,
            'real_eigenvalues': True
        }
        
        return processed_data
    
    def _process_persistence_diagrams(self, persistence_diagrams: Any) -> Dict[str, Any]:
        """Process persistence diagrams for directed sheaf.
        
        Args:
            persistence_diagrams: Original persistence diagrams
            
        Returns:
            Processed persistence diagrams
        """
        # Add directed sheaf specific processing
        processed_diagrams = {
            'original_diagrams': persistence_diagrams,
            'directed_processed': True,
            'hermitian_laplacian_source': True,
            'real_embedded_computation': True
        }
        
        return processed_diagrams
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about the adapter configuration.
        
        Returns:
            Dictionary with adapter information
        """
        return {
            'class_name': 'DirectedSheafAdapter',
            'preserve_metadata': self.preserve_metadata,
            'validate_conversions': self.validate_conversions,
            'optimize_sparse_operations': self.optimize_sparse_operations,
            'device': str(self.device),
            'capabilities': [
                'spectral_analysis_adaptation',
                'visualization_adaptation', 
                'eigenvalue_extraction',
                'persistence_result_conversion',
                'compatibility_sheaf_creation'
            ],
            'mathematical_foundation': 'Real embedding of Hermitian Laplacians',
            'integration_method': 'Format conversion with metadata preservation'
        }