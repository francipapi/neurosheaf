"""Main API for Neurosheaf analysis.

This module provides the high-level interface for conducting neural network
similarity analysis using persistent sheaf Laplacians.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import platform

from .utils.logging import setup_logger
from .utils.exceptions import ValidationError, ComputationError, ArchitectureError
from .utils.profiling import profile_memory, profile_time

# Import directed sheaf components
from .directed_sheaf import DirectedSheafBuilder, DirectedSheafAdapter
from .sheaf.assembly.builder import SheafBuilder
from .spectral.persistent import PersistentSpectralAnalyzer


class NeurosheafAnalyzer:
    """Main interface for neural network similarity analysis.
    
    This class provides a high-level API for analyzing neural networks using
    persistent sheaf Laplacians. It automatically handles device detection,
    memory management, and provides Mac-specific optimizations.
    
    Examples:
        >>> analyzer = NeurosheafAnalyzer()
        >>> model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU())
        >>> data = torch.randn(1000, 100)
        >>> results = analyzer.analyze(model, data)
        >>> cka_matrix = results['cka_matrix']
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        memory_limit_gb: float = 8.0,
        enable_profiling: bool = True,
        log_level: str = "INFO"
    ):
        """Initialize the Neurosheaf analyzer.
        
        Args:
            device: Device to use ('cpu', 'mps', 'cuda', or None for auto-detection)
            memory_limit_gb: Memory limit in GB for computations
            enable_profiling: Whether to enable performance profiling
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.logger = setup_logger("neurosheaf.analyzer", level=log_level)
        self.device = self._detect_device(device)
        self.memory_limit_gb = memory_limit_gb
        self.enable_profiling = enable_profiling
        
        # Mac-specific initialization
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = platform.processor() == "arm"
        
        # Initialize spectral analyzer for DTW comparisons
        self.spectral_analyzer = PersistentSpectralAnalyzer()
        
        self.logger.info(f"Initialized NeurosheafAnalyzer on {self.device}")
        if self.is_mac:
            self.logger.info(f"Mac detected: Apple Silicon = {self.is_apple_silicon}")
    
    def _detect_device(self, device: Optional[str] = None) -> torch.device:
        """Detect the optimal device for computation.
        
        Args:
            device: Optional device specification
            
        Returns:
            torch.device: The selected device
        """
        if device is not None:
            return torch.device(device)
        
        # Mac-specific device detection
        if platform.system() == "Darwin":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        
        # Other platforms
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        return torch.device("cpu")
    
    def analyze(
        self,
        model: nn.Module,
        data: torch.Tensor,
        batch_size: Optional[int] = None,
        layers: Optional[List[str]] = None,
        directed: bool = False,
        directionality_parameter: float = 0.25
    ) -> Dict[str, Any]:
        """Perform complete neurosheaf analysis.
        
        Supports both directed and undirected sheaf analysis. For directed analysis,
        the method constructs complex-valued stalks with Hermitian Laplacians to
        capture directional information in neural networks.
        
        Args:
            model: PyTorch neural network model
            data: Input data tensor
            batch_size: Batch size for processing (auto-detected if None)
            layers: Specific layers to analyze (all if None)
            directed: Whether to perform directed sheaf analysis
            directionality_parameter: q parameter controlling directional strength (0.0-1.0)
            
        Returns:
            Dictionary containing analysis results:
                - 'analysis_type': 'directed' or 'undirected'
                - 'sheaf': Constructed sheaf object
                - 'laplacian_metadata': Metadata about Laplacian construction
                - 'spectral_analysis': Spectral analysis results (if applicable)
                - 'device_info': Device and hardware information
                - 'performance': Performance metrics
                - 'directionality_parameter': q parameter (if directed)
                
        Raises:
            ValidationError: If input validation fails
            ComputationError: If analysis computation fails
            ArchitectureError: If model architecture is unsupported
        """
        analysis_type = "directed" if directed else "undirected"
        self.logger.info(f"Starting {analysis_type} neurosheaf analysis...")
        
        # Validate inputs
        self._validate_inputs(model, data)
        
        # Validate directionality parameter
        if directed and not (0.0 <= directionality_parameter <= 1.0):
            raise ValidationError("directionality_parameter must be between 0.0 and 1.0")
        
        # Move model and data to device
        model = model.to(self.device)
        data = data.to(self.device)
        
        try:
            if directed:
                return self._analyze_directed(model, data, directionality_parameter)
            else:
                return self._analyze_undirected(model, data)
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise ComputationError(f"Analysis failed: {e}")
    
    def _validate_inputs(self, model: nn.Module, data: torch.Tensor) -> None:
        """Validate input parameters.
        
        Args:
            model: PyTorch model to validate
            data: Input data to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(model, nn.Module):
            raise ValidationError("Model must be a PyTorch nn.Module")
        
        if not isinstance(data, torch.Tensor):
            raise ValidationError("Data must be a torch.Tensor")
        
        if data.dim() < 2:
            raise ValidationError("Data must have at least 2 dimensions")
        
        if data.shape[0] == 0:
            raise ValidationError("Data cannot be empty")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device and hardware information.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'device': str(self.device),
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__
        }
        
        # Mac-specific information
        if self.is_mac:
            info['is_apple_silicon'] = self.is_apple_silicon
            if self.device.type == 'mps':
                info['mps_available'] = torch.backends.mps.is_available()
        
        # CUDA information
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['cuda_device_count'] = torch.cuda.device_count()
            if self.device.type == 'cuda':
                info['cuda_device_name'] = torch.cuda.get_device_name()
        
        return info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information.
        
        Returns:
            Dictionary with memory information
        """
        import psutil
        
        # System memory
        memory_info = {
            'system_total_gb': psutil.virtual_memory().total / (1024**3),
            'system_available_gb': psutil.virtual_memory().available / (1024**3),
            'system_used_gb': psutil.virtual_memory().used / (1024**3),
            'system_percent': psutil.virtual_memory().percent
        }
        
        # Device-specific memory
        if self.device.type == 'cuda' and torch.cuda.is_available():
            memory_info['cuda_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_info['cuda_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['cuda_cached_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        elif self.device.type == 'mps' and self.is_apple_silicon:
            # Apple Silicon uses unified memory
            memory_info['unified_memory'] = True
            memory_info['mps_allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
        
        return memory_info
    
    def _analyze_directed(self, model: nn.Module, data: torch.Tensor, directionality_parameter: float) -> Dict[str, Any]:
        """Perform directed sheaf analysis.
        
        Args:
            model: PyTorch model
            data: Input data tensor
            directionality_parameter: q parameter controlling directional strength
            
        Returns:
            Dictionary with directed analysis results
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Building directed sheaf with q={directionality_parameter}")
        
        # Step 1: Build base real sheaf
        sheaf_builder = SheafBuilder()
        base_sheaf = sheaf_builder.build_from_activations(model, data)
        
        # Step 2: Convert to directed sheaf
        directed_builder = DirectedSheafBuilder(
            directionality_parameter=directionality_parameter,
            device=self.device
        )
        directed_sheaf = directed_builder.build_from_sheaf(base_sheaf)
        
        # Step 3: Adapt for spectral analysis if needed
        adapter = DirectedSheafAdapter(device=self.device)
        real_laplacian, laplacian_metadata = adapter.adapt_for_spectral_analysis(directed_sheaf)
        
        construction_time = time.time() - start_time
        
        results = {
            'analysis_type': 'directed',
            'directed_sheaf': directed_sheaf,
            'base_sheaf': base_sheaf,
            'real_laplacian': real_laplacian,
            'laplacian_metadata': laplacian_metadata,
            'directionality_parameter': directionality_parameter,
            'construction_time': construction_time,
            'device_info': self._get_device_info(),
            'memory_info': self._get_memory_info(),
            'performance': {
                'construction_time': construction_time,
                'complex_dimension': laplacian_metadata.complex_dimension,
                'real_dimension': laplacian_metadata.real_dimension,
                'sparsity': laplacian_metadata.sparsity
            }
        }
        
        self.logger.info(f"Directed analysis completed in {construction_time:.3f}s")
        return results
    
    def _analyze_undirected(self, model: nn.Module, data: torch.Tensor) -> Dict[str, Any]:
        """Perform undirected sheaf analysis.
        
        Args:
            model: PyTorch model
            data: Input data tensor
            
        Returns:
            Dictionary with undirected analysis results
        """
        import time
        start_time = time.time()
        
        self.logger.info("Building undirected sheaf")
        
        # Build standard real sheaf
        sheaf_builder = SheafBuilder()
        sheaf = sheaf_builder.build_from_activations(model, data)
        
        construction_time = time.time() - start_time
        
        results = {
            'analysis_type': 'undirected',
            'sheaf': sheaf,
            'construction_time': construction_time,
            'device_info': self._get_device_info(),
            'memory_info': self._get_memory_info(),
            'performance': {
                'construction_time': construction_time,
                'num_nodes': len(sheaf.stalks),
                'num_edges': len(sheaf.restrictions)
            }
        }
        
        self.logger.info(f"Undirected analysis completed in {construction_time:.3f}s")
        return results
    
    def analyze_directed(
        self,
        model: nn.Module,
        data: torch.Tensor,
        directionality_parameter: float = 0.25,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform directed sheaf analysis.
        
        Convenience method that calls analyze() with directed=True.
        
        Args:
            model: PyTorch neural network model
            data: Input data tensor
            directionality_parameter: q parameter controlling directional strength (0.0-1.0)
            **kwargs: Additional arguments passed to analyze()
            
        Returns:
            Dictionary containing directed analysis results
        """
        return self.analyze(
            model=model,
            data=data,
            directed=True,
            directionality_parameter=directionality_parameter,
            **kwargs
        )
    
    def compare_directed_undirected(
        self,
        model: nn.Module,
        data: torch.Tensor,
        directionality_parameter: float = 0.25
    ) -> Dict[str, Any]:
        """Compare directed vs undirected analysis.
        
        Performs both directed and undirected sheaf analysis and compares
        the results, providing insights into the directional structure
        of the neural network.
        
        Args:
            model: PyTorch neural network model
            data: Input data tensor
            directionality_parameter: q parameter controlling directional strength (0.0-1.0)
            
        Returns:
            Dictionary containing comparison results:
                - 'directed_results': Results from directed analysis
                - 'undirected_results': Results from undirected analysis
                - 'comparison': Comparison metrics and insights
                - 'performance_comparison': Performance comparison
        """
        self.logger.info("Starting directed vs undirected comparison...")
        
        # Perform both analyses
        directed_results = self.analyze_directed(model, data, directionality_parameter)
        undirected_results = self.analyze(model, data, directed=False)
        
        # Compute comparison metrics
        comparison = self._compute_comparison_metrics(directed_results, undirected_results)
        
        results = {
            'directed_results': directed_results,
            'undirected_results': undirected_results,
            'comparison': comparison,
            'performance_comparison': {
                'directed_time': directed_results['construction_time'],
                'undirected_time': undirected_results['construction_time'],
                'time_ratio': directed_results['construction_time'] / undirected_results['construction_time'],
                'memory_overhead': directed_results['performance'].get('real_dimension', 0) / max(1, undirected_results['performance'].get('num_nodes', 1)),
                'directionality_parameter': directionality_parameter
            }
        }
        
        self.logger.info("Comparison completed")
        return results
    
    def _compute_comparison_metrics(self, directed_results: Dict[str, Any], undirected_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comparison metrics between directed and undirected results.
        
        Args:
            directed_results: Results from directed analysis
            undirected_results: Results from undirected analysis
            
        Returns:
            Dictionary with comparison metrics
        """
        # Extract basic metrics
        directed_sheaf = directed_results['directed_sheaf']
        undirected_sheaf = undirected_results['sheaf']
        
        # Compute structural differences
        comparison = {
            'structural_comparison': {
                'same_nodes': directed_sheaf.poset.nodes() == undirected_sheaf.poset.nodes(),
                'same_edges': directed_sheaf.poset.edges() == undirected_sheaf.poset.edges(),
                'directed_nodes': len(directed_sheaf.complex_stalks),
                'undirected_nodes': len(undirected_sheaf.stalks),
                'directed_edges': len(directed_sheaf.directed_restrictions),
                'undirected_edges': len(undirected_sheaf.restrictions)
            },
            'mathematical_properties': {
                'directed_laplacian_type': 'hermitian',
                'undirected_laplacian_type': 'symmetric',
                'directed_stalks_type': 'complex',
                'undirected_stalks_type': 'real',
                'directionality_parameter': directed_results['directionality_parameter']
            },
            'performance_insights': {
                'memory_factor': directed_results['laplacian_metadata'].real_dimension / max(1, undirected_results['performance'].get('num_nodes', 1)),
                'time_factor': directed_results['construction_time'] / undirected_results['construction_time'],
                'sparsity_directed': directed_results['laplacian_metadata'].sparsity,
                'complexity_increase': 'Real embedding doubles dimensions'
            }
        }
        
        return comparison
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.
        
        Returns:
            Dictionary with system and hardware information
        """
        return {
            'device_info': self._get_device_info(),
            'memory_info': self._get_memory_info(),
            'analyzer_config': {
                'device': str(self.device),
                'memory_limit_gb': self.memory_limit_gb,
                'enable_profiling': self.enable_profiling,
                'is_mac': self.is_mac,
                'is_apple_silicon': self.is_apple_silicon
            }
        }
    
    def profile_memory_usage(self, model: nn.Module, data: torch.Tensor) -> Dict[str, Any]:
        """Profile memory usage for the given model and data.
        
        Args:
            model: PyTorch model to profile
            data: Input data for profiling
            
        Returns:
            Dictionary with memory profiling results
        """
        if not self.enable_profiling:
            self.logger.warning("Profiling is disabled")
            return {'status': 'disabled'}
        
        @profile_memory()
        def _profile_forward_pass():
            model.eval()
            with torch.no_grad():
                return model(data)
        
        # Run profiling
        output = _profile_forward_pass()
        
        return {
            'status': 'completed',
            'output_shape': output.shape,
            'memory_info': self._get_memory_info(),
            'device_info': self._get_device_info()
        }
    
    def compare_networks(self,
                        model1: nn.Module,
                        model2: nn.Module,
                        data: torch.Tensor,
                        method: str = 'dtw',
                        eigenvalue_index: Optional[int] = None,
                        multivariate: bool = False,
                        filtration_type: str = 'threshold',
                        n_steps: int = 50,
                        **kwargs) -> Dict[str, Any]:
        """Compare two neural networks using eigenvalue evolution analysis.
        
        This method analyzes both networks, constructs sheaves, and compares their
        eigenvalue evolution patterns using Dynamic Time Warping (DTW) or other
        similarity measures.
        
        Args:
            model1: First neural network to compare
            model2: Second neural network to compare
            data: Input data for activation analysis
            method: Comparison method ('dtw', 'euclidean', 'cosine')
            eigenvalue_index: Index of eigenvalue to compare (None = all)
            multivariate: Whether to use multivariate DTW
            filtration_type: Type of filtration ('threshold', 'cka_based', 'custom')
            n_steps: Number of filtration steps (default: 50)
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing:
            - similarity_score: Overall similarity score
            - dtw_comparison: Detailed DTW comparison results
            - model1_analysis: Full analysis of model1
            - model2_analysis: Full analysis of model2
            - comparison_metadata: Comparison metadata
            
        Examples:
            >>> analyzer = NeurosheafAnalyzer()
            >>> resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
            >>> vgg = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
            >>> data = torch.randn(100, 3, 224, 224)
            >>> comparison = analyzer.compare_networks(resnet, vgg, data, 
            ...                                      filtration_type='threshold', n_steps=100)
            >>> print(f"Similarity: {comparison['similarity_score']:.3f}")
        """
        self.logger.info(f"Comparing two networks using {method} method")
        
        # Analyze both models to get sheaves
        analysis1 = self.analyze(model1, data, **kwargs)
        analysis2 = self.analyze(model2, data, **kwargs)
        
        # Extract sheaves for spectral comparison
        sheaf1 = analysis1['sheaf']
        sheaf2 = analysis2['sheaf']
        
        if method == 'dtw':
            # Use DTW for eigenvalue evolution comparison
            comparison_result = self.spectral_analyzer.compare_filtration_evolution(
                sheaf1, sheaf2,
                eigenvalue_index=eigenvalue_index,
                multivariate=multivariate,
                filtration_type=filtration_type,
                n_steps=n_steps
            )
            similarity_score = comparison_result['similarity_metrics']['combined_similarity']
            
        else:
            # Fallback to simpler comparison methods
            comparison_result = self._compare_networks_simple(analysis1, analysis2, method)
            similarity_score = comparison_result.get('similarity_score', 0.0)
        
        self.logger.info(f"Network comparison completed: similarity = {similarity_score:.3f}")
        
        return {
            'similarity_score': similarity_score,
            'method': method,
            'dtw_comparison': comparison_result if method == 'dtw' else None,
            'model1_analysis': analysis1,
            'model2_analysis': analysis2,
            'comparison_metadata': {
                'eigenvalue_index': eigenvalue_index,
                'multivariate': multivariate,
                'filtration_type': filtration_type,
                'n_steps': n_steps,
                'data_shape': data.shape,
                'device': str(self.device)
            }
        }
    
    def compare_multiple_networks(self,
                                 models: List[nn.Module],
                                 data: torch.Tensor,
                                 method: str = 'dtw',
                                 eigenvalue_index: Optional[int] = None,
                                 multivariate: bool = False,
                                 filtration_type: str = 'threshold',
                                 n_steps: int = 50,
                                 **kwargs) -> Dict[str, Any]:
        """Compare multiple neural networks pairwise.
        
        Args:
            models: List of neural networks to compare
            data: Input data for activation analysis
            method: Comparison method ('dtw', 'euclidean', 'cosine')
            eigenvalue_index: Index of eigenvalue to compare (None = all)
            multivariate: Whether to use multivariate DTW
            filtration_type: Type of filtration ('threshold', 'cka_based', 'custom')
            n_steps: Number of filtration steps (default: 50)
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing:
            - distance_matrix: Pairwise similarity matrix
            - similarity_rankings: Ranked similarity results
            - individual_analyses: Analysis results for each model
            - cluster_analysis: Clustering results based on similarity
            
        Examples:
            >>> analyzer = NeurosheafAnalyzer()
            >>> models = [resnet18, resnet50, vgg16, densenet121]
            >>> data = torch.randn(100, 3, 224, 224)
            >>> comparison = analyzer.compare_multiple_networks(models, data,
            ...                                               filtration_type='cka_based', n_steps=30)
            >>> print(comparison['distance_matrix'])
        """
        self.logger.info(f"Comparing {len(models)} networks using {method} method")
        
        # Analyze all models to get sheaves
        analyses = []
        sheaves = []
        
        for i, model in enumerate(models):
            self.logger.debug(f"Analyzing model {i+1}/{len(models)}")
            analysis = self.analyze(model, data, **kwargs)
            analyses.append(analysis)
            sheaves.append(analysis['sheaf'])
        
        if method == 'dtw':
            # Use DTW for multiple sheaf comparison
            comparison_result = self.spectral_analyzer.compare_multiple_sheaves(
                sheaves,
                eigenvalue_index=eigenvalue_index,
                multivariate=multivariate,
                filtration_type=filtration_type,
                n_steps=n_steps
            )
            
            distance_matrix = comparison_result['distance_matrix']
            similarity_rankings = comparison_result['similarity_rankings']
            
        else:
            # Fallback to simpler comparison methods
            distance_matrix, similarity_rankings = self._compare_multiple_simple(
                analyses, method
            )
        
        # Perform cluster analysis on similarity matrix
        cluster_analysis = self._perform_cluster_analysis(distance_matrix)
        
        self.logger.info(f"Multiple network comparison completed")
        
        return {
            'distance_matrix': distance_matrix,
            'similarity_rankings': similarity_rankings,
            'individual_analyses': analyses,
            'cluster_analysis': cluster_analysis,
            'method': method,
            'comparison_metadata': {
                'n_models': len(models),
                'eigenvalue_index': eigenvalue_index,
                'multivariate': multivariate,
                'filtration_type': filtration_type,
                'n_steps': n_steps,
                'data_shape': data.shape,
                'device': str(self.device)
            }
        }
    
    def _compare_networks_simple(self,
                                analysis1: Dict[str, Any],
                                analysis2: Dict[str, Any],
                                method: str) -> Dict[str, Any]:
        """Simple network comparison using basic metrics."""
        if method == 'euclidean':
            # Compare CKA matrices using Euclidean distance
            cka1 = analysis1.get('cka_matrix', torch.zeros(1, 1))
            cka2 = analysis2.get('cka_matrix', torch.zeros(1, 1))
            
            if cka1.shape != cka2.shape:
                # Pad smaller matrix
                max_size = max(cka1.shape[0], cka2.shape[0])
                cka1_padded = torch.zeros(max_size, max_size)
                cka2_padded = torch.zeros(max_size, max_size)
                
                cka1_padded[:cka1.shape[0], :cka1.shape[1]] = cka1
                cka2_padded[:cka2.shape[0], :cka2.shape[1]] = cka2
                
                cka1, cka2 = cka1_padded, cka2_padded
            
            distance = torch.norm(cka1 - cka2).item()
            max_norm = max(torch.norm(cka1).item(), torch.norm(cka2).item())
            similarity_score = 1.0 - (distance / max_norm) if max_norm > 0 else 1.0
            
        elif method == 'cosine':
            # Compare using cosine similarity
            cka1 = analysis1.get('cka_matrix', torch.zeros(1, 1)).flatten()
            cka2 = analysis2.get('cka_matrix', torch.zeros(1, 1)).flatten()
            
            # Pad to same length
            max_len = max(len(cka1), len(cka2))
            cka1_padded = torch.zeros(max_len)
            cka2_padded = torch.zeros(max_len)
            
            cka1_padded[:len(cka1)] = cka1
            cka2_padded[:len(cka2)] = cka2
            
            cosine_sim = torch.nn.functional.cosine_similarity(
                cka1_padded.unsqueeze(0), cka2_padded.unsqueeze(0)
            )
            similarity_score = cosine_sim.item()
            
        else:
            raise ValueError(f"Unknown comparison method: {method}")
        
        return {
            'similarity_score': similarity_score,
            'method': method,
            'distance': distance if method == 'euclidean' else 1.0 - similarity_score
        }
    
    def _compare_multiple_simple(self,
                                analyses: List[Dict[str, Any]],
                                method: str) -> Tuple[torch.Tensor, List[Dict]]:
        """Simple multiple network comparison."""
        n_models = len(analyses)
        distance_matrix = torch.zeros(n_models, n_models)
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                comparison = self._compare_networks_simple(analyses[i], analyses[j], method)
                distance = comparison['distance']
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        # Create similarity rankings
        similarity_rankings = []
        for i in range(n_models):
            distances = distance_matrix[i, :].clone()
            distances[i] = float('inf')  # Exclude self
            
            sorted_indices = torch.argsort(distances)
            ranking = {
                'model_index': i,
                'most_similar': [
                    {
                        'model_index': int(idx),
                        'distance': float(distances[idx]),
                        'similarity': 1.0 - float(distances[idx])
                    }
                    for idx in sorted_indices[:min(5, n_models-1)]
                ]
            }
            similarity_rankings.append(ranking)
        
        return distance_matrix.numpy(), similarity_rankings
    
    def _perform_cluster_analysis(self, distance_matrix: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Perform clustering analysis on similarity matrix."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            
            # Convert to numpy if needed
            if isinstance(distance_matrix, torch.Tensor):
                distance_matrix = distance_matrix.numpy()
            
            # Convert distance to similarity
            similarity_matrix = 1.0 - distance_matrix
            
            # Perform clustering for different numbers of clusters
            n_samples = distance_matrix.shape[0]
            best_score = -1
            best_n_clusters = 2
            
            for n_clusters in range(2, min(n_samples, 6)):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='precomputed',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(distance_matrix)
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
                except Exception:
                    continue
            
            # Final clustering with best parameters
            clustering = AgglomerativeClustering(
                n_clusters=best_n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
            
            return {
                'n_clusters': best_n_clusters,
                'labels': labels.tolist(),
                'silhouette_score': best_score,
                'cluster_assignments': {
                    f'cluster_{i}': [idx for idx, label in enumerate(labels) if label == i]
                    for i in range(best_n_clusters)
                }
            }
            
        except ImportError:
            self.logger.warning("sklearn not available, skipping cluster analysis")
            return {'status': 'sklearn_not_available'}
        except Exception as e:
            self.logger.warning(f"Cluster analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}