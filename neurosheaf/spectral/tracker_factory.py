# neurosheaf/spectral/tracker_factory.py
"""
Factory for creating appropriate subspace tracker based on construction method.

This factory enables seamless integration of different tracking methods without
modifying existing code. It automatically routes to the appropriate tracker
based on the sheaf's construction method.
"""

from typing import Dict, Any, Optional
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class SubspaceTrackerFactory:
    """
    Factory for creating appropriate subspace tracker based on construction method.
    
    Ensures seamless integration without modifying existing code by providing
    a clean abstraction layer that routes to method-specific implementations.
    """
    
    # Registry of available tracker implementations
    _tracker_registry = {
        'standard': 'neurosheaf.spectral.tracker.SubspaceTracker',
        'scaled_procrustes': 'neurosheaf.spectral.tracker.SubspaceTracker', 
        'whitened_procrustes': 'neurosheaf.spectral.tracker.SubspaceTracker',
        'fx_unified_whitened': 'neurosheaf.spectral.tracker.SubspaceTracker',
        'gromov_wasserstein': 'neurosheaf.spectral.gw.gw_subspace_tracker.GWSubspaceTracker'
    }
    
    @classmethod
    def create_tracker(cls, construction_method: str = 'standard', **kwargs):
        """
        Create appropriate tracker based on construction method.
        
        Args:
            construction_method: Construction method used for sheaf
                - 'standard': Standard Procrustes-based construction
                - 'scaled_procrustes': Scaled Procrustes variant
                - 'whitened_procrustes': Whitened Procrustes variant  
                - 'fx_unified_whitened': FX-based unified whitened construction
                - 'gromov_wasserstein': Gromov-Wasserstein based construction
            **kwargs: Additional arguments passed to tracker constructor
            
        Returns:
            Appropriate SubspaceTracker instance
            
        Raises:
            ValueError: If construction_method is not supported
            ImportError: If required tracker module cannot be imported
        """
        if construction_method not in cls._tracker_registry:
            supported_methods = ', '.join(cls._tracker_registry.keys())
            raise ValueError(
                f"Unsupported construction method: '{construction_method}'. "
                f"Supported methods: {supported_methods}"
            )
        
        tracker_class_path = cls._tracker_registry[construction_method]
        
        try:
            # Import and instantiate the appropriate tracker
            tracker_class = cls._import_tracker_class(tracker_class_path)
            tracker_instance = tracker_class(**kwargs)
            
            logger.info(f"Created {tracker_class.__name__} for construction method: {construction_method}")
            return tracker_instance
            
        except ImportError as e:
            logger.error(f"Failed to import tracker for method {construction_method}: {e}")
            # Fallback to standard tracker if GW-specific tracker fails
            if construction_method == 'gromov_wasserstein':
                logger.warning("Falling back to standard SubspaceTracker for GW method")
                return cls._create_fallback_tracker(**kwargs)
            raise
        
        except Exception as e:
            logger.error(f"Failed to create tracker for method {construction_method}: {e}")
            raise
    
    @classmethod 
    def _import_tracker_class(cls, class_path: str):
        """
        Dynamically import tracker class from module path.
        
        Args:
            class_path: Full module path to tracker class (e.g., 'module.Class')
            
        Returns:
            Tracker class
        """
        module_path, class_name = class_path.rsplit('.', 1)
        
        # Handle different import scenarios
        if module_path.startswith('neurosheaf.spectral.tracker'):
            from ..spectral.tracker import SubspaceTracker
            return SubspaceTracker
        elif module_path.startswith('neurosheaf.spectral.gw'):
            from ..spectral.gw.gw_subspace_tracker import GWSubspaceTracker
            return GWSubspaceTracker
        else:
            # Generic dynamic import
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
    
    @classmethod
    def _create_fallback_tracker(cls, **kwargs):
        """
        Create fallback standard tracker when specialized tracker fails.
        
        Args:
            **kwargs: Arguments for tracker constructor
            
        Returns:
            Standard SubspaceTracker instance
        """
        from ..spectral.tracker import SubspaceTracker
        return SubspaceTracker(**kwargs)
    
    @classmethod
    def register_tracker(cls, construction_method: str, tracker_class_path: str):
        """
        Register a new tracker implementation for a construction method.
        
        Args:
            construction_method: Name of construction method
            tracker_class_path: Full module path to tracker class
        """
        cls._tracker_registry[construction_method] = tracker_class_path
        logger.info(f"Registered tracker for method '{construction_method}': {tracker_class_path}")
    
    @classmethod
    def get_supported_methods(cls) -> Dict[str, str]:
        """
        Get dictionary of supported construction methods and their tracker classes.
        
        Returns:
            Dictionary mapping construction method names to tracker class paths
        """
        return cls._tracker_registry.copy()
    
    @classmethod
    def is_method_supported(cls, construction_method: str) -> bool:
        """
        Check if a construction method is supported.
        
        Args:
            construction_method: Name of construction method to check
            
        Returns:
            True if method is supported, False otherwise
        """
        return construction_method in cls._tracker_registry


def create_tracker_for_sheaf(sheaf, **kwargs):
    """
    Convenience function to create appropriate tracker for a given sheaf.
    
    Automatically detects construction method from sheaf metadata and
    creates the appropriate tracker instance.
    
    Args:
        sheaf: Sheaf object with metadata containing construction_method
        **kwargs: Additional arguments for tracker constructor
        
    Returns:
        Appropriate SubspaceTracker instance
    """
    construction_method = getattr(sheaf, 'metadata', {}).get('construction_method', 'standard')
    
    logger.debug(f"Auto-detected construction method from sheaf metadata: {construction_method}")
    
    return SubspaceTrackerFactory.create_tracker(
        construction_method=construction_method, 
        **kwargs
    )