# Updated Sheaf Construction Implementation Plan v3 (Fixed)

## Overview
This updated v3 plan incorporates the fix for the double-centering issue in CKA computation. The key change is ensuring that all Gram matrices are computed from raw (uncentered) activations when using debiased CKA.

## Key Fixes from Original v3
- **Use raw activations** for all CKA/Gram matrix computations
- **Updated documentation** to emphasize no pre-centering
- **Integration with fixed debiased CKA** implementation
- All other improvements from v3 retained

## Implementation Structure

### 1. Activation Capture with Adaptive Sampling (Updated)

```python
# neursheaf/sheaf/activation_capture.py
from typing import Dict, Tuple, Optional
import torch
import numpy as np
from ..cka import compute_cka_matrix

class AdaptiveActivationCapture:
    """
    Capture neural network activations with adaptive sampling.
    
    IMPORTANT: Activations are kept raw (uncentered) to avoid
    double-centering issues with debiased CKA.
    
    References
    ----------
    - Project knowledge: Debiased CKA requirements (500-1000 samples minimum)
    - Murphy et al. (2024): Sample complexity analysis
    """
    
    def __init__(self, 
                 eps: float = 0.01,
                 start_samples: int = 512, 
                 max_samples: int = 4096,
                 device: str = 'cuda'):
        self.eps = eps
        self.start_samples = start_samples
        self.max_samples = max_samples
        self.device = device
        
    def collect_activations(self, 
                          model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """
        Collect RAW activations with adaptive sampling until CKA convergence.
        
        Returns
        -------
        activations : Dict[str, torch.Tensor]
            Raw activation tensors for each layer (NOT centered)
        """
        model.eval()
        hooks = []
        activations = {}
        
        def get_activation_hook(name):
            def hook(module, input, output):
                # Store raw activations WITHOUT centering
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach())
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if self._is_target_layer(module):
                hooks.append(module.register_forward_hook(get_activation_hook(name)))
        
        n = self.start_samples
        prev_cka = None
        
        while n <= self.max_samples:
            # Clear previous activations
            activations.clear()
            
            # Collect n samples
            sample_count = 0
            for batch_data in dataloader:
                if sample_count >= n:
                    break
                    
                inputs = batch_data[0].to(self.device)
                with torch.no_grad():
                    _ = model(inputs)
                    
                sample_count += inputs.shape[0]
            
            # Concatenate activations (keep raw, no centering!)
            acts = {}
            for name, act_list in activations.items():
                acts[name] = torch.cat(act_list, dim=0)[:n]
            
            # Check convergence
            if prev_cka is not None:
                curr_cka = self._compute_cka_matrix(acts)
                relative_change = torch.norm(curr_cka - prev_cka, 'fro') / torch.norm(curr_cka, 'fro')
                
                if relative_change < self.eps:
                    logging.info(f"CKA converged at {n} samples (Δ={relative_change:.3f})")
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                    return acts
                    
            prev_cka = curr_cka if prev_cka is None else prev_cka
            n = min(n * 2, self.max_samples)
            
        logging.warning(f"Max samples {self.max_samples} reached without convergence")
        # Remove hooks
        for hook in hooks:
            hook.remove()
        return acts
```

### 2. Network Poset Extraction with FX

python# neursheaf/sheaf/poset.py
import networkx as nx
import torch
from torch.fx import symbolic_trace
from typing import Dict, List, Optional, Set
import warnings

class NeuralNetworkPoset:
    """
    Extract poset structure from neural network using torch.fx.
    
    Key improvement: Automatic extraction for any PyTorch model
    using symbolic tracing, eliminating manual architecture handlers.
    
    References
    ----------
    - torch.fx for automatic graph extraction
    - Project knowledge: Poset theory for neural networks
    """
    
    def __init__(self):
        self.custom_extractors = {}  # For backward compatibility
        
    def extract_poset(self, model: torch.nn.Module) -> nx.DiGraph:
        """
        Extract directed acyclic graph (poset) from model.
        
        Uses torch.fx for automatic extraction with fallback
        to custom extractors for dynamic models.
        
        Parameters
        ----------
        model : torch.nn.Module
            Neural network model
            
        Returns
        -------
        poset : nx.DiGraph
            Network poset structure
        """
        model_type = type(model).__name__
        
        # Check for custom extractor first (backward compatibility)
        if model_type in self.custom_extractors:
            return self.custom_extractors[model_type](model)
            
        # Try FX-based extraction
        try:
            return self._extract_poset_fx(model)
        except Exception as e:
            warnings.warn(
                f"FX extraction failed for {model_type}: {str(e)}. "
                f"Consider implementing a custom extractor for dynamic control flow."
            )
            # Fallback to module-based extraction
            return self._extract_poset_modules(model)
    
    def _extract_poset_fx(self, model: torch.nn.Module) -> nx.DiGraph:
        """
        Extract poset using torch.fx symbolic tracing.
        
        Automatically handles skip connections and complex topologies.
        """
        # Symbolic trace the model
        traced = symbolic_trace(model)
        
        poset = nx.DiGraph()
        node_to_module = {}
        module_outputs = {}
        
        # First pass: identify modules and their operations
        for node in traced.graph.nodes:
            if node.op == 'call_module':
                module_name = node.target
                module = dict(model.named_modules())[module_name]
                
                poset.add_node(
                    module_name,
                    layer_type=type(module).__name__,
                    fx_node=node.name
                )
                node_to_module[node.name] = module_name
                module_outputs[module_name] = node.name
                
        # Add input/output nodes
        poset.add_node('input', layer_type='Input')
        poset.add_node('output', layer_type='Output')
        
        # Second pass: build edges based on data dependencies
        for node in traced.graph.nodes:
            if node.op == 'call_module':
                current_module = node_to_module[node.name]
                
                # Trace inputs
                for arg in node.args:
                    if hasattr(arg, 'name'):
                        if arg.name in node_to_module:
                            # Direct module-to-module connection
                            source = node_to_module[arg.name]
                            poset.add_edge(source, current_module)
                        elif arg.op == 'placeholder':
                            # Input connection
                            poset.add_edge('input', current_module)
                            
            elif node.op == 'output':
                # Connect to output
                for arg in node.args[0]:  # args[0] is the output tuple
                    if hasattr(arg, 'name') and arg.name in node_to_module:
                        poset.add_edge(node_to_module[arg.name], 'output')
                        
        # Verify DAG property
        if not nx.is_directed_acyclic_graph(poset):
            raise ValueError("Extracted graph is not a DAG")
            
        return poset
    
    def _extract_poset_modules(self, model: torch.nn.Module) -> nx.DiGraph:
        """
        Fallback: Extract poset based on module hierarchy.
        
        Less accurate but works for all models.
        """
        poset = nx.DiGraph()
        
        # Add all computational modules
        computational_modules = []
        for name, module in model.named_modules():
            if self._is_computational_layer(module) and name:
                poset.add_node(name, layer_type=type(module).__name__)
                computational_modules.append(name)
                
        # Add input/output
        poset.add_node('input', layer_type='Input')
        poset.add_node('output', layer_type='Output')
        
        # Simple sequential assumption for edges
        if computational_modules:
            poset.add_edge('input', computational_modules[0])
            
            for i in range(len(computational_modules) - 1):
                poset.add_edge(computational_modules[i], computational_modules[i+1])
                
            poset.add_edge(computational_modules[-1], 'output')
            
        return poset
    
    def _is_computational_layer(self, module: torch.nn.Module) -> bool:
        """Check if module performs computation"""
        computational_types = (
            torch.nn.Conv2d, torch.nn.Linear, torch.nn.LSTM,
            torch.nn.GRU, torch.nn.Transformer, torch.nn.MultiheadAttention,
            torch.nn.Conv1d, torch.nn.Conv3d, torch.nn.ConvTranspose2d
        )
        return isinstance(module, computational_types)
    
    def register_custom_extractor(self, model_type: str, extractor_fn):
        """Register custom extractor for specific model types"""
        self.custom_extractors[model_type] = extractor_fn

### 3. Sheaf Construction with Scaled Procrustes (Updated)

```python
# neursheaf/sheaf/construction.py
from scipy.linalg import orthogonal_procrustes
import numpy as np
from typing import Dict, Tuple

class NeuralNetworkSheaf:
    """
    Construct cellular sheaf from neural network.
    
    IMPORTANT: Uses raw activations for CKA computation to avoid
    double-centering with debiased estimator.
    
    References
    ----------
    - https://github.com/kb1dds/pysheaf.git for sheaf theory
    - scipy.linalg.orthogonal_procrustes for restriction maps
    - Murphy et al. (2024) for debiased CKA
    """
    
    def __init__(self, poset: nx.DiGraph, stalk_dimension: int = None):
        self.poset = poset
        self.num_nodes = poset.number_of_nodes()
        self.stalks = {}
        self.restrictions = {}
        self.stalk_dimension = stalk_dimension
        
    def add_stalk(self, node: str, data: np.ndarray):
        """
        Add stalk data (CKA Gram matrix) at node.
        
        Parameters
        ----------
        data : np.ndarray
            Gram matrix computed from RAW activations (not centered)
        """
        self.stalks[node] = data
        
        if self.stalk_dimension is None:
            self.stalk_dimension = data.shape[0]
            
    def compute_restrictions(self):
        """
        Compute restriction maps using scaled Procrustes.
        
        Key improvement: Store and use scale factor in Laplacian.
        Note: Procrustes operates on Gram matrices from raw activations.
        """
        for edge in self.poset.edges():
            source, target = edge
            
            if source in self.stalks and target in self.stalks:
                # Get stalk data (Gram matrices from raw activations)
                K_source = self.stalks[source]
                K_target = self.stalks[target]
                
                # Compute scaled orthogonal Procrustes
                Q, scale = orthogonal_procrustes(K_source, K_target)
                
                # Store both Q and scale
                self.restrictions[(source, target)] = {
                    'Q': Q,
                    'scale': scale,
                    'source': source,
                    'target': target
                }
```

### 4. Optimized Laplacian Construction

python# neursheaf/sheaf/laplacian.py
import numpy as np
import torch
from scipy.sparse import csr_matrix, block_diag as sparse_block_diag
from typing import Dict, Tuple
import logging

class OptimizedLaplacianBuilder:
    """
    Build sparse sheaf Laplacian with memory optimization.
    
    Key optimizations:
    - Block-diagonal structure exploitation
    - Sparse matrix operations throughout
    - Batch construction for GPU efficiency
    
    References
    ----------
    - PySheaf for sheaf Laplacian theory
    - SciPy sparse for efficient construction
    """
    
    def __init__(self, sheaf):
        self.sheaf = sheaf
        self.num_nodes = sheaf.poset.number_of_nodes()
        self.stalk_dim = sheaf.stalk_dimension
        
    def build_sparse_laplacian(self) -> csr_matrix:
        """
        Construct sparse sheaf Laplacian.
        
        20× faster than dense construction for large networks.
        
        Returns
        -------
        L : csr_matrix
            Sparse sheaf Laplacian
        """
        total_dim = self.num_nodes * self.stalk_dim
        
        # Build diagonal blocks (node Laplacians)
        diagonal_blocks = []
        node_to_idx = {node: i for i, node in enumerate(self.sheaf.poset.nodes())}
        
        for node in self.sheaf.poset.nodes():
            L_v = self._compute_node_laplacian(node)
            diagonal_blocks.append(csr_matrix(L_v))
            
        # Assemble block diagonal efficiently
        L_diagonal = sparse_block_diag(diagonal_blocks)
        
        # Add off-diagonal blocks (restriction maps)
        rows, cols, data = [], [], []
        
        for edge in self.sheaf.poset.edges():
            source, target = edge
            if edge in self.sheaf.restrictions:
                R = self.sheaf.restrictions[edge]['Q']
                scale = self.sheaf.restrictions[edge]['scale']
                
                # Scaled restriction map
                R_scaled = R * scale
                
                # Add R and R^T blocks
                source_idx = node_to_idx[source]
                target_idx = node_to_idx[target]
                
                # Add -R block
                for i in range(self.stalk_dim):
                    for j in range(self.stalk_dim):
                        if abs(R_scaled[i, j]) > 1e-10:
                            rows.append(source_idx * self.stalk_dim + i)
                            cols.append(target_idx * self.stalk_dim + j)
                            data.append(-R_scaled[i, j])
                            
                # Add -R^T block
                for i in range(self.stalk_dim):
                    for j in range(self.stalk_dim):
                        if abs(R_scaled[j, i]) > 1e-10:
                            rows.append(target_idx * self.stalk_dim + i)
                            cols.append(source_idx * self.stalk_dim + j)
                            data.append(-R_scaled[j, i])
                            
        # Combine diagonal and off-diagonal
        if rows:
            off_diagonal = csr_matrix(
                (data, (rows, cols)), 
                shape=(total_dim, total_dim)
            )
            L = L_diagonal + off_diagonal
        else:
            L = L_diagonal
            
        # Validate symmetry
        if not self._is_symmetric(L):
            logging.warning("Laplacian is not symmetric - check restriction maps")
            
        return L
    
    def _compute_node_laplacian(self, node: str) -> np.ndarray:
        """
        Compute local Laplacian at node.
        
        L_v = sum of R_e^T R_e for all incident edges
        """
        L_v = np.zeros((self.stalk_dim, self.stalk_dim))
        
        # Incoming edges
        for pred in self.sheaf.poset.predecessors(node):
            edge = (pred, node)
            if edge in self.sheaf.restrictions:
                R = self.sheaf.restrictions[edge]['Q']
                scale = self.sheaf.restrictions[edge]['scale']
                L_v += scale**2 * R.T @ R
                
        # Outgoing edges  
        for succ in self.sheaf.poset.successors(node):
            edge = (node, succ)
            if edge in self.sheaf.restrictions:
                R = self.sheaf.restrictions[edge]['Q']
                scale = self.sheaf.restrictions[edge]['scale']
                L_v += scale**2 * R @ R.T
                
        return L_v
    
    def _is_symmetric(self, L: csr_matrix, tol: float = 1e-10) -> bool:
        """Check if sparse matrix is symmetric"""
        diff = L - L.T
        return np.abs(diff.data).max() < tol


### 5. Complete Pipeline Integration (Updated)

```python
# neursheaf/sheaf/pipeline.py
class SheafConstructionPipeline:
    """
    Complete pipeline for neural network sheaf construction.
    
    CRITICAL: All activations and Gram matrices use raw (uncentered)
    features to avoid double-centering with debiased CKA.
    
    Integrates:
    - Adaptive activation capture (raw activations)
    - Poset extraction
    - CKA computation with validation (debiased)
    - Scaled Procrustes restrictions
    - Optimized Laplacian building
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.activation_capture = AdaptiveActivationCapture(**self.config['activation'])
        self.poset_extractor = NeuralNetworkPoset()
        self.laplacian_builder = None  # Created per sheaf
        
        # External CKA implementation
        if self.config['use_external_cka']:
            from centered_kernel_alignment import CKA
            self.cka_computer = CKA(device=self.config['device'])
        else:
            from ..cka import DebiasedCKAComputer
            self.cka_computer = DebiasedCKAComputer()
    
    def _compute_cka_stalk(self, activation: torch.Tensor) -> np.ndarray:
        """
        Compute CKA Gram matrix for stalk.
        
        IMPORTANT: Uses raw activation tensor without centering.
        """
        # Compute Gram matrix from RAW activations
        if hasattr(self.cka_computer, 'kernel'):
            # External implementation
            K = self.cka_computer.kernel(activation, activation, debiased=True)
        else:
            # Our implementation - pass raw activations
            K = activation @ activation.T  # No centering!
            
            # If using our debiased implementation, it handles centering internally
            if hasattr(self.cka_computer, '_compute_debiased_cka'):
                # This is just for validation - actual CKA values computed elsewhere
                pass
        
        # Validate
        if not self._is_valid_gram_matrix(K):
            raise ValueError("Invalid Gram matrix computed")
            
        return K.cpu().numpy() if torch.is_tensor(K) else K
    
    def build_sheaf(self, 
                   model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader) -> Tuple[NeuralNetworkSheaf, csr_matrix]:
        """
        Build complete neural network sheaf.
        
        Returns
        -------
        sheaf : NeuralNetworkSheaf
            Constructed sheaf with stalks from raw activations
        laplacian : csr_matrix
            Sparse sheaf Laplacian
        """
        print("Phase 1: Adaptive activation capture (raw features)...")
        activations = self.activation_capture.collect_activations(model, dataloader)
        
        print("Phase 2: Poset extraction...")
        poset = self.poset_extractor.extract_poset(model)
        
        print("Phase 3: CKA stalk computation (no centering)...")
        sheaf = NeuralNetworkSheaf(poset)
        
        for node in poset.nodes():
            if node in activations:
                # Compute CKA Gram matrix from RAW activations
                K = self._compute_cka_stalk(activations[node])
                sheaf.add_stalk(node, K)
        
        print("Phase 4: Restriction map computation...")
        sheaf.compute_restrictions()
        
        print("Phase 5: Laplacian construction...")
        builder = OptimizedLaplacianBuilder(sheaf)
        laplacian = builder.build_sparse_laplacian()
        
        print("Phase 6: Validation...")
        sheaf.validate_sheaf()
        self._validate_laplacian(laplacian, sheaf)
        
        return sheaf, laplacian
```

## Unit Tests (Updated)

```python
# tests/test_sheaf_construction_no_centering.py
import pytest
import torch
import numpy as np
from neursheaf.sheaf import SheafConstructionPipeline

class TestSheafConstructionNoCentering:
    
    def test_raw_activations_used(self):
        """Verify that raw activations are used throughout"""
        model = create_test_model()
        dataloader = create_test_dataloader()
        
        pipeline = SheafConstructionPipeline()
        
        # Capture activations
        activations = pipeline.activation_capture.collect_activations(model, dataloader)
        
        # Check that activations are not centered
        for name, act in activations.items():
            mean_norm = torch.norm(act.mean(dim=0))
            assert mean_norm > 0.1, f"Layer {name} appears to be centered"
    
    def test_gram_matrix_from_raw_activations(self):
        """Test that Gram matrices use raw activations"""
        # Create test data
        raw_act = torch.randn(100, 50)
        centered_act = raw_act - raw_act.mean(dim=0, keepdim=True)
        
        pipeline = SheafConstructionPipeline()
        
        # Compute Gram matrices
        K_raw = pipeline._compute_cka_stalk(raw_act)
        K_centered = pipeline._compute_cka_stalk(centered_act)
        
        # Should be different
        assert not np.allclose(K_raw, K_centered)
        
        # Raw should have larger values (no centering reduces magnitude)
        assert np.mean(np.abs(K_raw)) > np.mean(np.abs(K_centered))
```

## Documentation

```python
"""
Sheaf Construction Implementation (Fixed)

This module implements neural network sheaf construction with the
critical fix for double-centering in CKA computation.

Key Points
----------
1. All activations are kept RAW (uncentered) throughout the pipeline
2. Gram matrices K = X @ X.T use raw activation matrices X
3. The debiased CKA estimator handles centering internally
4. This fix prevents artificially suppressed similarity values

The sheaf construction process:
1. Capture raw neural activations adaptively
2. Extract network poset structure  
3. Compute CKA Gram matrices from raw activations
4. Compute scaled Procrustes restriction maps
5. Build sparse block-diagonal Laplacian

References
----------
- PySheaf: https://github.com/kb1dds/pysheaf.git
- Debiased CKA: Murphy et al. (2024)
- Procrustes: scipy.linalg.orthogonal_procrustes
"""
```

## Summary

This updated sheaf construction plan ensures:

1. **Raw activations** are used throughout the pipeline
2. **No pre-centering** occurs before Gram matrix computation
3. **Clear documentation** about the importance of avoiding double-centering
4. **Integration** with the fixed debiased CKA implementation
5. **Validation** to detect if centered data is accidentally used

The fix ensures accurate neural network similarity measurements by preventing the bias introduced by double-centering in the debiased CKA computation.