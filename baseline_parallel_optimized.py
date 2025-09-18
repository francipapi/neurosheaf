#!/usr/bin/env python3
"""
Optimized parallel baseline representational distances between saved PyTorch models.

This is a highly optimized version that fixes the performance bottlenecks in baseline_parallel.py:
- Individual pair-based computation (not row-based)
- Minimal data transfer per worker
- Perfect load balancing across workers
- Progress tracking at pair level

Expected speedup: 10-15x faster than serial baseline.py

Usage:
  python baseline_parallel_optimized.py \
      --models-dir baseline_models \
      --outputs-dir baseline_output_optimized \
      --num-workers 10 \
      --cache-activations \
      --labels-csv labels.csv
"""

import argparse
import os
import json
import csv
import glob
import re
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, OrderedDict as ODType, Optional
from collections import OrderedDict
from functools import partial
import time
import itertools

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes, subspace_angles
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import ot  # POT: Python Optimal Transport
    from ot.gromov import entropic_gromov_wasserstein, gromov_wasserstein2
    POT_AVAILABLE = True
except Exception:
    POT_AVAILABLE = False

try:
    from tslearn.metrics import dtw as dtw_distance
    TSLEARN_AVAILABLE = True
except Exception:
    TSLEARN_AVAILABLE = False


# -----------------------
# Model registry (same as before)
# -----------------------
class MLPModel(nn.Module):
    """MLP model architecture matching the configuration."""
    def __init__(
        self,
        input_dim: int = 3,
        num_hidden_layers: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 1,
        activation_fn_name: str = 'relu',
        output_activation_fn_name: str = 'sigmoid',
        dropout_rate: float = 0.0012
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(self.output_activation_fn)
        
        self.layers = nn.Sequential(*layers_list)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'softmax': nn.Softmax(dim=-1),
            'none': nn.Identity()
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name.lower()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CustomModel(nn.Module):
    """Model class that matches the actual saved weights structure with Conv1D layers."""
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(3, 32),                                    # layers.0
            nn.ReLU(),                                           # layers.1 (activation)
            nn.Linear(32, 32),                                   # layers.2
            nn.ReLU(),                                           # layers.3 (activation)
            nn.Dropout(0.0),                                     # layers.4 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.5
            nn.ReLU(),                                           # layers.6 (activation)
            nn.Dropout(0.0),                                     # layers.7 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.8
            nn.ReLU(),                                           # layers.9 (activation)
            nn.Dropout(0.0),                                     # layers.10 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.11
            nn.ReLU(),                                           # layers.12 (activation)
            nn.Dropout(0.0),                                     # layers.13 (dropout)
            nn.Linear(32, 1),                                    # layers.14
            nn.Sigmoid()                                         # layers.15 (activation)
        )
    
    def forward(self, x):
        # Input: [batch_size, 3]
        
        # Layer 0: Linear(3 -> 32) + ReLU
        x = self.layers[1](self.layers[0](x))  # [batch_size, 32]
        
        # Layer 2: Linear(32 -> 32) + ReLU + Dropout
        x = self.layers[4](self.layers[3](self.layers[2](x)))  # [batch_size, 32]
        
        # Reshape for Conv1D: [batch_size, 32] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 5: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[7](self.layers[6](self.layers[5](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 8: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[10](self.layers[9](self.layers[8](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 11: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[13](self.layers[12](self.layers[11](x)))  # [batch_size, 32, 1]
        
        # Flatten for final layer: [batch_size, 32, 1] -> [batch_size, 32]
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        
        # Layer 14: Linear(32 -> 1) + Sigmoid
        x = self.layers[15](self.layers[14](x))  # [batch_size, 1]
        
        return x


class PyramidMLP(nn.Module):
    """Pyramid MLP with decreasing layer sizes for digits classification."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_layers = 4
        self.layer_dims = [128, 64, 32, 16]
        self.num_classes = num_classes
        self.arch_type = "pyramid"
        
        layers = []
        prev_dim = 64
        
        for dim in self.layer_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.GELU()
            ])
            prev_dim = dim
        
        # Output layer: 16 -> 10
        layers.append(nn.Linear(16, num_classes))
        
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.layers(x)


class HourglassMLP(nn.Module):
    """Hourglass MLP with expanding-contracting-expanding pattern for digits classification."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_layers = 5
        self.layer_dims = [96, 48, 24, 48, 96]
        self.num_classes = num_classes
        self.arch_type = "hourglass"
        
        layers = []
        prev_dim = 64
        
        for i, dim in enumerate(self.layer_dims):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.Dropout(0.1),
                nn.SiLU()
            ])
            prev_dim = dim
        
        # Output layer: 96 -> 10
        layers.append(nn.Linear(96, num_classes))
        
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.layers(x)


MODEL_REGISTRY = {
    'mlp': lambda: MLPModel(),
    'custom': lambda: CustomModel(),
    'pyramid': lambda: PyramidMLP(),
    'hourglass': lambda: HourglassMLP(),
}

# Optional filename pattern mapping (updated for neurosheaf models and digits models)
FILENAME_TO_ARCH = [
    (re.compile(r"mlp_", re.I), 'mlp'),
    (re.compile(r"custom_", re.I), 'custom'),
    (re.compile(r"pyramid", re.I), 'pyramid'),
    (re.compile(r"hourglass", re.I), 'hourglass'),
]


# -----------------------
# Data loader for digits dataset
# -----------------------
def get_eval_loader(batch_size: int) -> torch.utils.data.DataLoader:
    """Load digits dataset for activation extraction."""
    from sklearn.datasets import load_digits
    
    # Load the digits dataset
    digits = load_digits()
    
    # Convert to torch tensor and take first batch_size samples
    # digits.data is already flattened (64 features per sample)
    data = torch.from_numpy(digits.data[:batch_size]).float()
    
    # Normalize to [0, 16] range (same as original 8x8 pixel values)
    data = data / data.max() * 16.0
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


# -----------------------
# Activation capture (same as before)
# -----------------------
def get_activation_extractor(model: nn.Module, spatial_reduce: str = "gap"):
    """Registers forward hooks to capture outputs of modules with weights."""
    handles = []
    buffers: ODType[str, List[torch.Tensor]] = OrderedDict()

    def hook(name):
        def _hook(module, inp, out):
            with torch.no_grad():
                x = out
                if isinstance(x, (tuple, list)):
                    x = x[0]
                if x.dim() == 4:  # N,C,H,W
                    if spatial_reduce == "gap":
                        x = x.mean(dim=(2,3))  # N,C
                    else:
                        x = x.flatten(1)
                elif x.dim() > 2:
                    x = x.flatten(1)
                buffers[name].append(x.detach().cpu().T)  # (units, batch)
        return _hook

    chosen = []
    for n, m in model.named_modules():
        is_leaf = (len(list(m.children())) == 0)
        has_weight = hasattr(m, 'weight') and m.weight is not None
        if is_leaf and has_weight:
            chosen.append(n)

    for n in chosen:
        buffers[n] = []
        handles.append(dict(model.named_modules())[n].register_forward_hook(hook(n)))

    class Extractor:
        def __enter__(self):
            return buffers
        def __exit__(self, exc_type, exc, tb):
            for h in handles: h.remove()
        def stack(self):
            return OrderedDict((k, torch.cat(v, dim=1).numpy()) for k, v in buffers.items())

    return Extractor()


# -----------------------
# Utility: load models
# -----------------------
def infer_arch_from_filename(fname: str, default_arch: str) -> str:
    base = os.path.basename(fname)
    for rx, arch in FILENAME_TO_ARCH:
        if rx.search(base):
            return arch
    return default_arch

def build_model(arch_key: str) -> nn.Module:
    if arch_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture key '{arch_key}'. Edit MODEL_REGISTRY.")
    return MODEL_REGISTRY[arch_key]()

def load_state(model: nn.Module, path: str):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    # Support either full state dict or {'model_state_dict': ...} or {'state_dict': ...}
    if isinstance(sd, dict):
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        elif 'state_dict' in sd:
            sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    return model


# -----------------------
# OPTIMIZED PARALLEL ACTIVATION EXTRACTION
# -----------------------
def extract_activations_single_model(args_tuple):
    """Extract activations for a single model. Used by multiprocessing."""
    model_path, arch_default, batch_size, spatial_reduce, device_str = args_tuple
    
    try:
        # Build model
        arch_key = infer_arch_from_filename(model_path, arch_default)
        model = build_model(arch_key)
        model = load_state(model, model_path)
        model.eval()
        
        # Get data loader
        loader = get_eval_loader(batch_size)
        
        # Extract activations
        device = torch.device(device_str)
        model = model.to(device)
        
        extractor = get_activation_extractor(model, spatial_reduce=spatial_reduce)
        with extractor as buf:
            with torch.no_grad():
                for batch in loader:
                    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                    xb = xb.to(device, non_blocking=True)
                    _ = model(xb)
        acts = extractor.stack()
        
        key = os.path.basename(model_path)
        return key, acts, len(acts)
        
    except Exception as e:
        return os.path.basename(model_path), None, str(e)


def extract_all_activations_parallel(model_paths: List[str], args, num_workers: int, 
                                   cache_dir: Optional[Path] = None) -> Dict[str, OrderedDict]:
    """Extract activations from all models in parallel."""
    acts_all = {}
    
    # Check cache first
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "activations_cache.pkl"
        if cache_file.exists():
            print(f"Loading cached activations from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_acts = pickle.load(f)
                # Check if all models are cached
                cached_keys = set(cached_acts.keys())
                required_keys = {os.path.basename(mp) for mp in model_paths}
                if required_keys.issubset(cached_keys):
                    print(f"All {len(required_keys)} models found in cache!")
                    return {k: cached_acts[k] for k in required_keys}
                else:
                    missing = required_keys - cached_keys
                    print(f"Cache incomplete. Missing {len(missing)} models: {sorted(list(missing))[:5]}...")
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    # Prepare arguments for parallel processing
    device_str = 'cpu'  # Use CPU for parallel processing to avoid GPU memory conflicts
    args_list = [
        (mp, args.arch_default, args.batch_size, args.spatial_reduce, device_str) 
        for mp in model_paths
    ]
    
    print(f"Extracting activations from {len(model_paths)} models using {num_workers} workers...")
    
    # Use multiprocessing pool
    with mp.Pool(num_workers) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(pool.imap(extract_activations_single_model, args_list), 
                              total=len(args_list), desc="Extracting activations"))
        else:
            results = list(pool.map(extract_activations_single_model, args_list))
    
    # Process results
    for key, acts, info in results:
        if acts is not None:
            acts_all[key] = acts
            if isinstance(info, int):
                print(f"  - {key}: captured {info} layers")
            else:
                print(f"  - {key}: info = {info}")
        else:
            print(f"  - ERROR processing {key}: {info}")
    
    # Save to cache
    if cache_dir is not None:
        cache_file = cache_dir / "activations_cache.pkl"
        print(f"Saving activations to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(acts_all, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    return acts_all


# -----------------------
# Distance computations with numerical stability
# -----------------------
def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    # X,Y: (samples, units)
    if X.shape[0] < X.shape[1]: X = X.T
    if Y.shape[0] < Y.shape[1]: Y = Y.T
    Kx = _center_gram(X @ X.T)
    Ky = _center_gram(Y @ Y.T)
    num = (Kx * Ky).sum()
    den = np.sqrt((Kx*Kx).sum() * (Ky*Ky).sum()) + 1e-12
    return float(num / den)

def rdm_from_activations(A: np.ndarray, metric='correlation') -> np.ndarray:
    # A: (samples, units)
    D = squareform(pdist(A, metric=metric))
    return D

def rdm_spearman(D1: np.ndarray, D2: np.ndarray) -> float:
    i = np.triu_indices_from(D1, k=1)
    rho, _ = spearmanr(D1[i], D2[i])
    # Handle NaN cases (e.g., when all values are identical)
    if np.isnan(rho):
        rho = 0.0
    return float(rho)

def hungarian_match_matrix(M: np.ndarray, maximize: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    from scipy.optimize import linear_sum_assignment
    cost = -M if maximize else M.copy()
    
    # Handle NaN/inf values
    if np.any(~np.isfinite(cost)):
        # Replace NaN and inf with extreme values
        cost = np.nan_to_num(cost, nan=1e6 if not maximize else -1e6, 
                            posinf=1e6, neginf=-1e6)
    
    r, c = linear_sum_assignment(cost)
    score = M[r, c].mean()
    
    # Handle NaN in final score
    if np.isnan(score):
        score = 0.0
    
    return r, c, score

def procrustes_residual(A: np.ndarray, B: np.ndarray, p: int = 64) -> float:
    A = A - A.mean(0); B = B - B.mean(0)
    
    # Ensure we have enough samples for PCA
    if A.shape[0] < 2 or B.shape[0] < 2:
        return 1.0  # Maximum distance for degenerate cases
        
    p = min(p, A.shape[1], B.shape[1], A.shape[0]-1, B.shape[0]-1)
    if p <= 0:
        return 1.0  # Maximum distance when no valid subspace
    
    try:
        PA = PCA(p).fit_transform(A)
        PB = PCA(p).fit_transform(B)
        R, _ = orthogonal_procrustes(PA, PB)
        res = np.linalg.norm(PA @ R - PB, 'fro') / (np.linalg.norm(PB, 'fro') + 1e-12)
        return float(res)
    except Exception:
        # Fallback to a simple distance measure
        return 1.0

def grassmann_distance(A: np.ndarray, B: np.ndarray, p: int = 64) -> float:
    A = A - A.mean(0); B = B - B.mean(0)
    # Ensure we have enough samples for PCA
    if A.shape[0] < 2 or B.shape[0] < 2:
        return 1.0  # Maximum distance for degenerate cases
    
    p = min(p, A.shape[1], B.shape[1], A.shape[0]-1, B.shape[0]-1)
    if p <= 0:
        return 1.0  # Maximum distance when no valid subspace
    
    PA = PCA(p).fit(A).components_.T  # (units, p)
    PB = PCA(p).fit(B).components_.T
    
    # Ensure same number of rows by padding with zeros if necessary
    if PA.shape[0] != PB.shape[0]:
        max_rows = max(PA.shape[0], PB.shape[0])
        if PA.shape[0] < max_rows:
            PA = np.vstack([PA, np.zeros((max_rows - PA.shape[0], PA.shape[1]))])
        if PB.shape[0] < max_rows:
            PB = np.vstack([PB, np.zeros((max_rows - PB.shape[0], PB.shape[1]))])
    
    try:
        thetas = subspace_angles(PA, PB)
        return float(np.sqrt((thetas**2).sum()))
    except Exception:
        # Fallback to a simple distance measure
        return 1.0

# Module-level functions for multiprocessing
def rsa_similarity(XA: np.ndarray, XB: np.ndarray, rdm_metric: str = 'correlation') -> float:
    """RSA similarity function that can be pickled for multiprocessing."""
    D1 = rdm_from_activations(XA, metric=rdm_metric)
    D2 = rdm_from_activations(XB, metric=rdm_metric)
    return rdm_spearman(D1, D2)

def procrustes_distance_wrapper(A: np.ndarray, B: np.ndarray, pca_dim: int = 64) -> float:
    """Procrustes distance wrapper that can be pickled for multiprocessing."""
    return procrustes_residual(A, B, p=pca_dim)

def grassmann_distance_wrapper(A: np.ndarray, B: np.ndarray, pca_dim: int = 64) -> float:
    """Grassmann distance wrapper that can be pickled for multiprocessing."""
    return grassmann_distance(A, B, p=pca_dim)

def compute_vector_pair_distance(args_tuple):
    """Compute distance between two GW cost vectors. Used by multiprocessing."""
    pair_idx, i, j, vec_i, vec_j, use_dtw = args_tuple
    
    if use_dtw and TSLEARN_AVAILABLE:
        d = float(dtw_distance(vec_i, vec_j))
    else:
        # L2 after linear interpolation to equal length
        L = max(len(vec_i), len(vec_j))
        xi = np.interp(np.linspace(0,1,L), np.linspace(0,1,len(vec_i)), vec_i)
        yi = np.interp(np.linspace(0,1,L), np.linspace(0,1,len(vec_j)), vec_j)
        d = float(np.linalg.norm(xi - yi))
    
    return pair_idx, i, j, d


# -----------------------
# OPTIMIZED PAIR-BASED PARALLEL COMPUTATIONS
# -----------------------
def compute_single_pair_distance(args_tuple):
    """
    Compute distance for a single model pair. This is the key optimization!
    Instead of passing entire activation dictionary, we only pass the two models needed.
    """
    pair_idx, i, j, key_i, key_j, acts_i, acts_j, distance_function, function_kwargs = args_tuple
    
    try:
        # Get layer lists
        layersA, layersB = list(acts_i.keys()), list(acts_j.keys())
        M = np.zeros((len(layersA), len(layersB)), dtype=float)
        
        # Compute layer-wise distances/similarities
        for ia, la in enumerate(layersA):
            XA = acts_i[la].T  # samples x units
            for jb, lb in enumerate(layersB):
                XB = acts_j[lb].T
                # Remove is_similarity from kwargs before calling function
                func_kwargs = {k: v for k, v in function_kwargs.items() if k != 'is_similarity'}
                M[ia, jb] = distance_function(XA, XB, **func_kwargs)
        
        # Hungarian matching
        is_similarity = function_kwargs.get('is_similarity', False)
        r, c, score = hungarian_match_matrix(M, maximize=is_similarity)
        
        if is_similarity:  # similarity -> distance
            distance = 1.0 - score
        else:  # direct distance
            distance = score
        
        return pair_idx, i, j, distance
        
    except Exception as e:
        return pair_idx, i, j, f"Error: {e}"


def compute_pairwise_distances_optimized(acts_all: Dict[str, OrderedDict], 
                                        distance_function, function_kwargs: dict,
                                        method_name: str, num_workers: int) -> Tuple[np.ndarray, List[str]]:
    """
    Highly optimized pairwise distance computation.
    Key optimization: Only send the two needed activation sets per worker, not all activations.
    """
    keys = list(acts_all.keys())
    N = len(keys)
    
    # Generate all unique pairs
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    print(f"Computing {method_name} distances for {len(pairs)} pairs using {num_workers} workers...")
    
    # Prepare arguments - KEY OPTIMIZATION: minimal data per worker
    args_list = []
    for pair_idx, (i, j) in enumerate(pairs):
        key_i, key_j = keys[i], keys[j]
        # Only send the two activation sets needed for this pair!
        args_tuple = (
            pair_idx, i, j, key_i, key_j,
            acts_all[key_i], acts_all[key_j],  # Only these two!
            distance_function, function_kwargs
        )
        args_list.append(args_tuple)
    
    # Process in parallel
    with mp.Pool(num_workers) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(pool.imap(compute_single_pair_distance, args_list), 
                              total=len(args_list), desc=f"Computing {method_name}"))
        else:
            results = list(pool.map(compute_single_pair_distance, args_list))
    
    # Build distance matrix
    D = np.zeros((N, N), dtype=float)
    for pair_idx, i, j, distance in results:
        if isinstance(distance, str):  # Error case
            print(f"Error in pair {i},{j}: {distance}")
            distance = 1.0  # Use maximum distance for errors
        
        D[i, j] = distance
        D[j, i] = distance  # Symmetric
    
    return D, keys


# -----------------------
# GW/OT functions
# -----------------------
def gw_cost_between_layers(H1: np.ndarray, H2: np.ndarray, reg: float = 0.1, metric='cosine') -> float:
    """
    H1,H2: (units, samples) activations for two layers.
    Build mm-spaces on units with distances over neuron activation patterns.
    """
    if not POT_AVAILABLE:
        raise RuntimeError("POT (ot) is not installed. pip install pot")
    # distance matrices between neuron rows
    def cosine_dists(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
        return 1.0 - S  # cosine distance
    if metric == 'cosine':
        C1 = cosine_dists(H1)
        C2 = cosine_dists(H2)
    else:
        # fallback: correlation distances
        C1 = squareform(pdist(H1, metric='correlation'))
        C2 = squareform(pdist(H2, metric='correlation'))

    m1, m2 = H1.shape[0], H2.shape[0]
    mu1 = np.ones(m1) / m1
    mu2 = np.ones(m2) / m2

    try:
        # entropic GW returns coupling; we report the squared GW loss
        _ = entropic_gromov_wasserstein(C1, C2, mu1, mu2, epsilon=reg, verbose=False, max_iter=100)
        # POT doesn't directly return the cost with entropic version; compute GW2 with plan if needed.
        # Here, fallback: use GW2 (without entropic) to get the value.
        loss = gromov_wasserstein2(C1, C2, mu1, mu2, 'square_loss')
    except Exception:
        loss = gromov_wasserstein2(C1, C2, mu1, mu2, 'square_loss')
    return float(loss)

def compute_ot_vectors_and_distances(acts_all: Dict[str, OrderedDict], use_dtw: bool = False, 
                                   num_workers: int = None) -> Tuple[np.ndarray, List[str]]:
    """Compute OT vector distances with some parallelization for the vector comparison part."""
    keys = list(acts_all.keys())
    N = len(keys)
    
    # First, compute GW cost vectors for all models (sequential per model)
    print("Computing GW cost vectors...")
    vectors = []
    for k in tqdm(keys, desc="GW vectors") if TQDM_AVAILABLE else keys:
        layers = list(acts_all[k].keys())
        costs = []
        for idx in range(len(layers)-1):
            H1 = acts_all[k][layers[idx]]
            H2 = acts_all[k][layers[idx+1]]
            c = gw_cost_between_layers(H1, H2, reg=0.1, metric='cosine')
            costs.append(c)
        vectors.append(np.array(costs, dtype=float))
    
    # Now compute pairwise distances between vectors
    
    # Generate all unique pairs
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Prepare arguments
    args_list = []
    for pair_idx, (i, j) in enumerate(pairs):
        args_tuple = (pair_idx, i, j, vectors[i], vectors[j], use_dtw)
        args_list.append(args_tuple)
    
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, len(pairs))
    
    print(f"Computing pairwise vector distances for {len(pairs)} pairs using {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(pool.imap(compute_vector_pair_distance, args_list), 
                              total=len(args_list), desc="OT distances"))
        else:
            results = list(pool.map(compute_vector_pair_distance, args_list))
    
    # Fill distance matrix
    D = np.zeros((N, N), dtype=float)
    for pair_idx, i, j, distance in results:
        D[i, j] = distance
        D[j, i] = distance  # Symmetric
    
    return D, keys


# -----------------------
# Metrics and saving (same as before)
# -----------------------
def save_matrix_and_metrics(method: str, D: np.ndarray, keys: List[str], outdir: Path,
                            labels_map: Dict[str, str] = None, n_clusters: int = 2):
    outdir_dist = outdir / "distances"
    outdir_dist.mkdir(parents=True, exist_ok=True)
    np.save(outdir_dist / f"{method}_dist.npy", D)
    # CSV with header
    with open(outdir_dist / f"{method}_dist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + keys)
        for i, k in enumerate(keys):
            w.writerow([k] + [f"{D[i,j]:.6f}" for j in range(len(keys))])

    if labels_map is None:
        return

    # Build labels array in same order
    y_true = [labels_map.get(os.path.basename(k), None) for k in keys]
    if any(v is None for v in y_true):
        print(f"[{method}] WARNING: some labels missing; skipping metrics.")
        return

    # Encode labels to ints
    classes = {c:i for i,c in enumerate(sorted(set(y_true)))}
    y_true_int = np.array([classes[c] for c in y_true])

    # Clustering from distance matrix
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    y_pred = agg.fit_predict(D)

    metrics = {
        "ARI": float(adjusted_rand_score(y_true_int, y_pred)),
        "AMI": float(adjusted_mutual_info_score(y_true_int, y_pred)),
        "V_measure": float(v_measure_score(y_true_int, y_pred)),
    }

    # Silhouette with precomputed distances (requires >1 cluster)
    try:
        sil = silhouette_score(D, y_pred, metric='precomputed')
        metrics["Silhouette"] = float(sil)
    except Exception:
        pass

    # Embed with MDS to compute DBI / CH (they require features)
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4, max_iter=300)
        X2 = mds.fit_transform(D)
        metrics["DaviesBouldin_on_MDS2"] = float(davies_bouldin_score(X2, y_pred))
        metrics["CalinskiHarabasz_on_MDS2"] = float(calinski_harabasz_score(X2, y_pred))
    except Exception:
        pass

    outdir_metrics = outdir / "metrics"
    outdir_metrics.mkdir(parents=True, exist_ok=True)
    with open(outdir_metrics / f"{method}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def read_labels_csv(path: str) -> Dict[str, str]:
    mapping = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mapping[row["filename"]] = row["label"]
    return mapping


# -----------------------
# Main
# -----------------------
def main():
    start_time = time.time()
    
    ap = argparse.ArgumentParser(description="Compute baseline model-to-model distances (OPTIMIZED PARALLEL VERSION).")
    ap.add_argument("--models-dir", required=True, help="Folder with .pth model files")
    ap.add_argument("--outputs-dir", required=True, help="Where to write distance matrices and metrics")
    ap.add_argument("--arch-default", default="pyramid", help="Architecture key to use if filename pattern doesn't match")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--spatial-reduce", default="gap", choices=["gap", "flatten"])
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--rdm-metric", default="correlation", choices=["correlation", "euclidean", "cosine"])
    ap.add_argument("--labels-csv", default=None, help="CSV mapping filename->label")
    ap.add_argument("--n-clusters", type=int, default=2, help="Clusters for AgglomerativeClustering (if labels provided, set to len(unique(labels)))")
    ap.add_argument("--use-dtw-ot", action="store_true", help="Use DTW for OT-vector distance (requires tslearn)")
    
    # Parallel processing arguments
    ap.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 2)")
    ap.add_argument("--cache-activations", action="store_true", help="Cache activations to disk for faster reruns")
    ap.add_argument("--cache-dir", default="activation_cache", help="Directory to store activation cache")
    
    args = ap.parse_args()

    # Set number of workers
    if args.num_workers is None:
        args.num_workers = max(1, mp.cpu_count() - 2)
    
    print(f"Using {args.num_workers} parallel workers on {mp.cpu_count()} CPU cores")

    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path(args.cache_dir) if args.cache_activations else None

    # Find models
    model_paths = sorted(glob.glob(str(models_dir / "*.pth")))
    if len(model_paths) == 0:
        print("No .pth files found in", models_dir)
        return

    print(f"Found {len(model_paths)} models")

    # Extract activations in parallel
    acts_all = extract_all_activations_parallel(model_paths, args, args.num_workers, cache_dir)
    print(f"Successfully extracted activations from {len(acts_all)} models")
    
    extraction_time = time.time() - start_time
    print(f"Activation extraction took {extraction_time:.1f} seconds")

    # Optional labels
    labels_map = read_labels_csv(args.labels_csv) if args.labels_csv else None
    if labels_map is not None and args.n_clusters is None:
        # set to number of unique labels
        n_unique = len(set(labels_map.values()))
        args.n_clusters = n_unique

    distance_start = time.time()

    # ---- CKA (similarity -> distance = 1 - mean-matched-CKA)
    D_cka, keys = compute_pairwise_distances_optimized(
        acts_all, linear_cka, {'is_similarity': True}, "CKA", args.num_workers
    )
    save_matrix_and_metrics("cka", D_cka, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- RSA (RDM Spearman)
    D_rsa, _ = compute_pairwise_distances_optimized(
        acts_all, rsa_similarity, {'rdm_metric': args.rdm_metric, 'is_similarity': True}, 
        "RSA", args.num_workers
    )
    save_matrix_and_metrics("rsa", D_rsa, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- Procrustes (distance = residual)
    D_proc, _ = compute_pairwise_distances_optimized(
        acts_all, procrustes_distance_wrapper, {'pca_dim': args.pca_dim, 'is_similarity': False},
        "Procrustes", args.num_workers
    )
    save_matrix_and_metrics("procrustes", D_proc, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- Grassmann (distance = sqrt sum of principal angles^2)
    D_grass, _ = compute_pairwise_distances_optimized(
        acts_all, grassmann_distance_wrapper, {'pca_dim': args.pca_dim, 'is_similarity': False},
        "Grassmann", args.num_workers
    )
    save_matrix_and_metrics("grassmann", D_grass, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- OT (no-sheaf): vector of adjacent-layer GW costs, compare vectors
    if not POT_AVAILABLE:
        print("POT not installed; skipping OT baseline. pip install pot")
    else:
        D_ot, _ = compute_ot_vectors_and_distances(acts_all, use_dtw=args.use_dtw_ot, num_workers=args.num_workers)
        save_matrix_and_metrics("ot_vector", D_ot, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    distance_time = time.time() - distance_start
    total_time = time.time() - start_time
    
    print(f"\nTiming summary:")
    print(f"  Activation extraction: {extraction_time:.1f} seconds")
    print(f"  Distance computation: {distance_time:.1f} seconds")
    print(f"  Total runtime: {total_time:.1f} seconds")
    print(f"\nDone. Matrices saved in: {outputs_dir / 'distances'}")
    if labels_map:
        print(f"Metrics saved in: {outputs_dir / 'metrics'}")


if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    mp.set_start_method('spawn', force=True)
    main()