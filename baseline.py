#!/usr/bin/env python3
"""
Baseline representational distances between saved PyTorch models.

Baselines computed:
  - CKA (linear, centered)
  - RSA (RDM Spearman, layer-matched)
  - Orthogonal Procrustes (PCA to p dims, orthogonal fit residual)
  - Grassmann / Subspace distance (principal angles of PCA subspaces)
  - OT (no-sheaf): vector of adjacent-layer GW costs; compare vectors

Outputs per method:
  - distances/{method}_dist.npy  (N x N matrix)
  - distances/{method}_dist.csv
  - metrics/{method}_metrics.json (if labels provided)

Usage:
  python baselines_eval.py \
      --models-dir path/to/models \
      --outputs-dir out/ \
      --arch-default mlp1 \
      --batch-size 256 \
      --device cuda:0 \
      --n-clusters 2 \
      --labels-csv labels.csv

labels.csv (optional):
  filename,label
  model1.pth,trained
  model2.pth,random
  ...

IMPORTANT: Implement get_eval_loader() below to return a DataLoader that
emits the inputs you want to use for activation extraction.
"""

import argparse
import os
import json
import csv
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, OrderedDict as ODType
from collections import OrderedDict

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
# Model registry (updated for neurosheaf models)
# -----------------------
class MLPModel(nn.Module):
    """MLP model architecture matching the configuration:
    - input_dim: 3 (torus data)
    - num_hidden_layers: 8 
    - hidden_dim: 32
    - output_dim: 1 (binary classification)
    - activation_fn: relu
    - output_activation_fn: sigmoid
    - dropout_rate: 0.0012
    """
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
        
        # Store configuration
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Get activation functions
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        # Build the network
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
        
        # Use 'layers' as the attribute name to match saved weights
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
        
        # Based on the error messages, the model has:
        # layers.0: Linear(3, 32) 
        # layers.2: Linear(32, 32)
        # layers.5: Conv1D with weight shape [32, 16, 2] 
        # layers.8: Conv1D with weight shape [32, 16, 2]
        # layers.11: Conv1D with weight shape [32, 16, 2]
        # layers.14: Final layer
        
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

MODEL_REGISTRY = {
    'mlp': lambda: MLPModel(),
    'custom': lambda: CustomModel(),
}

# Optional filename pattern mapping (updated for neurosheaf models)
FILENAME_TO_ARCH = [
    (re.compile(r"mlp_", re.I), 'mlp'),
    (re.compile(r"custom_", re.I), 'custom'),
]


# -----------------------
# Data loader for torus inputs
# -----------------------
def get_eval_loader(batch_size: int) -> torch.utils.data.DataLoader:
    """
    Generate random torus data for activation extraction.
    Data generation: 12 * torch.rand((batch_size, 3))
    """
    # Generate a reasonably sized dataset for activation extraction
    # Using multiple batches to ensure we get a good representation
    num_samples = batch_size * 8  # Use 8 batches worth of data
    
    # Generate torus data: 12 * torch.rand((num_samples, 3))
    data = 12 * torch.rand((num_samples, 3), dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


# -----------------------
# Activation capture
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
# Distances: helpers (layer matching etc.)
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


# -----------------------
# Baselines: pairwise distances between models
# -----------------------
def pairwise_from_layerwise_similarity(acts_all: Dict[str, OrderedDict], sim_fn, maximize=True) -> np.ndarray:
    keys = list(acts_all.keys())
    N = len(keys)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i+1, N):
            A = acts_all[keys[i]]
            B = acts_all[keys[j]]
            layersA, layersB = list(A.keys()), list(B.keys())
            M = np.zeros((len(layersA), len(layersB)), dtype=float)
            for ia, la in enumerate(layersA):
                XA = A[la].T  # samples x units
                for jb, lb in enumerate(layersB):
                    XB = B[lb].T
                    M[ia, jb] = sim_fn(XA, XB)
            r, c, score = hungarian_match_matrix(M, maximize=maximize)
            dist = 1.0 - score if maximize else score
            D[i, j] = D[j, i] = dist
    return D, keys

def pairwise_from_layerwise_distance(acts_all: Dict[str, OrderedDict], dist_fn) -> np.ndarray:
    keys = list(acts_all.keys())
    N = len(keys)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i+1, N):
            A = acts_all[keys[i]]
            B = acts_all[keys[j]]
            layersA, layersB = list(A.keys()), list(B.keys())
            M = np.zeros((len(layersA), len(layersB)), dtype=float)
            for ia, la in enumerate(layersA):
                XA = A[la].T
                for jb, lb in enumerate(layersB):
                    XB = B[lb].T
                    M[ia, jb] = dist_fn(XA, XB)
            r, c, score = hungarian_match_matrix(M, maximize=False)
            D[i, j] = D[j, i] = score
    return D, keys

def pairwise_ot_vector(acts_all: Dict[str, OrderedDict], use_dtw: bool = False) -> Tuple[np.ndarray, List[str]]:
    keys = list(acts_all.keys())
    N = len(keys)
    # Build GW cost vector per model (adjacent layers)
    vectors = []
    for k in keys:
        layers = list(acts_all[k].keys())
        costs = []
        for idx in range(len(layers)-1):
            H1 = acts_all[k][layers[idx]]
            H2 = acts_all[k][layers[idx+1]]
            c = gw_cost_between_layers(H1, H2, reg=0.1, metric='cosine')
            costs.append(c)
        vectors.append(np.array(costs, dtype=float))

    # Pairwise distance between vectors
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i+1, N):
            x, y = vectors[i], vectors[j]
            if use_dtw and TSLEARN_AVAILABLE:
                d = float(dtw_distance(x, y))
            else:
                # L2 after linear interpolation to equal length
                L = max(len(x), len(y))
                xi = np.interp(np.linspace(0,1,L), np.linspace(0,1,len(x)), x)
                yi = np.interp(np.linspace(0,1,L), np.linspace(0,1,len(y)), y)
                d = float(np.linalg.norm(xi - yi))
            D[i, j] = D[j, i] = d
    return D, keys


# -----------------------
# Metrics and saving
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
    ap = argparse.ArgumentParser(description="Compute baseline model-to-model distances from activations.")
    ap.add_argument("--models-dir", required=True, help="Folder with .pth model files")
    ap.add_argument("--outputs-dir", required=True, help="Where to write distance matrices and metrics")
    ap.add_argument("--arch-default", default="mlp", help="Architecture key to use if filename pattern doesn't match")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--spatial-reduce", default="gap", choices=["gap", "flatten"])
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--rdm-metric", default="correlation", choices=["correlation", "euclidean", "cosine"])
    ap.add_argument("--labels-csv", default=None, help="CSV mapping filename->label")
    ap.add_argument("--n-clusters", type=int, default=2, help="Clusters for AgglomerativeClustering (if labels provided, set to len(unique(labels)))")
    ap.add_argument("--use-dtw-ot", action="store_true", help="Use DTW for OT-vector distance (requires tslearn)")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    try:
        loader = get_eval_loader(args.batch_size)
    except NotImplementedError as e:
        print(e)
        print("\nEdit get_eval_loader() in this script to return a DataLoader of inputs.")
        return

    # Load models and extract activations
    device = torch.device(args.device)
    model_paths = sorted(glob.glob(str(models_dir / "*.pth")))
    if len(model_paths) == 0:
        print("No .pth files found in", models_dir)
        return

    acts_all: Dict[str, OrderedDict] = {}
    print(f"Found {len(model_paths)} models. Extracting activations...")
    for mp in model_paths:
        arch_key = infer_arch_from_filename(mp, args.arch_default)
        model = build_model(arch_key).to(device)
        model = load_state(model, mp)
        model.eval()

        extractor = get_activation_extractor(model, spatial_reduce=args.spatial_reduce)
        with extractor as buf:
            with torch.no_grad():
                for batch in loader:
                    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                    xb = xb.to(device, non_blocking=True)
                    _ = model(xb)
        acts = extractor.stack()
        key = os.path.basename(mp)
        acts_all[key] = acts
        print(f"  - {key}: captured {len(acts)} layers")

    # Optional labels
    labels_map = read_labels_csv(args.labels_csv) if args.labels_csv else None
    if labels_map is not None and args.n_clusters is None:
        # set to number of unique labels
        n_unique = len(set(labels_map.values()))
        args.n_clusters = n_unique

    # ---- CKA (similarity -> distance = 1 - mean-matched-CKA)
    print("Computing CKA distances...")
    D_cka, keys = pairwise_from_layerwise_similarity(
        acts_all,
        sim_fn=linear_cka,
        maximize=True
    )
    save_matrix_and_metrics("cka", D_cka, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- RSA (RDM Spearman)
    print("Computing RSA distances...")
    def rsa_sim(XA, XB):
        D1 = rdm_from_activations(XA, metric=args.rdm_metric)
        D2 = rdm_from_activations(XB, metric=args.rdm_metric)
        return rdm_spearman(D1, D2)
    D_rsa, _ = pairwise_from_layerwise_similarity(
        acts_all,
        sim_fn=rsa_sim,
        maximize=True
    )
    save_matrix_and_metrics("rsa", D_rsa, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- Procrustes (distance = residual)
    print("Computing Procrustes distances...")
    D_proc, _ = pairwise_from_layerwise_distance(
        acts_all,
        dist_fn=lambda A,B: procrustes_residual(A, B, p=args.pca_dim)
    )
    save_matrix_and_metrics("procrustes", D_proc, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- Grassmann (distance = sqrt sum of principal angles^2)
    print("Computing Grassmann distances...")
    D_grass, _ = pairwise_from_layerwise_distance(
        acts_all,
        dist_fn=lambda A,B: grassmann_distance(A, B, p=args.pca_dim)
    )
    save_matrix_and_metrics("grassmann", D_grass, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    # ---- OT (no-sheaf): vector of adjacent-layer GW costs, compare vectors
    if not POT_AVAILABLE:
        print("POT not installed; skipping OT baseline. pip install pot")
    else:
        print("Computing OT (no-sheaf) vector distances...")
        D_ot, _ = pairwise_ot_vector(acts_all, use_dtw=args.use_dtw_ot)
        save_matrix_and_metrics("ot_vector", D_ot, keys, outputs_dir, labels_map, n_clusters=args.n_clusters)

    print("\nDone. Matrices saved in:", outputs_dir / "distances")
    if labels_map:
        print("Metrics saved in:", outputs_dir / "metrics")


if __name__ == "__main__":
    main()
