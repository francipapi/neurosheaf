#!/usr/bin/env python3
"""
Comprehensive Clustering Metrics Analysis (fixed & validated)

- Uses precomputed distance matrices.
- Internal metrics use predicted cluster labels (agglomerative on D).
- Bootstraps re-cluster each resample.
- Robust stability via ARI and label-aligned Jaccard.
- PERMANOVA / ANOSIM prefer scikit-bio; solid fallbacks included.

Usage:
    python compute_comprehensive_clustering_metrics.py \
      --distance-file new_comparison_corrected_dtw_distance.npy \
      --index-file new_comparison_corrected_dtw_index.json \
      --output-prefix dtw_comprehensive
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment

# sklearn
try:
    from sklearn.metrics import (
        silhouette_score, davies_bouldin_score, calinski_harabasz_score,
        adjusted_rand_score, adjusted_mutual_info_score, v_measure_score,
        confusion_matrix
    )
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.manifold import MDS
    _HAS_SKLEARN = True
except ImportError:
    print("Warning: scikit-learn not available. Some metrics will be unavailable.")
    _HAS_SKLEARN = False

# scikit-bio
try:
    from skbio.stats.distance import permanova as skbio_permanova, anosim as skbio_anosim
    import skbio
    _HAS_SKBIO = True
except ImportError:
    print("Warning: scikit-bio not available. Using fallbacks for PERMANOVA/ANOSIM.")
    _HAS_SKBIO = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_BOOTSTRAP_DEFAULT = 1000
N_PERMUTATIONS_DEFAULT = 999
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42  # for MDS reproducibility


# ---------------------------
# Utilities & hygiene
# ---------------------------
def _hygienize_distance(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def extract_labels_from_filenames(filenames: List[str]) -> Tuple[List[str], np.ndarray, Dict[str, List[int]]]:
    labels = []
    indices = {'random': [], 'trained': []}
    for i, filename in enumerate(filenames):
        fn = filename.lower()
        if 'random' in fn:
            label = 'random'
        elif any(k in fn for k in ['trained', 'seed', 'acc']):
            label = 'trained'
        else:
            label = 'unknown'
        labels.append(label)
        if label in indices:
            indices[label].append(i)
    numeric_labels = np.array([0 if l == 'random' else 1 if l == 'trained' else -1 for l in labels], dtype=int)
    return labels, numeric_labels, indices


def _agglomerative_labels(D: np.ndarray, n_clusters: int = 2) -> Optional[np.ndarray]:
    if not _HAS_SKLEARN:
        return None
    clu = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return clu.fit_predict(D)


# ---------------------------
# Metrics
# ---------------------------
def dunn_index(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    lbls = np.asarray(labels)
    unique_labels = np.unique(lbls)
    if len(unique_labels) < 2:
        return np.nan
    # min inter
    min_inter = np.inf
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            I = np.where(lbls == unique_labels[i])[0]
            J = np.where(lbls == unique_labels[j])[0]
            if len(I) == 0 or len(J) == 0:
                continue
            inter = distance_matrix[np.ix_(I, J)]
            if inter.size:
                m = np.min(inter)
                if np.isfinite(m):
                    min_inter = min(min_inter, m)
    # max intra
    max_intra = 0.0
    for lab in unique_labels:
        idx = np.where(lbls == lab)[0]
        if len(idx) > 1:
            A = distance_matrix[np.ix_(idx, idx)]
            triu = np.triu_indices_from(A, k=1)
            if len(triu[0]) > 0:
                val = np.max(A[triu])
                if np.isfinite(val):
                    max_intra = max(max_intra, val)
    if max_intra == 0.0:
        return np.nan if not np.isfinite(min_inter) else np.inf
    return float(min_inter / max_intra)


def cophenetic_correlation(distance_matrix: np.ndarray) -> float:
    try:
        D = _hygienize_distance(distance_matrix)
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method='average')
        c, _ = cophenet(Z, condensed)
        return float(c) if np.isfinite(c) else np.nan
    except Exception:
        return np.nan


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / max(n1 + n2 - 2, 1))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return float((mean1 - mean2) / pooled)


# ---------------------------
# Hypothesis tests
# ---------------------------
def permanova_test(distance_matrix: np.ndarray, labels: np.ndarray,
                   n_permutations: int = N_PERMUTATIONS_DEFAULT) -> Tuple[float, float]:
    D = _hygienize_distance(distance_matrix)
    y = np.asarray(labels)
    mask = y != -1
    D = D[np.ix_(mask, mask)]
    y = y[mask]
    if len(np.unique(y)) < 2 or D.shape[0] < 4:
        return np.nan, np.nan

    if _HAS_SKBIO:
        try:
            dm = skbio.DistanceMatrix(D)
            groups = pd.Series(y, name='group')
            res = skbio_permanova(dm, groups, permutations=n_permutations)
            return float(res['test statistic']), float(res['p-value'])
        except Exception:
            pass

    # Fallback PERMANOVA (Anderson 2001 style; uses squared distances)
    # SST = (1/N) * sum_{i<j} d_ij^2
    # SSW = sum_g (1/n_g) * sum_{i<j in g} d_ij^2
    # SSB = SST - SSW
    # F = (SSB/(G-1)) / (SSW/(N-G))
    N = D.shape[0]
    if N < 4:
        return np.nan, np.nan
    d2 = D**2
    triu = np.triu_indices(N, k=1)
    SST = np.sum(d2[triu]) / N

    unique = np.unique(y)
    G = len(unique)
    if G < 2:
        return np.nan, np.nan

    SSW = 0.0
    for g in unique:
        idx = np.where(y == g)[0]
        n_g = len(idx)
        if n_g > 1:
            A = d2[np.ix_(idx, idx)]
            t = np.triu_indices(n_g, k=1)
            SSW += np.sum(A[t]) / n_g

    SSB = SST - SSW
    dfB = G - 1
    dfW = N - G
    if dfW <= 0 or SSW <= 0:
        return np.nan, np.nan
    F_obs = (SSB / dfB) / (SSW / dfW)

    # permutations
    perm_F = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        SSW_p = 0.0
        for g in unique:
            idx = np.where(y_perm == g)[0]
            n_g = len(idx)
            if n_g > 1:
                A = d2[np.ix_(idx, idx)]
                t = np.triu_indices(n_g, k=1)
                SSW_p += np.sum(A[t]) / n_g
        SSB_p = SST - SSW_p
        if SSW_p > 0:
            F_p = (SSB_p / dfB) / (SSW_p / dfW)
            perm_F.append(F_p)
    if len(perm_F) == 0:
        return F_obs, np.nan
    perm_F = np.asarray(perm_F)
    p_val = (np.sum(perm_F >= F_obs) + 1) / (len(perm_F) + 1)
    return float(F_obs), float(p_val)


def anosim_test(distance_matrix: np.ndarray, labels: np.ndarray,
                n_permutations: int = N_PERMUTATIONS_DEFAULT) -> Tuple[float, float]:
    D = _hygienize_distance(distance_matrix)
    y = np.asarray(labels)
    mask = y != -1
    D = D[np.ix_(mask, mask)]
    y = y[mask]
    if len(np.unique(y)) < 2 or D.shape[0] < 4:
        return np.nan, np.nan

    if _HAS_SKBIO:
        try:
            dm = skbio.DistanceMatrix(D)
            groups = pd.Series(y, name='group')
            res = skbio_anosim(dm, groups, permutations=n_permutations)
            return float(res['test statistic']), float(res['p-value'])
        except Exception:
            pass

    # Fallback ANOSIM
    # Rank all upper-triangular distances; compute mean rank within vs between groups.
    # R = (mean_between - mean_within) / (L/2) where L = #pairs = N(N-1)/2
    N = D.shape[0]
    triu = np.triu_indices(N, k=1)
    dist = D[triu]
    ranks = stats.rankdata(dist, method='average')
    L = len(ranks)

    def _means(D, y, ranks):
        within, between = [], []
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if y[i] != -1 and y[j] != -1:
                    if y[i] == y[j]:
                        within.append(ranks[idx])
                    else:
                        between.append(ranks[idx])
                idx += 1
        return (np.mean(within) if within else np.nan,
                np.mean(between) if between else np.nan)

    mw, mb = _means(D, y, ranks)
    if not np.isfinite(mw) or not np.isfinite(mb) or L == 0:
        return np.nan, np.nan
    R_obs = (mb - mw) / (L / 2)

    perm_R = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        mw_p, mb_p = _means(D, y_perm, ranks)
        if np.isfinite(mw_p) and np.isfinite(mb_p):
            perm_R.append((mb_p - mw_p) / (L / 2))
    if len(perm_R) == 0:
        return R_obs, np.nan
    perm_R = np.asarray(perm_R)
    p_val = (np.sum(perm_R >= R_obs) + 1) / (len(perm_R) + 1)
    return float(R_obs), float(p_val)


# ---------------------------
# Bootstrap helpers
# ---------------------------
def _ci_from_samples(samples: List[float], conf: float = CONFIDENCE_LEVEL) -> Tuple[float, float]:
    x = np.asarray([s for s in samples if np.isfinite(s)], dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    alpha = 1 - conf
    return (float(np.percentile(x, 100 * alpha / 2)),
            float(np.percentile(x, 100 * (1 - alpha / 2))))


def _bootstrap_internal_silhouette(D: np.ndarray, n_bootstrap: int) -> Tuple[float, Tuple[float, float]]:
    if not _HAS_SKLEARN:
        return np.nan, (np.nan, np.nan)
    sil_full = np.nan
    try:
        pred_full = _agglomerative_labels(D)
        if pred_full is not None and len(np.unique(pred_full)) >= 2:
            sil_full = silhouette_score(D, pred_full, metric='precomputed')
    except Exception:
        pass

    vals = []
    N = D.shape[0]
    for _ in range(n_bootstrap):
        idx = np.random.choice(N, size=N, replace=True)
        Db = D[np.ix_(idx, idx)]
        try:
            pb = _agglomerative_labels(Db)
            if pb is not None and len(np.unique(pb)) >= 2:
                vals.append(silhouette_score(Db, pb, metric='precomputed'))
        except Exception:
            continue
    return float(sil_full), _ci_from_samples(vals)


def _bootstrap_db_ch(D: np.ndarray, n_bootstrap: int, which: str) -> Tuple[float, Tuple[float, float]]:
    if not _HAS_SKLEARN:
        return np.nan, (np.nan, np.nan)
    def _embed(dm):
        m = min(10, dm.shape[0] - 1) if dm.shape[0] > 1 else 1
        return MDS(n_components=m, dissimilarity='precomputed', random_state=RANDOM_STATE).fit_transform(dm)

    full_val = np.nan
    try:
        coords = _embed(D)
        pred = _agglomerative_labels(D)
        if pred is not None and len(np.unique(pred)) >= 2:
            if which == 'db':
                full_val = davies_bouldin_score(coords, pred)
            else:
                full_val = calinski_harabasz_score(coords, pred)
    except Exception:
        pass

    vals = []
    N = D.shape[0]
    for _ in range(n_bootstrap):
        idx = np.random.choice(N, size=N, replace=True)
        Db = D[np.ix_(idx, idx)]
        try:
            coords_b = _embed(Db)
            pb = _agglomerative_labels(Db)
            if pb is not None and len(np.unique(pb)) >= 2:
                v = davies_bouldin_score(coords_b, pb) if which == 'db' else calinski_harabasz_score(coords_b, pb)
                vals.append(v)
        except Exception:
            continue
    return float(full_val), _ci_from_samples(vals)


def _bootstrap_dunn(D: np.ndarray, n_bootstrap: int) -> Tuple[float, Tuple[float, float]]:
    full_val = np.nan
    try:
        pred = _agglomerative_labels(D)
        if pred is not None and len(np.unique(pred)) >= 2:
            full_val = dunn_index(D, pred)
    except Exception:
        pass

    vals = []
    N = D.shape[0]
    for _ in range(n_bootstrap):
        idx = np.random.choice(N, size=N, replace=True)
        Db = D[np.ix_(idx, idx)]
        try:
            pb = _agglomerative_labels(Db)
            if pb is not None and len(np.unique(pb)) >= 2:
                vals.append(dunn_index(Db, pb))
        except Exception:
            continue
    return float(full_val), _ci_from_samples(vals)


def _cluster_stability(D: np.ndarray, n_bootstrap: int = 100) -> Dict[str, Any]:
    """Returns ARI stability and Jaccard (after Hungarian alignment)."""
    if not _HAS_SKLEARN:
        return {'ari_mean': np.nan, 'ari_ci': (np.nan, np.nan),
                'jaccard_mean': np.nan, 'jaccard_ci': (np.nan, np.nan)}
    try:
        orig = _agglomerative_labels(D)
        if orig is None or len(np.unique(orig)) < 2:
            raise ValueError("Could not form 2 clusters for stability baseline.")
    except Exception:
        return {'ari_mean': np.nan, 'ari_ci': (np.nan, np.nan),
                'jaccard_mean': np.nan, 'jaccard_ci': (np.nan, np.nan)}

    N = D.shape[0]
    ari_vals, jac_vals = [], []

    for _ in range(n_bootstrap):
        idx = np.random.choice(N, size=N, replace=True)
        Db = D[np.ix_(idx, idx)]
        try:
            boot = _agglomerative_labels(Db)
            if boot is None or len(np.unique(boot)) < 2:
                continue
            # Align labels (Hungarian) to handle label switching
            cm = confusion_matrix(orig[idx], boot, labels=[0, 1])
            r, c = linear_sum_assignment(-cm)
            mapping = {c[i]: r[i] for i in range(len(r))}
            boot_aligned = np.array([mapping[b] for b in boot])

            ari_vals.append(adjusted_rand_score(orig[idx], boot_aligned))

            # Jaccard on membership of cluster 1 (after alignment)
            A = set(np.where(orig[idx] == 1)[0])
            B = set(np.where(boot_aligned == 1)[0])
            denom = len(A | B)
            jacc = (len(A & B) / denom) if denom > 0 else 1.0
            jac_vals.append(jacc)
        except Exception:
            continue

    return {
        'ari_mean': float(np.nanmean(ari_vals)) if len(ari_vals) else np.nan,
        'ari_ci': _ci_from_samples(ari_vals),
        'jaccard_mean': float(np.nanmean(jac_vals)) if len(jac_vals) else np.nan,
        'jaccard_ci': _ci_from_samples(jac_vals),
    }


# ---------------------------
# Main computation
# ---------------------------
def compute_all_metrics(distance_matrix: np.ndarray, labels: np.ndarray,
                        filenames: List[str], n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
                        n_permutations: int = N_PERMUTATIONS_DEFAULT) -> Dict[str, Any]:
    results = {}

    # Filter to known labels for external tests
    known = labels != -1
    if not np.any(known):
        raise SystemExit("No valid labels found (all unknown).")

    D = _hygienize_distance(distance_matrix)
    D_valid = D[np.ix_(known, known)]
    y_valid = labels[known]

    if len(np.unique(y_valid)) < 2:
        raise SystemExit("Need at least two groups (random/trained).")

    # INTERNAL QUALITY (predicted labels on D_valid)
    if _HAS_SKLEARN:
        sil_val, sil_ci = _bootstrap_internal_silhouette(D_valid, n_bootstrap)
        results['silhouette_coefficient'] = {
            'value': sil_val, 'ci': sil_ci,
            'interpretation': 'Higher is better (range: -1 to 1)'
        }

        db_val, db_ci = _bootstrap_db_ch(D_valid, min(100, n_bootstrap), which='db')
        results['davies_bouldin_index'] = {
            'value': db_val, 'ci': db_ci,
            'interpretation': 'Lower is better (>0)'
        }

        ch_val, ch_ci = _bootstrap_db_ch(D_valid, min(100, n_bootstrap), which='ch')
        results['calinski_harabasz_index'] = {
            'value': ch_val, 'ci': ch_ci,
            'interpretation': 'Higher is better (>0)'
        }
    else:
        results['silhouette_coefficient'] = {'value': np.nan, 'ci': (np.nan, np.nan)}
        results['davies_bouldin_index'] = {'value': np.nan, 'ci': (np.nan, np.nan)}
        results['calinski_harabasz_index'] = {'value': np.nan, 'ci': (np.nan, np.nan)}

    dunn_val, dunn_ci = _bootstrap_dunn(D_valid, n_bootstrap)
    results['dunn_index'] = {'value': dunn_val, 'ci': dunn_ci, 'interpretation': 'Higher is better (>0)'}

    coph_val = cophenetic_correlation(D_valid)
    # Bootstrap CCC (no labels): resample rows/cols jointly
    coph_samples = []
    N = D_valid.shape[0]
    for _ in range(n_bootstrap):
        idx = np.random.choice(N, size=N, replace=True)
        Db = D_valid[np.ix_(idx, idx)]
        coph_samples.append(cophenetic_correlation(Db))
    results['cophenetic_correlation'] = {'value': coph_val, 'ci': _ci_from_samples(coph_samples),
                                         'interpretation': 'Higher is better (range: -1 to 1)'}

    # HYPOTHESIS TESTS (use GT groups)
    F_stat, F_p = permanova_test(D_valid, y_valid, n_permutations=n_permutations)
    results['permanova'] = {'f_statistic': F_stat, 'p_value': F_p, 'interpretation': 'Tests group centroid differences'}

    R_stat, R_p = anosim_test(D_valid, y_valid, n_permutations=n_permutations)
    results['anosim'] = {'r_statistic': R_stat, 'p_value': R_p,
                         'interpretation': 'Within-group vs between-group distances'}

    # EXTERNAL VALIDATION (compare predicted vs GT on full D_valid)
    if _HAS_SKLEARN:
        try:
            pred = _agglomerative_labels(D_valid)
            if pred is None or len(np.unique(pred)) < 2:
                raise ValueError("Agglomerative failed to produce 2 clusters.")
            ari = adjusted_rand_score(y_valid, pred)
            ami = adjusted_mutual_info_score(y_valid, pred)
            vms = v_measure_score(y_valid, pred)
        except Exception:
            ari = ami = vms = np.nan
    else:
        ari = ami = vms = np.nan

    results['adjusted_rand_index'] = {'value': ari, 'ci': (ari, ari),
                                      'interpretation': 'Higher is better (-1 to 1, 1=perfect)'}
    results['adjusted_mutual_info'] = {'value': ami, 'ci': (ami, ami),
                                       'interpretation': 'Higher is better (0 to 1, 1=perfect)'}
    results['v_measure'] = {'value': vms, 'ci': (vms, vms),
                            'interpretation': 'Higher is better (0 to 1, 1=perfect)'}

    # EFFECT SIZE (inter vs intra distances under GT groups)
    rand_idx = np.where(y_valid == 0)[0]
    trnd_idx = np.where(y_valid == 1)[0]
    intra = []
    if len(rand_idx) > 1:
        A = D_valid[np.ix_(rand_idx, rand_idx)]
        t = np.triu_indices_from(A, k=1)
        intra.extend(A[t])
    if len(trnd_idx) > 1:
        B = D_valid[np.ix_(trnd_idx, trnd_idx)]
        t = np.triu_indices_from(B, k=1)
        intra.extend(B[t])
    inter = D_valid[np.ix_(rand_idx, trnd_idx)].ravel() if len(rand_idx) and len(trnd_idx) else np.array([])
    results['cohens_d'] = {'value': cohens_d(np.asarray(inter), np.asarray(intra)),
                           'interpretation': '|d|>0.8=large, >0.5=medium, >0.2=small'}

    # STABILITY (internal reclustering)
    stab = _cluster_stability(D_valid, n_bootstrap=min(100, n_bootstrap))
    results['stability_ari'] = {'value': stab['ari_mean'], 'ci': stab['ari_ci'],
                                'interpretation': 'ARI stability under bootstrap re-clustering'}
    results['stability_jaccard'] = {'value': stab['jaccard_mean'], 'ci': stab['jaccard_ci'],
                                    'interpretation': 'Set Jaccard (label-aligned) under bootstrap'}

    return results


# ---------------------------
# Output helpers
# ---------------------------
def format_latex_table(results: Dict[str, Any], output_path: str, title: str = "Distance Matrix Cluster Analysis"):
    """Generate LaTeX table. Avoid f-strings for LaTeX braces to prevent NameError."""
    import math
    def fmt_ci(res: Dict[str, Any]):
        v = res.get('value', float('nan'))
        ci = res.get('ci', (float('nan'), float('nan')))
        if v is None or not np.isfinite(v):
            return "N/A & [N/A, N/A]"
        lo, hi = ci if isinstance(ci, (list, tuple)) and len(ci) == 2 else (float('nan'), float('nan'))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return f"{v:.3f} & [N/A, N/A]"
        return f"{v:.3f} & [{lo:.3f}, {hi:.3f}]"

    def fmt_stat_p(v, p):
        if v is None or not np.isfinite(v):
            return "N/A & N/A"
        if p is None or not np.isfinite(p):
            return f"{v:.3f} & N/A"
        return f"{v:.3f} & $p={p:.2e}$" if p < 1e-3 else f"{v:.3f} & $p={p:.3f}$"

    # Safely escape braces in title to avoid breaking LaTeX
    safe_title = str(title).replace("{", "\\{").replace("}", "\\}")

    lines = []
    lines += [
        r"\begin{table}[h!]",
        r"\centering",
        rf"\caption{{{safe_title}}}",
        r"\label{tab:cluster_analysis}",
        r"\begin{minipage}[t]{0.49\textwidth}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Value & 95\% CI \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Internal Quality}} \\",
        # values inserted below (no LaTeX braces in the f-strings themselves)
        f"Silhouette Coefficient & {fmt_ci(results.get('silhouette_coefficient', {}))} \\",
        f"Davies-Bouldin Index & {fmt_ci(results.get('davies_bouldin_index', {}))} \\",
        f"Calinski-Harabasz & {fmt_ci(results.get('calinski_harabasz_index', {}))} \\",
        f"Dunn Index & {fmt_ci(results.get('dunn_index', {}))} \\",
        f"Cophenetic Correlation & {fmt_ci(results.get('cophenetic_correlation', {}))} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Stability}} \\",
        f"ARI Stability & {fmt_ci(results.get('stability_ari', {}))} \\",
        f"Jaccard Stability & {fmt_ci(results.get('stability_jaccard', {}))} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Value & 95\% CI \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{External Validation}} \\",
        f"Adjusted Rand Index & {fmt_ci(results.get('adjusted_rand_index', {}))} \\",
        f"Adjusted Mutual Info & {fmt_ci(results.get('adjusted_mutual_info', {}))} \\",
        f"V-measure & {fmt_ci(results.get('v_measure', {}))} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Hypothesis Tests}} \\",
        # stats + p-values (no braces in f-strings)
        f"PERMANOVA F & {fmt_stat_p(results.get('permanova', {}).get('f_statistic', float('nan'),), results.get('permanova', {}).get('p_value', float('nan')))} \\",
        f"ANOSIM R & {fmt_stat_p(results.get('anosim', {}).get('r_statistic', float('nan')), results.get('anosim', {}).get('p_value', float('nan')))} \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Effect Size}} \\",
        # Cohen's d has no CI here
        (f"Cohen's $d$ & {results.get('cohens_d', {}).get('value', float('nan')):.3f} & -- \\"
         if np.isfinite(results.get('cohens_d', {}).get('value', float('nan')))
         else "Cohen's $d$ & N/A & -- \\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{minipage}",
        r"\end{table}",
        ""
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))



def save_results(results: Dict[str, Any], output_prefix: str):
    # JSON
    json_path = f"{output_prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
    print(f"Detailed results saved to: {json_path}")

    # CSV summary
    rows = []
    for k, v in results.items():
        row = {'Metric': k.replace('_', ' ').title()}
        if isinstance(v, dict):
            if 'value' in v:
                row['Value'] = v['value']
            if 'ci' in v:
                ci = v['ci']
                row['CI_Lower'] = ci[0] if isinstance(ci, (list, tuple)) else np.nan
                row['CI_Upper'] = ci[1] if isinstance(ci, (list, tuple)) else np.nan
            if 'p_value' in v:
                row['P_Value'] = v['p_value']
            row['Interpretation'] = v.get('interpretation', '')
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{output_prefix}_summary.csv", index=False, float_format="%.6f")
    print(f"Summary table saved to: {output_prefix}_summary.csv")

    # LaTeX
    format_latex_table(results, f"{output_prefix}_table.tex")


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute comprehensive clustering metrics from distance matrix (fixed).")
    ap.add_argument('--distance-file', type=str, required=True, help='Path to distance matrix (.npy)')
    ap.add_argument('--index-file', type=str, required=True, help='Path to file index (.json of filenames)')
    ap.add_argument('--output-prefix', type=str, default='comprehensive_clustering', help='Output prefix')
    ap.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP_DEFAULT, help='Bootstrap iterations')
    ap.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS_DEFAULT, help='Permutations for tests')
    args = ap.parse_args()

    D = np.load(args.distance_file)
    with open(args.index_file, 'r') as f:
        filenames = json.load(f)

    if D.shape[0] != D.shape[1]:
        raise SystemExit(f"Distance matrix must be square, got {D.shape}")
    if D.shape[0] != len(filenames):
        print(f"Warning: matrix size {D.shape[0]} != filenames {len(filenames)} (will still proceed)")

    _, y, idxs = extract_labels_from_filenames(filenames)
    if len(idxs['random']) == 0 or len(idxs['trained']) == 0:
        raise SystemExit("Error: Need both random and trained models for clustering analysis.")

    results = compute_all_metrics(D, y, filenames, n_bootstrap=args.n_bootstrap, n_permutations=args.n_permutations)
    save_results(results, args.output_prefix)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE CLUSTERING ANALYSIS SUMMARY")
    print("=" * 80)
    for m, d in results.items():
        print(f"\n{m.replace('_', ' ').title()}:")
        if isinstance(d, dict):
            if 'value' in d:
                v = d['value']
                ci = d.get('ci', (np.nan, np.nan))
                if np.isfinite(v) and isinstance(ci, (tuple, list)) and np.isfinite(ci[0]) and np.isfinite(ci[1]):
                    print(f"  Value: {v:.6f} [95% CI: {ci[0]:.6f}, {ci[1]:.6f}]")
                elif np.isfinite(v):
                    print(f"  Value: {v:.6f}")
                else:
                    print("  Value: N/A")
            if 'f_statistic' in d or 'r_statistic' in d:
                stat = d.get('f_statistic', d.get('r_statistic', np.nan))
                pv = d.get('p_value', np.nan)
                if np.isfinite(stat):
                    print(f"  Stat: {stat:.6f}, p-value: {pv if not np.isfinite(pv) else f'{pv:.6f}'}")
            if 'interpretation' in d:
                print(f"  Interpretation: {d['interpretation']}")
    print("\n" + "=" * 80)
    print("Analysis complete. See output files for details.")


if __name__ == "__main__":
    main()
