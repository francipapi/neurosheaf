#!/usr/bin/env python3
"""nystrom_validate_resnet.py

Validate the Nyström CKA implementation by comparing performance against
the exact debiased CKA on ResNet-50 layer similarity patterns.

This script tests:
1. Approximation accuracy vs exact CKA
2. Memory efficiency improvements
3. Computational speed improvements
4. Mathematical property preservation
5. ResNet similarity pattern reproduction

Expected results:
- High correlation (>0.9) with exact CKA patterns
- 5-10x memory savings for large activations
- Similar or faster computation time
- Preserved mathematical properties

Usage:
-----
python nystrom_validate_resnet.py --num-images 500 --device cuda --plot
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as dsets
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import psutil

# Make sure the local CKA package is importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # Add parent directory to path
from neurosheaf.cka.debiased import DebiasedCKA
from neurosheaf.cka.nystrom import NystromCKA


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Nyström CKA on ResNet-50")

    parser.add_argument("--dataset", choices=["cifar10", "imagenet"],
                        default="cifar10", help="Dataset to use")
    parser.add_argument("--num-images", type=int, default=500,
                        help="Number of images to process")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size during forward pass")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    parser.add_argument("--plot", action="store_true",
                        help="Show comparison plots")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to files")
    parser.add_argument("--imagenet-val-root", type=str, default=None,
                        help="Path to ImageNet validation set root (when --dataset imagenet)")
    parser.add_argument("--n-landmarks", type=int, default=256,
                        help="Number of landmarks for Nyström approximation")
    parser.add_argument("--test-landmark-counts", nargs="+", type=int, 
                        default=[64, 128, 256, 512],
                        help="Different landmark counts to test")
    parser.add_argument("--compare-exact", action="store_true",
                        help="Compare against exact CKA (memory intensive)")

    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """Return a DataLoader yielding *args.num_images* images."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    if args.dataset == "cifar10":
        ds = dsets.CIFAR10(root="~/.torch/data", train=False,
                           transform=transform, download=True)
    else:
        if args.imagenet_val_root is None:
            raise ValueError("--imagenet-val-root must be set when dataset=imagenet")
        ds = dsets.ImageFolder(root=args.imagenet_val_root, transform=transform)

    # Subset to the desired number of images
    idx = torch.randperm(len(ds))[:args.num_images]
    subset = torch.utils.data.Subset(ds, idx.tolist())

    loader = torch.utils.data.DataLoader(subset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)
    return loader


def register_hooks(model: nn.Module, layer_names: List[str],
                   activations: Dict[str, List[torch.Tensor]]) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks and append handles so we can remove them later."""

    def get_module_by_name(root: nn.Module, dotted: str) -> nn.Module:
        obj = root
        for attr in dotted.split('.'):
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        return obj

    handles = []

    for name in layer_names:
        module = get_module_by_name(model, name)
        handle = module.register_forward_hook(
            lambda m, inp, out, n=name: activations.setdefault(n, []).append(out.detach().cpu())
        )
        handles.append(handle)

    return handles


def collate_activations(activations: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate batch-wise activation lists into (N, F) tensors."""
    final = {}
    for name, tensors in activations.items():
        act = torch.cat(tensors, dim=0)             # (N, C, H, W) or (N, F)
        if act.dim() > 2:
            act = torch.flatten(act.mean(dim=(2, 3)), 1)  # Global-average across spatial dims
        final[name] = act  # (N, F)
    return final


def measure_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def compute_cka_matrix(activations: Dict[str, torch.Tensor], 
                      cka_computer, layer_names: List[str],
                      method_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute CKA matrix and track performance metrics."""
    n_layers = len(layer_names)
    cka_matrix = torch.zeros((n_layers, n_layers), dtype=torch.float32)
    
    start_time = time.time()
    start_memory = measure_memory_usage()
    peak_memory = start_memory
    
    print(f"\nComputing {method_name} CKA matrix for {n_layers} layers...")
    
    for i, layer_i in enumerate(layer_names):
        for j, layer_j in enumerate(layer_names):
            if i <= j:  # Only compute upper triangle (including diagonal)
                act_i = activations[layer_i]
                act_j = activations[layer_j]
                
                # Track memory during computation
                current_memory = measure_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                
                # Compute CKA between these two layers
                cka_val = cka_computer.compute(act_i, act_j, validate_properties=False)
                cka_matrix[i, j] = cka_val
                
                # Matrix is symmetric
                if i != j:
                    cka_matrix[j, i] = cka_val
                    
                print(f"  {method_name} CKA({layer_i}, {layer_j}) = {cka_val:.4f}")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    metrics = {
        'computation_time': end_time - start_time,
        'peak_memory_gb': peak_memory,
        'memory_increase_gb': peak_memory - start_memory,
        'final_memory_gb': end_memory
    }
    
    print(f"{method_name} computation completed in {metrics['computation_time']:.2f}s")
    print(f"Peak memory: {metrics['peak_memory_gb']:.2f}GB (+ {metrics['memory_increase_gb']:.2f}GB)")
    
    return cka_matrix, metrics


def compare_matrices(exact_matrix: torch.Tensor, 
                    approx_matrix: torch.Tensor,
                    layer_names: List[str],
                    method_name: str) -> Dict[str, float]:
    """Compare exact and approximate CKA matrices."""
    exact_np = exact_matrix.cpu().numpy()
    approx_np = approx_matrix.cpu().numpy()
    
    # Flatten upper triangle (excluding diagonal) for correlation analysis
    n = exact_np.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    exact_upper = exact_np[upper_indices]
    approx_upper = approx_np[upper_indices]
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(exact_upper, approx_upper)
    spearman_r, spearman_p = spearmanr(exact_upper, approx_upper)
    
    # Compute errors
    mae = np.mean(np.abs(exact_np - approx_np))
    rmse = np.sqrt(np.mean((exact_np - approx_np)**2))
    max_error = np.max(np.abs(exact_np - approx_np))
    
    # Diagonal accuracy (should be 1.0 for both)
    exact_diag = np.diag(exact_np)
    approx_diag = np.diag(approx_np)
    diag_error = np.mean(np.abs(exact_diag - approx_diag))
    
    metrics = {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'diagonal_error': diag_error
    }
    
    print(f"\n=== {method_name} vs Exact CKA Comparison ===")
    print(f"Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"Mean Absolute Error:  {mae:.4f}")
    print(f"Root Mean Sq. Error:  {rmse:.4f}")
    print(f"Maximum Error:        {max_error:.4f}")
    print(f"Diagonal Error:       {diag_error:.4f}")
    
    # Quality assessment
    if pearson_r > 0.95:
        print("✅ Excellent approximation quality")
    elif pearson_r > 0.90:
        print("✅ Good approximation quality")
    elif pearson_r > 0.80:
        print("⚠️  Acceptable approximation quality")
    else:
        print("❌ Poor approximation quality")
    
    return metrics


def validate_mathematical_properties(cka_matrix: torch.Tensor, 
                                    layer_names: List[str],
                                    method_name: str) -> Dict[str, bool]:
    """Validate mathematical properties of CKA matrix."""
    cka_np = cka_matrix.cpu().numpy()
    n = cka_np.shape[0]
    
    print(f"\n=== {method_name} Mathematical Properties ===")
    
    # Test 1: Diagonal elements should be close to 1.0
    diag_elements = np.diag(cka_np)
    diag_ok = np.allclose(diag_elements, 1.0, atol=0.05)  # 5% tolerance for Nyström
    diag_mean = np.mean(diag_elements)
    diag_std = np.std(diag_elements)
    print(f"Diagonal elements: mean={diag_mean:.4f}, std={diag_std:.4f} {'✅' if diag_ok else '❌'}")
    
    # Test 2: Matrix should be symmetric
    symmetry_error = np.max(np.abs(cka_np - cka_np.T))
    symmetric_ok = symmetry_error < 1e-6
    print(f"Symmetry error: {symmetry_error:.2e} {'✅' if symmetric_ok else '❌'}")
    
    # Test 3: All values should be in [0, 1]
    min_val = np.min(cka_np)
    max_val = np.max(cka_np)
    bounded_ok = (min_val >= -1e-6) and (max_val <= 1 + 1e-6)
    print(f"Bounds: [{min_val:.4f}, {max_val:.4f}] {'✅' if bounded_ok else '❌'}")
    
    # Test 4: Positive semidefinite (for Gram matrix interpretation)
    try:
        eigenvals = np.linalg.eigvals(cka_np)
        min_eigenval = np.min(eigenvals)
        psd_ok = min_eigenval >= -1e-6
        print(f"Min eigenvalue: {min_eigenval:.2e} {'✅' if psd_ok else '❌'}")
    except:
        psd_ok = False
        print("Could not compute eigenvalues ❌")
    
    properties = {
        'diagonal_correct': diag_ok,
        'symmetric': symmetric_ok,
        'bounded': bounded_ok,
        'positive_semidefinite': psd_ok
    }
    
    passed = sum(properties.values())
    total = len(properties)
    print(f"Mathematical properties: {passed}/{total} passed")
    
    return properties


def test_landmark_scaling(activations: Dict[str, torch.Tensor],
                         layer_names: List[str],
                         landmark_counts: List[int],
                         exact_matrix: Optional[torch.Tensor] = None) -> Dict[int, Dict]:
    """Test how approximation quality varies with number of landmarks."""
    print(f"\n=== Testing Landmark Scaling ===")
    
    results = {}
    
    for n_landmarks in landmark_counts:
        print(f"\nTesting with {n_landmarks} landmarks...")
        
        # Create Nyström CKA with all fixes enabled
        nystrom_cka = NystromCKA(
            n_landmarks=n_landmarks,
            use_qr_approximation=True,
            enable_psd_projection=True,
            spectral_regularization=True,
            adaptive_landmarks=True,
            enable_profiling=True,
            device='cpu'  # Force CPU for consistent memory measurement
        )
        
        # Compute CKA matrix
        matrix, metrics = compute_cka_matrix(
            activations, nystrom_cka, layer_names, 
            f"Nyström-{n_landmarks}"
        )
        
        # Validate properties
        properties = validate_mathematical_properties(matrix, layer_names, f"Nyström-{n_landmarks}")
        
        result = {
            'matrix': matrix,
            'metrics': metrics,
            'properties': properties,
            'n_landmarks': n_landmarks
        }
        
        # Compare with exact if available
        if exact_matrix is not None:
            comparison = compare_matrices(exact_matrix, matrix, layer_names, f"Nyström-{n_landmarks}")
            result['comparison'] = comparison
        
        results[n_landmarks] = result
    
    return results


def plot_comparison_results(exact_matrix: torch.Tensor,
                           nystrom_results: Dict[int, Dict],
                           layer_names: List[str],
                           args: argparse.Namespace) -> None:
    """Create comprehensive comparison plots."""
    if not args.plot:
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: CKA Matrix Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Exact CKA matrix
    im1 = axes[0, 0].imshow(exact_matrix.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[0, 0].set_title('Exact CKA')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Layer')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Best Nyström result (highest landmark count)
    best_landmarks = max(nystrom_results.keys())
    best_matrix = nystrom_results[best_landmarks]['matrix']
    im2 = axes[0, 1].imshow(best_matrix.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Nyström CKA ({best_landmarks} landmarks)')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Layer')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference matrix
    diff_matrix = torch.abs(exact_matrix - best_matrix).cpu().numpy()
    im3 = axes[1, 0].imshow(diff_matrix, cmap='Reds', vmin=0, vmax=np.max(diff_matrix))
    axes[1, 0].set_title('Absolute Difference')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Layer')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Correlation plot
    exact_upper = exact_matrix.cpu().numpy()[np.triu_indices(len(layer_names), k=1)]
    best_upper = best_matrix.cpu().numpy()[np.triu_indices(len(layer_names), k=1)]
    axes[1, 1].scatter(exact_upper, best_upper, alpha=0.6)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[1, 1].set_xlabel('Exact CKA')
    axes[1, 1].set_ylabel('Nyström CKA')
    axes[1, 1].set_title('Correlation Plot')
    
    # Add correlation coefficient
    corr = np.corrcoef(exact_upper, best_upper)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 1].transAxes, 
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if args.save_results:
        plt.savefig('nystrom_cka_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Landmark Scaling Analysis
    if len(nystrom_results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        landmarks = sorted(nystrom_results.keys())
        
        # Correlation vs landmarks
        correlations = [nystrom_results[n]['comparison']['pearson_r'] for n in landmarks]
        axes[0, 0].plot(landmarks, correlations, 'o-')
        axes[0, 0].set_xlabel('Number of Landmarks')
        axes[0, 0].set_ylabel('Pearson Correlation')
        axes[0, 0].set_title('Approximation Quality vs Landmarks')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error vs landmarks
        maes = [nystrom_results[n]['comparison']['mae'] for n in landmarks]
        axes[0, 1].plot(landmarks, maes, 'o-', color='red')
        axes[0, 1].set_xlabel('Number of Landmarks')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('Error vs Landmarks')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Computation time vs landmarks
        times = [nystrom_results[n]['metrics']['computation_time'] for n in landmarks]
        axes[1, 0].plot(landmarks, times, 'o-', color='green')
        axes[1, 0].set_xlabel('Number of Landmarks')
        axes[1, 0].set_ylabel('Computation Time (s)')
        axes[1, 0].set_title('Speed vs Landmarks')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory usage vs landmarks
        memories = [nystrom_results[n]['metrics']['peak_memory_gb'] for n in landmarks]
        axes[1, 1].plot(landmarks, memories, 'o-', color='purple')
        axes[1, 1].set_xlabel('Number of Landmarks')
        axes[1, 1].set_ylabel('Peak Memory (GB)')
        axes[1, 1].set_title('Memory Usage vs Landmarks')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if args.save_results:
            plt.savefig('nystrom_scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_results_summary(exact_metrics: Dict, 
                        nystrom_results: Dict[int, Dict],
                        layer_names: List[str],
                        args: argparse.Namespace) -> None:
    """Save comprehensive results summary."""
    if not args.save_results:
        return
    
    summary = {
        'experiment_config': {
            'num_images': args.num_images,
            'dataset': args.dataset,
            'device': args.device,
            'layer_names': layer_names
        },
        'exact_cka_metrics': exact_metrics,
        'nystrom_results': {}
    }
    
    # Convert torch tensors to numpy for JSON serialization
    for n_landmarks, result in nystrom_results.items():
        summary['nystrom_results'][n_landmarks] = {
            'metrics': result['metrics'],
            'properties': result['properties'],
            'comparison': result.get('comparison', {}),
            'matrix_shape': list(result['matrix'].shape)
        }
    
    # Save summary as JSON
    import json
    with open('nystrom_validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save matrices as numpy arrays
    np.save('exact_cka_matrix.npy', exact_metrics.get('matrix', np.array([])))
    for n_landmarks, result in nystrom_results.items():
        np.save(f'nystrom_matrix_{n_landmarks}.npy', result['matrix'].cpu().numpy())
    
    print(f"\nResults saved:")
    print(f"  - Summary: nystrom_validation_summary.json")
    print(f"  - Matrices: exact_cka_matrix.npy, nystrom_matrix_*.npy")


def main() -> None:
    args = get_args()
    
    print("Nyström CKA Validation on ResNet-50")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Images: {args.num_images}")
    print(f"  Device: {args.device}")
    print(f"  Landmarks: {args.n_landmarks}")
    print(f"  Compare exact: {args.compare_exact}")

    # ----- Model & layers ----------------------------------------------------
    model = models.resnet50(pretrained=True)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    # Representative layers from ResNet-50
    layer_names = [
        "conv1",
        "layer1.0.conv1", "layer1.2.conv3",
        "layer2.0.conv1", "layer2.3.conv3",
        "layer3.0.conv1", "layer3.5.conv3",
        "layer4.0.conv1", "layer4.2.conv3",
        "avgpool", "fc",
    ]

    # ----- Data --------------------------------------------------------------
    loader = load_dataset(args)
    print(f"\nLoaded {args.num_images} images from {args.dataset}")

    # ----- Activation capture ------------------------------------------------
    raw_acts: Dict[str, List[torch.Tensor]] = {}
    hooks = register_hooks(model, layer_names, raw_acts)

    print("Extracting activations...")
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            model(images)
            if (i + 1) % 5 == 0:
                print(f"  Processed {(i + 1) * args.batch_size} images")

    # Remove hooks
    for h in hooks:
        h.remove()

    # ---- Collate activations -----------------------------------------------
    activations = collate_activations(raw_acts)
    print(f"\nActivation shapes:")
    for name, act in activations.items():
        print(f"  {name}: {act.shape}")
    
    # Move to CPU for consistent memory measurement
    activations = {k: v.cpu().float() for k, v in activations.items()}

    # ----- Compute exact CKA (if requested) ---------------------------------
    exact_matrix = None
    exact_metrics = {}
    
    if args.compare_exact:
        print("\n" + "="*50)
        print("Computing Exact CKA Matrix")
        print("="*50)
        
        exact_cka = DebiasedCKA(device='cpu', enable_profiling=True)
        exact_matrix, exact_metrics = compute_cka_matrix(
            activations, exact_cka, layer_names, "Exact"
        )
        exact_metrics['matrix'] = exact_matrix

    # ----- Test Nyström approximation ---------------------------------------
    print("\n" + "="*50)
    print("Testing Nyström CKA Approximation")
    print("="*50)
    
    nystrom_results = test_landmark_scaling(
        activations, layer_names, args.test_landmark_counts, exact_matrix
    )

    # ----- Performance Summary ----------------------------------------------
    print("\n" + "="*50)
    print("Performance Summary")
    print("="*50)
    
    if exact_matrix is not None:
        print(f"Exact CKA:")
        print(f"  Time: {exact_metrics['computation_time']:.2f}s")
        print(f"  Peak Memory: {exact_metrics['peak_memory_gb']:.2f}GB")
    
    print(f"\nNyström CKA Results:")
    for n_landmarks in sorted(nystrom_results.keys()):
        result = nystrom_results[n_landmarks]
        metrics = result['metrics']
        properties = result['properties']
        
        print(f"\n  {n_landmarks} landmarks:")
        print(f"    Time: {metrics['computation_time']:.2f}s")
        print(f"    Peak Memory: {metrics['peak_memory_gb']:.2f}GB")
        print(f"    Properties: {sum(properties.values())}/{len(properties)} passed")
        
        if 'comparison' in result:
            comp = result['comparison']
            print(f"    Correlation: {comp['pearson_r']:.3f}")
            print(f"    MAE: {comp['mae']:.4f}")
            
            if exact_matrix is not None:
                speedup = exact_metrics['computation_time'] / metrics['computation_time']
                memory_ratio = exact_metrics['peak_memory_gb'] / metrics['peak_memory_gb']
                print(f"    Speedup: {speedup:.1f}x")
                print(f"    Memory efficiency: {memory_ratio:.1f}x")

    # ----- Validation and plotting ------------------------------------------
    if exact_matrix is not None and args.plot:
        plot_comparison_results(exact_matrix, nystrom_results, layer_names, args)
    
    # Save results
    save_results_summary(exact_metrics, nystrom_results, layer_names, args)
    
    # Final assessment
    if nystrom_results:
        best_landmarks = max(nystrom_results.keys())
        best_result = nystrom_results[best_landmarks]
        
        print(f"\n" + "="*50)
        print("Final Assessment")
        print("="*50)
        
        if 'comparison' in best_result:
            correlation = best_result['comparison']['pearson_r']
            mae = best_result['comparison']['mae']
            
            if correlation > 0.95 and mae < 0.05:
                print("🎉 Excellent Nyström approximation quality!")
            elif correlation > 0.90 and mae < 0.10:
                print("✅ Good Nyström approximation quality!")
            elif correlation > 0.80:
                print("⚠️  Acceptable Nyström approximation quality")
            else:
                print("❌ Poor approximation quality - consider more landmarks")
                
            print(f"Best configuration: {best_landmarks} landmarks")
            print(f"Correlation with exact: {correlation:.3f}")
            print(f"Mean absolute error: {mae:.4f}")
        
        properties_passed = sum(best_result['properties'].values())
        total_properties = len(best_result['properties'])
        print(f"Mathematical properties: {properties_passed}/{total_properties} passed")
        
        print("\nNyström CKA validation complete!")


if __name__ == "__main__":
    main()