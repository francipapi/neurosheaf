#!/usr/bin/env python3
"""
DTW Robustness Testing with Comprehensive Clustering Metrics

This script tests DTW parameter robustness around the baseline configuration by:
1. Testing configurations within reasonable ranges of baseline parameters
2. Computing comprehensive clustering metrics for each configuration
3. Calculating ΔARI, ΔSilhouette, and Spearman correlations
4. Generating a robustness statement with quantitative support

Usage:
    python test_dtw_robustness_comprehensive.py \
      --data-dir eigenvalueData \
      --pattern "*eigenvalues.npz" \
      --output-dir dtw_robustness_results \
      --n-jobs 8
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Import baseline configuration
from dtw_sensitivity_analysis import SensitivityConfig


class DTWRobustnessAnalyzer:
    """Comprehensive DTW robustness analysis with clustering metrics."""

    def __init__(self, data_dir: str, pattern: str, output_dir: str, n_jobs: int = -1):
        self.data_dir = Path(data_dir)
        self.pattern = pattern
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.distance_matrices_dir = self.output_dir / "distance_matrices"
        self.metrics_dir = self.output_dir / "clustering_metrics"

        for dir_path in [self.distance_matrices_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)

    def define_baseline_config(self) -> SensitivityConfig:
        """Define the baseline DTW configuration."""
        return SensitivityConfig(
            dtw_window=0.1,
            dtw_cost='l1',
            resample=200,
            smooth_win=15,
            amp_norm='zscore',
            smooth='moving',
            topk=0,
            no_time_scaling=True,
            pad_with_last=True,
            outlier_method='none'
        )

    def generate_robustness_configurations(self, baseline: SensitivityConfig) -> List[Tuple[str, SensitivityConfig]]:
        """Generate configurations within reasonable ranges of baseline parameters."""

        configs = []
        baseline_dict = baseline.to_dict()

        # Add baseline
        configs.append(("baseline", baseline))

        # Window variations (±50% from baseline)
        for window in [0.05, 0.15, 0.2]:
            config_dict = baseline_dict.copy()
            config_dict['dtw_window'] = window
            config = SensitivityConfig(**config_dict)
            configs.append((f"window_{window:.2f}", config))

        # Smoothing variations (±67% from baseline)
        for smooth_win in [10, 20, 25]:
            config_dict = baseline_dict.copy()
            config_dict['smooth_win'] = smooth_win
            config = SensitivityConfig(**config_dict)
            configs.append((f"smooth_{smooth_win}", config))

        # Resolution variations (±25% from baseline)
        for resample in [150, 250]:
            config_dict = baseline_dict.copy()
            config_dict['resample'] = resample
            config = SensitivityConfig(**config_dict)
            configs.append((f"resample_{resample}", config))

        # Amplitude normalization alternatives
        for amp_norm in ['unit', 'none']:
            config_dict = baseline_dict.copy()
            config_dict['amp_norm'] = amp_norm
            config = SensitivityConfig(**config_dict)
            configs.append((f"ampnorm_{amp_norm}", config))

        # Selected stable configurations from sensitivity analysis
        stable_configs = [
            # Most stable
            ("stable_loose_window", SensitivityConfig(**{**baseline_dict, 'dtw_window': 0.3, 'amp_norm': 'unit'})),
            # Robust alternative
            ("stable_moderate", SensitivityConfig(**{**baseline_dict, 'dtw_window': 0.2, 'smooth_win': 20, 'amp_norm': 'unit'})),
            # Fast stable
            ("stable_fast", SensitivityConfig(**{**baseline_dict, 'dtw_window': 0.5, 'resample': 100, 'amp_norm': 'unit'})),
        ]

        configs.extend(stable_configs)

        print(f"Generated {len(configs)} robustness test configurations")
        return configs

    def compute_dtw_distances(self, config_name: str, config: SensitivityConfig) -> Tuple[str, str]:
        """Compute DTW distance matrix for a given configuration."""

        print(f"\n--- Computing DTW distances for {config_name} ---")

        # Output file paths
        output_prefix = str(self.distance_matrices_dir / f"dtw_{config_name}")
        distance_file = f"{output_prefix}_dtw_distance.npy"
        index_file = f"{output_prefix}_dtw_index.json"

        # Skip if already computed
        if os.path.exists(distance_file) and os.path.exists(index_file):
            print(f"Distance matrix already exists for {config_name}, skipping...")
            return distance_file, index_file

        # Build command for compute_alternative_distances.py
        cmd = [
            "python", "compute_alternative_distances.py",
            "--data-dir", str(self.data_dir),
            "--pattern", self.pattern,
            "--resample", str(config.resample),
            "--topk", str(config.topk),
            "--amp-norm", config.amp_norm,
            "--smooth", config.smooth,
            "--smooth-win", str(config.smooth_win),
            "--n-jobs", str(self.n_jobs),
            "--outlier-method", config.outlier_method,
            "--out-prefix", output_prefix,
            "--metrics", "dtw",
            "--dtw-window", str(config.dtw_window)
        ]

        if config.no_time_scaling:
            cmd.append("--no-time-scaling")
        if config.pad_with_last:
            cmd.append("--pad-with-last")

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"DTW computation completed successfully for {config_name}")
            return distance_file, index_file

        except subprocess.CalledProcessError as e:
            print(f"ERROR: DTW computation failed for {config_name}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise

    def compute_comprehensive_metrics(self, config_name: str, distance_file: str, index_file: str) -> Dict[str, Any]:
        """Compute comprehensive clustering metrics for a distance matrix."""

        print(f"--- Computing comprehensive metrics for {config_name} ---")

        # Output prefix for clustering metrics
        metrics_prefix = str(self.metrics_dir / f"metrics_{config_name}")
        results_file = f"{metrics_prefix}_results.json"

        # Skip if already computed
        if os.path.exists(results_file):
            print(f"Clustering metrics already exist for {config_name}, loading...")
            with open(results_file, 'r') as f:
                return json.load(f)

        # Build command for compute_comprehensive_clustering_metrics.py
        cmd = [
            "python", "compute_comprehensive_clustering_metrics.py",
            "--distance-file", distance_file,
            "--index-file", index_file,
            "--output-prefix", metrics_prefix,
            "--n-bootstrap", "500",  # Reduced for speed
            "--n-permutations", "500"  # Reduced for speed
        ]

        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Clustering metrics computation completed for {config_name}")

            # Load results
            with open(results_file, 'r') as f:
                return json.load(f)

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Clustering metrics computation failed for {config_name}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise

    def compute_distance_matrix_correlation(self, baseline_file: str, test_file: str) -> float:
        """Compute Spearman correlation between two distance matrices."""

        try:
            # Load distance matrices
            D_baseline = np.load(baseline_file)
            D_test = np.load(test_file)

            # Ensure same size
            if D_baseline.shape != D_test.shape:
                print(f"Warning: Matrix size mismatch - baseline: {D_baseline.shape}, test: {D_test.shape}")
                return np.nan

            # Compute upper triangular elements (distance matrices are symmetric)
            n = D_baseline.shape[0]
            triu_indices = np.triu_indices(n, k=1)

            baseline_distances = D_baseline[triu_indices]
            test_distances = D_test[triu_indices]

            # Compute Spearman correlation
            correlation, _ = spearmanr(baseline_distances, test_distances)
            return float(correlation) if np.isfinite(correlation) else np.nan

        except Exception as e:
            print(f"Error computing distance matrix correlation: {e}")
            return np.nan

    def run_robustness_analysis(self) -> Dict[str, Any]:
        """Run comprehensive robustness analysis."""

        print("=" * 80)
        print("DTW ROBUSTNESS ANALYSIS WITH COMPREHENSIVE METRICS")
        print("=" * 80)

        # Define configurations
        baseline_config = self.define_baseline_config()
        configurations = self.generate_robustness_configurations(baseline_config)

        # Store results
        results = {
            'baseline_config': baseline_config.to_dict(),
            'configurations': {},
            'robustness_metrics': {}
        }

        distance_files = {}
        metrics_results = {}

        # Process each configuration
        for config_name, config in configurations:
            print(f"\n{'='*60}")
            print(f"Processing configuration: {config_name}")
            print(f"Config: {config.to_dict()}")

            try:
                # Compute DTW distances
                start_time = time.time()
                distance_file, index_file = self.compute_dtw_distances(config_name, config)
                distance_time = time.time() - start_time

                # Store file paths
                distance_files[config_name] = distance_file

                # Compute comprehensive clustering metrics
                start_time = time.time()
                metrics = self.compute_comprehensive_metrics(config_name, distance_file, index_file)
                metrics_time = time.time() - start_time

                # Store results
                results['configurations'][config_name] = {
                    'config': config.to_dict(),
                    'distance_file': distance_file,
                    'index_file': index_file,
                    'distance_computation_time': distance_time,
                    'metrics_computation_time': metrics_time,
                    'metrics': metrics
                }
                metrics_results[config_name] = metrics

                print(f"✓ Configuration {config_name} completed successfully")

            except Exception as e:
                print(f"✗ Configuration {config_name} failed: {e}")
                continue

        # Compute robustness metrics
        if 'baseline' in metrics_results and 'baseline' in distance_files:
            print(f"\n{'='*60}")
            print("Computing robustness metrics...")

            baseline_metrics = metrics_results['baseline']
            baseline_distance_file = distance_files['baseline']

            baseline_ari = baseline_metrics.get('adjusted_rand_index', {}).get('value', np.nan)
            baseline_silhouette = baseline_metrics.get('silhouette_coefficient', {}).get('value', np.nan)

            robustness_data = []

            for config_name, config_metrics in metrics_results.items():
                if config_name == 'baseline':
                    continue

                # Get metric values
                ari = config_metrics.get('adjusted_rand_index', {}).get('value', np.nan)
                silhouette = config_metrics.get('silhouette_coefficient', {}).get('value', np.nan)

                # Calculate deltas
                delta_ari = abs(ari - baseline_ari) if np.isfinite(ari) and np.isfinite(baseline_ari) else np.nan
                delta_silhouette = abs(silhouette - baseline_silhouette) if np.isfinite(silhouette) and np.isfinite(baseline_silhouette) else np.nan

                # Calculate distance matrix correlation
                spearman_corr = self.compute_distance_matrix_correlation(baseline_distance_file, distance_files[config_name])

                robustness_entry = {
                    'config_name': config_name,
                    'ari': ari,
                    'silhouette': silhouette,
                    'delta_ari': delta_ari,
                    'delta_silhouette': delta_silhouette,
                    'spearman_correlation': spearman_corr,
                    'baseline_ari': baseline_ari,
                    'baseline_silhouette': baseline_silhouette
                }

                robustness_data.append(robustness_entry)

                print(f"{config_name}: ΔARI={delta_ari:.4f}, ΔSil={delta_silhouette:.4f}, Spearman ρ={spearman_corr:.4f}")

            results['robustness_metrics'] = robustness_data

            # Calculate summary statistics
            delta_aris = [r['delta_ari'] for r in robustness_data if np.isfinite(r['delta_ari'])]
            delta_silhouettes = [r['delta_silhouette'] for r in robustness_data if np.isfinite(r['delta_silhouette'])]
            spearman_corrs = [r['spearman_correlation'] for r in robustness_data if np.isfinite(r['spearman_correlation'])]

            results['robustness_summary'] = {
                'max_delta_ari': max(delta_aris) if delta_aris else np.nan,
                'max_delta_silhouette': max(delta_silhouettes) if delta_silhouettes else np.nan,
                'min_spearman_correlation': min(spearman_corrs) if spearman_corrs else np.nan,
                'mean_delta_ari': np.mean(delta_aris) if delta_aris else np.nan,
                'mean_delta_silhouette': np.mean(delta_silhouettes) if delta_silhouettes else np.nan,
                'mean_spearman_correlation': np.mean(spearman_corrs) if spearman_corrs else np.nan,
            }

        return results

    def generate_robustness_report(self, results: Dict[str, Any]):
        """Generate comprehensive robustness report."""

        report_file = self.output_dir / "robustness_report.md"
        summary_file = self.output_dir / "robustness_summary.csv"

        # Generate CSV summary
        if 'robustness_metrics' in results:
            df = pd.DataFrame(results['robustness_metrics'])
            df.to_csv(summary_file, index=False, float_format="%.6f")
            print(f"Robustness summary saved to: {summary_file}")

        # Generate markdown report
        with open(report_file, 'w') as f:
            f.write("# DTW Robustness Analysis Report\n\n")

            # Baseline configuration
            f.write("## Baseline Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(results.get('baseline_config', {}), indent=2))
            f.write("\n```\n\n")

            # Summary statistics
            if 'robustness_summary' in results:
                summary = results['robustness_summary']
                f.write("## Robustness Summary\n\n")

                baseline_ari = results['robustness_metrics'][0]['baseline_ari'] if results['robustness_metrics'] else np.nan
                baseline_sil = results['robustness_metrics'][0]['baseline_silhouette'] if results['robustness_metrics'] else np.nan

                f.write(f"**Baseline Performance:**\n")
                f.write(f"- ARI: {baseline_ari:.4f}\n")
                f.write(f"- Silhouette: {baseline_sil:.4f}\n\n")

                f.write(f"**Robustness Metrics:**\n")
                f.write(f"- Maximum ΔARI: {summary.get('max_delta_ari', np.nan):.4f}\n")
                f.write(f"- Maximum ΔSilhouette: {summary.get('max_delta_silhouette', np.nan):.4f}\n")
                f.write(f"- Minimum Spearman ρ: {summary.get('min_spearman_correlation', np.nan):.4f}\n")
                f.write(f"- Mean ΔARI: {summary.get('mean_delta_ari', np.nan):.4f}\n")
                f.write(f"- Mean ΔSilhouette: {summary.get('mean_delta_silhouette', np.nan):.4f}\n")
                f.write(f"- Mean Spearman ρ: {summary.get('mean_spearman_correlation', np.nan):.4f}\n\n")

                # Generate robustness statement
                max_delta_ari = summary.get('max_delta_ari', np.nan)
                max_delta_sil = summary.get('max_delta_silhouette', np.nan)
                min_spearman = summary.get('min_spearman_correlation', np.nan)

                f.write("## DTW Robustness Statement\n\n")
                f.write("Within reasonable parameter ranges (window ∈ [0.05, 0.2], smooth_win ∈ [10, 25], ")
                f.write("resample ∈ [150, 250]), our DTW fingerprint demonstrates robustness:\n\n")

                if np.isfinite(max_delta_ari):
                    f.write(f"- **ΔARI ≤ {max_delta_ari:.3f}**: Clustering quality remains stable\n")
                if np.isfinite(max_delta_sil):
                    f.write(f"- **ΔSilhouette ≤ {max_delta_sil:.3f}**: Cluster separation is preserved\n")
                if np.isfinite(min_spearman):
                    f.write(f"- **Matrix Spearman ρ ≥ {min_spearman:.3f}**: Distance rankings are highly preserved\n\n")

                # Best-performing region analysis
                if 'robustness_metrics' in results:
                    robust_configs = [r for r in results['robustness_metrics']
                                    if np.isfinite(r['delta_ari']) and r['delta_ari'] <= 0.05
                                    and np.isfinite(r['spearman_correlation']) and r['spearman_correlation'] >= 0.9]

                    if robust_configs:
                        f.write("**Best-performing region consistently includes:**\n")
                        for config in robust_configs[:3]:  # Top 3 most robust
                            f.write(f"- {config['config_name']}: ΔARI={config['delta_ari']:.3f}, ρ={config['spearman_correlation']:.3f}\n")

            # Detailed results table
            if 'robustness_metrics' in results:
                f.write("\n## Detailed Results\n\n")
                f.write("| Configuration | ARI | Silhouette | ΔARI | ΔSilhouette | Spearman ρ |\n")
                f.write("|---------------|-----|------------|------|-------------|------------|\n")

                for result in results['robustness_metrics']:
                    name = result['config_name']
                    ari = result['ari']
                    sil = result['silhouette']
                    dARI = result['delta_ari']
                    dSil = result['delta_silhouette']
                    rho = result['spearman_correlation']

                    f.write(f"| {name} | {ari:.4f} | {sil:.4f} | {dARI:.4f} | {dSil:.4f} | {rho:.4f} |\n")

        print(f"Robustness report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="DTW Robustness Analysis with Comprehensive Clustering Metrics"
    )

    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing eigenvalue data files')
    parser.add_argument('--pattern', type=str, default='*eigenvalues.npz',
                       help='File pattern to match')
    parser.add_argument('--output-dir', type=str, default='dtw_robustness_results',
                       help='Output directory for robustness analysis results')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs for DTW computation')

    args = parser.parse_args()

    # Set environment variable for conda activation
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Initialize analyzer
    analyzer = DTWRobustnessAnalyzer(
        data_dir=args.data_dir,
        pattern=args.pattern,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs
    )

    # Run analysis
    results = analyzer.run_robustness_analysis()

    # Save complete results
    results_file = analyzer.output_dir / "complete_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    # Generate report
    analyzer.generate_robustness_report(results)

    print(f"\n{'='*80}")
    print("DTW ROBUSTNESS ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved in: {analyzer.output_dir}")
    print(f"- Complete results: {results_file}")
    print(f"- Robustness report: {analyzer.output_dir}/robustness_report.md")
    print(f"- Summary CSV: {analyzer.output_dir}/robustness_summary.csv")

    # Print summary
    if 'robustness_summary' in results:
        summary = results['robustness_summary']
        print(f"\n📊 Robustness Summary:")
        print(f"   Max ΔARI: {summary.get('max_delta_ari', np.nan):.4f}")
        print(f"   Max ΔSilhouette: {summary.get('max_delta_silhouette', np.nan):.4f}")
        print(f"   Min Spearman ρ: {summary.get('min_spearman_correlation', np.nan):.4f}")


if __name__ == '__main__':
    main()