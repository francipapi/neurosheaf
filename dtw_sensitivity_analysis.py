#!/usr/bin/env python3
"""
DTW Sensitivity Analysis for Ablation Study

This script performs comprehensive sensitivity analysis of DTW parameters using the exact
implementation from compute_alternative_distances.py. Tests both DTW-specific parameters
and preprocessing parameters to understand their impact on distance matrices and clustering.

Usage:
    python dtw_sensitivity_analysis.py \
      --data-dir eigenvalueData \
      --pattern "*eigenvalues.npz" \
      --baseline-config baseline_config.json \
      --output-dir dtw_sensitivity_results \
      --n-jobs 8

Baseline parameters (matching your provided command):
    --data-dir eigenvalueData --pattern "*eigenvalues.npz" --resample 200 --topk 0
    --amp-norm zscore --smooth moving --smooth-win 15 --no-time-scaling --pad-with-last
    --out-prefix comparison --n-jobs 8
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict

# Import exact implementations from compute_alternative_distances.py
from compute_alternative_distances import (
    dtw_distance, l2_distance, compute_distance_matrix,
    _find_files, load_run, remove_tinycnn_outliers,
    mean_curve, find_maximum_range, resample_to_grid,
    smooth_series, normalize_amplitude
)

@dataclass
class SensitivityConfig:
    """Configuration for a single sensitivity test run."""
    # DTW-specific parameters
    dtw_window: float
    dtw_cost: str
    # dtw_normalize: removed - data already normalized

    # Preprocessing parameters that affect DTW
    resample: int
    smooth_win: int
    amp_norm: str
    smooth: str = 'moving'
    topk: int = 0
    no_time_scaling: bool = True
    pad_with_last: bool = True
    outlier_method: str = 'none'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_identifier(self) -> str:
        """Get unique identifier string for this configuration."""
        return f"w{self.dtw_window:.3f}_{self.dtw_cost}_r{self.resample}_s{self.smooth_win}_{self.amp_norm}"

@dataclass
class SensitivityResult:
    """Results from a single sensitivity test run."""
    config: SensitivityConfig
    distance_matrix: np.ndarray
    computation_time: float

    # Distance matrix statistics
    mean_distance: float
    std_distance: float
    min_distance: float
    max_distance: float
    median_distance: float
    q25_distance: float
    q75_distance: float

    # Matrix properties
    condition_number: float
    matrix_rank: int
    sparsity: float  # fraction of near-zero distances

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving/analysis."""
        result_dict = {
            'config_id': self.config.get_identifier(),
            'computation_time': self.computation_time,
            'mean_distance': self.mean_distance,
            'std_distance': self.std_distance,
            'min_distance': self.min_distance,
            'max_distance': self.max_distance,
            'median_distance': self.median_distance,
            'q25_distance': self.q25_distance,
            'q75_distance': self.q75_distance,
            'condition_number': self.condition_number,
            'matrix_rank': self.matrix_rank,
            'sparsity': self.sparsity,
        }
        result_dict.update(self.config.to_dict())
        return result_dict


class DTWSensitivityAnalyzer:
    """Main class for conducting DTW sensitivity analysis."""

    def __init__(self, data_dir: str, pattern: str, output_dir: str, n_jobs: int = -1):
        self.data_dir = Path(data_dir)
        self.pattern = pattern
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.raw_dir = self.output_dir / "raw_distances"
        self.stats_dir = self.output_dir / "statistics"
        self.plots_dir = self.output_dir / "plots"

        for dir_path in [self.raw_dir, self.stats_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)

        # Store loaded data for reuse
        self.loaded_files: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None

    def generate_parameter_grid(self, baseline_config: SensitivityConfig) -> List[SensitivityConfig]:
        """Generate grid of parameter combinations for sensitivity analysis."""

        # Define parameter variations
        parameter_grids = {
            # DTW-specific parameters (primary focus)
            'dtw_window': [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],  # 1.0 = no constraint
            'dtw_cost': ['l1', 'l2'],
            # dtw_normalize removed - data already normalized

            # Preprocessing parameters (secondary focus)
            'resample': [50, 100, 200, 400],
            'smooth_win': [5, 10, 15, 20, 30],
            'amp_norm': ['zscore', 'unit', 'none'],
        }

        # Generate all combinations
        configs = []

        # Start with baseline
        baseline_dict = baseline_config.to_dict()

        # One-at-a-time sensitivity analysis (more interpretable)
        for param_name, values in parameter_grids.items():
            for value in values:
                config_dict = baseline_dict.copy()
                config_dict[param_name] = value
                configs.append(SensitivityConfig(**config_dict))

        # Also add some key combinations to test interaction effects
        interaction_configs = [
            # High resolution + tight window
            SensitivityConfig(**{**baseline_dict, 'resample': 400, 'dtw_window': 0.05}),
            # Low resolution + loose window
            SensitivityConfig(**{**baseline_dict, 'resample': 50, 'dtw_window': 0.5}),
            # No smoothing + L2 cost
            SensitivityConfig(**{**baseline_dict, 'smooth_win': 1, 'dtw_cost': 'l2'}),
            # Heavy smoothing + tight window
            SensitivityConfig(**{**baseline_dict, 'smooth_win': 30, 'dtw_window': 0.05}),
            # No amplitude normalization + L2 cost
            SensitivityConfig(**{**baseline_dict, 'amp_norm': 'none', 'dtw_cost': 'l2'}),
        ]

        configs.extend(interaction_configs)

        # Remove duplicates while preserving order
        seen_ids = set()
        unique_configs = []
        for config in configs:
            config_id = config.get_identifier()
            if config_id not in seen_ids:
                seen_ids.add(config_id)
                unique_configs.append(config)

        print(f"Generated {len(unique_configs)} unique parameter configurations")
        return unique_configs

    def load_and_preprocess_data(self, config: SensitivityConfig) -> Tuple[List[np.ndarray], List[str]]:
        """Load and preprocess data according to configuration."""

        # Find files if not already done
        if self.loaded_files is None:
            files = _find_files(self.data_dir, self.pattern)
            if not files:
                raise SystemExit(f"No files found in {self.data_dir} matching pattern '{self.pattern}'")

            print(f"Loading {len(files)} files...")
            loaded_data = []
            for p in files:
                try:
                    E, t = load_run(p)
                    E, t = remove_tinycnn_outliers(E, t, p.name, outlier_method=config.outlier_method)
                    loaded_data.append((p.name, E, t))
                except Exception as e:
                    print(f"[WARN] Failed to load {p.name}: {e}")
                    continue

            self.loaded_files = loaded_data
            print(f"Successfully loaded {len(self.loaded_files)} files")

        # Preprocessing pipeline using exact same logic as compute_alternative_distances.py
        curves: List[np.ndarray] = []
        index: List[str] = []
        common_time: Optional[np.ndarray] = None

        # Determine time normalization settings
        normalize_time = not config.no_time_scaling
        use_padding = config.pad_with_last
        common_range = None

        # If not normalizing time, determine the common range first
        if not normalize_time:
            files_data = [(t, E) for _, E, t in self.loaded_files]
            if use_padding:
                common_range = find_maximum_range(files_data)
            else:
                common_range = find_maximum_range(files_data)

        # Process all files with preprocessing
        for filename, E, t in self.loaded_files:
            try:
                # Compute mean curve
                y = mean_curve(E, topk=config.topk)

                # Resample to common grid
                t_new, y_new = resample_to_grid(
                    t, y, N=config.resample,
                    normalize_time=normalize_time,
                    common_range=common_range,
                    use_padding=use_padding
                )

                # Smooth
                y_smooth = smooth_series(y_new, method=config.smooth, win=config.smooth_win)

                # Normalize amplitude
                y_final = normalize_amplitude(y_smooth, mode=config.amp_norm)

                curves.append(y_final)
                index.append(filename)

                if common_time is None:
                    common_time = t_new

            except Exception as e:
                print(f"[WARN] Failed to process {filename}: {e}")
                continue

        if len(curves) == 0:
            raise RuntimeError("No curves could be processed successfully")

        return curves, index

    def compute_distance_matrix_with_stats(self, config: SensitivityConfig) -> SensitivityResult:
        """Compute DTW distance matrix and associated statistics."""

        print(f"Processing config: {config.get_identifier()}")

        # Load and preprocess data
        start_time = time.time()
        curves, index = self.load_and_preprocess_data(config)

        # Compute DTW distance matrix
        distance_matrix = compute_distance_matrix(
            curves,
            dtw_distance,
            n_jobs=self.n_jobs,
            window=config.dtw_window,
            cost=config.dtw_cost,
            normalize=False  # Data already normalized, no need for DTW normalization
        )

        computation_time = time.time() - start_time

        # Compute statistics (upper triangular part only)
        upper_tri = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

        # Basic statistics
        mean_dist = np.mean(upper_tri)
        std_dist = np.std(upper_tri)
        min_dist = np.min(upper_tri)
        max_dist = np.max(upper_tri)
        median_dist = np.median(upper_tri)
        q25_dist = np.percentile(upper_tri, 25)
        q75_dist = np.percentile(upper_tri, 75)

        # Matrix properties
        try:
            # Add small regularization for condition number computation
            regularized_matrix = distance_matrix + np.eye(distance_matrix.shape[0]) * 1e-10
            condition_number = np.linalg.cond(regularized_matrix)
        except:
            condition_number = float('inf')

        matrix_rank = np.linalg.matrix_rank(distance_matrix)

        # Sparsity (fraction of very small distances)
        sparsity_threshold = 1e-6 if min_dist > 0 else 1e-10
        sparsity = np.sum(upper_tri < sparsity_threshold) / len(upper_tri)

        result = SensitivityResult(
            config=config,
            distance_matrix=distance_matrix,
            computation_time=computation_time,
            mean_distance=mean_dist,
            std_distance=std_dist,
            min_distance=min_dist,
            max_distance=max_dist,
            median_distance=median_dist,
            q25_distance=q25_dist,
            q75_distance=q75_dist,
            condition_number=condition_number,
            matrix_rank=matrix_rank,
            sparsity=sparsity,
        )

        # Save distance matrix
        matrix_file = self.raw_dir / f"{config.get_identifier()}_distance.npy"
        np.save(matrix_file, distance_matrix)

        # Save index
        index_file = self.raw_dir / f"{config.get_identifier()}_index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

        print(f"  Computation time: {computation_time:.2f}s")
        print(f"  Mean distance: {mean_dist:.6f}")
        print(f"  Distance range: [{min_dist:.6f}, {max_dist:.6f}]")

        return result

    def run_sensitivity_analysis(self, baseline_config: SensitivityConfig) -> List[SensitivityResult]:
        """Run complete sensitivity analysis."""

        print("Starting DTW sensitivity analysis...")

        # Generate parameter grid
        configs = self.generate_parameter_grid(baseline_config)

        # Run analysis for each configuration
        results = []
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Running sensitivity test...")
            try:
                result = self.compute_distance_matrix_with_stats(config)
                results.append(result)
            except Exception as e:
                print(f"ERROR: Failed to process config {config.get_identifier()}: {e}")
                continue

        print(f"\nCompleted sensitivity analysis: {len(results)}/{len(configs)} configurations successful")

        # Save results summary
        self.save_results_summary(results, baseline_config)

        return results

    def save_results_summary(self, results: List[SensitivityResult], baseline_config: SensitivityConfig):
        """Save summary statistics and generate report."""

        # Convert to DataFrame
        df_data = [result.to_dict() for result in results]
        df = pd.DataFrame(df_data)

        # Save raw statistics
        stats_file = self.stats_dir / "sensitivity_statistics.csv"
        df.to_csv(stats_file, index=False)
        print(f"Saved statistics to {stats_file}")

        # Save baseline configuration
        baseline_file = self.stats_dir / "baseline_config.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_config.to_dict(), f, indent=2)

        # Generate summary report
        self.generate_summary_report(df, baseline_config)

        # Generate visualizations
        self.generate_sensitivity_plots(df, baseline_config)

    def generate_summary_report(self, df: pd.DataFrame, baseline_config: SensitivityConfig):
        """Generate markdown summary report."""

        report_file = self.output_dir / "sensitivity_report.md"

        # Find baseline result
        baseline_id = baseline_config.get_identifier()
        baseline_row = df[df['config_id'] == baseline_id]

        if len(baseline_row) == 0:
            print(f"Warning: Baseline configuration {baseline_id} not found in results")
            baseline_stats = None
        else:
            baseline_stats = baseline_row.iloc[0]

        with open(report_file, 'w') as f:
            f.write("# DTW Sensitivity Analysis Report\n\n")

            f.write("## Baseline Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(baseline_config.to_dict(), indent=2))
            f.write("\n```\n\n")

            if baseline_stats is not None:
                f.write("### Baseline Results\n\n")
                f.write(f"- Mean distance: {baseline_stats['mean_distance']:.6f}\n")
                f.write(f"- Distance std: {baseline_stats['std_distance']:.6f}\n")
                f.write(f"- Distance range: [{baseline_stats['min_distance']:.6f}, {baseline_stats['max_distance']:.6f}]\n")
                f.write(f"- Computation time: {baseline_stats['computation_time']:.2f}s\n")
                f.write(f"- Matrix rank: {baseline_stats['matrix_rank']}\n")
                f.write(f"- Condition number: {baseline_stats['condition_number']:.2e}\n\n")

            f.write("## Parameter Sensitivity Summary\n\n")

            # Analyze each parameter
            parameters = ['dtw_window', 'dtw_cost', 'resample', 'smooth_win', 'amp_norm']

            for param in parameters:
                if param in df.columns:
                    f.write(f"### {param}\n\n")

                    # Group by parameter value
                    param_analysis = df.groupby(param)['mean_distance'].agg(['mean', 'std', 'min', 'max']).round(6)

                    f.write("| Value | Mean Distance | Std | Min | Max |\n")
                    f.write("|-------|---------------|-----|-----|-----|\n")

                    for value, stats in param_analysis.iterrows():
                        f.write(f"| {value} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |\n")

                    f.write("\n")

            f.write("## Key Findings\n\n")

            # Identify most sensitive parameters
            sensitivity_analysis = {}
            for param in parameters:
                if param in df.columns:
                    param_groups = df.groupby(param)['mean_distance'].agg(['mean', 'std'])
                    # Coefficient of variation across parameter values
                    cv = param_groups['mean'].std() / param_groups['mean'].mean()
                    sensitivity_analysis[param] = cv

            # Sort by sensitivity
            sorted_sensitivity = sorted(sensitivity_analysis.items(), key=lambda x: x[1], reverse=True)

            f.write("### Most Sensitive Parameters (by coefficient of variation)\n\n")
            for param, cv in sorted_sensitivity:
                f.write(f"1. **{param}**: CV = {cv:.4f}\n")

            f.write("\n### Computational Performance\n\n")
            f.write(f"- Fastest configuration: {df.loc[df['computation_time'].idxmin(), 'config_id']} ({df['computation_time'].min():.2f}s)\n")
            f.write(f"- Slowest configuration: {df.loc[df['computation_time'].idxmax(), 'config_id']} ({df['computation_time'].max():.2f}s)\n")
            f.write(f"- Average computation time: {df['computation_time'].mean():.2f}s\n")

            f.write("\n### Distance Matrix Properties\n\n")
            f.write(f"- Best conditioned matrix: {df.loc[df['condition_number'].idxmin(), 'config_id']} (cond = {df['condition_number'].min():.2e})\n")
            f.write(f"- Highest rank matrices: {df['matrix_rank'].max()}/{df.shape[0]} configurations\n")
            f.write(f"- Average sparsity: {df['sparsity'].mean():.4f}\n")

        print(f"Generated report: {report_file}")

    def generate_sensitivity_plots(self, df: pd.DataFrame, baseline_config: SensitivityConfig):
        """Generate sensitivity visualization plots."""

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Parameters to analyze
        parameters = {
            'dtw_window': 'DTW Window Constraint',
            'dtw_cost': 'DTW Cost Function',
            'resample': 'Resampling Resolution',
            'smooth_win': 'Smoothing Window',
            'amp_norm': 'Amplitude Normalization'
        }

        # Create sensitivity plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (param, title) in enumerate(parameters.items()):
            ax = axes[i]

            if param in df.columns:
                # Box plot for categorical, scatter for numerical
                if df[param].dtype == 'object' or df[param].nunique() <= 10:
                    sns.boxplot(data=df, x=param, y='mean_distance', ax=ax)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.scatter(df[param], df['mean_distance'], alpha=0.6)
                    ax.set_xlabel(param)

                ax.set_ylabel('Mean Distance')
                ax.set_title(f'Sensitivity to {title}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Computation time analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Computation time vs mean distance
        ax1.scatter(df['mean_distance'], df['computation_time'], alpha=0.6)
        ax1.set_xlabel('Mean Distance')
        ax1.set_ylabel('Computation Time (s)')
        ax1.set_title('Accuracy vs Speed Trade-off')
        ax1.grid(True, alpha=0.3)

        # Computation time by resample parameter
        if 'resample' in df.columns:
            sns.boxplot(data=df, x='resample', y='computation_time', ax=ax2)
            ax2.set_title('Computation Time vs Resolution')
            ax2.set_ylabel('Computation Time (s)')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_data = df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Generated plots in {self.plots_dir}")


def parse_baseline_config(baseline_args: str) -> SensitivityConfig:
    """Parse baseline configuration from command line string."""

    # Default configuration matching the user's baseline
    config = SensitivityConfig(
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

    return config


def main():
    parser = argparse.ArgumentParser(
        description="DTW Sensitivity Analysis - Test parameter sensitivity using exact compute_alternative_distances.py implementation"
    )

    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing eigenvalue data files')
    parser.add_argument('--pattern', type=str, default='*eigenvalues.npz',
                       help='File pattern to match')
    parser.add_argument('--output-dir', type=str, default='dtw_sensitivity_results',
                       help='Output directory for results')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs')
    parser.add_argument('--baseline-config', type=str,
                       help='Path to JSON file with baseline configuration (optional)')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DTWSensitivityAnalyzer(
        data_dir=args.data_dir,
        pattern=args.pattern,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs
    )

    # Load or create baseline configuration
    if args.baseline_config and os.path.exists(args.baseline_config):
        with open(args.baseline_config) as f:
            baseline_dict = json.load(f)
            baseline_config = SensitivityConfig(**baseline_dict)
    else:
        # Use default baseline matching user's provided parameters
        baseline_config = parse_baseline_config("")
        print("Using default baseline configuration:")
        print(json.dumps(baseline_config.to_dict(), indent=2))

    # Run sensitivity analysis
    results = analyzer.run_sensitivity_analysis(baseline_config)

    print(f"\n{'='*60}")
    print("DTW SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(results)} parameter configurations")
    print(f"Results saved in: {analyzer.output_dir}")
    print(f"- Distance matrices: {analyzer.raw_dir}")
    print(f"- Statistics: {analyzer.stats_dir}")
    print(f"- Visualizations: {analyzer.plots_dir}")
    print(f"- Report: {analyzer.output_dir}/sensitivity_report.md")


if __name__ == '__main__':
    main()