#!/usr/bin/env python3
"""
Run DTW sensitivity analysis with the exact baseline parameters provided by the user.

This script replicates the user's baseline command:
python compute_alternative_distances.py \
  --data-dir eigenvalueData \
  --pattern "*eigenvalues.npz" \
  --resample 200 \
  --topk 0 \
  --amp-norm zscore \
  --smooth moving \
  --smooth-win 15 \
  --no-time-scaling \
  --pad-with-last \
  --out-prefix comparison \
  --n-jobs 8

And tests sensitivity of DTW parameters around this baseline.
"""

import json
from pathlib import Path
from dtw_sensitivity_analysis import DTWSensitivityAnalyzer, SensitivityConfig

def main():
    """Run DTW sensitivity analysis with user's baseline parameters."""

    print("DTW Sensitivity Analysis")
    print("Using exact baseline from user's command")
    print("=" * 60)

    # Create baseline configuration matching user's exact parameters
    baseline_config = SensitivityConfig(
        # DTW-specific parameters (using defaults from compute_alternative_distances.py)
        dtw_window=0.1,          # Default DTW window constraint
        dtw_cost='l1',           # Default DTW cost function
        # dtw_normalize removed - data already normalized

        # User's exact preprocessing parameters
        resample=200,            # --resample 200
        topk=0,                  # --topk 0
        amp_norm='zscore',       # --amp-norm zscore
        smooth='moving',         # --smooth moving
        smooth_win=15,           # --smooth-win 15
        no_time_scaling=True,    # --no-time-scaling
        pad_with_last=True,      # --pad-with-last
        outlier_method='none'    # Default outlier method
    )

    print("Baseline Configuration:")
    print(json.dumps(baseline_config.to_dict(), indent=2))
    print()

    # Save baseline configuration for reference
    baseline_file = Path("dtw_baseline_config.json")
    with open(baseline_file, 'w') as f:
        json.dump(baseline_config.to_dict(), f, indent=2)
    print(f"Saved baseline config to: {baseline_file}")

    # Initialize analyzer
    analyzer = DTWSensitivityAnalyzer(
        data_dir="eigenvalueData",
        pattern="*eigenvalues.npz",
        output_dir="dtw_sensitivity_results",
        n_jobs=8  # Match user's --n-jobs 8
    )

    print("\nStarting comprehensive DTW sensitivity analysis...")
    print("This will test the following parameter variations:")

    print("\n1. DTW-specific parameters:")
    print("   - window: [0.05, 0.1, 0.2, 0.3, 0.5, 1.0] (Sakoe-Chiba band)")
    print("   - cost: ['l1', 'l2'] (local cost function)")
    print("   - normalize: Fixed at False (data already normalized)")

    print("\n2. Preprocessing parameters affecting DTW:")
    print("   - resample: [50, 100, 200, 400] (time grid resolution)")
    print("   - smooth_win: [5, 10, 15, 20, 30] (smoothing window size)")
    print("   - amp_norm: ['zscore', 'unit', 'none'] (amplitude normalization)")

    print("\n3. Interaction effects:")
    print("   - High resolution + tight window")
    print("   - Low resolution + loose window")
    print("   - No smoothing + L2 cost")
    print("   - Heavy smoothing + tight window")
    print("   - And more...")

    print(f"\nEstimated total configurations: ~30-40")
    print(f"Estimated runtime: 15-30 minutes (depending on data size)")

    # Run the analysis
    results = analyzer.run_sensitivity_analysis(baseline_config)

    print(f"\n{'='*60}")
    print("DTW SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successfully processed {len(results)} parameter configurations")
    print(f"✓ Results saved in: dtw_sensitivity_results/")
    print(f"  - Distance matrices: dtw_sensitivity_results/raw_distances/")
    print(f"  - Statistics CSV: dtw_sensitivity_results/statistics/sensitivity_statistics.csv")
    print(f"  - Visualizations: dtw_sensitivity_results/plots/")
    print(f"  - Full report: dtw_sensitivity_results/sensitivity_report.md")

    print(f"\n📊 Key Results Summary:")
    if len(results) > 0:
        # Find baseline result
        baseline_id = baseline_config.get_identifier()
        baseline_result = next((r for r in results if r.config.get_identifier() == baseline_id), None)

        if baseline_result:
            print(f"   Baseline mean distance: {baseline_result.mean_distance:.6f}")
            print(f"   Baseline computation time: {baseline_result.computation_time:.2f}s")

        # Find min/max distances
        min_result = min(results, key=lambda r: r.mean_distance)
        max_result = max(results, key=lambda r: r.mean_distance)

        print(f"   Minimum mean distance: {min_result.mean_distance:.6f} ({min_result.config.get_identifier()})")
        print(f"   Maximum mean distance: {max_result.mean_distance:.6f} ({max_result.config.get_identifier()})")

        # Find fastest/slowest
        fastest_result = min(results, key=lambda r: r.computation_time)
        slowest_result = max(results, key=lambda r: r.computation_time)

        print(f"   Fastest computation: {fastest_result.computation_time:.2f}s ({fastest_result.config.get_identifier()})")
        print(f"   Slowest computation: {slowest_result.computation_time:.2f}s ({slowest_result.config.get_identifier()})")

    print(f"\n🎯 Next Steps:")
    print(f"1. Review the sensitivity report: dtw_sensitivity_results/sensitivity_report.md")
    print(f"2. Examine parameter sensitivity plots: dtw_sensitivity_results/plots/")
    print(f"3. Choose optimal DTW parameters based on your accuracy/speed requirements")
    print(f"4. Use the optimal parameters in your main analysis pipeline")

    print(f"\n📝 Citation: This sensitivity analysis uses the exact DTW implementation")
    print(f"   from compute_alternative_distances.py with identical preprocessing.")

if __name__ == '__main__':
    main()