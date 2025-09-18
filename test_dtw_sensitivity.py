#!/usr/bin/env python3
"""
Quick test script for DTW sensitivity analysis.
Tests with a small subset of configurations to verify functionality.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add current directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from dtw_sensitivity_analysis import DTWSensitivityAnalyzer, SensitivityConfig

def test_dtw_sensitivity():
    """Run a quick test of the DTW sensitivity analysis."""

    # Check if data directory exists
    data_dir = Path("eigenvalueData")
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found. Please run from repository root.")
        return False

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_results"

        print("Testing DTW sensitivity analysis...")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")

        # Initialize analyzer
        analyzer = DTWSensitivityAnalyzer(
            data_dir=str(data_dir),
            pattern="*eigenvalues.npz",
            output_dir=str(output_dir),
            n_jobs=2  # Use limited parallelism for testing
        )

        # Create a minimal baseline configuration
        baseline_config = SensitivityConfig(
            dtw_window=0.1,
            dtw_cost='l1',
            dtw_normalize=False,
            resample=50,  # Reduced for faster testing
            smooth_win=10,
            amp_norm='zscore'
        )

        print(f"Baseline config: {baseline_config.get_identifier()}")

        # Override parameter grid generation for testing
        def test_parameter_grid(self, baseline_config):
            """Generate minimal parameter grid for testing."""
            configs = [
                baseline_config,  # Baseline
                # Test DTW window sensitivity
                SensitivityConfig(**{**baseline_config.to_dict(), 'dtw_window': 0.2}),
                # Test cost function sensitivity
                SensitivityConfig(**{**baseline_config.to_dict(), 'dtw_cost': 'l2'}),
                # Test normalization sensitivity
                SensitivityConfig(**{**baseline_config.to_dict(), 'dtw_normalize': True}),
            ]
            print(f"Generated {len(configs)} test configurations")
            return configs

        # Monkey patch for testing
        analyzer.generate_parameter_grid = test_parameter_grid.__get__(analyzer, DTWSensitivityAnalyzer)

        try:
            # Run sensitivity analysis
            results = analyzer.run_sensitivity_analysis(baseline_config)

            # Verify results
            if len(results) > 0:
                print(f"✓ Successfully processed {len(results)} configurations")

                # Check output files exist
                stats_file = output_dir / "statistics" / "sensitivity_statistics.csv"
                report_file = output_dir / "sensitivity_report.md"

                if stats_file.exists():
                    print(f"✓ Statistics file created: {stats_file.stat().st_size} bytes")
                else:
                    print("✗ Statistics file not created")

                if report_file.exists():
                    print(f"✓ Report file created: {report_file.stat().st_size} bytes")
                else:
                    print("✗ Report file not created")

                # Print sample results
                print("\nSample results:")
                for result in results[:3]:
                    print(f"  {result.config.get_identifier()}: mean_dist={result.mean_distance:.6f}, time={result.computation_time:.2f}s")

                return True
            else:
                print("✗ No results generated")
                return False

        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("DTW Sensitivity Analysis Test")
    print("=" * 40)

    success = test_dtw_sensitivity()

    if success:
        print("\n✓ Test completed successfully!")
        print("\nTo run full sensitivity analysis:")
        print("python dtw_sensitivity_analysis.py \\")
        print("  --data-dir eigenvalueData \\")
        print("  --pattern '*eigenvalues.npz' \\")
        print("  --output-dir dtw_sensitivity_results \\")
        print("  --n-jobs 8")
    else:
        print("\n✗ Test failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()