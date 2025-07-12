#!/usr/bin/env python3
"""Comprehensive test runner for Phase 4 persistent homology testing strategy.

This script provides different testing configurations for comprehensive validation
of the persistent spectral analysis implementation with predefined success metrics.

Usage:
    python run_comprehensive_tests.py [--mode MODE] [--verbose] [--report]

Modes:
    - quick: Fast tests for development (unit tests only)
    - standard: Standard validation tests (unit + integration)  
    - comprehensive: Full test suite including performance and literature validation
    - stability: Focus on stability and robustness tests
    - performance: Performance benchmarks only
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List


class TestRunner:
    """Comprehensive test runner for Phase 4 validation."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.success_metrics = self._define_success_metrics()
    
    def _define_success_metrics(self) -> Dict:
        """Define fixed success metrics that tests must achieve."""
        return {
            # Mathematical Correctness (must pass 100%)
            'eigenvalue_non_negativity': {'target': 100, 'tolerance': 0},
            'persistence_ordering': {'target': 100, 'tolerance': 0},
            'spectral_gap_consistency': {'target': 95, 'tolerance': 5},
            'birth_death_ordering': {'target': 100, 'tolerance': 0},
            
            # Topological Validation (must pass 95%+)
            'connected_components_detection': {'target': 100, 'tolerance': 0},
            'barcode_validity': {'target': 95, 'tolerance': 5},
            'infinite_bar_detection': {'target': 90, 'tolerance': 10},
            'crossing_detection_accuracy': {'target': 95, 'tolerance': 5},
            
            # Performance Benchmarks (from CLAUDE.md)
            'memory_usage_gb': {'target': 3.0, 'tolerance': 0.5},
            'computation_time_minutes': {'target': 5.0, 'tolerance': 1.0},
            'cache_efficiency_percent': {'target': 90, 'tolerance': 10},
            
            # Stability Properties
            'stability_bottleneck_distance': {'target': 0.1, 'tolerance': 0.05},
            'numerical_precision_agreement': {'target': 95, 'tolerance': 5},
            'parameter_robustness': {'target': 90, 'tolerance': 10}
        }
    
    def run_quick_tests(self, verbose: bool = False) -> Dict:
        """Run quick tests for development (unit tests only)."""
        print("🚀 Running Quick Test Suite (Unit Tests Only)")
        print("=" * 60)
        
        test_commands = [
            # Unit tests only
            f"pytest {self.test_dir}/unit/ -v {'--tb=short' if not verbose else ''}",
        ]
        
        results = {}
        for i, cmd in enumerate(test_commands):
            print(f"\n📋 Running test batch {i+1}/{len(test_commands)}")
            result = self._run_test_command(cmd)
            results[f'unit_tests_{i+1}'] = result
        
        return self._generate_report(results, 'quick')
    
    def run_standard_tests(self, verbose: bool = False) -> Dict:
        """Run standard validation tests (unit + integration)."""
        print("🧪 Running Standard Test Suite (Unit + Integration)")
        print("=" * 60)
        
        test_commands = [
            # Unit tests
            f"pytest {self.test_dir}/unit/ -v {'--tb=short' if not verbose else ''}",
            # Integration tests  
            f"pytest {self.test_dir}/integration/ -v {'--tb=short' if not verbose else ''}",
        ]
        
        results = {}
        for i, cmd in enumerate(test_commands):
            print(f"\n📋 Running test batch {i+1}/{len(test_commands)}")
            result = self._run_test_command(cmd)
            results[f'batch_{i+1}'] = result
        
        return self._generate_report(results, 'standard')
    
    def run_comprehensive_tests(self, verbose: bool = False) -> Dict:
        """Run full comprehensive test suite."""
        print("🏆 Running Comprehensive Test Suite (All Tests)")
        print("=" * 60)
        
        test_commands = [
            # Unit tests
            f"pytest {self.test_dir}/unit/ -v {'--tb=short' if not verbose else ''}",
            # Integration tests
            f"pytest {self.test_dir}/integration/ -v {'--tb=short' if not verbose else ''}",  
            # Validation tests (excluding slow performance tests)
            f"pytest {self.test_dir}/validation/ -v -m 'not slow' {'--tb=short' if not verbose else ''}",
            # Performance tests (marked as slow)
            f"pytest {self.test_dir}/performance/ -v -m 'slow' {'--tb=short' if not verbose else ''}",
        ]
        
        results = {}
        for i, cmd in enumerate(test_commands):
            suite_name = ['unit', 'integration', 'validation', 'performance'][i]
            print(f"\n📋 Running {suite_name} tests ({i+1}/{len(test_commands)})")
            result = self._run_test_command(cmd)
            results[suite_name] = result
        
        return self._generate_report(results, 'comprehensive')
    
    def run_stability_tests(self, verbose: bool = False) -> Dict:
        """Run stability and robustness focused tests."""
        print("🛡️ Running Stability Test Suite")
        print("=" * 60)
        
        test_commands = [
            # Stability validation tests
            f"pytest {self.test_dir}/validation/test_stability_analysis.py -v {'--tb=short' if not verbose else ''}",
            # Subspace tracking stability
            f"pytest {self.test_dir}/unit/test_subspace_tracking_validation.py::TestSubspaceTracking::test_tracking_consistency -v",
            # Eigenvalue stability
            f"pytest {self.test_dir}/unit/test_eigenvalue_validation.py::TestEigenvalueValidation::test_numerical_stability -v",
        ]
        
        results = {}
        for i, cmd in enumerate(test_commands):
            print(f"\n📋 Running stability test batch {i+1}/{len(test_commands)}")
            result = self._run_test_command(cmd)
            results[f'stability_{i+1}'] = result
        
        return self._generate_report(results, 'stability')
    
    def run_performance_tests(self, verbose: bool = False) -> Dict:
        """Run performance benchmarks only."""
        print("⚡ Running Performance Benchmark Suite")
        print("=" * 60)
        
        test_commands = [
            # Performance benchmarks
            f"pytest {self.test_dir}/performance/ -v -m 'slow' {'--tb=short' if not verbose else ''}",
        ]
        
        results = {}
        for i, cmd in enumerate(test_commands):
            print(f"\n📋 Running performance batch {i+1}/{len(test_commands)}")
            result = self._run_test_command(cmd)
            results[f'performance_{i+1}'] = result
        
        return self._generate_report(results, 'performance')
    
    def _run_test_command(self, cmd: str) -> Dict:
        """Run a single test command and capture results."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test timed out after 600 seconds',
                'execution_time': 600,
                'success': False
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Error running test: {e}',
                'execution_time': time.time() - start_time,
                'success': False
            }
    
    def _generate_report(self, results: Dict, mode: str) -> Dict:
        """Generate comprehensive test report."""
        print(f"\n📊 Test Report - {mode.upper()} Mode")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        total_time = 0
        
        for suite_name, result in results.items():
            success_icon = "✅" if result['success'] else "❌"
            print(f"{success_icon} {suite_name.upper()}: "
                  f"{'PASSED' if result['success'] else 'FAILED'} "
                  f"({result['execution_time']:.1f}s)")
            
            if result['success']:
                passed_tests += 1
            total_tests += 1
            total_time += result['execution_time']
            
            # Show errors if any
            if not result['success'] and result['stderr']:
                print(f"   Error: {result['stderr'][:100]}...")
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 Summary:")
        print(f"   Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Mode: {mode}")
        
        # Validate against success metrics
        metrics_passed = self._validate_success_metrics(results, mode)
        
        if success_rate >= 90 and metrics_passed:
            print(f"🎉 {mode.upper()} TEST SUITE PASSED!")
        else:
            print(f"⚠️  {mode.upper()} TEST SUITE NEEDS ATTENTION")
        
        return {
            'mode': mode,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'total_time': total_time,
            'metrics_passed': metrics_passed,
            'detailed_results': results
        }
    
    def _validate_success_metrics(self, results: Dict, mode: str) -> bool:
        """Validate results against predefined success metrics."""
        print(f"\n🎯 Validating Success Metrics for {mode} mode:")
        
        # This is a simplified validation - in practice would parse test output
        # to extract specific metric values
        
        all_passed = True
        
        # Basic validation based on test success
        for suite_name, result in results.items():
            if not result['success']:
                print(f"   ❌ {suite_name}: Failed basic execution")
                all_passed = False
            else:
                print(f"   ✅ {suite_name}: Passed basic execution")
        
        # Performance validation (for modes that include performance tests)
        if mode in ['comprehensive', 'performance']:
            performance_results = [r for name, r in results.items() if 'performance' in name]
            if performance_results:
                avg_time = sum(r['execution_time'] for r in performance_results) / len(performance_results)
                target_time = self.success_metrics['computation_time_minutes']['target'] * 60
                
                if avg_time <= target_time:
                    print(f"   ✅ Performance: Average time {avg_time:.1f}s ≤ target {target_time}s")
                else:
                    print(f"   ❌ Performance: Average time {avg_time:.1f}s > target {target_time}s")
                    all_passed = False
        
        return all_passed


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description='Comprehensive test runner for Phase 4 persistent homology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_comprehensive_tests.py --mode quick
    python run_comprehensive_tests.py --mode comprehensive --verbose
    python run_comprehensive_tests.py --mode stability --report
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'standard', 'comprehensive', 'stability', 'performance'],
        default='standard',
        help='Test mode to run (default: standard)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output from tests'
    )
    
    parser.add_argument(
        '--report',
        action='store_true', 
        help='Generate detailed report file'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    print(f"🧬 Phase 4 Persistent Homology Testing Strategy")
    print(f"Mode: {args.mode}")
    print(f"Verbose: {args.verbose}")
    print(f"Report: {args.report}")
    print()
    
    # Run tests based on mode
    if args.mode == 'quick':
        report = runner.run_quick_tests(args.verbose)
    elif args.mode == 'standard':
        report = runner.run_standard_tests(args.verbose)
    elif args.mode == 'comprehensive':
        report = runner.run_comprehensive_tests(args.verbose)
    elif args.mode == 'stability':
        report = runner.run_stability_tests(args.verbose)
    elif args.mode == 'performance':
        report = runner.run_performance_tests(args.verbose)
    
    # Generate report file if requested
    if args.report:
        import json
        report_file = f"test_report_{args.mode}_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if report['success_rate'] >= 90 and report['metrics_passed']:
        print(f"\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed or metrics not met.")
        sys.exit(1)


if __name__ == '__main__':
    main()