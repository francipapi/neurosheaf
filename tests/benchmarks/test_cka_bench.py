"""Integration tests for CKA benchmark suite.

This module provides pytest integration for the CKA benchmark suite,
allowing automated testing of benchmark configurations and validation
of benchmark pipeline functionality.
"""

import pytest
import yaml
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[2] / "benchmarks"))

from cka_bench.scripts.benchmark_runner import BenchmarkRunner


class TestCKABenchmarkSuite:
    """Test suite for CKA benchmark functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_paths(self):
        """Setup paths for benchmark tests."""
        self.repo_root = Path(__file__).parents[2]
        self.benchmark_dir = self.repo_root / "benchmarks" / "cka_bench"
        self.configs_dir = self.benchmark_dir / "configs"
        self.scripts_dir = self.benchmark_dir / "scripts"
        
        # Ensure directories exist
        assert self.benchmark_dir.exists(), f"Benchmark directory not found: {self.benchmark_dir}"
        assert self.configs_dir.exists(), f"Configs directory not found: {self.configs_dir}"
        assert self.scripts_dir.exists(), f"Scripts directory not found: {self.scripts_dir}"
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for test results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def benchmark_runner(self):
        """Create benchmark runner instance."""
        return BenchmarkRunner(verbose=False)
    
    def test_config_files_exist(self):
        """Test that all required configuration files exist."""
        required_configs = [
            'rotation_sweep.yaml',
            'mix_curve.yaml',
            'resnet_pair.yaml',
            'resnet_ft.yaml',
            'bert_layers.yaml',
            'bootstrap_stability.yaml'
        ]
        
        for config_name in required_configs:
            config_path = self.configs_dir / config_name
            assert config_path.exists(), f"Configuration file missing: {config_path}"
    
    def test_script_files_exist(self):
        """Test that all required script files exist."""
        required_scripts = [
            'gen_synthetic.py',
            'extract_activations.py',
            'run_cka.py',
            'analyse_results.py',
            'benchmark_runner.py'
        ]
        
        for script_name in required_scripts:
            script_path = self.scripts_dir / script_name
            assert script_path.exists(), f"Script file missing: {script_path}"
            assert script_path.is_file(), f"Script is not a file: {script_path}"
    
    def test_config_validation(self):
        """Test that all configuration files are valid YAML."""
        for config_file in self.configs_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                    assert isinstance(config, dict), f"Config is not a dictionary: {config_file}"
                    assert 'benchmark' in config, f"Missing 'benchmark' section: {config_file}"
                    assert 'name' in config['benchmark'], f"Missing benchmark name: {config_file}"
                    assert 'type' in config['benchmark'], f"Missing benchmark type: {config_file}"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_file}: {e}")
    
    @pytest.mark.parametrize("config_name", [
        'rotation_sweep.yaml',
        'mix_curve.yaml'
    ])
    def test_synthetic_benchmark_configs(self, config_name):
        """Test synthetic benchmark configurations."""
        config_path = self.configs_dir / config_name
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check synthetic benchmark requirements
        assert config['benchmark']['type'] == 'synthetic'
        assert 'data' in config
        assert 'cka_methods' in config
        assert 'validation' in config
        assert 'reproducibility' in config
        
        # Check CKA methods
        for method in config['cka_methods']:
            assert 'name' in method
            assert 'type' in method
            assert method['type'] in ['exact', 'nystrom']
    
    @pytest.mark.parametrize("config_name", [
        'resnet_pair.yaml',
        'resnet_ft.yaml',
        'bert_layers.yaml'
    ])
    def test_real_model_benchmark_configs(self, config_name):
        """Test real model benchmark configurations."""
        config_path = self.configs_dir / config_name
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check real model benchmark requirements
        assert config['benchmark']['type'] == 'real_model'
        assert 'model' in config
        assert 'dataset' in config
        assert 'layers' in config
        assert 'cka_methods' in config
        assert 'validation' in config
        
        # Check model configuration
        assert 'name' in config['model']
        
        # Check dataset configuration
        assert 'name' in config['dataset']
        assert 'batch_size' in config['dataset']
    
    def test_benchmark_runner_initialization(self, benchmark_runner):
        """Test that benchmark runner initializes correctly."""
        assert benchmark_runner is not None
        assert benchmark_runner.synthetic_benchmarks
        assert benchmark_runner.real_model_benchmarks
        assert benchmark_runner.all_benchmarks
        
        # Check that benchmark lists are correct
        assert 'rotation_sweep' in benchmark_runner.synthetic_benchmarks
        assert 'mix_curve' in benchmark_runner.synthetic_benchmarks
        assert 'resnet_pair' in benchmark_runner.real_model_benchmarks
        assert 'bert_layers' in benchmark_runner.real_model_benchmarks
    
    def test_script_execution_help(self):
        """Test that scripts can be executed and show help."""
        scripts_to_test = [
            'gen_synthetic.py',
            'extract_activations.py',
            'run_cka.py',
            'analyse_results.py',
            'benchmark_runner.py'
        ]
        
        for script_name in scripts_to_test:
            script_path = self.scripts_dir / script_name
            
            # Run script with --help flag
            result = subprocess.run(
                [sys.executable, str(script_path), '--help'],
                capture_output=True,
                text=True,
                cwd=self.benchmark_dir
            )
            
            assert result.returncode == 0, f"Script {script_name} failed with --help"
            assert 'usage:' in result.stdout.lower(), f"No usage info in {script_name} help"
    
    @pytest.mark.slow
    def test_rotation_sweep_minimal(self, temp_results_dir):
        """Test rotation sweep benchmark with minimal configuration."""
        # Create minimal test configuration
        test_config = {
            'benchmark': {
                'name': 'rotation_sweep_test',
                'description': 'Minimal rotation sweep test',
                'type': 'synthetic'
            },
            'data': {
                'seed': 0,
                'n_samples': 100,  # Reduced for testing
                'dim': 32,         # Reduced for testing
                'dtype': 'float32'
            },
            'rotation': {
                'angles_deg': [0, 45, 90]  # Reduced for testing
            },
            'cka_methods': [
                {
                    'name': 'exact',
                    'type': 'exact',
                    'use_unbiased': True
                }
            ],
            'validation': {
                'theoretical_curve': 'cos_squared_half_theta',
                'min_r_squared': 0.80,  # Relaxed for testing
                'max_approx_error': 0.10,
                'monotonicity_check': True,
                'angle_range': [0, 90]
            },
            'output': {
                'save_raw_data': True,
                'save_plots': True,
                'plot_format': 'png',
                'results_dir': 'results/rotation_sweep_test'
            },
            'resources': {
                'max_memory_gb': 2.0,
                'timeout_minutes': 5
            },
            'reproducibility': {
                'torch_deterministic': True,
                'numpy_seed': 0,
                'torch_seed': 0
            }
        }
        
        # Save test config
        test_config_path = temp_results_dir / 'rotation_sweep_test.yaml'
        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Run synthetic data generation
        gen_script = self.scripts_dir / 'gen_synthetic.py'
        result = subprocess.run(
            [sys.executable, str(gen_script), str(test_config_path)],
            capture_output=True,
            text=True,
            cwd=self.benchmark_dir
        )
        
        assert result.returncode == 0, f"Data generation failed: {result.stderr}"
        
        # Check that data was generated
        data_dir = self.benchmark_dir / 'data' / 'synthetic' / 'rotation_sweep_test'
        assert data_dir.exists(), "Data directory not created"
        
        data_files = list(data_dir.glob('*.pkl'))
        assert len(data_files) > 0, "No data files generated"
    
    def test_baseline_files_exist(self):
        """Test that baseline reference files exist."""
        baselines_dir = self.benchmark_dir / 'baselines'
        assert baselines_dir.exists(), "Baselines directory not found"
        
        required_baselines = [
            'resnet_pair_baseline.csv',
            'resnet_ft_baseline.csv',
            'bert_layers_baseline.csv'
        ]
        
        for baseline_name in required_baselines:
            baseline_path = baselines_dir / baseline_name
            assert baseline_path.exists(), f"Baseline file missing: {baseline_path}"
    
    def test_baseline_files_format(self):
        """Test that baseline files have correct format."""
        baselines_dir = self.benchmark_dir / 'baselines'
        
        for baseline_file in baselines_dir.glob('*.csv'):
            with open(baseline_file, 'r') as f:
                content = f.read()
                assert content.strip(), f"Baseline file is empty: {baseline_file}"
                
                # Check for header
                lines = content.strip().split('\n')
                assert len(lines) > 1, f"Baseline file has no data: {baseline_file}"
                
                # Check that it's valid CSV format
                header = lines[0]
                assert ',' in header, f"Baseline file not CSV format: {baseline_file}"
    
    @pytest.mark.parametrize("suite_name", ['synthetic', 'real_model'])
    def test_suite_configuration(self, benchmark_runner, suite_name):
        """Test that benchmark suites are properly configured."""
        if suite_name == 'synthetic':
            benchmarks = benchmark_runner.synthetic_benchmarks
        else:
            benchmarks = benchmark_runner.real_model_benchmarks
        
        assert len(benchmarks) > 0, f"No benchmarks in {suite_name} suite"
        
        # Check that all benchmark configs exist
        for benchmark_name in benchmarks:
            config_path = self.configs_dir / f"{benchmark_name}.yaml"
            assert config_path.exists(), f"Config missing for {benchmark_name}"
    
    def test_directory_structure(self):
        """Test that the benchmark directory structure is correct."""
        expected_dirs = [
            'configs',
            'data',
            'scripts',
            'results',
            'baselines'
        ]
        
        for dir_name in expected_dirs:
            dir_path = self.benchmark_dir / dir_name
            assert dir_path.exists(), f"Directory missing: {dir_path}"
            assert dir_path.is_dir(), f"Not a directory: {dir_path}"
    
    def test_data_directories_created(self):
        """Test that data directories are created correctly."""
        data_dir = self.benchmark_dir / 'data'
        
        # Check for subdirectories
        synthetic_dir = data_dir / 'synthetic'
        activations_dir = data_dir / 'activations'
        cache_dir = data_dir / 'cache'
        
        # These should be created by the benchmark scripts
        assert synthetic_dir.exists() or not list(data_dir.iterdir()), "Data directory structure incorrect"
    
    def test_results_directories_created(self):
        """Test that results directories are created correctly."""
        results_dir = self.benchmark_dir / 'results'
        
        # Directory should exist (might be empty)
        assert results_dir.exists(), "Results directory missing"
        assert results_dir.is_dir(), "Results path is not a directory"
    
    @pytest.mark.integration
    def test_benchmark_runner_suite_dry_run(self, benchmark_runner):
        """Test benchmark runner suite functionality without actually running."""
        # This tests the suite configuration logic without running the benchmarks
        
        # Test suite name validation
        with pytest.raises(Exception):  # Should raise ValidationError
            benchmark_runner.run_benchmark_suite('invalid_suite')
        
        # Test that suite configurations are valid
        for suite_name in ['synthetic', 'real_model', 'all']:
            if suite_name == 'synthetic':
                benchmarks = benchmark_runner.synthetic_benchmarks
            elif suite_name == 'real_model':
                benchmarks = benchmark_runner.real_model_benchmarks
            else:
                benchmarks = benchmark_runner.all_benchmarks
            
            assert isinstance(benchmarks, list), f"Suite {suite_name} is not a list"
            assert len(benchmarks) > 0, f"Suite {suite_name} is empty"
    
    def test_memory_and_resource_configs(self):
        """Test that resource configurations are reasonable."""
        for config_file in self.configs_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'resources' in config:
                resources = config['resources']
                
                # Check memory limits are reasonable
                if 'max_memory_gb' in resources:
                    max_memory = resources['max_memory_gb']
                    assert 0.5 <= max_memory <= 32, f"Unreasonable memory limit in {config_file}: {max_memory}GB"
                
                # Check timeout limits are reasonable
                if 'timeout_minutes' in resources:
                    timeout = resources['timeout_minutes']
                    assert 1 <= timeout <= 120, f"Unreasonable timeout in {config_file}: {timeout}min"


@pytest.mark.benchmark
class TestBenchmarkPerformance:
    """Performance-related tests for benchmark suite."""
    
    def test_config_loading_performance(self):
        """Test that configuration loading is fast."""
        import time
        
        configs_dir = Path(__file__).parents[2] / "benchmarks" / "cka_bench" / "configs"
        
        start_time = time.time()
        
        for config_file in configs_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
        
        end_time = time.time()
        
        # Should load all configs in less than 1 second
        assert end_time - start_time < 1.0, "Config loading too slow"
    
    def test_script_import_performance(self):
        """Test that script imports are fast."""
        import time
        
        scripts_dir = Path(__file__).parents[2] / "benchmarks" / "cka_bench" / "scripts"
        
        start_time = time.time()
        
        # Test that we can import the runner quickly
        from cka_bench.scripts.benchmark_runner import BenchmarkRunner
        
        end_time = time.time()
        
        # Should import in less than 2 seconds
        assert end_time - start_time < 2.0, "Script import too slow"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])