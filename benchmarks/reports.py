#!/usr/bin/env python3
"""Mac-aware performance reporting for Neurosheaf baseline measurements.

This module generates comprehensive performance reports with Mac-specific
insights and optimization recommendations.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import platform

from neurosheaf.utils.logging import setup_logger


class MacPerformanceReporter:
    """Mac-aware performance reporter for Neurosheaf benchmarks."""
    
    def __init__(
        self,
        results_dir: str = "baseline_results",
        output_dir: str = "reports",
        log_level: str = "INFO"
    ):
        """Initialize the Mac performance reporter.
        
        Args:
            results_dir: Directory containing benchmark results
            output_dir: Directory to save reports
            log_level: Logging level
        """
        self.logger = setup_logger("neurosheaf.benchmarks.reports", level=log_level)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Mac-specific initialization
        self.is_mac = platform.system() == "Darwin"
        self.is_apple_silicon = platform.processor() == "arm"
        
        self.logger.info(f"Initialized MacPerformanceReporter")
        self.logger.info(f"Results dir: {self.results_dir}")
        self.logger.info(f"Output dir: {self.output_dir}")
    
    def generate_comprehensive_report(
        self,
        results_files: Optional[List[str]] = None
    ) -> str:
        """Generate a comprehensive performance report.
        
        Args:
            results_files: List of result files to analyze (auto-detect if None)
            
        Returns:
            Path to the generated report
        """
        if results_files is None:
            results_files = self._find_result_files()
        
        self.logger.info(f"Generating comprehensive report from {len(results_files)} files")
        
        # Load all results
        all_results = {}
        for file in results_files:
            file_path = self.results_dir / file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    all_results[file] = json.load(f)
        
        # Generate report sections
        report_sections = [
            self._generate_executive_summary(all_results),
            self._generate_system_info_section(all_results),
            self._generate_baseline_analysis(all_results),
            self._generate_scaling_analysis(all_results),
            self._generate_mac_specific_insights(all_results),
            self._generate_optimization_recommendations(all_results),
            self._generate_next_steps(all_results)
        ]
        
        # Combine report
        report = "\n\n".join(report_sections)
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"neurosheaf_performance_report_{timestamp}.md"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
        return str(report_path)
    
    def _find_result_files(self) -> List[str]:
        """Find all result files in the results directory.
        
        Returns:
            List of result file names
        """
        if not self.results_dir.exists():
            return []
        
        result_files = []
        for file in self.results_dir.glob("*.json"):
            result_files.append(file.name)
        
        return sorted(result_files)
    
    def _generate_executive_summary(self, all_results: Dict[str, Any]) -> str:
        """Generate executive summary section.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            Executive summary section
        """
        summary = [
            "# Neurosheaf Baseline Performance Report",
            "",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Platform**: {platform.system()} {platform.release()}",
            f"**Architecture**: {platform.machine()}",
            f"**Processor**: {platform.processor()}",
            "",
            "## Executive Summary",
            "",
            "This report analyzes the baseline performance of Neurosheaf CKA computation",
            "on Mac hardware, providing insights for Phase 2 optimization.",
            "",
            "### Key Findings",
            ""
        ]
        
        # Extract key metrics
        total_experiments = len(all_results)
        successful_experiments = sum(1 for r in all_results.values() if not r.get('error'))
        
        summary.extend([
            f"- **Total Experiments**: {total_experiments}",
            f"- **Successful Experiments**: {successful_experiments}",
            f"- **Mac System**: {self.is_mac}",
            f"- **Apple Silicon**: {self.is_apple_silicon}",
            ""
        ])
        
        # Find peak memory usage
        peak_memory = 0
        peak_experiment = None
        
        for exp_name, result in all_results.items():
            if isinstance(result, dict) and 'cka_profiling' in result:
                memory = result['cka_profiling'].get('memory_increase_gb', 0)
                if memory > peak_memory:
                    peak_memory = memory
                    peak_experiment = exp_name
        
        if peak_memory > 0:
            summary.extend([
                f"- **Peak Memory Usage**: {peak_memory:.2f} GB",
                f"- **Peak Memory Experiment**: {peak_experiment}",
                f"- **Target Progress**: {peak_memory/1536:.1%} of 1.5TB target",
                ""
            ])
        
        return "\n".join(summary)
    
    def _generate_system_info_section(self, all_results: Dict[str, Any]) -> str:
        """Generate system information section.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            System information section
        """
        section = [
            "## System Information",
            "",
            f"**Operating System**: {platform.system()} {platform.release()}",
            f"**Architecture**: {platform.machine()}",
            f"**Processor**: {platform.processor()}",
            f"**Python Version**: {platform.python_version()}",
            ""
        ]
        
        # Extract device info from results
        device_info = None
        for result in all_results.values():
            if isinstance(result, dict) and 'system_info' in result:
                device_info = result['system_info'].get('device_info')
                break
        
        if device_info:
            section.extend([
                "### Device Information",
                "",
                f"**Device**: {device_info.get('device', 'Unknown')}",
                f"**MPS Available**: {device_info.get('mps_available', 'Unknown')}",
                f"**MPS Built**: {device_info.get('mps_built', 'Unknown')}",
                f"**CUDA Available**: {device_info.get('cuda_available', 'Unknown')}",
                ""
            ])
        
        # Extract memory info
        memory_info = None
        for result in all_results.values():
            if isinstance(result, dict) and 'system_info' in result:
                memory_info = result['system_info'].get('initial_memory')
                break
        
        if memory_info:
            section.extend([
                "### Memory Information",
                "",
                f"**Total System Memory**: {memory_info.get('system_total_gb', 0):.1f} GB",
                f"**Available Memory**: {memory_info.get('system_available_gb', 0):.1f} GB",
                f"**Unified Memory**: {memory_info.get('unified_memory', 'Unknown')}",
                ""
            ])
        
        return "\n".join(section)
    
    def _generate_baseline_analysis(self, all_results: Dict[str, Any]) -> str:
        """Generate baseline analysis section.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            Baseline analysis section
        """
        section = [
            "## Baseline Performance Analysis",
            "",
            "### Memory Usage Baseline",
            ""
        ]
        
        # Find baseline experiments
        baseline_results = {}
        for exp_name, result in all_results.items():
            if 'baseline' in exp_name.lower() or 'resnet50' in exp_name.lower():
                baseline_results[exp_name] = result
        
        if baseline_results:
            section.append("| Experiment | Memory (GB) | Time (s) | Layers | Status |")
            section.append("|------------|-------------|----------|---------|--------|")
            
            for exp_name, result in baseline_results.items():
                if isinstance(result, dict) and 'cka_profiling' in result:
                    profiling = result['cka_profiling']
                    memory = profiling.get('memory_increase_gb', 0)
                    time_s = profiling.get('computation_time_seconds', 0)
                    layers = profiling.get('n_layers', 0)
                    status = "✅ Success" if memory > 0 else "❌ Failed"
                    
                    section.append(f"| {exp_name} | {memory:.2f} | {time_s:.1f} | {layers} | {status} |")
            
            section.append("")
        
        # Memory breakdown analysis
        section.extend([
            "### Memory Breakdown Analysis",
            "",
            "The baseline CKA computation memory usage consists of:",
            "",
            "1. **Activation Storage**: Raw neural network activations",
            "2. **Gram Matrices**: O(n²) memory for similarity computation",
            "3. **Intermediate Matrices**: HSIC computation intermediates",
            "4. **CKA Matrix**: Final similarity matrix",
            ""
        ])
        
        return "\n".join(section)
    
    def _generate_scaling_analysis(self, all_results: Dict[str, Any]) -> str:
        """Generate scaling analysis section.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            Scaling analysis section
        """
        section = [
            "## Scaling Analysis",
            "",
            "### Memory Scaling Behavior",
            ""
        ]
        
        # Find scaling experiments
        scaling_results = {}
        for exp_name, result in all_results.items():
            if 'scaling' in exp_name.lower():
                scaling_results[exp_name] = result
        
        if scaling_results:
            for exp_name, result in scaling_results.items():
                if isinstance(result, dict) and 'analysis' in result:
                    analysis = result['analysis']
                    
                    section.extend([
                        f"#### {exp_name}",
                        "",
                        f"**Scaling Variable**: {analysis.get('scaling_variable', 'Unknown')}",
                        f"**Measurements**: {analysis.get('num_measurements', 0)}",
                        ""
                    ])
                    
                    if 'memory_analysis' in analysis:
                        mem_analysis = analysis['memory_analysis']
                        section.extend([
                            "**Memory Analysis**:",
                            f"- Expected Scaling: {mem_analysis.get('expected_scaling', 'Unknown')}",
                            f"- R² Score: {mem_analysis.get('r_squared', 0):.3f}",
                            f"- Memory Range: {mem_analysis.get('min_memory_gb', 0):.2f} - {mem_analysis.get('max_memory_gb', 0):.2f} GB",
                            ""
                        ])
                    
                    if 'optimization_targets' in analysis:
                        targets = analysis['optimization_targets']
                        if targets:
                            section.extend([
                                "**Optimization Targets**:",
                                *[f"- {target}" for target in targets],
                                ""
                            ])
        
        return "\n".join(section)
    
    def _generate_mac_specific_insights(self, all_results: Dict[str, Any]) -> str:
        """Generate Mac-specific insights section.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            Mac-specific insights section
        """
        section = [
            "## Mac-Specific Performance Insights",
            "",
            f"**System Type**: {'Apple Silicon' if self.is_apple_silicon else 'Intel Mac'}",
            ""
        ]
        
        if self.is_apple_silicon:
            section.extend([
                "### Apple Silicon Optimizations",
                "",
                "Apple Silicon Macs have unique characteristics that affect performance:",
                "",
                "1. **Unified Memory Architecture**: CPU and GPU share the same memory pool",
                "2. **MPS Backend**: Metal Performance Shaders for GPU acceleration",
                "3. **Memory Bandwidth**: High bandwidth unified memory",
                "4. **Thermal Management**: Performance throttling under sustained load",
                ""
            ])
            
            # Check for MPS usage
            mps_used = False
            for result in all_results.values():
                if isinstance(result, dict) and 'system_info' in result:
                    device_info = result['system_info'].get('device_info', {})
                    if device_info.get('device') == 'mps':
                        mps_used = True
                        break
            
            if mps_used:
                section.extend([
                    "**MPS Usage Detected**: ✅",
                    "- GPU acceleration is being utilized",
                    "- Memory usage includes MPS allocations",
                    "- Consider MPS-specific optimizations in Phase 2",
                    ""
                ])
            else:
                section.extend([
                    "**MPS Usage**: ❌ Not detected",
                    "- Running on CPU only",
                    "- Consider enabling MPS for GPU acceleration",
                    "- Potential performance improvement opportunity",
                    ""
                ])
        
        else:
            section.extend([
                "### Intel Mac Considerations",
                "",
                "Intel Macs have different optimization opportunities:",
                "",
                "1. **Discrete GPU**: Separate GPU memory pool",
                "2. **CPU Optimization**: Focus on CPU-based optimizations",
                "3. **Memory Management**: Careful memory allocation patterns",
                ""
            ])
        
        return "\n".join(section)
    
    def _generate_optimization_recommendations(self, all_results: Dict[str, Any]) -> str:
        """Generate optimization recommendations section.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            Optimization recommendations section
        """
        section = [
            "## Optimization Recommendations for Phase 2",
            "",
            "Based on the baseline analysis, here are the key optimization priorities:",
            "",
            "### High Priority Optimizations",
            "",
            "1. **Gram Matrix Optimization**",
            "   - Implement Nyström approximation for memory reduction",
            "   - Use block-wise computation for large matrices",
            "   - Consider sparse representations where applicable",
            "",
            "2. **Memory Management**",
            "   - Implement adaptive batch sizing",
            "   - Use gradient checkpointing techniques",
            "   - Add memory-efficient HSIC computation",
            "",
            "3. **Mac-Specific Optimizations**"
        ]
        
        if self.is_apple_silicon:
            section.extend([
                "   - Optimize for unified memory architecture",
                "   - Implement MPS-accelerated operations",
                "   - Use Metal-optimized kernels where possible",
                ""
            ])
        else:
            section.extend([
                "   - Optimize for Intel CPU architecture",
                "   - Consider AVX/AVX2 optimizations",
                "   - Implement efficient memory prefetching",
                ""
            ])
        
        section.extend([
            "### Medium Priority Optimizations",
            "",
            "1. **Algorithmic Improvements**",
            "   - Implement incremental CKA computation",
            "   - Add early stopping for convergence",
            "   - Use hierarchical similarity computation",
            "",
            "2. **Numerical Stability**",
            "   - Improve numerical precision handling",
            "   - Add robust eigenvalue computation",
            "   - Implement adaptive epsilon selection",
            "",
            "### Low Priority Optimizations",
            "",
            "1. **User Experience**",
            "   - Add progress bars for long computations",
            "   - Implement result caching",
            "   - Add visualization optimizations",
            ""
        ])
        
        return "\n".join(section)
    
    def _generate_next_steps(self, all_results: Dict[str, Any]) -> str:
        """Generate next steps section.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            Next steps section
        """
        section = [
            "## Next Steps",
            "",
            "### Phase 2 Implementation Plan",
            "",
            "1. **Week 1-2**: Implement debiased CKA with Nyström approximation",
            "2. **Week 3**: Add adaptive sampling for memory efficiency",
            "3. **Week 4**: Implement Mac-specific optimizations",
            "",
            "### Performance Targets for Phase 2",
            "",
            "- **Memory Reduction**: Achieve 10x reduction from baseline",
            "- **Speed Improvement**: 5x faster computation",
            "- **Scalability**: Handle 2x larger networks",
            "",
            "### Validation Requirements",
            "",
            "- All optimizations must maintain mathematical correctness",
            "- CKA properties must be preserved (symmetry, boundedness)",
            "- Regression tests must pass on Mac hardware",
            "",
            "### Success Metrics",
            "",
            "- ResNet50 analysis under 3GB memory usage",
            "- Complete analysis in under 1 minute",
            "- Support for 100+ layer networks",
            "",
            "---",
            "",
            f"*Report generated by Neurosheaf MacPerformanceReporter on {time.strftime('%Y-%m-%d %H:%M:%S')}*"
        ]
        
        return "\n".join(section)
    
    def generate_plots(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate performance plots.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            List of generated plot filenames
        """
        plot_files = []
        
        # Memory usage comparison plot
        memory_plot = self._generate_memory_comparison_plot(all_results)
        if memory_plot:
            plot_files.append(memory_plot)
        
        # Scaling analysis plots
        scaling_plots = self._generate_scaling_plots(all_results)
        plot_files.extend(scaling_plots)
        
        return plot_files
    
    def _generate_memory_comparison_plot(self, all_results: Dict[str, Any]) -> Optional[str]:
        """Generate memory usage comparison plot.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            Plot filename if successful, None otherwise
        """
        # Extract memory data
        experiments = []
        memory_values = []
        
        for exp_name, result in all_results.items():
            if isinstance(result, dict) and 'cka_profiling' in result:
                profiling = result['cka_profiling']
                memory = profiling.get('memory_increase_gb', 0)
                if memory > 0:
                    experiments.append(exp_name.replace('_', ' ').title())
                    memory_values.append(memory)
        
        if not experiments:
            return None
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(experiments, memory_values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, memory_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f} GB', ha='center', va='bottom')
        
        plt.xlabel('Experiment')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Neurosheaf Baseline Memory Usage Comparison')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add 20GB target line
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20GB Target')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"memory_comparison_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Memory comparison plot saved to {plot_path}")
        return plot_filename
    
    def _generate_scaling_plots(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate scaling analysis plots.
        
        Args:
            all_results: Dictionary of all loaded results
            
        Returns:
            List of generated plot filenames
        """
        plot_files = []
        
        # Find scaling results
        for exp_name, result in all_results.items():
            if 'scaling' in exp_name.lower() and isinstance(result, dict):
                if 'measurements' in result:
                    plot_file = self._create_scaling_plot(exp_name, result)
                    if plot_file:
                        plot_files.append(plot_file)
        
        return plot_files
    
    def _create_scaling_plot(self, exp_name: str, result: Dict[str, Any]) -> Optional[str]:
        """Create individual scaling plot.
        
        Args:
            exp_name: Experiment name
            result: Result dictionary
            
        Returns:
            Plot filename if successful, None otherwise
        """
        measurements = result.get('measurements', [])
        successful = [m for m in measurements if m.get('success', False)]
        
        if len(successful) < 2:
            return None
        
        # Determine x-axis variable
        if 'batch_size' in successful[0]:
            x_values = [m['batch_size'] for m in successful]
            x_label = 'Batch Size'
        elif 'num_layers' in successful[0]:
            x_values = [m['num_layers'] for m in successful]
            x_label = 'Number of Layers'
        elif 'feature_dim' in successful[0]:
            x_values = [m['feature_dim'] for m in successful]
            x_label = 'Feature Dimension'
        else:
            return None
        
        memory_values = [m['total_memory_increase_gb'] for m in successful]
        time_values = [m['computation_time_seconds'] for m in successful]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory scaling
        ax1.plot(x_values, memory_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title(f'Memory Scaling - {exp_name}')
        ax1.grid(True, alpha=0.3)
        
        # Time scaling
        ax2.plot(x_values, time_values, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title(f'Time Scaling - {exp_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"scaling_{exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Scaling plot saved to {plot_path}")
        return plot_filename


def main():
    """Main entry point for report generation."""
    reporter = MacPerformanceReporter()
    
    print("Generating comprehensive performance report...")
    
    # Generate main report
    report_path = reporter.generate_comprehensive_report()
    print(f"Report generated: {report_path}")
    
    # Generate plots
    results_files = reporter._find_result_files()
    if results_files:
        all_results = {}
        for file in results_files:
            file_path = reporter.results_dir / file
            if file_path.exists():
                with open(file_path, 'r') as f:
                    all_results[file] = json.load(f)
        
        plots = reporter.generate_plots(all_results)
        print(f"Generated {len(plots)} plots")
    
    print("Report generation complete!")


if __name__ == "__main__":
    main()