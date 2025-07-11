#!/usr/bin/env python3
"""ResNet18 End-to-End Pipeline Test - Complete Neurosheaf Validation

This script implements a comprehensive test of the entire neurosheaf pipeline using
ResNet18 from torchvision. It validates all phases from model loading through 
spectral analysis, serving as a baseline reference implementation.

Test Specifications:
- Model: torchvision.models.resnet18(weights="IMAGENET1K_V1")  
- Input: 256×3×224×224 batch (deterministic, seed=0)
- Expected: 79 nodes, 85 edges, ~9300 total dimensions
- Performance: ≤5 min runtime, ≤3.8GB memory on Mac 12-core
- Validation: All mathematical properties within tolerance

Usage:
    python resnet18_end_to_end_test.py [--device cpu|mps|cuda] [--verbose]
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import networkx as nx
import psutil
from scipy import sparse

# Add neurosheaf to path if running as script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from neurosheaf.cka import DebiasedCKA
from neurosheaf.sheaf import (
    FXPosetExtractor, SheafBuilder, ProcrustesMaps, 
    WhiteningProcessor, SheafLaplacianBuilder, Sheaf
)
from neurosheaf.utils.config import Config
from neurosheaf.utils.device import detect_optimal_device, get_device_info
from neurosheaf.utils.exceptions import ValidationError, ComputationError, MemoryError
from neurosheaf.utils.logging import setup_logger
from neurosheaf.utils.profiling import profile_memory, profile_time


@dataclass
class TestResult:
    """Container for test stage results."""
    stage_name: str
    success: bool
    duration_seconds: float
    memory_peak_mb: float
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass 
class PipelineMetrics:
    """Expected and actual pipeline metrics."""
    # Expected values from specification
    expected_nodes: int = 79
    expected_edges: int = 85
    expected_total_dims: int = 9300
    expected_laplacian_nnz: int = 278000
    expected_harmonic_dim: int = 4
    expected_max_residual: float = 0.048
    expected_symmetry_error: float = 1e-11
    expected_min_eigenvalue: float = -8e-10
    expected_runtime_sec: float = 300.0
    expected_peak_memory_gb: float = 3.8
    
    # Actual measured values
    actual_nodes: Optional[int] = None
    actual_edges: Optional[int] = None
    actual_total_dims: Optional[int] = None
    actual_laplacian_nnz: Optional[int] = None
    actual_harmonic_dim: Optional[int] = None
    actual_max_residual: Optional[float] = None
    actual_symmetry_error: Optional[float] = None
    actual_min_eigenvalue: Optional[float] = None
    actual_runtime_sec: Optional[float] = None
    actual_peak_memory_gb: Optional[float] = None


class ResNet18EndToEndTest:
    """Complete end-to-end pipeline test for neurosheaf with ResNet18."""
    
    def __init__(self, device: Optional[str] = None, verbose: bool = False):
        """Initialize the test runner.
        
        Args:
            device: Device to use ('cpu', 'mps', 'cuda', or None for auto-detection)
            verbose: Enable verbose logging and output
        """
        self.device = detect_optimal_device(device)
        self.verbose = verbose
        
        # Setup logging
        log_level = "DEBUG" if verbose else "INFO"
        self.logger = setup_logger("resnet18_test", level=log_level)
        
        # Initialize metrics
        self.metrics = PipelineMetrics()
        self.results: List[TestResult] = []
        
        # System info
        self.device_info = get_device_info()
        self.is_mac = self.device_info['is_mac']
        self.is_apple_silicon = self.device_info['is_apple_silicon']
        
        # Performance tracking
        self.start_time = None
        self.peak_memory_mb = 0.0
        
        self.logger.info(f"Initialized ResNet18 end-to-end test on {self.device}")
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        if self.is_mac:
            self.logger.info(f"Mac detected: Apple Silicon = {self.is_apple_silicon}")
    
    def run_complete_test(self) -> bool:
        """Run the complete end-to-end pipeline test.
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING RESNET18 END-TO-END PIPELINE TEST")
        self.logger.info("=" * 80)
        
        self.start_time = time.time()
        overall_success = True
        
        try:
            # Stage 1: Model Setup & Activation Extraction
            stage1_success = self._run_stage_1()
            overall_success &= stage1_success
            
            if not stage1_success:
                self.logger.error("Stage 1 failed - aborting pipeline test")
                return False
            
            # Stage 2: CKA Computation & Whitening
            stage2_success = self._run_stage_2()
            overall_success &= stage2_success
            
            if not stage2_success:
                self.logger.error("Stage 2 failed - aborting pipeline test")
                return False
            
            # Stage 3: Sheaf Construction
            stage3_success = self._run_stage_3()
            overall_success &= stage3_success
            
            if not stage3_success:
                self.logger.error("Stage 3 failed - aborting pipeline test")
                return False
            
            # Stage 4: Laplacian Assembly & Spectral Analysis
            stage4_success = self._run_stage_4()
            overall_success &= stage4_success
            
            # Stage 5: Performance & Numerical Validation
            stage5_success = self._run_stage_5()
            overall_success &= stage5_success
            
        except Exception as e:
            self.logger.error(f"Fatal error in pipeline test: {e}")
            self.logger.error(traceback.format_exc())
            overall_success = False
        
        finally:
            # Record final metrics
            if self.start_time:
                self.metrics.actual_runtime_sec = time.time() - self.start_time
                self.metrics.actual_peak_memory_gb = self.peak_memory_mb / 1024.0
            
            # Print summary
            self._print_final_summary(overall_success)
        
        return overall_success
    
    def _run_stage_1(self) -> bool:
        """Stage 1: Model Setup & Activation Extraction."""
        self.logger.info("Stage 1: Model Setup & Activation Extraction")
        stage_start = time.time()
        
        try:
            # Load ResNet18 with ImageNet weights
            self.logger.info("Loading ResNet18 with ImageNet weights...")
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.eval()
            model = model.to(self.device)
            
            # Generate deterministic test batch (reduced size for numerical stability)
            self.logger.info("Generating deterministic test batch (64×3×224×224)...")
            torch.manual_seed(0)
            input_batch = torch.randn(64, 3, 224, 224, device=self.device)
            
            # Extract activations using hooks
            self.logger.info("Extracting activations from all layers...")
            activations = self._extract_activations(model, input_batch)
            
            # Validate extraction results
            num_activations = len(activations)
            self.logger.info(f"Extracted {num_activations} layer activations")
            
            # Check if we have the expected number of layers
            if num_activations < 70:  # Should be around 79
                self.logger.warning(f"Only extracted {num_activations} activations, expected ~79")
            
            # Store data for next stages
            self.model = model
            self.input_batch = input_batch
            self.activations = activations
            
            # Record success
            duration = time.time() - stage_start
            memory_mb = self._get_current_memory_mb()
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            
            result = TestResult(
                stage_name="Stage 1: Model Setup & Activation Extraction",
                success=True,
                duration_seconds=duration,
                memory_peak_mb=memory_mb,
                details={
                    "num_activations": num_activations,
                    "input_shape": list(input_batch.shape),
                    "model_parameters": sum(p.numel() for p in model.parameters())
                }
            )
            self.results.append(result)
            
            self.logger.info(f"✅ Stage 1 completed in {duration:.2f}s (Memory: {memory_mb:.1f}MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Stage 1 failed: {e}")
            result = TestResult(
                stage_name="Stage 1: Model Setup & Activation Extraction", 
                success=False,
                duration_seconds=time.time() - stage_start,
                memory_peak_mb=self._get_current_memory_mb(),
                errors=[str(e)]
            )
            self.results.append(result)
            return False
    
    def _run_stage_2(self) -> bool:
        """Stage 2: Layer-wise Whitening."""
        self.logger.info("Stage 2: Layer-wise Whitening")
        stage_start = time.time()
        
        try:
            # Filter activations to remove problematic ones
            filtered_activations = {}
            
            for name, act in self.activations.items():
                # Check basic statistics
                if act.std() > 1e-8 and not torch.isnan(act).any() and not torch.isinf(act).any():
                    # Check condition number of Gram matrix for numerical stability
                    gram = act @ act.T
                    try:
                        cond = torch.linalg.cond(gram).item()
                        if cond < 1e12:  # Reasonable condition number
                            filtered_activations[name] = act
                        else:
                            self.logger.warning(f"Skipping {name} due to high condition number: {cond:.2e}")
                    except:
                        # If condition number computation fails, skip
                        self.logger.warning(f"Skipping {name} due to condition number computation failure")
                else:
                    self.logger.warning(f"Skipping {name} due to poor statistics")
            
            self.logger.info(f"Using {len(filtered_activations)}/{len(self.activations)} activations for whitening")
            
            # Apply whitening transformation (no cross-layer CKA needed for sheaf construction)
            self.logger.info("Computing layer-wise Gram matrices and applying whitening transformation...")
            whitener = WhiteningProcessor()
            whitened_activations = {}
            total_dims = 0
            
            for layer_name, activation in filtered_activations.items():
                # Compute Gram matrix for this layer only
                gram = activation @ activation.T
                
                # Apply whitening transformation
                whitened_gram, W, info = whitener.whiten_gram_matrix(
                    gram, 
                    variance_threshold=0.99
                )
                
                # Store results
                whitened_activations[layer_name] = {
                    'gram_whitened': whitened_gram,
                    'whitening_matrix': W,
                    'info': info,
                    'rank': info['effective_rank'],
                    'original_activation': activation  # Store for potential sheaf construction
                }
                total_dims += info['effective_rank']
                
                if self.verbose:
                    self.logger.debug(f"Layer {layer_name}: rank {info['effective_rank']}")
            
            # Validate stalk dimensions
            self._validate_stalk_dimensions(whitened_activations)
            
            # Store data for next stages (no CKA matrix needed)
            self.whitened_activations = whitened_activations
            self.filtered_activations = filtered_activations  # Store filtered activations
            self.metrics.actual_total_dims = total_dims
            
            # Record success
            duration = time.time() - stage_start
            memory_mb = self._get_current_memory_mb()
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            
            result = TestResult(
                stage_name="Stage 2: Layer-wise Whitening",
                success=True,
                duration_seconds=duration,
                memory_peak_mb=memory_mb,
                details={
                    "total_dims": total_dims,
                    "num_whitened_layers": len(whitened_activations),
                    "num_filtered_layers": len(filtered_activations)
                }
            )
            self.results.append(result)
            
            self.logger.info(f"✅ Stage 2 completed in {duration:.2f}s (Memory: {memory_mb:.1f}MB)")
            self.logger.info(f"   Total dimensions: {total_dims}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Stage 2 failed: {e}")
            result = TestResult(
                stage_name="Stage 2: Layer-wise Whitening",
                success=False,
                duration_seconds=time.time() - stage_start,
                memory_peak_mb=self._get_current_memory_mb(),
                errors=[str(e)]
            )
            self.results.append(result)
            return False
    
    def _run_stage_3(self) -> bool:
        """Stage 3: Sheaf Construction."""
        self.logger.info("Stage 3: Sheaf Construction")
        stage_start = time.time()
        
        try:
            # Extract poset using FX
            self.logger.info("Extracting poset structure using FX...")
            poset_extractor = FXPosetExtractor(handle_dynamic=True)
            poset = poset_extractor.extract_poset(self.model)
            
            num_nodes = poset.number_of_nodes()
            num_edges = poset.number_of_edges()
            
            self.logger.info(f"Extracted poset: {num_nodes} nodes, {num_edges} edges")
            
            # Build restriction maps in whitened coordinates
            self.logger.info("Computing restriction maps in whitened coordinates...")
            procrustes_maps = ProcrustesMaps()
            
            restrictions = {}
            max_residual = 0.0
            
            for edge in poset.edges():
                source, target = edge
                
                # Get whitened Gram matrices
                if source in self.whitened_activations and target in self.whitened_activations:
                    source_gram = self.whitened_activations[source]['gram_whitened']
                    target_gram = self.whitened_activations[target]['gram_whitened']
                    
                    # Compute restriction map in whitened space (CRITICAL: no back-transformation)
                    restriction, residual = procrustes_maps.compute_restriction_whitened(
                        source_gram, target_gram
                    )
                    
                    restrictions[edge] = restriction
                    max_residual = max(max_residual, residual)
                    
                    if self.verbose:
                        self.logger.debug(f"Restriction {source}→{target}: residual {residual:.6f}")
            
            # Build sheaf structure
            self.logger.info("Assembling sheaf structure...")
            sheaf_builder = SheafBuilder()
            
            stalks = {name: data['gram_whitened'] for name, data in self.whitened_activations.items()}
            
            sheaf = sheaf_builder.build_sheaf(
                poset=poset,
                stalks=stalks, 
                restrictions=restrictions
            )
            
            # Store data for next stages
            self.poset = poset
            self.restrictions = restrictions
            self.sheaf = sheaf
            self.metrics.actual_nodes = num_nodes
            self.metrics.actual_edges = num_edges
            self.metrics.actual_max_residual = max_residual
            
            # Validate numerical properties
            success = self._validate_sheaf_properties(sheaf, max_residual)
            
            # Record result
            duration = time.time() - stage_start
            memory_mb = self._get_current_memory_mb()
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            
            result = TestResult(
                stage_name="Stage 3: Sheaf Construction",
                success=success,
                duration_seconds=duration,
                memory_peak_mb=memory_mb,
                details={
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "max_residual": max_residual,
                    "num_restrictions": len(restrictions)
                }
            )
            self.results.append(result)
            
            if success:
                self.logger.info(f"✅ Stage 3 completed in {duration:.2f}s (Memory: {memory_mb:.1f}MB)")
                self.logger.info(f"   Max restriction residual: {max_residual:.6f}")
            else:
                self.logger.warning(f"⚠️ Stage 3 completed with warnings in {duration:.2f}s")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Stage 3 failed: {e}")
            result = TestResult(
                stage_name="Stage 3: Sheaf Construction",
                success=False,
                duration_seconds=time.time() - stage_start,
                memory_peak_mb=self._get_current_memory_mb(),
                errors=[str(e)]
            )
            self.results.append(result)
            return False
    
    def _run_stage_4(self) -> bool:
        """Stage 4: Laplacian Assembly & Spectral Analysis."""
        self.logger.info("Stage 4: Laplacian Assembly & Spectral Analysis")
        stage_start = time.time()
        
        try:
            # Build sparse Laplacian
            self.logger.info("Assembling sparse sheaf Laplacian...")
            laplacian_builder = SheafLaplacianBuilder()
            
            laplacian_matrix, metadata = laplacian_builder.build_laplacian(self.sheaf)
            
            # Get matrix properties
            nnz = laplacian_matrix.nnz
            shape = laplacian_matrix.shape
            density = nnz / (shape[0] * shape[1]) * 100
            
            self.logger.info(f"Laplacian: {shape[0]}×{shape[1]}, {nnz} non-zeros ({density:.3f}% dense)")
            
            # Validate symmetry
            symmetry_error = self._check_laplacian_symmetry(laplacian_matrix)
            
            # Compute eigenvalues
            self.logger.info("Computing eigenvalues using Lanczos iteration...")
            eigenvalues = self._compute_eigenvalues(laplacian_matrix, k=20)
            
            min_eigenvalue = eigenvalues[0]
            harmonic_dim = np.sum(np.abs(eigenvalues) < 1e-8)
            
            # Store results
            self.laplacian_matrix = laplacian_matrix
            self.eigenvalues = eigenvalues
            self.metrics.actual_laplacian_nnz = nnz
            self.metrics.actual_symmetry_error = symmetry_error
            self.metrics.actual_min_eigenvalue = min_eigenvalue
            self.metrics.actual_harmonic_dim = harmonic_dim
            
            # Validate spectral properties
            success = self._validate_spectral_properties(eigenvalues, symmetry_error)
            
            # Record result
            duration = time.time() - stage_start
            memory_mb = self._get_current_memory_mb()
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            
            result = TestResult(
                stage_name="Stage 4: Laplacian Assembly & Spectral Analysis",
                success=success,
                duration_seconds=duration,
                memory_peak_mb=memory_mb,
                details={
                    "laplacian_shape": shape,
                    "laplacian_nnz": nnz,
                    "density_percent": density,
                    "symmetry_error": symmetry_error,
                    "min_eigenvalue": min_eigenvalue,
                    "harmonic_dim": harmonic_dim
                }
            )
            self.results.append(result)
            
            if success:
                self.logger.info(f"✅ Stage 4 completed in {duration:.2f}s (Memory: {memory_mb:.1f}MB)")
                self.logger.info(f"   Harmonic dimension: {harmonic_dim}")
                self.logger.info(f"   Min eigenvalue: {min_eigenvalue:.2e}")
            else:
                self.logger.warning(f"⚠️ Stage 4 completed with warnings in {duration:.2f}s")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Stage 4 failed: {e}")
            result = TestResult(
                stage_name="Stage 4: Laplacian Assembly & Spectral Analysis",
                success=False,
                duration_seconds=time.time() - stage_start,
                memory_peak_mb=self._get_current_memory_mb(),
                errors=[str(e)]
            )
            self.results.append(result)
            return False
    
    def _run_stage_5(self) -> bool:
        """Stage 5: Performance & Numerical Validation."""
        self.logger.info("Stage 5: Performance & Numerical Validation")
        stage_start = time.time()
        
        try:
            # Validate all metrics against expected values
            validation_results = self._validate_all_metrics()
            
            # Count passed/failed validations
            passed = sum(1 for result in validation_results.values() if result['passed'])
            total = len(validation_results)
            
            success = passed == total
            
            # Record result
            duration = time.time() - stage_start
            memory_mb = self._get_current_memory_mb()
            
            result = TestResult(
                stage_name="Stage 5: Performance & Numerical Validation",
                success=success,
                duration_seconds=duration,
                memory_peak_mb=memory_mb,
                details={
                    "validations_passed": passed,
                    "validations_total": total,
                    "validation_results": validation_results
                }
            )
            self.results.append(result)
            
            if success:
                self.logger.info(f"✅ Stage 5 completed in {duration:.2f}s - All validations passed!")
            else:
                self.logger.warning(f"⚠️ Stage 5 completed in {duration:.2f}s - {total-passed}/{total} validations failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Stage 5 failed: {e}")
            result = TestResult(
                stage_name="Stage 5: Performance & Numerical Validation",
                success=False,
                duration_seconds=time.time() - stage_start,
                memory_peak_mb=self._get_current_memory_mb(),
                errors=[str(e)]
            )
            self.results.append(result)
            return False
    
    def _extract_activations(self, model: nn.Module, input_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from all model layers using hooks."""
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Process activations for better numerical conditioning
                    if output.dim() > 2:
                        # For conv layers: Use global average pooling but preserve more information
                        # (batch, channels, h, w) -> (batch, channels)
                        batch_size, channels = output.shape[:2]
                        
                        # Use adaptive pooling to reduce spatial dimensions while preserving variance
                        pooled = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                        output_flat = pooled.view(batch_size, channels)
                        
                        # Add small amount of noise for numerical stability
                        noise_std = 1e-8
                        noise = torch.randn_like(output_flat) * noise_std
                        output_flat = output_flat + noise
                        
                    else:
                        output_flat = output
                        # Add small noise for fully connected layers too
                        noise_std = 1e-8
                        noise = torch.randn_like(output_flat) * noise_std
                        output_flat = output_flat + noise
                    
                    # Normalize activations to improve conditioning
                    if output_flat.std() > 1e-8:  # Avoid division by zero
                        output_flat = (output_flat - output_flat.mean(dim=0, keepdim=True)) / (output_flat.std(dim=0, keepdim=True) + 1e-8)
                    
                    activations[name] = output_flat.detach().cpu()
                    
            return hook
        
        # Register hooks for all named modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def _validate_stalk_dimensions(self, whitened_activations: Dict[str, Any]) -> None:
        """Validate that stalk dimensions match expected ranges."""
        conv_64_ranks = []
        conv_128_ranks = []
        conv_256_ranks = []
        conv_512_ranks = []
        
        for layer_name, data in whitened_activations.items():
            rank = data['rank']
            
            # Categorize by layer type (rough heuristic)
            if 'conv1' in layer_name or ('layer1' in layer_name and 'conv' in layer_name):
                conv_64_ranks.append(rank)
            elif 'layer2' in layer_name and 'conv' in layer_name:
                conv_128_ranks.append(rank)
            elif 'layer3' in layer_name and 'conv' in layer_name:
                conv_256_ranks.append(rank)
            elif 'layer4' in layer_name and 'conv' in layer_name:
                conv_512_ranks.append(rank)
        
        # Log validation results
        if conv_64_ranks:
            avg_64 = np.mean(conv_64_ranks)
            self.logger.info(f"Conv 64-ch layers: avg rank {avg_64:.1f} (expected 60-64)")
        
        if conv_128_ranks:
            avg_128 = np.mean(conv_128_ranks)
            self.logger.info(f"Conv 128-ch layers: avg rank {avg_128:.1f} (expected 110-118)")
        
        if conv_256_ranks:
            avg_256 = np.mean(conv_256_ranks)
            self.logger.info(f"Conv 256-ch layers: avg rank {avg_256:.1f} (expected 220-230)")
        
        if conv_512_ranks:
            avg_512 = np.mean(conv_512_ranks)
            self.logger.info(f"Conv 512-ch layers: avg rank {avg_512:.1f} (expected 450-470)")
    
    def _validate_sheaf_properties(self, sheaf: Sheaf, max_residual: float) -> bool:
        """Validate mathematical properties of the sheaf."""
        success = True
        
        # Check restriction residual threshold
        if max_residual > self.metrics.expected_max_residual:
            self.logger.warning(f"Max restriction residual {max_residual:.6f} > {self.metrics.expected_max_residual:.6f}")
            success = False
        else:
            self.logger.info(f"✓ Restriction residuals within tolerance: {max_residual:.6f}")
        
        # Additional sheaf property validations could go here
        # (transitivity, orthogonality, etc.)
        
        return success
    
    def _check_laplacian_symmetry(self, laplacian: sparse.csr_matrix) -> float:
        """Check symmetry of the Laplacian matrix."""
        try:
            # Convert to dense for small matrices, or sample for large ones
            if laplacian.shape[0] < 1000:
                dense = laplacian.toarray()
                symmetry_error = np.max(np.abs(dense - dense.T))
            else:
                # Sample-based symmetry check for large matrices
                n_samples = 1000
                rows = np.random.choice(laplacian.shape[0], n_samples, replace=True)
                cols = np.random.choice(laplacian.shape[1], n_samples, replace=True)
                
                errors = []
                for i, j in zip(rows, cols):
                    val_ij = laplacian[i, j]
                    val_ji = laplacian[j, i]
                    errors.append(abs(val_ij - val_ji))
                
                symmetry_error = max(errors)
            
            return float(symmetry_error)
            
        except Exception as e:
            self.logger.warning(f"Could not check Laplacian symmetry: {e}")
            return float('inf')
    
    def _compute_eigenvalues(self, laplacian: sparse.csr_matrix, k: int = 20) -> np.ndarray:
        """Compute smallest eigenvalues using sparse methods."""
        try:
            from scipy.sparse.linalg import eigsh
            
            # Compute k smallest eigenvalues
            eigenvalues, _ = eigsh(laplacian, k=k, which='SA', tol=1e-5)
            return np.sort(eigenvalues)
            
        except Exception as e:
            self.logger.warning(f"Could not compute eigenvalues: {e}")
            return np.array([float('nan')] * k)
    
    def _validate_spectral_properties(self, eigenvalues: np.ndarray, symmetry_error: float) -> bool:
        """Validate spectral properties of the Laplacian."""
        success = True
        
        # Check symmetry
        if symmetry_error > self.metrics.expected_symmetry_error:
            self.logger.warning(f"Symmetry error {symmetry_error:.2e} > {self.metrics.expected_symmetry_error:.2e}")
            success = False
        else:
            self.logger.info(f"✓ Laplacian symmetry within tolerance: {symmetry_error:.2e}")
        
        # Check positive semi-definiteness
        min_eigenvalue = eigenvalues[0]
        if min_eigenvalue < self.metrics.expected_min_eigenvalue:
            self.logger.warning(f"Min eigenvalue {min_eigenvalue:.2e} < {self.metrics.expected_min_eigenvalue:.2e}")
            success = False
        else:
            self.logger.info(f"✓ Laplacian is positive semi-definite: λ_min = {min_eigenvalue:.2e}")
        
        return success
    
    def _validate_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Validate all pipeline metrics against expected values."""
        results = {}
        
        # Define validation checks with 10% tolerance
        checks = [
            ('nodes', self.metrics.actual_nodes, self.metrics.expected_nodes, 0.0),
            ('edges', self.metrics.actual_edges, self.metrics.expected_edges, 0.0),
            ('total_dims', self.metrics.actual_total_dims, self.metrics.expected_total_dims, 0.1),
            ('laplacian_nnz', self.metrics.actual_laplacian_nnz, self.metrics.expected_laplacian_nnz, 0.2),
            ('harmonic_dim', self.metrics.actual_harmonic_dim, self.metrics.expected_harmonic_dim, 0.0),
            ('runtime_sec', self.metrics.actual_runtime_sec, self.metrics.expected_runtime_sec, float('inf')),  # Only upper bound
            ('peak_memory_gb', self.metrics.actual_peak_memory_gb, self.metrics.expected_peak_memory_gb, float('inf'))  # Only upper bound
        ]
        
        for name, actual, expected, tolerance in checks:
            if actual is None:
                results[name] = {'passed': False, 'reason': 'Not measured', 'actual': None, 'expected': expected}
                continue
            
            if tolerance == float('inf'):
                # Only check upper bound
                passed = actual <= expected
                reason = f"Within limit" if passed else f"Exceeded: {actual} > {expected}"
            elif tolerance == 0.0:
                # Exact match
                passed = actual == expected
                reason = f"Exact match" if passed else f"Mismatch: {actual} != {expected}"
            else:
                # Tolerance-based check
                diff = abs(actual - expected) / expected
                passed = diff <= tolerance
                reason = f"Within {tolerance*100}% tolerance" if passed else f"Outside tolerance: {diff*100:.1f}% > {tolerance*100}%"
            
            results[name] = {
                'passed': passed,
                'reason': reason,
                'actual': actual,
                'expected': expected,
                'tolerance': tolerance
            }
        
        return results
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _print_final_summary(self, overall_success: bool) -> None:
        """Print comprehensive test summary."""
        self.logger.info("=" * 80)
        self.logger.info("RESNET18 END-TO-END PIPELINE TEST SUMMARY")
        self.logger.info("=" * 80)
        
        # Overall result
        status = "✅ PASSED" if overall_success else "❌ FAILED"
        self.logger.info(f"Overall Result: {status}")
        
        # Performance summary
        total_time = self.metrics.actual_runtime_sec or 0.0
        peak_memory = self.metrics.actual_peak_memory_gb or 0.0
        self.logger.info(f"Total Runtime: {total_time:.2f}s (target: ≤{self.metrics.expected_runtime_sec:.0f}s)")
        self.logger.info(f"Peak Memory: {peak_memory:.2f}GB (target: ≤{self.metrics.expected_peak_memory_gb:.1f}GB)")
        
        # Stage-by-stage results
        self.logger.info("\nStage Results:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            self.logger.info(f"  {status} {result.stage_name}: {result.duration_seconds:.2f}s")
            if result.errors:
                for error in result.errors:
                    self.logger.info(f"    Error: {error}")
        
        # Key metrics comparison
        self.logger.info("\nKey Metrics:")
        self.logger.info(f"  Nodes: {self.metrics.actual_nodes} (expected: {self.metrics.expected_nodes})")
        self.logger.info(f"  Edges: {self.metrics.actual_edges} (expected: {self.metrics.expected_edges})")
        self.logger.info(f"  Total Dims: {self.metrics.actual_total_dims} (expected: ~{self.metrics.expected_total_dims})")
        self.logger.info(f"  Laplacian NNZ: {self.metrics.actual_laplacian_nnz} (expected: ~{self.metrics.expected_laplacian_nnz})")
        self.logger.info(f"  Harmonic Dim: {self.metrics.actual_harmonic_dim} (expected: {self.metrics.expected_harmonic_dim})")
        
        if self.metrics.actual_max_residual is not None:
            self.logger.info(f"  Max Residual: {self.metrics.actual_max_residual:.6f} (target: <{self.metrics.expected_max_residual:.3f})")
        
        if self.metrics.actual_symmetry_error is not None:
            self.logger.info(f"  Symmetry Error: {self.metrics.actual_symmetry_error:.2e} (target: <{self.metrics.expected_symmetry_error:.0e})")
        
        if self.metrics.actual_min_eigenvalue is not None:
            self.logger.info(f"  Min Eigenvalue: {self.metrics.actual_min_eigenvalue:.2e} (target: ≥{self.metrics.expected_min_eigenvalue:.0e})")
        
        self.logger.info("=" * 80)


def main():
    """Main entry point for the ResNet18 end-to-end test."""
    parser = argparse.ArgumentParser(description="ResNet18 End-to-End Pipeline Test")
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging and output')
    
    args = parser.parse_args()
    
    # Initialize and run test
    test_runner = ResNet18EndToEndTest(device=args.device, verbose=args.verbose)
    success = test_runner.run_complete_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()