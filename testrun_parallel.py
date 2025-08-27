#!/usr/bin/env python3
"""testrun_parallel.py

Parallel processing script for analyzing all models in the models/ folder
and saving results with consistent naming to eigenvalueData/ folder.

Usage:
    python testrun_parallel.py                  # Process all models with defaults
    python testrun_parallel.py --workers 4      # Use 4 parallel workers  
    python testrun_parallel.py --resume         # Skip already processed models
    python testrun_parallel.py --dry-run        # Preview without processing
    python testrun_parallel.py --pattern "mlp*" # Process only MLP models
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import psutil
from tqdm import tqdm

# Core neurosheaf imports - exact same as testrun.py
from neurosheaf.sheaf.core.gw_config import GWConfig
from neurosheaf.sheaf.assembly.builder import SheafBuilder
from neurosheaf.spectral.persistent import PersistentSpectralAnalyzer, AlphaFlowSpec, StaticBuildConfig
from neurosheaf.spectral.flows.alpha_flow import AlphaGroupingPolicy
from neurosheaf.utils import load_model
from neurosheaf.api import NeurosheafAnalyzer
from neurosheaf.visualization import EnhancedVisualizationFactory
from neurosheaf.io import save_eigenvalue_evolution

# NEW: GW Subspace Tracker imports (from testrun.py)
from neurosheaf.spectral.gw.gw_subspace_tracker import GWSubspaceTracker
from neurosheaf.spectral.tracker_factory import SubspaceTrackerFactory
from neurosheaf.spectral.gw.pes_computation import PESComputer
from neurosheaf.spectral import UnifiedStaticLaplacian

# Set up logging exactly like testrun.py
import logging
logging.getLogger('neurosheaf').setLevel(logging.DEBUG)


class MLPModel(nn.Module):
    """MLP model architecture matching the saved weights configuration."""
    
    def __init__(
        self,
        input_dim: int = 3,
        num_hidden_layers: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 1,
        activation_fn_name: str = 'relu',
        output_activation_fn_name: str = 'sigmoid',
        dropout_rate: float = 0.0012
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Get activation functions
        self.activation_fn = self._get_activation_fn(activation_fn_name)
        self.output_activation_fn = self._get_activation_fn(output_activation_fn_name)
        
        # Build the network
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(input_dim, hidden_dim))
        layers_list.append(self.activation_fn)
        if dropout_rate > 0:
            layers_list.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(self.activation_fn)
            if dropout_rate > 0:
                layers_list.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        if output_activation_fn_name != 'none':
            layers_list.append(self.output_activation_fn)
        
        # Use 'layers' as the attribute name to match saved weights
        self.layers = nn.Sequential(*layers_list)
    
    def _get_activation_fn(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'softmax': nn.Softmax(dim=-1),
            'none': nn.Identity()
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name.lower()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CustomModel(nn.Module):
    """Custom model class with Conv1D layers matching saved weights structure."""
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(3, 32),                                    # layers.0
            nn.ReLU(),                                           # layers.1 (activation)
            nn.Linear(32, 32),                                   # layers.2
            nn.ReLU(),                                           # layers.3 (activation)
            nn.Dropout(0.0),                                     # layers.4 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.5
            nn.ReLU(),                                           # layers.6 (activation)
            nn.Dropout(0.0),                                     # layers.7 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.8
            nn.ReLU(),                                           # layers.9 (activation)
            nn.Dropout(0.0),                                     # layers.10 (dropout)
            nn.Conv1d(in_channels=16, out_channels=32, 
                     kernel_size=2, stride=1, padding=0),        # layers.11
            nn.ReLU(),                                           # layers.12 (activation)
            nn.Dropout(0.0),                                     # layers.13 (dropout)
            nn.Linear(32, 1),                                    # layers.14
            nn.Sigmoid()                                         # layers.15 (activation)
        )
    
    def forward(self, x):
        # Input: [batch_size, 3]
        
        # Layer 0: Linear(3 -> 32) + ReLU
        x = self.layers[1](self.layers[0](x))  # [batch_size, 32]
        
        # Layer 2: Linear(32 -> 32) + ReLU + Dropout
        x = self.layers[4](self.layers[3](self.layers[2](x)))  # [batch_size, 32]
        
        # Reshape for Conv1D: [batch_size, 32] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 5: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[7](self.layers[6](self.layers[5](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 8: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[10](self.layers[9](self.layers[8](x)))  # [batch_size, 32, 1]
        
        # Reshape for next Conv1D: [batch_size, 32, 1] -> [batch_size, 16, 2]
        x = x.view(-1, 16, 2)  # [batch_size, 16, 2]
        
        # Layer 11: Conv1D(16->32, k=2) + ReLU + Dropout
        x = self.layers[13](self.layers[12](self.layers[11](x)))  # [batch_size, 32, 1]
        
        # Flatten for final layer: [batch_size, 32, 1] -> [batch_size, 32]
        x = x.view(x.size(0), -1)  # [batch_size, 32]
        
        # Layer 14: Linear(32 -> 1) + Sigmoid
        x = self.layers[15](self.layers[14](x))  # [batch_size, 1]
        
        return x

class MLP4x256(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        dims = [784] + [hidden_dim] * num_layers
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.head(x)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Channels chosen to minimize units while staying usable for MNIST
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=7, padding=0, bias=True)   # 28x28 -> 4x4
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=1, padding=0, bias=True)  # 4x4   -> 3x3
        self.conv3 = nn.Conv2d(16, 20, kernel_size=2, stride=1, padding=0, bias=True) # 3x3   -> 2x2
        self.conv4 = nn.Conv2d(20, 24, kernel_size=2, stride=1, padding=0, bias=True) # 2x2   -> 1x1
        self.fc    = nn.Linear(24, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [B, 4, 4, 4]
        x = F.relu(self.conv2(x))   # [B, 16, 3, 3]
        x = F.relu(self.conv3(x))   # [B, 20, 2, 2]
        x = F.relu(self.conv4(x))   # [B, 24, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 24]
        return self.fc(x)


class ModelProcessor:
    """Handles model loading, processing, and result saving."""
    
    def __init__(self, batch_size: int = 1000, n_steps: int = 30):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility - exact same as testrun.py
        self.random_seed = 30  # testrun.py uses 30, not 42
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        
        # Initialize MNIST dataset for MNIST models (lazy loading)
        self.mnist_data = None
        self.mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
        ])
    
    def is_mnist_model(self, model_name: str) -> bool:
        """Check if a model is MNIST-based by its filename pattern."""
        return (model_name.startswith('mlp4layer_mnist') or 
                model_name.startswith('mnist_mlp') or
                model_name.startswith('tinycnn'))
    
    def load_mnist_data(self, model) -> torch.Tensor:
        """Load MNIST test data with appropriate transform based on model type."""
        # Choose transform based on model type
        if isinstance(model, TinyCNN):
            # CNN models need 2D images: [batch_size, 1, 28, 28]
            transform = transforms.ToTensor()
            self.logger.info("Using 2D transform for CNN model")
        else:
            # MLP models need flattened input: [batch_size, 784]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            self.logger.info("Using flattened transform for MLP model")
        
        self.logger.info("Loading MNIST dataset...")
        try:
            mnist_test = datasets.MNIST(
                root='./data', 
                train=False, 
                download=True, 
                transform=transform
            )
            
            # Create data loader and get first batch_size samples
            test_loader = torch.utils.data.DataLoader(
                mnist_test, 
                batch_size=self.batch_size, 
                shuffle=False
            )
            data, labels = next(iter(test_loader))
            self.logger.info(f"Loaded MNIST data shape: {data.shape}")
            self.logger.info(f"Labels for first 10 samples: {labels[:10].tolist()}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load MNIST data: {e}")
            return None
    
    def load_model_by_type(self, model_path: Path) -> Optional[torch.nn.Module]:
        """Auto-detect model type from filename and load with appropriate architecture."""
        model_name = model_path.stem
        
        try:
            if model_name.startswith('mlp_'):
                return load_model(MLPModel, str(model_path))
            elif model_name.startswith('custom_'):
                return load_model(CustomModel, str(model_path))
            elif model_name.startswith('mlp4layer_mnist'):
                return load_model(MLP4x256, str(model_path))
            elif model_name.startswith('mnist_mlp'):
                return load_model(MLP4x256, str(model_path))
            elif model_name.startswith('tinycnn'):
                return load_model(TinyCNN, str(model_path))
            elif model_name.startswith('mlp4'):
                return load_model(MLP4x256, str(model_path))
            else:
                self.logger.warning(f"Unknown model type for {model_name}, trying as MLP")
                return load_model(MLPModel, str(model_path))
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def extract_model_metadata(self, model_path: Path) -> Dict[str, str]:
        """Extract metadata from model filename using new naming convention."""
        model_name = model_path.stem
        metadata = {
            'model_name': model_name,
            'architecture': 'unknown',
            'type': 'unknown',
            'accuracy': 'unknown',
            'epochs': 'unknown',
            'seed': 'unknown',
            'version': 'unknown'
        }
        
        # Parse architecture from prefix
        if model_name.startswith('mlp_'):
            metadata['architecture'] = 'mlp'
        elif model_name.startswith('custom_'):
            metadata['architecture'] = 'custom'
        elif model_name.startswith('mlp4layer_mnist'):
            metadata['architecture'] = 'mlp4layer_mnist'
        elif model_name.startswith('mnist_mlp'):
            metadata['architecture'] = 'mnist_mlp'
        elif model_name.startswith('tinycnn_mnist'):
            metadata['architecture'] = 'tinycnn_mnist'
        elif model_name.startswith('tinycnn_random'):
            metadata['architecture'] = 'tinycnn_mnist'  # Same architecture, different training
        elif model_name.startswith('tinycnn'):
            metadata['architecture'] = 'tinycnn'
        elif model_name.startswith('mlp4'):
            metadata['architecture'] = 'mlp4'
        
        # Handle different naming patterns
        parts = model_name.split('_')
        
        if model_name.startswith('mlp4layer_mnist'):
            # Pattern: mlp4layer_mnist_seedXX
            metadata['type'] = 'trained'  # These are trained models
            seed_match = re.search(r'seed(\d+)', model_name)
            if seed_match:
                metadata['seed'] = seed_match.group(1)
        elif model_name.startswith('tinycnn_mnist'):
            # Pattern: tinycnn_mnist.pth
            metadata['type'] = 'trained'  # These are trained models
        elif model_name.startswith('tinycnn_random'):
            # Pattern: tinycnn_random_XXX
            metadata['type'] = 'random'
            # Extract version number (last part)
            if len(parts) >= 3:
                metadata['version'] = parts[2]  # '001', '005', etc.
        elif model_name.startswith('mnist_mlp'):
            # Pattern: mnist_mlp_random_XXX
            if len(parts) >= 3:
                metadata['type'] = parts[2]  # 'random'
            # Extract version number (last part)
            if len(parts) >= 4:
                metadata['version'] = parts[3]  # '001', '002', etc.
        else:
            # Original patterns: mlp_*, custom_*
            if len(parts) >= 2:
                metadata['type'] = parts[1]  # 'random' or 'trained'
            
            # Extract version number (always present as vXX)
            version_match = re.search(r'_v(\d+)', model_name)
            if version_match:
                metadata['version'] = version_match.group(1)
            
            # For trained models, extract accuracy and epochs
            if metadata['type'] == 'trained':
                acc_match = re.search(r'_acc([0-9.]+)', model_name)
                if acc_match:
                    metadata['accuracy'] = acc_match.group(1)
                
                ep_match = re.search(r'_ep(\d+)', model_name)
                if ep_match:
                    metadata['epochs'] = ep_match.group(1)
        
        return metadata
    
    def process_single_model(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single model and return results."""
        self.logger.info(f"Processing model: {model_path.name}")
        
        try:
            # Load model
            model = self.load_model_by_type(model_path)
            if model is None:
                return None
            
            model_name = model_path.stem
            
            # Generate appropriate test data based on model type
            if self.is_mnist_model(model_name):
                # Load actual MNIST data for MNIST models
                data = self.load_mnist_data(model)
                if data is None:
                    self.logger.error(f"Failed to load MNIST data for {model_name}")
                    return None
            else:
                # Generate random data for other models - exact same as testrun.py
                data = 10 * torch.randn(self.batch_size, 3)
                self.logger.info(f"Generated random data shape: {data.shape}")
            
            # Use high-level API exactly like testrun.py
            analyzer = NeurosheafAnalyzer(device='cpu')
            
            self.logger.info("Running analysis WITH layer filtering (exclude final single-output layers):")
            analysis = analyzer.analyze(
                model, data, 
                method='gromov_wasserstein',
                use_normalized_laplacian=False,
                exclude_final_single_output=True  # Enable layer filtering to reduce degeneracy
            )
            sheaf = analysis['sheaf']
            
            # Run spectral analysis exactly like testrun.py
            spectral_analyzer = PersistentSpectralAnalyzer(
                default_n_steps=self.n_steps,
                default_filtration_type='threshold'
            )
            
            results = spectral_analyzer.analyze(
                sheaf,
                filtration_type='threshold',
                n_steps=self.n_steps
            )
            
            # Extract model metadata
            metadata = self.extract_model_metadata(model_path)
            
            return {
                'results': results,
                'metadata': metadata,
                'model_path': model_path,
                'processing_time': time.time(),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {model_path.name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {
                'model_path': model_path,
                'metadata': self.extract_model_metadata(model_path),
                'error': str(e),
                'success': False
            }
    
    def save_results(self, processing_result: Dict[str, Any], output_dir: Path) -> bool:
        """Save processing results with consistent naming."""
        if not processing_result['success']:
            return False
        
        try:
            model_name = processing_result['metadata']['model_name']
            output_file = output_dir / f"{model_name}_eigenvalues.npz"
            
            # Save eigenvalue evolution
            results = processing_result['results']
            save_eigenvalue_evolution(
                results['persistence_result']['eigenvalue_sequences'],
                results['filtration_params'],
                str(output_file)
            )
            
            self.logger.info(f"Saved results to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results for {processing_result['model_path'].name}: {e}")
            return False


def process_model_worker(args: Tuple[Path, Dict]) -> Dict[str, Any]:
    """Worker function for parallel processing."""
    model_path, config = args
    processor = ModelProcessor(
        batch_size=config['batch_size'],
        n_steps=config['n_steps']
    )
    return processor.process_single_model(model_path)


def discover_models(models_dir: Path, pattern: str = "*") -> List[Path]:
    """Discover all model files matching the pattern."""
    model_files = []
    for file_path in models_dir.glob("*.pth"):
        if fnmatch.fnmatch(file_path.name, pattern):
            model_files.append(file_path)
    
    return sorted(model_files)


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set neurosheaf logging to WARNING to reduce noise
    logging.getLogger('neurosheaf').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def create_summary_csv(results: List[Dict], output_dir: Path) -> None:
    """Create a CSV summary of all processing results."""
    csv_file = output_dir / "processing_summary.csv"
    
    fieldnames = [
        'model_name', 'architecture', 'type', 'accuracy', 'epochs', 
        'seed', 'version', 'success', 'processing_time', 'error'
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = result['metadata'].copy()
            row['success'] = result['success']
            row['processing_time'] = result.get('processing_time', 'N/A')
            row['error'] = result.get('error', '')
            writer.writerow(row)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process all models in parallel and save eigenvalue data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--models-dir', 
        type=Path, 
        default=Path('models'),
        help='Directory containing model files'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=Path, 
        default=Path('eigenvalueData'),
        help='Directory to save eigenvalue data'
    )
    
    parser.add_argument(
        '--workers', 
        type=int, 
        default=mp.cpu_count() // 2,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1000,
        help='Batch size for model input data'
    )
    
    parser.add_argument(
        '--n-steps', 
        type=int, 
        default=30,
        help='Number of filtration steps'
    )
    
    parser.add_argument(
        '--pattern', 
        type=str, 
        default='*',
        help='Filename pattern to match (e.g., "mlp*", "custom*")'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Skip models that already have results'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    # Setup directories
    args.models_dir = Path(args.models_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = args.output_dir / "processing_log.txt"
    logger = setup_logging(log_file)
    
    # Discover models
    model_files = discover_models(args.models_dir, args.pattern)
    logger.info(f"Discovered {len(model_files)} model files")
    
    if args.resume:
        # Filter out already processed models
        existing_results = set(f.stem.replace('_eigenvalues', '') 
                              for f in args.output_dir.glob('*_eigenvalues.npz'))
        model_files = [f for f in model_files if f.stem not in existing_results]
        logger.info(f"After resume filter: {len(model_files)} models to process")
    
    if args.dry_run:
        print(f"Would process {len(model_files)} models:")
        for model_file in model_files:
            print(f"  {model_file.name}")
        return
    
    if not model_files:
        logger.info("No models to process")
        return
    
    # Check memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"Available memory: {memory_gb:.1f} GB")
    
    if memory_gb < 4:
        logger.warning("Low memory detected. Consider reducing batch size or workers.")
    
    # Process models
    logger.info(f"Starting parallel processing with {args.workers} workers")
    start_time = time.time()
    
    config = {
        'batch_size': args.batch_size,
        'n_steps': args.n_steps
    }
    
    all_results = []
    processor = ModelProcessor()  # For saving results
    
    with mp.Pool(args.workers) as pool:
        # Create work items
        work_items = [(model_path, config) for model_path in model_files]
        
        # Process with progress bar
        with tqdm(total=len(work_items), desc="Processing models") as pbar:
            for result in pool.imap_unordered(process_model_worker, work_items):
                if result:
                    all_results.append(result)
                    
                    # Save results immediately
                    if result['success']:
                        processor.save_results(result, args.output_dir)
                
                pbar.update(1)
    
    # Create summary
    create_summary_csv(all_results, args.output_dir)
    
    # Final statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r['success'])
    failed = len(all_results) - successful
    
    logger.info(f"Processing complete!")
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()