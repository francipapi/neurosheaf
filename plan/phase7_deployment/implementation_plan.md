# Phase 7: Deployment Implementation Plan (Week 16)

## Overview
Final deployment preparations including Docker containerization, PyPI package release, documentation hosting, and production-ready distribution.

## Week 16: Deployment and Release

### Day 1-2: Package Preparation
- [ ] Finalize package structure and metadata
- [ ] Update version information and changelog
- [ ] Create comprehensive README.md
- [ ] Add license and contribution guidelines
- [ ] Prepare setup.py and pyproject.toml

### Day 3-4: Docker Containerization
- [ ] Create production Docker image
- [ ] Add development Docker setup
- [ ] Create docker-compose for dashboard
- [ ] Optimize image size and security
- [ ] Test container deployment

### Day 5-6: PyPI Release
- [ ] Test package installation in clean environment
- [ ] Create release notes and changelog
- [ ] Upload to PyPI test repository
- [ ] Validate package installation and functionality
- [ ] Release to production PyPI

### Day 7: Documentation and Final Testing
- [ ] Deploy documentation to ReadTheDocs
- [ ] Create release announcement
- [ ] Final integration testing
- [ ] Performance validation
- [ ] Security audit

## Implementation Details

### Package Structure for Release
```
neurosheaf/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── setup.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── MANIFEST.in
├── neurosheaf/
│   ├── __init__.py
│   ├── api.py
│   ├── cka/
│   ├── sheaf/
│   ├── spectral/
│   ├── visualization/
│   └── utils/
├── tests/
├── examples/
├── docs/
├── scripts/
│   ├── install_dependencies.sh
│   ├── run_tests.sh
│   └── benchmark.py
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   ├── docker-compose.yml
│   └── requirements-docker.txt
└── .github/
    └── workflows/
```

### Setup.py Configuration
```python
# setup.py
from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from __init__.py
def get_version():
    with open("neurosheaf/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"

# Read requirements
def get_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="neurosheaf",
    version=get_version(),
    author="Neurosheaf Team",
    author_email="contact@neurosheaf.org",
    description="Persistent Sheaf Laplacians for Neural Network Similarity Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neurosheaf/neurosheaf",
    project_urls={
        "Bug Tracker": "https://github.com/neurosheaf/neurosheaf/issues",
        "Documentation": "https://neurosheaf.readthedocs.io/",
        "Source Code": "https://github.com/neurosheaf/neurosheaf",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "jupyter>=1.0",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "kaleido>=0.2",  # For static image export
        ],
        "full": [
            "plotly>=5.0",
            "dash>=2.0",
            "kaleido>=0.2",
            "jupyter>=1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "neurosheaf=neurosheaf.cli:main",
            "neurosheaf-dashboard=neurosheaf.visualization.dashboard:main",
        ],
    },
    package_data={
        "neurosheaf": [
            "data/*.json",
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
```

### PyProject.toml (Modern Python Packaging)
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neurosheaf"
version = "1.0.0"
description = "Persistent Sheaf Laplacians for Neural Network Similarity Analysis"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Neurosheaf Team", email = "contact@neurosheaf.org"},
]
maintainers = [
    {name = "Neurosheaf Team", email = "contact@neurosheaf.org"},
]
keywords = [
    "neural networks",
    "similarity analysis", 
    "persistent homology",
    "sheaf theory",
    "deep learning",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
]
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "networkx>=2.6",
    "matplotlib>=3.5.0",
    "scikit-learn>=1.0",
    "tqdm>=4.62.0",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=4.0",
    "mypy>=0.950",
    "pre-commit>=2.15",
]
docs = [
    "sphinx>=5.0",
    "sphinx-rtd-theme>=1.0",
    "nbsphinx>=0.8",
    "myst-parser>=0.18",
    "sphinx-autodoc-typehints>=1.19",
]
viz = [
    "plotly>=5.0",
    "dash>=2.0",
    "kaleido>=0.2",
    "jupyter>=1.0",
]
all = [
    "plotly>=5.0",
    "dash>=2.0", 
    "kaleido>=0.2",
    "jupyter>=1.0",
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=22.0",
    "sphinx>=5.0",
]

[project.urls]
Homepage = "https://github.com/neurosheaf/neurosheaf"
Documentation = "https://neurosheaf.readthedocs.io/"
Repository = "https://github.com/neurosheaf/neurosheaf"
"Bug Tracker" = "https://github.com/neurosheaf/neurosheaf/issues"
Changelog = "https://github.com/neurosheaf/neurosheaf/blob/main/CHANGELOG.md"

[project.scripts]
neurosheaf = "neurosheaf.cli:main"
neurosheaf-dashboard = "neurosheaf.visualization.dashboard:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["neurosheaf*"]
exclude = ["tests*"]

[tool.setuptools.package-data]
neurosheaf = ["data/*.json", "templates/*.html", "static/css/*.css", "static/js/*.js"]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["neurosheaf"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short",
    "--cov=neurosheaf",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "benchmark: marks tests as benchmarks",
    "gpu: marks tests that require GPU",
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
```

### Docker Configuration
```dockerfile
# docker/Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash neurosheaf && \
    chown -R neurosheaf:neurosheaf /app
USER neurosheaf

# Expose port for dashboard
EXPOSE 8050

# Default command
CMD ["neurosheaf-dashboard"]
```

```dockerfile
# docker/Dockerfile.dev
FROM python:3.9-slim

# Development environment with additional tools
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Install Jupyter extensions
RUN pip install --no-cache-dir jupyter-contrib-nbextensions && \
    jupyter contrib nbextension install --user

# Copy source code
COPY . .

# Install package in development mode
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash neurosheaf && \
    chown -R neurosheaf:neurosheaf /app
USER neurosheaf

# Expose ports
EXPOSE 8050 8888

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  neurosheaf:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8050:8050"
    volumes:
      - ../examples:/app/examples
      - ../data:/app/data
      - neurosheaf_results:/app/results
    environment:
      - NEUROSHEAF_LOG_LEVEL=INFO
      - NEUROSHEAF_CACHE_DIR=/app/cache
    restart: unless-stopped
    
  neurosheaf-dev:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dev
    ports:
      - "8050:8050"
      - "8888:8888"
    volumes:
      - ../:/app
      - jupyter_data:/home/neurosheaf/.jupyter
    environment:
      - NEUROSHEAF_LOG_LEVEL=DEBUG
      - JUPYTER_ENABLE_LAB=yes
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
    restart: unless-stopped

volumes:
  neurosheaf_results:
  jupyter_data:
```

### CLI Interface
```python
# neurosheaf/cli.py
"""Command-line interface for Neurosheaf."""

import argparse
import sys
import os
import json
import torch
from pathlib import Path
from .api import NeurosheafAnalyzer
from .utils.logging import setup_logger
from .visualization.dashboard import NeurosheafDashboard

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Neurosheaf: Persistent Sheaf Laplacians for Neural Network Analysis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a neural network")
    analyze_parser.add_argument("model_path", help="Path to saved model")
    analyze_parser.add_argument("--data", help="Path to input data")
    analyze_parser.add_argument("--output", "-o", help="Output directory", default="./results")
    analyze_parser.add_argument("--cka-samples", type=int, default=1000, help="Number of CKA samples")
    analyze_parser.add_argument("--n-steps", type=int, default=20, help="Number of persistence steps")
    analyze_parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    analyze_parser.add_argument("--config", help="Path to configuration file")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start interactive dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8050, help="Port for dashboard")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="Host for dashboard")
    dashboard_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument("--output", "-o", help="Output file for results")
    benchmark_parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert between formats")
    convert_parser.add_argument("input_file", help="Input file")
    convert_parser.add_argument("output_file", help="Output file")
    convert_parser.add_argument("--format", choices=["json", "pickle", "hdf5"], default="json")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "convert":
        run_convert(args)
    else:
        parser.print_help()
        sys.exit(1)

def run_analyze(args):
    """Run analysis command."""
    logger = setup_logger("neurosheaf.cli")
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location='cpu')
    
    # Load data
    if args.data:
        logger.info(f"Loading data from {args.data}")
        data = torch.load(args.data, map_location='cpu')
    else:
        # Generate synthetic data
        logger.info("Generating synthetic data")
        data = torch.randn(100, 3, 224, 224)  # Default image size
    
    # Create analyzer
    analyzer = NeurosheafAnalyzer(
        cka_samples=args.cka_samples,
        n_persistence_steps=args.n_steps,
        use_gpu=args.gpu,
        **config
    )
    
    # Run analysis
    logger.info("Starting analysis...")
    try:
        results = analyzer.analyze(model, data)
        
        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = convert_tensors_to_lists(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

def run_dashboard(args):
    """Run dashboard command."""
    logger = setup_logger("neurosheaf.dashboard")
    
    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    
    dashboard = NeurosheafDashboard(port=args.port)
    dashboard.run(debug=args.debug)

def run_benchmark(args):
    """Run benchmark command."""
    logger = setup_logger("neurosheaf.benchmark")
    
    from .benchmarks import run_benchmarks
    
    logger.info("Running benchmarks...")
    results = run_benchmarks(quick=args.quick)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

def run_convert(args):
    """Run convert command."""
    logger = setup_logger("neurosheaf.convert")
    
    # Implementation would depend on specific conversion needs
    logger.info(f"Converting {args.input_file} to {args.output_file}")
    # ... conversion logic ...

def convert_tensors_to_lists(obj):
    """Convert tensors to lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    main()
```

### Release Automation Script
```python
# scripts/release.py
"""Automated release script for Neurosheaf."""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, check=True):
    """Run shell command and return result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result

def get_version():
    """Get current version from package."""
    with open("neurosheaf/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return "unknown"

def update_version(new_version):
    """Update version in package files."""
    # Update __init__.py
    with open("neurosheaf/__init__.py", "r") as f:
        content = f.read()
    
    content = content.replace(
        f'__version__ = "{get_version()}"',
        f'__version__ = "{new_version}"'
    )
    
    with open("neurosheaf/__init__.py", "w") as f:
        f.write(content)
    
    # Update pyproject.toml
    with open("pyproject.toml", "r") as f:
        content = f.read()
    
    content = content.replace(
        f'version = "{get_version()}"',
        f'version = "{new_version}"'
    )
    
    with open("pyproject.toml", "w") as f:
        f.write(content)
    
    print(f"Version updated to {new_version}")

def run_tests():
    """Run complete test suite."""
    print("Running test suite...")
    
    # Unit tests
    run_command("pytest tests/unit/ -v")
    
    # Integration tests
    run_command("pytest tests/integration/ -v -m 'not slow'")
    
    # Style checks
    run_command("black --check neurosheaf/")
    run_command("isort --check-only neurosheaf/")
    run_command("flake8 neurosheaf/")
    
    print("All tests passed!")

def build_package():
    """Build package for distribution."""
    print("Building package...")
    
    # Clean previous builds
    run_command("rm -rf build/ dist/ *.egg-info/")
    
    # Build package
    run_command("python -m build")
    
    print("Package built successfully!")

def create_git_tag(version):
    """Create git tag for release."""
    print(f"Creating git tag v{version}...")
    
    run_command(f"git add .")
    run_command(f"git commit -m 'Release v{version}'")
    run_command(f"git tag v{version}")
    
    print(f"Git tag v{version} created!")

def upload_to_pypi(test=False):
    """Upload package to PyPI."""
    if test:
        print("Uploading to PyPI test repository...")
        run_command("python -m twine upload --repository testpypi dist/*")
    else:
        print("Uploading to PyPI...")
        run_command("python -m twine upload dist/*")
    
    print("Package uploaded successfully!")

def main():
    """Main release process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Release Neurosheaf package")
    parser.add_argument("version", help="New version number")
    parser.add_argument("--test", action="store_true", help="Upload to test PyPI")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test suite")
    parser.add_argument("--skip-tag", action="store_true", help="Skip git tag creation")
    
    args = parser.parse_args()
    
    current_version = get_version()
    print(f"Current version: {current_version}")
    print(f"New version: {args.version}")
    
    # Confirm release
    confirm = input("Continue with release? (y/N): ")
    if confirm.lower() != 'y':
        print("Release cancelled.")
        sys.exit(0)
    
    try:
        # Update version
        update_version(args.version)
        
        # Run tests
        if not args.skip_tests:
            run_tests()
        
        # Build package
        build_package()
        
        # Create git tag
        if not args.skip_tag:
            create_git_tag(args.version)
        
        # Upload to PyPI
        upload_to_pypi(test=args.test)
        
        print(f"Release v{args.version} completed successfully!")
        
    except Exception as e:
        print(f"Release failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### README.md Template
```markdown
# Neurosheaf

[![PyPI version](https://badge.fury.io/py/neurosheaf.svg)](https://badge.fury.io/py/neurosheaf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/neurosheaf/neurosheaf/workflows/Tests/badge.svg)](https://github.com/neurosheaf/neurosheaf/actions)
[![Coverage](https://codecov.io/gh/neurosheaf/neurosheaf/branch/main/graph/badge.svg)](https://codecov.io/gh/neurosheaf/neurosheaf)

**Persistent Sheaf Laplacians for Neural Network Similarity Analysis**

Neurosheaf is a Python framework for analyzing neural network similarity using persistent sheaf Laplacians. It provides a mathematically principled approach to understanding how neural networks represent and process information across different architectures and layers.

## Key Features

- **Debiased CKA Computation**: Correct implementation without double-centering
- **Automatic Architecture Analysis**: FX-based poset extraction works with any PyTorch model
- **Robust Spectral Analysis**: Subspace tracking handles eigenvalue crossings
- **Memory Efficient**: 500× memory reduction compared to baseline implementations
- **Interactive Visualization**: Dashboard with log-scale support and WebGL rendering
- **Production Ready**: Comprehensive testing, documentation, and Docker support

## Quick Start

### Installation

```bash
pip install neurosheaf
```

For development installation:
```bash
git clone https://github.com/neurosheaf/neurosheaf.git
cd neurosheaf
pip install -e ".[dev]"
```

### Basic Usage

```python
import torch
import torch.nn as nn
from neurosheaf.api import NeurosheafAnalyzer

# Create your model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# Generate or load your data
data = torch.randn(1000, 100)

# Analyze the model
analyzer = NeurosheafAnalyzer()
results = analyzer.analyze(model, data)

# Access results
cka_matrix = results['cka_matrix']
persistence_diagram = results['persistence']['diagrams']
sheaf = results['sheaf']
```

### Interactive Dashboard

```bash
neurosheaf dashboard
```

Then open your browser to `http://localhost:8050`

## Documentation

- [User Guide](https://neurosheaf.readthedocs.io/en/latest/user_guide/)
- [API Reference](https://neurosheaf.readthedocs.io/en/latest/api/)
- [Examples](https://neurosheaf.readthedocs.io/en/latest/examples/)
- [Theory Background](https://neurosheaf.readthedocs.io/en/latest/theory/)

## Architecture Support

Neurosheaf automatically handles:
- **ResNets**: Skip connections detected automatically
- **Transformers**: Attention patterns and layer normalization
- **CNNs**: Convolutional and pooling hierarchies
- **RNNs/LSTMs**: Recurrent state flow tracking
- **Custom Models**: Any PyTorch model via torch.fx

## Performance

- **Memory**: <3GB for ResNet50 analysis (vs 1.5TB baseline)
- **Speed**: 20× faster Laplacian assembly
- **Scalability**: Handles networks with 50+ layers
- **GPU Support**: CUDA acceleration for large models

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Neurosheaf in your research, please cite:

```bibtex
@software{neurosheaf2024,
  title={Neurosheaf: Persistent Sheaf Laplacians for Neural Network Similarity Analysis},
  author={Neurosheaf Team},
  year={2024},
  url={https://github.com/neurosheaf/neurosheaf}
}
```

## Acknowledgments

- Built on PyTorch and scientific Python ecosystem
- Inspired by topological data analysis and sheaf theory
- Thanks to the open source community for tools and libraries
```

## Deployment Checklist

### Pre-Release Validation
- [ ] All tests pass on multiple Python versions
- [ ] Documentation builds without errors
- [ ] Package installs correctly in clean environment
- [ ] CLI commands work as expected
- [ ] Docker containers build and run successfully
- [ ] Performance benchmarks meet requirements

### Release Process
- [ ] Update version numbers in all files
- [ ] Create comprehensive changelog
- [ ] Build and test package locally
- [ ] Upload to PyPI test repository
- [ ] Validate test installation
- [ ] Create git tag and push to repository
- [ ] Upload to production PyPI
- [ ] Deploy documentation to ReadTheDocs

### Post-Release
- [ ] Verify package availability on PyPI
- [ ] Test installation from PyPI
- [ ] Update documentation with new version
- [ ] Create release announcement
- [ ] Monitor for issues and bug reports

## Success Criteria

1. **Package Quality**: Successfully installs via pip on all supported platforms
2. **Documentation**: Complete docs hosted on ReadTheDocs
3. **Performance**: All benchmarks pass within specified limits
4. **Usability**: CLI and API work intuitively for new users
5. **Reliability**: No critical issues in first 48 hours post-release

## Phase 7 Deliverables

1. **Production Package**
   - PyPI-ready package with correct metadata
   - Comprehensive setup.py and pyproject.toml
   - All dependencies properly specified

2. **Docker Support**
   - Production Docker image
   - Development environment container
   - Docker Compose configuration

3. **Documentation Site**
   - ReadTheDocs deployment
   - API documentation
   - User guides and examples

4. **Release Infrastructure**
   - Automated release scripts
   - CI/CD pipeline for releases
   - Version management system

5. **Community Resources**
   - Contributing guidelines
   - Issue templates
   - Code of conduct
   - License and legal documentation

The Neurosheaf framework is now ready for production use and community adoption!