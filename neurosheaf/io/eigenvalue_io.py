"""Eigenvalue evolution I/O utilities for Neurosheaf.

This module provides functions for saving and loading eigenvalue evolution results
from the neurosheaf persistent spectral analysis pipeline.
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
import warnings
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def save_eigenvalue_evolution(
    eigenvalue_sequences: List[torch.Tensor],
    filtration_params: List[float],
    output_path: Union[str, Path],
    format: str = 'npz',
    metadata: Optional[Dict] = None,
    max_eigenvalues: Optional[int] = None
) -> None:
    """Save eigenvalue evolution results to file.
    
    This function takes the eigenvalue sequences produced by the neurosheaf pipeline
    and saves them as an eigenvalue matrix (T x n_eigs) and a time vector t of length T.
    
    Args:
        eigenvalue_sequences: List of eigenvalue tensors from neurosheaf analysis
        filtration_params: List of filtration parameter values (time points)
        output_path: Path to save the results
        format: Output format ('npz', 'npy', 'csv', 'mat')
        metadata: Optional metadata to save with the results
        max_eigenvalues: Maximum number of eigenvalues to save (None for auto-detect)
        
    Raises:
        ValueError: If input data is invalid or inconsistent
        IOError: If file writing fails
    """
    output_path = Path(output_path)
    
    # Validate inputs
    if not eigenvalue_sequences:
        raise ValueError("eigenvalue_sequences cannot be empty")
    if len(eigenvalue_sequences) != len(filtration_params):
        raise ValueError(f"Length mismatch: {len(eigenvalue_sequences)} eigenvalue sequences "
                        f"but {len(filtration_params)} filtration parameters")
    
    logger.info(f"Saving eigenvalue evolution with {len(eigenvalue_sequences)} time steps")
    
    # Convert to numpy arrays
    time_vector = np.array(filtration_params, dtype=np.float64)
    
    # Determine maximum number of eigenvalues
    if max_eigenvalues is None:
        max_eigenvalues = max(len(seq) for seq in eigenvalue_sequences if len(seq) > 0)
        if max_eigenvalues == 0:
            raise ValueError("All eigenvalue sequences are empty")
    
    logger.debug(f"Using {max_eigenvalues} eigenvalues per time step")
    
    # Create eigenvalue matrix (T x n_eigs)
    T = len(eigenvalue_sequences)
    eigenvalue_matrix = np.full((T, max_eigenvalues), np.nan, dtype=np.float64)
    
    for i, seq in enumerate(eigenvalue_sequences):
        if len(seq) > 0:
            # Convert tensor to numpy and handle dtype
            seq_np = seq.detach().cpu().numpy().astype(np.float64)
            n_eigs = min(len(seq_np), max_eigenvalues)
            eigenvalue_matrix[i, :n_eigs] = seq_np[:n_eigs]
    
    # Prepare metadata
    save_metadata = {
        'format_version': '1.0',
        'neurosheaf_version': 'current',
        'n_time_steps': T,
        'n_eigenvalues': max_eigenvalues,
        'filtration_range': (float(np.min(time_vector)), float(np.max(time_vector))),
        'eigenvalue_range': (float(np.nanmin(eigenvalue_matrix)), float(np.nanmax(eigenvalue_matrix))),
        'has_missing_eigenvalues': np.any(np.isnan(eigenvalue_matrix))
    }
    
    if metadata is not None:
        save_metadata.update(metadata)
    
    # Save based on format
    format = format.lower()
    
    try:
        if format == 'npz':
            output_path = output_path.with_suffix('.npz')
            np.savez_compressed(
                output_path,
                eigenvalue_matrix=eigenvalue_matrix,
                time_vector=time_vector,
                metadata=save_metadata
            )
            
        elif format == 'npy':
            # Save as separate .npy files
            base_path = output_path.with_suffix('')
            np.save(f"{base_path}_eigenvalues.npy", eigenvalue_matrix)
            np.save(f"{base_path}_time.npy", time_vector)
            np.save(f"{base_path}_metadata.npy", save_metadata)
            
        elif format == 'csv':
            output_path = output_path.with_suffix('.csv')
            # Create a DataFrame-like structure for CSV
            import pandas as pd
            
            # Combine time vector with eigenvalue matrix
            data = np.column_stack([time_vector, eigenvalue_matrix])
            
            # Create column names
            columns = ['time'] + [f'eigenvalue_{i}' for i in range(max_eigenvalues)]
            
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(output_path, index=False)
            
            # Save metadata separately
            metadata_path = output_path.with_suffix('.meta.json')
            import json
            
            # Convert numpy types to native Python types for JSON serialization
            json_metadata = {}
            for key, value in save_metadata.items():
                if isinstance(value, np.bool_):
                    json_metadata[key] = bool(value)
                elif isinstance(value, (np.integer, np.floating)):
                    json_metadata[key] = value.item()
                else:
                    json_metadata[key] = value
            
            with open(metadata_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
                
        elif format == 'mat':
            output_path = output_path.with_suffix('.mat')
            from scipy.io import savemat
            
            savemat(output_path, {
                'eigenvalue_matrix': eigenvalue_matrix,
                'time_vector': time_vector,
                'metadata': save_metadata
            })
            
        else:
            raise ValueError(f"Unsupported format: {format}. "
                           f"Supported formats: 'npz', 'npy', 'csv', 'mat'")
    
        logger.info(f"Successfully saved eigenvalue evolution to {output_path}")
        logger.debug(f"Matrix shape: {eigenvalue_matrix.shape}, "
                    f"Time vector length: {len(time_vector)}")
        
    except Exception as e:
        logger.error(f"Failed to save eigenvalue evolution: {e}")
        raise IOError(f"Failed to save eigenvalue evolution: {e}") from e


def load_eigenvalue_evolution(
    filepath: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """Load eigenvalue evolution results from file.
    
    Args:
        filepath: Path to the saved eigenvalue evolution file
        
    Returns:
        Tuple of (eigenvalue_matrix, time_vector, metadata)
        - eigenvalue_matrix: numpy array of shape (T, n_eigs)
        - time_vector: numpy array of length T
        - metadata: dictionary with saved metadata (if available)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
        IOError: If file reading fails
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading eigenvalue evolution from {filepath}")
    
    try:
        if filepath.suffix == '.npz':
            data = np.load(filepath, allow_pickle=True)
            eigenvalue_matrix = data['eigenvalue_matrix']
            time_vector = data['time_vector']
            metadata = data.get('metadata', {})
            if isinstance(metadata, np.ndarray):
                metadata = metadata.item()
                
        elif filepath.suffix == '.npy':
            # Load separate .npy files
            # Handle both direct path and base path scenarios
            if filepath.name.endswith('_eigenvalues.npy'):
                base_path = str(filepath)[:-len('_eigenvalues.npy')]
                eigenvalue_matrix = np.load(filepath)
            else:
                base_path = filepath.with_suffix('')
                eigenvalue_matrix = np.load(f"{base_path}_eigenvalues.npy")
            
            time_vector = np.load(f"{base_path}_time.npy")
            
            metadata_path = f"{base_path}_metadata.npy"
            if Path(metadata_path).exists():
                metadata = np.load(metadata_path, allow_pickle=True).item()
            else:
                metadata = {}
                
        elif filepath.suffix == '.csv':
            import pandas as pd
            
            df = pd.read_csv(filepath)
            time_vector = df['time'].values
            eigenvalue_matrix = df.drop('time', axis=1).values
            
            # Try to load metadata
            metadata_path = filepath.with_suffix('.meta.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
                
        elif filepath.suffix == '.mat':
            from scipy.io import loadmat
            
            data = loadmat(filepath)
            eigenvalue_matrix = data['eigenvalue_matrix']
            time_vector = data['time_vector'].flatten()
            metadata = data.get('metadata', {})
            if isinstance(metadata, np.ndarray):
                metadata = metadata.item()
                
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. "
                           f"Supported formats: .npz, .npy, .csv, .mat")
    
        logger.info(f"Successfully loaded eigenvalue evolution: "
                   f"matrix shape {eigenvalue_matrix.shape}, "
                   f"time vector length {len(time_vector)}")
        
        return eigenvalue_matrix, time_vector, metadata
        
    except Exception as e:
        logger.error(f"Failed to load eigenvalue evolution: {e}")
        raise IOError(f"Failed to load eigenvalue evolution: {e}") from e


def convert_eigenvalue_sequences_to_matrix(
    eigenvalue_sequences: List[torch.Tensor],
    max_eigenvalues: Optional[int] = None
) -> np.ndarray:
    """Convert list of eigenvalue tensors to a numpy matrix.
    
    Helper function to convert eigenvalue sequences to matrix format
    without saving to file.
    
    Args:
        eigenvalue_sequences: List of eigenvalue tensors
        max_eigenvalues: Maximum number of eigenvalues to include
        
    Returns:
        numpy array of shape (T, n_eigs) with NaN for missing values
    """
    if not eigenvalue_sequences:
        raise ValueError("eigenvalue_sequences cannot be empty")
    
    if max_eigenvalues is None:
        max_eigenvalues = max(len(seq) for seq in eigenvalue_sequences if len(seq) > 0)
        if max_eigenvalues == 0:
            raise ValueError("All eigenvalue sequences are empty")
    
    T = len(eigenvalue_sequences)
    matrix = np.full((T, max_eigenvalues), np.nan, dtype=np.float64)
    
    for i, seq in enumerate(eigenvalue_sequences):
        if len(seq) > 0:
            seq_np = seq.detach().cpu().numpy().astype(np.float64)
            n_eigs = min(len(seq_np), max_eigenvalues)
            matrix[i, :n_eigs] = seq_np[:n_eigs]
    
    return matrix