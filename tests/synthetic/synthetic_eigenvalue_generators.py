#!/usr/bin/env python3
"""
Synthetic Eigenvalue Sequence Generators for DTW Testing

This module provides generators for creating controlled synthetic eigenvalue sequences
that mimic patterns found in neural network spectral analysis. These patterns are used
to validate and benchmark the DTW implementation before applying it to real neural networks.

The generators create eigenvalue evolution patterns based on mathematical functions that
represent different types of spectral behavior during filtration processes.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class SequenceMetadata:
    """Metadata for synthetic eigenvalue sequences."""
    pattern_type: str
    parameters: Dict
    expected_properties: Dict
    description: str


class SyntheticEigenvalueGenerator:
    """Generator for synthetic eigenvalue evolution sequences."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        self.rng = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
    
    def exponential_decay(self, 
                         n_steps: int = 50, 
                         n_eigenvals: int = 10,
                         decay_rate: float = 0.1,
                         initial_value: float = 10.0,
                         noise_level: float = 0.0) -> Tuple[List[torch.Tensor], SequenceMetadata]:
        """Generate exponential decay eigenvalue sequence.
        
        Pattern: λ_k(t) = initial_value * exp(-decay_rate * k * t) + noise
        Simulates eigenvalue decay during filtration in neural networks.
        
        Args:
            n_steps: Number of filtration steps
            n_eigenvals: Number of eigenvalues per step
            decay_rate: Exponential decay rate parameter
            initial_value: Initial eigenvalue magnitude
            noise_level: Gaussian noise standard deviation
            
        Returns:
            Tuple of (eigenvalue_sequence, metadata)
        """
        sequence = []
        
        for step in range(n_steps):
            # Normalized time parameter [0, 1]
            t = step / (n_steps - 1) if n_steps > 1 else 0.0
            
            eigenvals = []
            for k in range(n_eigenvals):
                # Exponential decay with eigenvalue index dependency
                base_value = initial_value * np.exp(-decay_rate * (k + 1) * t)
                
                # Add controlled noise
                if noise_level > 0:
                    noise = self.rng.normal(0, noise_level * base_value)
                    value = max(base_value + noise, 1e-12)  # Ensure positive
                else:
                    value = base_value
                
                eigenvals.append(value)
            
            # Sort in descending order (typical eigenvalue ordering)
            eigenvals = sorted(eigenvals, reverse=True)
            sequence.append(torch.tensor(eigenvals, dtype=torch.float32))
        
        metadata = SequenceMetadata(
            pattern_type="exponential_decay",
            parameters={
                "decay_rate": decay_rate,
                "initial_value": initial_value,
                "noise_level": noise_level,
                "n_steps": n_steps,
                "n_eigenvals": n_eigenvals
            },
            expected_properties={
                "monotonic_decay": True,
                "largest_eigenval_range": (initial_value * np.exp(-decay_rate * n_steps), initial_value),
                "spectral_gap": "decreasing"
            },
            description=f"Exponential decay with rate {decay_rate}, {n_eigenvals} eigenvalues over {n_steps} steps"
        )
        
        return sequence, metadata
    
    def power_law(self,
                  n_steps: int = 50,
                  n_eigenvals: int = 10,
                  exponent: float = -0.5,
                  scaling: float = 10.0,
                  noise_level: float = 0.0) -> Tuple[List[torch.Tensor], SequenceMetadata]:
        """Generate power-law eigenvalue sequence.
        
        Pattern: λ_k(t) = scaling * (t + 1)^exponent * k^(-0.5) + noise
        Simulates power-law spectral decay common in complex networks.
        
        Args:
            n_steps: Number of filtration steps
            n_eigenvals: Number of eigenvalues per step  
            exponent: Power law exponent (typically negative)
            scaling: Overall scaling factor
            noise_level: Gaussian noise standard deviation
            
        Returns:
            Tuple of (eigenvalue_sequence, metadata)
        """
        sequence = []
        
        for step in range(n_steps):
            # Time parameter starting from 1 to avoid division by zero
            t = step + 1
            
            eigenvals = []
            for k in range(n_eigenvals):
                # Power law with both time and eigenvalue index dependency
                base_value = scaling * (t ** exponent) * ((k + 1) ** (-0.5))
                
                # Add controlled noise
                if noise_level > 0:
                    noise = self.rng.normal(0, noise_level * base_value)
                    value = max(base_value + noise, 1e-12)
                else:
                    value = base_value
                
                eigenvals.append(value)
            
            eigenvals = sorted(eigenvals, reverse=True)
            sequence.append(torch.tensor(eigenvals, dtype=torch.float32))
        
        metadata = SequenceMetadata(
            pattern_type="power_law",
            parameters={
                "exponent": exponent,
                "scaling": scaling,
                "noise_level": noise_level,
                "n_steps": n_steps,
                "n_eigenvals": n_eigenvals
            },
            expected_properties={
                "monotonic_decay": exponent < 0,
                "power_law_exponent": exponent,
                "scaling_factor": scaling
            },
            description=f"Power law with exponent {exponent}, {n_eigenvals} eigenvalues over {n_steps} steps"
        )
        
        return sequence, metadata
    
    def sinusoidal_modulated(self,
                           n_steps: int = 50,
                           n_eigenvals: int = 10,
                           frequency: float = 2.0,
                           amplitude: float = 5.0,
                           baseline: float = 5.0,
                           phase_shift: float = 0.0,
                           noise_level: float = 0.0) -> Tuple[List[torch.Tensor], SequenceMetadata]:
        """Generate sinusoidally modulated eigenvalue sequence.
        
        Pattern: λ_k(t) = baseline + amplitude * sin(2π * frequency * t + phase_shift) * exp(-0.1*k) + noise
        Simulates oscillatory behavior that might occur in certain neural network dynamics.
        
        Args:
            n_steps: Number of filtration steps
            n_eigenvals: Number of eigenvalues per step
            frequency: Oscillation frequency
            amplitude: Oscillation amplitude
            baseline: Baseline eigenvalue level
            phase_shift: Phase shift in radians
            noise_level: Gaussian noise standard deviation
            
        Returns:
            Tuple of (eigenvalue_sequence, metadata)
        """
        sequence = []
        
        for step in range(n_steps):
            t = step / (n_steps - 1) if n_steps > 1 else 0.0
            
            eigenvals = []
            for k in range(n_eigenvals):
                # Sinusoidal modulation with exponential eigenvalue decay
                base_modulation = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
                eigenval_scaling = np.exp(-0.1 * k)  # Decay with eigenvalue index
                base_value = baseline + base_modulation * eigenval_scaling
                
                # Ensure positive eigenvalues
                base_value = max(base_value, 0.1)
                
                # Add controlled noise
                if noise_level > 0:
                    noise = self.rng.normal(0, noise_level * base_value)
                    value = max(base_value + noise, 1e-12)
                else:
                    value = base_value
                
                eigenvals.append(value)
            
            eigenvals = sorted(eigenvals, reverse=True)
            sequence.append(torch.tensor(eigenvals, dtype=torch.float32))
        
        metadata = SequenceMetadata(
            pattern_type="sinusoidal_modulated",
            parameters={
                "frequency": frequency,
                "amplitude": amplitude,
                "baseline": baseline,
                "phase_shift": phase_shift,
                "noise_level": noise_level,
                "n_steps": n_steps,
                "n_eigenvals": n_eigenvals
            },
            expected_properties={
                "oscillatory": True,
                "frequency": frequency,
                "amplitude_range": (baseline - amplitude, baseline + amplitude)
            },
            description=f"Sinusoidal modulation with frequency {frequency}, {n_eigenvals} eigenvalues over {n_steps} steps"
        )
        
        return sequence, metadata
    
    def phase_transition(self,
                        n_steps: int = 50,
                        n_eigenvals: int = 10,
                        transition_point: float = 0.5,
                        pre_level: float = 10.0,
                        post_level: float = 1.0,
                        transition_width: float = 0.1,
                        noise_level: float = 0.0) -> Tuple[List[torch.Tensor], SequenceMetadata]:
        """Generate eigenvalue sequence with phase transition.
        
        Pattern: Smooth transition from pre_level to post_level using sigmoid function.
        Simulates abrupt spectral changes during filtration in neural networks.
        
        Args:
            n_steps: Number of filtration steps
            n_eigenvals: Number of eigenvalues per step
            transition_point: Normalized time of transition [0, 1]
            pre_level: Eigenvalue level before transition
            post_level: Eigenvalue level after transition
            transition_width: Width of transition region
            noise_level: Gaussian noise standard deviation
            
        Returns:
            Tuple of (eigenvalue_sequence, metadata)
        """
        sequence = []
        
        for step in range(n_steps):
            t = step / (n_steps - 1) if n_steps > 1 else 0.0
            
            # Sigmoid transition function
            transition_steepness = 1.0 / max(transition_width, 1e-6)
            sigmoid = 1.0 / (1.0 + np.exp(-transition_steepness * (t - transition_point)))
            level = pre_level * (1 - sigmoid) + post_level * sigmoid
            
            eigenvals = []
            for k in range(n_eigenvals):
                # Apply eigenvalue-dependent scaling
                eigenval_factor = np.exp(-0.2 * k)  # Larger eigenvalues decay faster
                base_value = level * eigenval_factor
                
                # Add controlled noise
                if noise_level > 0:
                    noise = self.rng.normal(0, noise_level * base_value)
                    value = max(base_value + noise, 1e-12)
                else:
                    value = base_value
                
                eigenvals.append(value)
            
            eigenvals = sorted(eigenvals, reverse=True)
            sequence.append(torch.tensor(eigenvals, dtype=torch.float32))
        
        metadata = SequenceMetadata(
            pattern_type="phase_transition",
            parameters={
                "transition_point": transition_point,
                "pre_level": pre_level,
                "post_level": post_level,
                "transition_width": transition_width,
                "noise_level": noise_level,
                "n_steps": n_steps,
                "n_eigenvals": n_eigenvals
            },
            expected_properties={
                "has_transition": True,
                "transition_location": transition_point,
                "level_change": abs(post_level - pre_level)
            },
            description=f"Phase transition at t={transition_point}, {n_eigenvals} eigenvalues over {n_steps} steps"
        )
        
        return sequence, metadata
    
    def constant_sequence(self,
                         n_steps: int = 50,
                         n_eigenvals: int = 10,
                         value: float = 5.0,
                         noise_level: float = 0.0) -> Tuple[List[torch.Tensor], SequenceMetadata]:
        """Generate constant eigenvalue sequence.
        
        Pattern: Constant eigenvalues (useful for baseline comparisons).
        
        Args:
            n_steps: Number of filtration steps
            n_eigenvals: Number of eigenvalues per step
            value: Constant eigenvalue value
            noise_level: Gaussian noise standard deviation
            
        Returns:
            Tuple of (eigenvalue_sequence, metadata)
        """
        sequence = []
        
        for step in range(n_steps):
            eigenvals = []
            for k in range(n_eigenvals):
                # Slight decay with eigenvalue index for realism
                base_value = value * np.exp(-0.05 * k)
                
                # Add controlled noise
                if noise_level > 0:
                    noise = self.rng.normal(0, noise_level * base_value)
                    final_value = max(base_value + noise, 1e-12)
                else:
                    final_value = base_value
                
                eigenvals.append(final_value)
            
            eigenvals = sorted(eigenvals, reverse=True)
            sequence.append(torch.tensor(eigenvals, dtype=torch.float32))
        
        metadata = SequenceMetadata(
            pattern_type="constant",
            parameters={
                "value": value,
                "noise_level": noise_level,
                "n_steps": n_steps,
                "n_eigenvals": n_eigenvals
            },
            expected_properties={
                "monotonic_decay": False,
                "constant_level": value,
                "variance": noise_level ** 2
            },
            description=f"Constant sequence with value {value}, {n_eigenvals} eigenvalues over {n_steps} steps"
        )
        
        return sequence, metadata
    
    def generate_test_pairs(self) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], float, str]]:
        """Generate a set of test sequence pairs with expected DTW properties.
        
        Returns:
            List of (seq1, seq2, expected_relative_distance, description) tuples
        """
        test_pairs = []
        
        # 1. Identical sequences (should have distance ≈ 0)
        seq1, _ = self.exponential_decay(decay_rate=0.1)
        seq2, _ = self.exponential_decay(decay_rate=0.1)  # Same parameters
        test_pairs.append((seq1, seq2, 0.0, "Identical exponential decay sequences"))
        
        # 2. Similar decay rates (should have small distance)
        seq1, _ = self.exponential_decay(decay_rate=0.1)
        seq2, _ = self.exponential_decay(decay_rate=0.12)
        test_pairs.append((seq1, seq2, 0.2, "Similar exponential decay rates (0.1 vs 0.12)"))
        
        # 3. Different decay rates (should have medium distance)
        seq1, _ = self.exponential_decay(decay_rate=0.1)
        seq2, _ = self.exponential_decay(decay_rate=0.3)
        test_pairs.append((seq1, seq2, 0.5, "Different exponential decay rates (0.1 vs 0.3)"))
        
        # 4. Very different patterns (should have large distance)
        seq1, _ = self.exponential_decay(decay_rate=0.1)
        seq2, _ = self.sinusoidal_modulated(frequency=3.0)
        test_pairs.append((seq1, seq2, 0.8, "Exponential decay vs sinusoidal modulation"))
        
        # 5. Phase transitions at different points (should have medium distance)
        seq1, _ = self.phase_transition(transition_point=0.3)
        seq2, _ = self.phase_transition(transition_point=0.7)
        test_pairs.append((seq1, seq2, 0.4, "Phase transitions at different points (0.3 vs 0.7)"))
        
        # 6. Constant vs dynamic (should have large distance)
        seq1, _ = self.constant_sequence(value=5.0)
        seq2, _ = self.exponential_decay(decay_rate=0.2)
        test_pairs.append((seq1, seq2, 0.9, "Constant sequence vs exponential decay"))
        
        # 7. Different power law exponents (should have medium distance)
        seq1, _ = self.power_law(exponent=-0.3)
        seq2, _ = self.power_law(exponent=-0.7)
        test_pairs.append((seq1, seq2, 0.3, "Power law with different exponents (-0.3 vs -0.7)"))
        
        # 8. Same pattern with noise vs without (should have small distance)
        seq1, _ = self.exponential_decay(decay_rate=0.15, noise_level=0.0)
        seq2, _ = self.exponential_decay(decay_rate=0.15, noise_level=0.05)
        test_pairs.append((seq1, seq2, 0.1, "Same pattern with and without noise"))
        
        return test_pairs


def demonstrate_generators():
    """Demonstrate the synthetic eigenvalue generators."""
    print("🧪 Synthetic Eigenvalue Generator Demo")
    print("=" * 50)
    
    generator = SyntheticEigenvalueGenerator(random_seed=42)
    
    # Generate examples of each pattern type
    patterns = [
        ("Exponential Decay", lambda: generator.exponential_decay(n_steps=10, decay_rate=0.2)),
        ("Power Law", lambda: generator.power_law(n_steps=10, exponent=-0.5)),
        ("Sinusoidal", lambda: generator.sinusoidal_modulated(n_steps=10, frequency=2.0)),
        ("Phase Transition", lambda: generator.phase_transition(n_steps=10, transition_point=0.6)),
        ("Constant", lambda: generator.constant_sequence(n_steps=10, value=3.0))
    ]
    
    for name, generator_func in patterns:
        sequence, metadata = generator_func()
        print(f"\n{name}:")
        print(f"  Description: {metadata.description}")
        print(f"  First step eigenvalues: {sequence[0][:5].tolist()}")  # Show first 5
        print(f"  Last step eigenvalues: {sequence[-1][:5].tolist()}")
        print(f"  Parameters: {metadata.parameters}")
    
    # Generate test pairs
    print(f"\n{'Test Pairs':<40} {'Expected Distance':<20}")
    print("-" * 60)
    test_pairs = generator.generate_test_pairs()
    for seq1, seq2, expected_dist, description in test_pairs:
        print(f"{description:<40} {expected_dist:<20.1f}")


if __name__ == "__main__":
    demonstrate_generators()