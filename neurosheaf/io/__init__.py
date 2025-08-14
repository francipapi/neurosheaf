"""Input/Output utilities and configuration for Neurosheaf.

This module provides configuration management and type definitions
for the Neurosheaf framework, including production-ready Global
Section tracking configuration.
"""

from .config import H0Config, DEFAULT_H0_CONFIG
from .types import (
    KernelResult,
    TransportResult,
    RRQRResult,
    PersistenceResult,
    StepDiagnostics,
    CertificateResult,
    WhiteningResult,
    TransportValidation
)

__all__ = [
    # Configuration
    "H0Config",
    "DEFAULT_H0_CONFIG",
    
    # Type definitions
    "KernelResult",
    "TransportResult", 
    "RRQRResult",
    "PersistenceResult",
    "StepDiagnostics",
    "CertificateResult",
    "WhiteningResult",
    "TransportValidation"
]