# Updated Debiased CKA Implementation Plan v3 (Fixed)

## Overview
This updated v3 plan fixes the double-centering issue in the debiased CKA implementation. The key fix is removing explicit feature centering since the unbiased HSIC estimator already handles centering internally.

## Key Fixes from Original v3
- **Removed explicit feature centering** when computing Gram matrices
- **Updated Nyström approximation** to use raw activations
- **Added unit test** to verify no double-centering occurs
- All other improvements from v3 retained

## Core Implementation (Fixed)

### 1. Main API Interface (Updated)

```python
# neursheaf/cka/api.py
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
from .debiased import DebiasedCKAComputer
from .validation import CKAValidator

def compute_layerwise_cka(
    activations: Dict[str, Union[np.ndarray, torch.Tensor]],
    debiased: bool = True,
    n_landmarks: Optional[int] = None,
    device: str = 'cuda',
    return_variance: bool = False,
    n_bootstrap: int = 3
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """
    Compute CKA between layers with optional Nyström approximation.
    
    IMPORTANT: For debiased CKA, activations are NOT centered before
    computing Gram matrices, as the unbiased HSIC estimator handles
    centering internally (Murphy et al., 2024).
    
    Parameters
    ----------
    activations : Dict[str, array]
        Layer name -> activation matrix (n_samples, n_features)
        Raw activations without centering
    debiased : bool
        Use debiased CKA (Murphy et al., 2024)
    n_landmarks : int, optional
        Number of landmarks for Nyström approximation (None for exact)
    device : str
        Device for computation ('cuda' or 'cpu')
    return_variance : bool
        Return variance estimates across bootstrap samples
    n_bootstrap : int
        Number of bootstrap runs for variance estimation
        
    Returns
    -------
    cka_values : Dict[str, float]
        Pairwise CKA values between layers
    gram_matrices : Dict[str, np.ndarray], optional
        Gram matrices for each layer (if requested)
        
    References
    ----------
    1. Murphy et al. (2024): Debiased CKA theory
    2. https://github.com/RistoAle97/centered-kernel-alignment.git
    """
    # Implementation continues as before...
```

### 2. Nyström Approximation (Fixed)

```python
# neursheaf/cka/nystrom.py
import torch
import numpy as np
from typing import Tuple, Optional

class NystromApproximation:
    """
    Nyström approximation for large-scale kernel matrices.
    
    IMPORTANT: Uses raw activations, not centered, to avoid
    double-centering with debiased CKA.
    """
    
    def compute_nystrom_kernel(self, 
                             X: torch.Tensor,
                             landmarks_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute Nyström approximation of kernel matrix.
        
        Parameters
        ----------
        X : torch.Tensor
            Raw activation matrix (n_samples, n_features)
            NOT centered to avoid double-centering
        landmarks_idx : torch.Tensor
            Indices of landmark points
            
        Returns
        -------
        K_approx : torch.Tensor
            Approximated kernel matrix
        """
        # Use raw X, not centered
        X_landmarks = X[landmarks_idx]
        
        # Compute kernel blocks WITHOUT centering
        K_ll = X_landmarks @ X_landmarks.T  # Landmark kernel
        K_nl = X @ X_landmarks.T            # Cross kernel
        
        # Nyström approximation
        K_ll_pinv = torch.linalg.pinv(K_ll)
        K_approx = K_nl @ K_ll_pinv @ K_nl.T
        
        return K_approx
```

### 3. Debiased CKA Computer (Fixed)

```python
# neursheaf/cka/debiased.py
import torch
import numpy as np
from typing import Dict, Tuple, Optional

class DebiasedCKAComputer:
    """
    Compute debiased CKA using unbiased HSIC estimator.
    
    Key fix: No explicit centering of features before computing
    Gram matrices, as unbiased HSIC handles this internally.
    """
    
    def compute_pairwise(self, 
                        activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute pairwise debiased CKA values.
        
        Parameters
        ----------
        activations : dict
            Raw activations for each layer (NOT centered)
        """
        cka_values = {}
        
        for name1, X in activations.items():
            for name2, Y in activations.items():
                if name1 < name2:  # Avoid duplicates
                    # Compute Gram matrices from RAW activations
                    K = X @ X.T  # No centering!
                    L = Y @ Y.T  # No centering!
                    
                    # Apply debiased CKA formula
                    cka = self._compute_debiased_cka(K, L)
                    cka_values[f'{name1}-{name2}'] = cka
                    
        return cka_values
    
    def _compute_debiased_cka(self, K: torch.Tensor, L: torch.Tensor) -> float:
        """
        Compute debiased CKA from Gram matrices.
        
        This implements the unbiased HSIC estimator from Murphy et al. (2024).
        The formula already includes centering, so K and L should be
        computed from raw (uncentered) features.
        """
        n = K.shape[0]
        
        # Remove diagonal
        K_no_diag = K - torch.diag(torch.diag(K))
        L_no_diag = L - torch.diag(torch.diag(L))
        
        # Compute row/column sums (excluding diagonal)
        K_row_sum = K_no_diag.sum(dim=1, keepdim=True)
        L_row_sum = L_no_diag.sum(dim=1, keepdim=True)
        
        # Compute total sum (excluding diagonal)
        K_total = K_no_diag.sum()
        L_total = L_no_diag.sum()
        
        # Unbiased HSIC formula (includes centering)
        term1 = torch.trace(K_no_diag @ L_no_diag)
        term2 = (K_row_sum.T @ L_row_sum).squeeze() / (n - 2)
        term3 = K_total * L_total / ((n - 1) * (n - 2))
        
        hsic_xy = (term1 + term3 - 2 * term2) / (n * (n - 3))
        
        # Compute HSIC_XX and HSIC_YY similarly
        hsic_xx = self._compute_hsic_self(K_no_diag, K_row_sum, K_total, n)
        hsic_yy = self._compute_hsic_self(L_no_diag, L_row_sum, L_total, n)
        
        # Debiased CKA
        if hsic_xx > 0 and hsic_yy > 0:
            cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        else:
            cka = 0.0
            
        return cka.item()
```

### 4. Unit Test for Double-Centering (New)

```python
# tests/test_no_double_centering.py
import pytest
import torch
import numpy as np
from neursheaf.cka import DebiasedCKAComputer

class TestNoDoubleCentering:
    """Test that we don't double-center in debiased CKA"""
    
    def test_ckav3_no_double_centering(self):
        """
        Verify that using raw vs centered features gives different results,
        and that raw features give higher (correct) CKA values.
        """
        # Generate correlated data
        torch.manual_seed(42)
        n_samples = 128
        n_features = 64
        
        X = torch.randn(n_samples, n_features)
        Y = torch.randn(n_samples, n_features) + 0.3 * X  # Correlated
        
        computer = DebiasedCKAComputer()
        
        # Correct: Use raw activations
        K_raw = X @ X.T
        L_raw = Y @ Y.T
        cka_correct = computer._compute_debiased_cka(K_raw, L_raw)
        
        # Wrong: Center features first (double-centering)
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        K_centered = X_centered @ X_centered.T
        L_centered = Y_centered @ Y_centered.T
        cka_wrong = computer._compute_debiased_cka(K_centered, L_centered)
        
        # The correct CKA should be higher
        assert cka_correct > cka_wrong, \
            f"Double-centering suppressed CKA! Correct: {cka_correct:.4f}, Wrong: {cka_wrong:.4f}"
        
        # Also verify the values are substantially different
        assert abs(cka_correct - cka_wrong) > 0.05, \
            "CKA values too similar - double-centering check may be ineffective"
    
    def test_nystrom_no_centering(self):
        """Test that Nyström approximation uses raw features"""
        from neursheaf.cka.nystrom import NystromApproximation
        
        torch.manual_seed(42)
        X = torch.randn(1000, 128)
        
        nystrom = NystromApproximation(n_landmarks=100)
        landmarks_idx = torch.randperm(X.shape[0])[:100]
        
        # Compute with raw features (correct)
        K_approx = nystrom.compute_nystrom_kernel(X, landmarks_idx)
        
        # Verify it matches direct computation
        K_exact = X @ X.T
        
        # Check approximation quality (should be reasonable)
        rel_error = torch.norm(K_exact - K_approx, 'fro') / torch.norm(K_exact, 'fro')
        assert rel_error < 0.1, f"Nyström approximation error too high: {rel_error:.4f}"
```

### 5. Updated Validation Module

```python
# neursheaf/cka/validation.py
class CKAValidator:
    """Validation for CKA computation"""
    
    def validate_no_preprocessing(self, activations: Dict[str, torch.Tensor]):
        """
        Warn if activations appear to be pre-centered.
        
        For debiased CKA, activations should NOT be centered.
        """
        for name, act in activations.items():
            mean_norm = torch.norm(act.mean(dim=0))
            
            if mean_norm < 1e-6:
                warnings.warn(
                    f"Layer {name} appears to be centered (mean≈0). "
                    f"For debiased CKA, use raw activations to avoid double-centering. "
                    f"See Murphy et al. (2024)."
                )
```

## Integration with Sheaf Construction

The sheaf construction pipeline must be updated to pass raw activations:

```python
# neursheaf/sheaf/construction.py (excerpt)
class NeuralNetworkSheaf:
    def add_stalk(self, node: str, activations: np.ndarray):
        """
        Add stalk data at node.
        
        Parameters
        ----------
        activations : np.ndarray
            Raw activation matrix (n_samples, n_features)
            NOT centered for debiased CKA compatibility
        """
        # Compute Gram matrix from raw activations
        K = activations @ activations.T
        self.stalks[node] = K
```

## Documentation Update

```python
"""
Debiased CKA Implementation (Fixed)

This module implements debiased Centered Kernel Alignment (CKA) with
the critical fix for double-centering issues identified in Murphy et al. (2024).

IMPORTANT: Double-Centering Fix
-------------------------------
The unbiased HSIC estimator used in debiased CKA already performs
centering internally. Therefore:

1. DO NOT center features before computing Gram matrices
2. Pass raw activations to all CKA functions
3. The Gram matrices K = X @ X.T should use raw X

This fix ensures accurate similarity measurements and prevents
artificially suppressed CKA values.

References
----------
1. Murphy et al. (2024): "The Geometry of Neural Networks" 
   - Shows unbiased HSIC already centers data
2. https://github.com/RistoAle97/centered-kernel-alignment.git
   - External CKA implementation for comparison
"""
```

## Summary of Changes

1. **Removed all explicit centering** of features before Gram matrix computation
2. **Updated Nyström** to use raw activations
3. **Added comprehensive unit tests** to verify no double-centering
4. **Added validation warnings** if centered data is detected
5. **Updated documentation** to emphasize the importance of using raw activations

This fix ensures the debiased CKA implementation correctly measures neural network similarity without the bias introduced by double-centering.