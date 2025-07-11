🔧 NYSTRÖM IMPLEMENTATION FIX PLAN

  📋 IDENTIFIED CRITICAL ISSUES

  1. Rank Oversampling: Using n_landmarks > rank(K) causes numerical instability
  2. Poor Rank Reduction: Standard Nyström doesn't handle rank reduction optimally
  3. Non-PSD Approximations: Current method produces negative eigenvalues
  4. Inverse Conditioning: SVD regularization is insufficient for numerical stability
  5. Landmark Selection: Random sampling doesn't guarantee good spectral coverage

  🎯 MATHEMATICAL FIXES REQUIRED

  1. Adaptive Rank-Aware Landmark Selection

  - Problem: Current implementation uses fixed n_landmarks regardless of intrinsic rank
  - Solution: Dynamically determine effective rank and limit landmarks accordingly
  - Formula: n_landmarks_effective = min(n_landmarks, rank(K), n_samples-1)

  2. QR-Decomposition Based Nyström (Improved Method)

  - Problem: Standard Nyström K ≈ C W⁻¹ C^T has poor rank reduction
  - Solution: Use QR decomposition for better landmark matrix factorization
  - Reference: "Improved fixed-rank Nyström approximation via QR decomposition"

  3. Spectral Regularization

  - Problem: SVD regularization alone insufficient for rank-deficient cases
  - Solution: Add spectral regularization based on effective rank
  - Method: Use eigenvalue thresholding instead of simple epsilon regularization

  4. Positive Semidefinite Projection

  - Problem: Approximated kernels have negative eigenvalues
  - Solution: Project approximation back to PSD cone
  - Method: Eigen-decomposition + negative eigenvalue clipping

  5. Condition Number Monitoring

  - Problem: No early warning for numerical instability
  - Solution: Monitor condition numbers and adapt regularization dynamically

  🛠️ IMPLEMENTATION PLAN

  Phase 1: Mathematical Foundation Fixes

  1. Rank Detection: Implement robust rank estimation for input matrices
  2. Adaptive Landmarks: Modify landmark selection to respect effective rank
  3. QR-based Approximation: Replace standard formula with QR-decomposition approach

  Phase 2: Numerical Stability Improvements

  1. Spectral Regularization: Implement eigenvalue-based regularization
  2. PSD Projection: Add positive semidefinite constraint enforcement
  3. Condition Monitoring: Add runtime stability checks

  Phase 3: Performance Optimization

  1. Smart Sampling: Implement spectral-aware landmark selection
  2. Adaptive Refinement: Add iterative improvement mechanism
  3. Memory Efficiency: Optimize for large-scale applications

  Phase 4: Validation & Testing

  1. Mathematical Properties: Verify PSD, symmetry, and approximation quality
  2. Convergence Tests: Ensure accuracy improves with more landmarks (up to rank limit)
  3. Benchmark Validation: Test against corrected CKA benchmark data

  🔬 SPECIFIC MATHEMATICAL CORRECTIONS

  Current (Broken) Formula:

  K_mm = X_landmarks @ X_landmarks.T  # [n_landmarks, n_landmarks]
  K_nm = X @ X_landmarks.T            # [n_samples, n_landmarks]  
  K_mm_inv = SVD_regularize(K_mm)
  K_approx = K_nm @ K_mm_inv @ K_nm.T

  Corrected Formula (QR-based):

  # Step 1: Rank-aware landmark selection
  effective_rank = min(torch.linalg.matrix_rank(X @ X.T), n_landmarks, X.shape[0]-1)
  landmarks = select_spectral_landmarks(X, effective_rank)

  # Step 2: QR-based approximation
  Q, R = torch.linalg.qr(K_nm)  # QR decomposition
  K_approx = Q @ Q.T @ K_exact @ Q @ Q.T  # Project to landmark subspace

  # Step 3: PSD projection
  eigenvals, eigenvecs = torch.linalg.eigh(K_approx)
  eigenvals_clipped = torch.clamp(eigenvals, min=0)  # Remove negative eigenvalues
  K_approx = eigenvecs @ torch.diag(eigenvals_clipped) @ eigenvecs.T

  📊 SUCCESS CRITERIA

  1. CKA(X,X) ≈ 1.0 for self-similarity
  2. Monotonic improvement with more landmarks (up to effective rank)
  3. Positive semidefinite approximated kernels (no negative eigenvalues)
  4. R² > 0.9 on rotation benchmark (vs current R² = -4.11)
  5. MAE < 0.05 on all validation tests

  ⚠️ CRITICAL CONSTRAINTS

  1. DO NOT MODIFY the working exact CKA implementation
  2. Maintain API compatibility with existing NystromCKA interface
  3. Preserve computational efficiency (don't make it slower than necessary)
  4. Robust error handling for edge cases (small matrices, rank deficiency)

  This plan addresses all identified mathematical and numerical issues while providing a clear implementation roadmap. The fixes are based on recent research in
  low-rank matrix approximation and should restore the Nyström method to its intended accuracy and reliability.