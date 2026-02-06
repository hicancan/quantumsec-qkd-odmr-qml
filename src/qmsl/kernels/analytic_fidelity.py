from __future__ import annotations
import numpy as np

def product_fidelity_kernel_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Analytic fidelity kernel for a product-state angle embedding:
      |psi(x)> = ⊗_i (cos(x_i/2)|0> + sin(x_i/2)|1>)
      K(x,y) = |<psi(x)|psi(y)>|^2 = ∏_i cos^2((x_i - y_i)/2)
    This is PSD and can be used as an SVM kernel.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    # shape: (nX, nY, d)
    dif = X[:, None, :] - Y[None, :, :]
    K = np.cos(dif / 2.0) ** 2
    return np.prod(K, axis=-1)
