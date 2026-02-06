from __future__ import annotations
import numpy as np
from .backend_kernel_interface import KernelConfig

def _require_cudaq():
    try:
        import cudaq
        return cudaq
    except Exception as e:
        raise ImportError("CUDA-Q is required for cudaq backend kernel.") from e

def cudaq_kernel_matrix(X: np.ndarray, Y: np.ndarray, cfg: KernelConfig) -> np.ndarray:
    """
    CUDA-Q kernel matrix estimated by sampling (stochastic).

    Kernel definition:
      K(x,y) = P(0..0) after applying U(y) then U(x)† on |0..0>.

    Feature map U(·):
      Ry(angle_i) on each qubit + optional CZ chain entanglement.
    """
    cudaq = _require_cudaq()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = int(cfg.n_qubits)
    shots = int(cfg.shots)

    # Prefer deterministic CPU simulator if available
    try:
        cudaq.set_target("qpp-cpu")
    except Exception:
        pass
    # Try set random seed if supported
    try:
        cudaq.set_random_seed(int(cfg.seed))  # not always available
    except Exception:
        pass

    if bool(cfg.entangle):
        @cudaq.kernel
        def kernel_circuit(x_params: list[float], y_params: list[float]):
            q = cudaq.qvector(n)
            # U(y)
            for k in range(n):
                ry(y_params[k], q[k])
            if n > 1:
                for k in range(n - 1):
                    cz(q[k], q[k + 1])
            # U(x)†
            if n > 1:
                for k in reversed(range(n - 1)):
                    cz(q[k], q[k + 1])
            for k in range(n):
                ry(-x_params[k], q[k])
            mz(q)
    else:
        @cudaq.kernel
        def kernel_circuit(x_params: list[float], y_params: list[float]):
            q = cudaq.qvector(n)
            for k in range(n):
                ry(y_params[k], q[k])
            for k in range(n):
                ry(-x_params[k], q[k])
            mz(q)

    K = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
    zero = "0" * n
    for i in range(X.shape[0]):
        x = X[i].tolist()
        for j in range(Y.shape[0]):
            y = Y[j].tolist()
            # API may accept shots_count; keep consistent with common CUDA-Q usage
            counts = cudaq.sample(kernel_circuit, x, y, shots_count=shots)
            c0 = int(counts.get(zero, 0))
            K[i, j] = c0 / float(shots)

    return K
