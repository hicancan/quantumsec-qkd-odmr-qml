from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Callable, Dict, Any
import numpy as np

BackendName = Literal["analytic", "pennylane", "qiskit", "cirq", "cudaq"]

@dataclass(frozen=True)
class KernelConfig:
    n_qubits: int = 4
    entangle: bool = True
    shots: int = 4096   # used by sampling backends (cudaq)
    seed: int = 0

def assert_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X

def clip_and_pad_features(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """Ensure feature dimension equals n_qubits by truncation/padding."""
    X = assert_2d(X)
    d = X.shape[1]
    if d == n_qubits:
        return X
    if d > n_qubits:
        return X[:, :n_qubits]
    # pad with zeros
    pad = np.zeros((X.shape[0], n_qubits - d), dtype=float)
    return np.hstack([X, pad])

def kernel_matrix(X: np.ndarray, Y: np.ndarray, backend: BackendName, cfg: KernelConfig) -> np.ndarray:
    X = clip_and_pad_features(X, cfg.n_qubits)
    Y = clip_and_pad_features(Y, cfg.n_qubits)

    if backend == "analytic":
        from .analytic_fidelity import product_fidelity_kernel_matrix
        # For analytic kernel we require entangle=False to match the product embedding assumption.
        if cfg.entangle:
            raise ValueError("Analytic kernel assumes product-state embedding; set entangle=False.")
        return product_fidelity_kernel_matrix(X, Y)

    if backend == "pennylane":
        from .kernel_pennylane import pennylane_kernel_matrix
        return pennylane_kernel_matrix(X, Y, cfg)

    if backend == "qiskit":
        from .kernel_qiskit import qiskit_kernel_matrix
        return qiskit_kernel_matrix(X, Y, cfg)

    if backend == "cirq":
        from .kernel_cirq import cirq_kernel_matrix
        return cirq_kernel_matrix(X, Y, cfg)

    if backend == "cudaq":
        from .kernel_cudaq import cudaq_kernel_matrix
        return cudaq_kernel_matrix(X, Y, cfg)

    raise ValueError(f"Unknown backend: {backend}")
