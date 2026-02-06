from __future__ import annotations
from functools import lru_cache
from typing import Optional
import numpy as np

from .backend_kernel_interface import KernelConfig

def _require_pl():
    try:
        import pennylane as qml
        return qml
    except Exception as e:
        raise ImportError("PennyLane is required for pennylane backend kernel.") from e

def _feature_map(qml, x: np.ndarray, wires, entangle: bool):
    # Angle embedding using RY per feature; optional CZ chain entanglement
    for i, w in enumerate(wires):
        qml.RY(float(x[i]), wires=w)
    if entangle and len(wires) > 1:
        for i in range(len(wires) - 1):
            qml.CZ(wires=[wires[i], wires[i+1]])

def pennylane_kernel_matrix(X: np.ndarray, Y: np.ndarray, cfg: KernelConfig) -> np.ndarray:
    qml = _require_pl()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_qubits = int(cfg.n_qubits)
    wires = list(range(n_qubits))

    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def kernel_prob_zero(x, y):
        _feature_map(qml, y, wires, cfg.entangle)
        qml.adjoint(lambda z: _feature_map(qml, z, wires, cfg.entangle))(x)  # U(x)^\dagger
        return qml.probs(wires=wires)

    K = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            probs = kernel_prob_zero(X[i], Y[j])
            K[i, j] = float(probs[0])
    return K
