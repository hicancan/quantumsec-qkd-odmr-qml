from __future__ import annotations
import numpy as np
from .backend_kernel_interface import KernelConfig

def _require_cirq():
    try:
        import cirq
        return cirq
    except Exception as e:
        raise ImportError("Cirq is required for cirq backend kernel.") from e

def _feature_map_cirq(cirq, qubits, x, entangle: bool):
    ops = []
    for q, xi in zip(qubits, x):
        ops.append(cirq.ry(float(xi))(q))
    if entangle and len(qubits) > 1:
        for i in range(len(qubits)-1):
            ops.append(cirq.CZ(qubits[i], qubits[i+1]))
    return ops

def cirq_kernel_matrix(X: np.ndarray, Y: np.ndarray, cfg: KernelConfig) -> np.ndarray:
    cirq = _require_cirq()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = int(cfg.n_qubits)
    qubits = [cirq.LineQubit(i) for i in range(n)]
    sim = cirq.Simulator(seed=int(cfg.seed))

    K = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            circuit = cirq.Circuit()
            circuit.append(_feature_map_cirq(cirq, qubits, Y[j], cfg.entangle))
            # append inverse of feature map for X[i]
            inv_ops = _feature_map_cirq(cirq, qubits, X[i], cfg.entangle)
            circuit.append(cirq.inverse(inv_ops))
            result = sim.simulate(circuit)
            sv = np.array(result.final_state_vector, dtype=complex)
            K[i, j] = float(np.abs(sv[0])**2)
    return K
