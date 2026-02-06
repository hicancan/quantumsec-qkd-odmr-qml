from __future__ import annotations
import numpy as np
from .backend_kernel_interface import KernelConfig

def _require_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        return QuantumCircuit, Aer
    except Exception as e:
        raise ImportError("Qiskit + Qiskit Aer are required for qiskit backend kernel.") from e

def _feature_map_qiskit(qc, x, entangle: bool):
    for i, xi in enumerate(x):
        qc.ry(float(xi), i)
    if entangle and qc.num_qubits > 1:
        for i in range(qc.num_qubits - 1):
            qc.cz(i, i+1)

def qiskit_kernel_matrix(X: np.ndarray, Y: np.ndarray, cfg: KernelConfig) -> np.ndarray:
    QuantumCircuit, Aer = _require_qiskit()
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = int(cfg.n_qubits)

    backend = Aer.get_backend("aer_simulator_statevector")

    K = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            qc = QuantumCircuit(n)
            _feature_map_qiskit(qc, Y[j], cfg.entangle)
            # append inverse of U(X[i])
            inv = QuantumCircuit(n)
            _feature_map_qiskit(inv, X[i], cfg.entangle)
            qc.compose(inv.inverse(), inplace=True)
            qc.save_statevector()
            result = backend.run(qc, seed_simulator=int(cfg.seed)).result()
            sv = np.array(result.get_statevector(qc), dtype=complex)
            # prob of |0..0> is |amp0|^2
            K[i, j] = float(np.abs(sv[0])**2)
    return K
