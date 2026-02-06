import numpy as np
import pytest

from qmsl.kernels.backend_kernel_interface import kernel_matrix, KernelConfig

def _have(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False

@pytest.mark.skipif(not (_have("pennylane") and _have("qiskit") and _have("qiskit_aer") and _have("cirq")),
                    reason="Quantum backends not installed in this environment.")
@pytest.mark.parametrize("entangle", [False, True])
def test_pl_qiskit_cirq_kernel_close(entangle):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(8, 4))
    cfg = KernelConfig(n_qubits=4, entangle=entangle, shots=8192, seed=0)

    K_pl = kernel_matrix(X, X, backend="pennylane", cfg=cfg)
    K_qk = kernel_matrix(X, X, backend="qiskit", cfg=cfg)
    K_cq = kernel_matrix(X, X, backend="cirq", cfg=cfg)

    assert np.max(np.abs(K_pl - K_qk)) < 1e-6
    assert np.max(np.abs(K_pl - K_cq)) < 1e-6

@pytest.mark.skipif(not _have("pennylane"), reason="PennyLane not installed.")
def test_analytic_matches_pl_when_product_embedding():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(10, 4))
    cfg = KernelConfig(n_qubits=4, entangle=False, seed=0)

    K_an = kernel_matrix(X, X, backend="analytic", cfg=cfg)
    K_pl = kernel_matrix(X, X, backend="pennylane", cfg=cfg)
    assert np.max(np.abs(K_an - K_pl)) < 1e-6
