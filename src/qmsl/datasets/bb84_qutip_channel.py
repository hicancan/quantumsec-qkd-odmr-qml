from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np

try:
    import qutip as qt
except Exception as e:  # pragma: no cover
    qt = None  # type: ignore

from .bb84_numpy import h2, BB84SessionFeatures

def _require_qutip():
    if qt is None:
        raise ImportError("QuTiP is required for bb84_qutip_channel backend. Install qutip.")

def _ket0():
    return qt.basis(2, 0)

def _ket1():
    return qt.basis(2, 1)

def _ket_plus():
    return (qt.basis(2, 0) + qt.basis(2, 1)).unit()

def _ket_minus():
    return (qt.basis(2, 0) - qt.basis(2, 1)).unit()

def _projectors(basis: int):
    # basis: 0->Z, 1->X
    if basis == 0:
        k0, k1 = _ket0(), _ket1()
    else:
        k0, k1 = _ket_plus(), _ket_minus()
    P0 = k0 * k0.dag()
    P1 = k1 * k1.dag()
    return P0, P1, k0, k1

def _apply_depolarizing(rho: "qt.Qobj", p: float) -> "qt.Qobj":
    # rho -> (1-p)rho + p I/2
    I = qt.qeye(2)
    return (1.0 - p) * rho + p * (I / 2.0)

def _apply_phase_damping(rho: "qt.Qobj", gamma: float) -> "qt.Qobj":
    # simple dephasing: off-diagonals shrink by (1-gamma)
    # Kraus: K0 = sqrt(1-gamma) I, K1 = sqrt(gamma) Z in a toy model
    Z = qt.sigmaz()
    K0 = np.sqrt(1.0 - gamma) * qt.qeye(2)
    K1 = np.sqrt(gamma) * Z
    return K0 * rho * K0.dag() + K1 * rho * K1.dag()

def _measure_in_basis(rho: "qt.Qobj", basis: int, rng: np.random.Generator) -> Tuple[int, "qt.Qobj"]:
    P0, P1, k0, k1 = _projectors(basis)
    p0 = float((P0 * rho).tr().real)
    p0 = min(max(p0, 0.0), 1.0)
    bit = 0 if rng.random() < p0 else 1
    if bit == 0:
        post = (P0 * rho * P0) / max(p0, 1e-12)
    else:
        p1 = 1.0 - p0
        post = (P1 * rho * P1) / max(p1, 1e-12)
    return bit, post

def _prepare_state(bit: int, basis: int) -> "qt.Qobj":
    P0, P1, k0, k1 = _projectors(basis)
    ket = k0 if bit == 0 else k1
    return ket * ket.dag()

def simulate_bb84_session_qutip(
    n_bits: int,
    p_eve: float,
    p_depol: float,
    p_dephase: float,
    rng: np.random.Generator,
    eve_threshold: float = 0.3,
) -> Tuple[BB84SessionFeatures, int, Dict[str, float]]:
    """
    QuTiP-based BB84 simulator with density matrices + toy depolarizing & dephasing noise.

    This is deliberately small-scale (slower than numpy). Use it for *physics-backed validation*.
    Returns (features, label, meta).
    """
    _require_qutip()
    n_bits = int(n_bits)
    a_bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    a_bases = rng.integers(0, 2, size=n_bits, dtype=np.int8)

    b_bases = rng.integers(0, 2, size=n_bits, dtype=np.int8)

    intercept = rng.random(n_bits) < float(p_eve)
    e_bases = rng.integers(0, 2, size=n_bits, dtype=np.int8)

    # raw records for sifting
    a_sift = []
    b_sift = []
    b_basis_sift = []

    for i in range(n_bits):
        a_bit = int(a_bits[i])
        a_basis = int(a_bases[i])
        rho = _prepare_state(a_bit, a_basis)

        # Eve intercept-resend
        in_basis = a_basis
        if bool(intercept[i]):
            e_basis = int(e_bases[i])
            e_bit, _post = _measure_in_basis(rho, e_basis, rng)
            rho = _prepare_state(e_bit, e_basis)
            in_basis = e_basis

        # Channel noise (physics-ish)
        rho = _apply_depolarizing(rho, float(p_depol))
        rho = _apply_phase_damping(rho, float(p_dephase))

        # Bob measurement
        b_basis = int(b_bases[i])
        b_bit, _ = _measure_in_basis(rho, b_basis, rng)

        # Sifting by comparing Bob basis with Alice basis (standard BB84)
        if b_basis == a_basis:
            a_sift.append(a_bit)
            b_sift.append(b_bit)
            b_basis_sift.append(b_basis)

    sift_rate = len(a_sift) / max(1, n_bits)
    if len(a_sift) == 0:
        feats = BB84SessionFeatures(0.0, 0.0, 0.0, float(sift_rate), 0.0, 0.0, 0.0)
    else:
        a_sift_arr = np.array(a_sift, dtype=np.int8)
        b_sift_arr = np.array(b_sift, dtype=np.int8)
        basis_arr = np.array(b_basis_sift, dtype=np.int8)

        err = (a_sift_arr != b_sift_arr).astype(np.int8)
        qber = float(err.mean())

        z_mask = (basis_arr == 0)
        x_mask = (basis_arr == 1)
        qber_z = float(err[z_mask].mean()) if z_mask.any() else qber
        qber_x = float(err[x_mask].mean()) if x_mask.any() else qber

        z_cnt = int(z_mask.sum())
        x_cnt = int(x_mask.sum())
        basis_imbalance = abs(z_cnt - x_cnt) / max(1, z_cnt + x_cnt)

        burst = float((err[1:] * err[:-1]).mean()) if err.size >= 2 else 0.0

        key_fraction = max(0.0, 1.0 - 2.0 * h2(qber))
        skr = float(sift_rate) * key_fraction
        feats = BB84SessionFeatures(qber, qber_z, qber_x, float(sift_rate), float(basis_imbalance), float(burst), float(skr))

    label = 1 if float(p_eve) >= eve_threshold else 0
    meta = {"p_eve": float(p_eve), "p_depol": float(p_depol), "p_dephase": float(p_dephase)}
    return feats, label, meta
