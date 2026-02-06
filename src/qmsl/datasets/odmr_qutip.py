from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np

try:
    import qutip as qt
except Exception as e:  # pragma: no cover
    qt = None  # type: ignore

try:
    import sympy as sp
except Exception as e:  # pragma: no cover
    sp = None  # type: ignore

def _require_qutip():
    if qt is None:
        raise ImportError("QuTiP is required for ODMR simulation. Install qutip.")

@dataclass(frozen=True)
class ODMRConfig:
    D: float = 2.87e9          # zero-field splitting (Hz)
    E: float = 0.0             # strain (Hz) simplified
    gamma: float = 28e9        # gyromagnetic ratio (Hz/T) ~ 28 GHz/T
    linewidth: float = 2.5e6   # Lorentzian HWHM (Hz)
    contrast: float = 0.02     # ODMR contrast depth

def spin_1_ops():
    _require_qutip()
    # QuTiP has jmat for spin operators
    Sx = qt.jmat(1, 'x')
    Sy = qt.jmat(1, 'y')
    Sz = qt.jmat(1, 'z')
    return Sx, Sy, Sz

def nv_ground_hamiltonian(Bz_T: float, cfg: ODMRConfig) -> "qt.Qobj":
    """Simplified NV ground-state Hamiltonian for spin-1:
    H = D Sz^2 + E(Sx^2 - Sy^2) + gamma * Bz * Sz
    """
    _require_qutip()
    Sx, Sy, Sz = spin_1_ops()
    H = cfg.D * (Sz * Sz) + cfg.E * (Sx*Sx - Sy*Sy) + cfg.gamma * float(Bz_T) * Sz
    return H

def transition_frequencies(Bz_T: float, cfg: ODMRConfig) -> Tuple[float, float]:
    """Return two main transition frequencies (approx) from ms=0 to ms=±1 as eigen-energy differences."""
    _require_qutip()
    H = nv_ground_hamiltonian(Bz_T, cfg)
    evals = np.array(H.eigenenergies(), dtype=float)
    # Sort energies; for spin-1 we have 3 levels
    evals.sort()
    # Transition frequencies: between middle and extremes (heuristic)
    f1 = abs(evals[1] - evals[0])
    f2 = abs(evals[2] - evals[1])
    return float(f1), float(f2)

def lorentzian(f: np.ndarray, f0: float, gamma: float) -> np.ndarray:
    return (gamma**2) / ((f - f0)**2 + gamma**2)

def simulate_odmr_spectrum(
    freqs: np.ndarray,
    Bz_T: float,
    cfg: ODMRConfig,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an ODMR spectrum (normalized fluorescence) with two Lorentzian dips."""
    f1, f2 = transition_frequencies(Bz_T, cfg)
    # ODMR dip model: baseline 1.0 minus contrast * sum Lorentzians (normalized)
    L = lorentzian(freqs, f1, cfg.linewidth) + lorentzian(freqs, f2, cfg.linewidth)
    L = L / max(L.max(), 1e-12)
    signal = 1.0 - cfg.contrast * L
    noise = rng.normal(0.0, noise_std, size=freqs.shape)
    return (signal + noise).astype(float)

def generate_odmr_dataset(
    n_per_class: int,
    n_freq: int,
    f_center: float,
    span: float,
    B0_T: float,
    B1_T: float,
    noise_std: float,
    rng: np.random.Generator,
    cfg: Optional[ODMRConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Binary dataset: class0 uses B0_T, class1 uses B1_T."""
    _require_qutip()
    if cfg is None:
        cfg = ODMRConfig(linewidth=2.5e6)
    freqs = np.linspace(f_center - span/2, f_center + span/2, int(n_freq), dtype=float)

    X0 = np.vstack([simulate_odmr_spectrum(freqs, B0_T, cfg, noise_std, rng) for _ in range(int(n_per_class))])
    X1 = np.vstack([simulate_odmr_spectrum(freqs, B1_T, cfg, noise_std, rng) for _ in range(int(n_per_class))])
    X = np.vstack([X0, X1])
    y = np.array([0]*int(n_per_class) + [1]*int(n_per_class), dtype=int)
    meta = {"freqs": freqs, "Bz_T": np.array([B0_T]*int(n_per_class) + [B1_T]*int(n_per_class), dtype=float)}
    return X, y, meta

def sympy_verify_eigs_example() -> str:
    """Symbolic sanity snippet: verify eigenvalues for the simplified spin-1 Hamiltonian with B along z.
    This produces a short text (not full derivation) suitable for logs.
    """
    if sp is None:
        return "SymPy not available."
    # Represent Sz for spin-1 in the standard basis: diag(1,0,-1)
    D, gamma, B = sp.symbols('D gamma B', positive=True, real=True)
    Sz = sp.Matrix([[1,0,0],[0,0,0],[0,0,-1]])
    H = D*(Sz*Sz) + gamma*B*Sz
    eigs = H.eigenvals()
    return f"SymPy eigvals of H=D*Sz^2 + gamma*B*Sz: {eigs}"
