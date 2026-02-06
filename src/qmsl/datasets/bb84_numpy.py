from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np

def h2(p: float) -> float:
    """Binary entropy H2(p)."""
    eps = 1e-12
    p = min(max(float(p), eps), 1.0 - eps)
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

@dataclass(frozen=True)
class BB84SessionFeatures:
    qber: float
    qber_z: float
    qber_x: float
    sift_rate: float
    basis_imbalance: float
    burst: float
    secret_key_rate: float

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.qber, self.qber_z, self.qber_x,
            self.sift_rate, self.basis_imbalance,
            self.burst, self.secret_key_rate
        ], dtype=float)

def simulate_bb84_session(
    n_bits: int,
    p_eve: float,
    p_chan: float,
    rng: np.random.Generator,
    basis_bias: float = 0.5,
) -> BB84SessionFeatures:
    """
    Simulate one BB84 session with intercept-resend eavesdropping and bit-flip channel noise.

    - Alice chooses random bit and basis
    - Eve intercepts each signal with probability p_eve, measures in random basis, resends
    - Channel flips bit with probability p_chan
    - Bob measures in random basis
    - Sifting keeps positions where Alice basis == Bob basis
    Returns interpretable security/measurement statistics.
    """
    n_bits = int(n_bits)
    p_eve = float(p_eve)
    p_chan = float(p_chan)
    basis_bias = float(basis_bias)

    # Alice bits
    a_bit = rng.integers(0, 2, size=n_bits, dtype=np.int8)

    # Alice bases (biased coin optional)
    a_basis = (rng.random(n_bits) >= basis_bias).astype(np.int8)  # 0 with prob=basis_bias, 1 otherwise

    # Eve intercept mask
    intercept = rng.random(n_bits) < p_eve
    e_basis = rng.integers(0, 2, size=n_bits, dtype=np.int8)

    e_random_bit = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    e_bit = np.where(e_basis == a_basis, a_bit, e_random_bit).astype(np.int8)

    in_bit = np.where(intercept, e_bit, a_bit).astype(np.int8)
    in_basis = np.where(intercept, e_basis, a_basis).astype(np.int8)

    # Channel noise: bit flip
    flip = rng.random(n_bits) < p_chan
    in_bit_noisy = np.where(flip, 1 - in_bit, in_bit).astype(np.int8)

    # Bob bases
    b_basis = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    b_random_bit = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    b_bit = np.where(b_basis == in_basis, in_bit_noisy, b_random_bit).astype(np.int8)

    # Sifting: keep where Bob basis == Alice basis
    sift_mask = (b_basis == a_basis)
    sift_rate = float(sift_mask.mean())

    a_sift = a_bit[sift_mask]
    b_sift = b_bit[sift_mask]
    b_basis_sift = b_basis[sift_mask]

    if a_sift.size == 0:
        # Degenerate case: no matches
        return BB84SessionFeatures(0.0, 0.0, 0.0, sift_rate, 0.0, 0.0, 0.0)

    err = (a_sift != b_sift).astype(np.int8)
    qber = float(err.mean())

    z_mask = (b_basis_sift == 0)
    x_mask = (b_basis_sift == 1)
    qber_z = float(err[z_mask].mean()) if z_mask.any() else qber
    qber_x = float(err[x_mask].mean()) if x_mask.any() else qber

    z_cnt = int(z_mask.sum())
    x_cnt = int(x_mask.sum())
    basis_imbalance = abs(z_cnt - x_cnt) / max(1, z_cnt + x_cnt)

    if err.size < 2:
        burst = 0.0
    else:
        # fraction of adjacent pairs both errors (simple burstiness proxy)
        burst = float((err[1:] * err[:-1]).mean())

    key_fraction = max(0.0, 1.0 - 2.0 * h2(qber))
    secret_key_rate = sift_rate * key_fraction

    return BB84SessionFeatures(qber, qber_z, qber_x, sift_rate, basis_imbalance, burst, secret_key_rate)

def generate_bb84_dataset(
    n_sessions: int,
    n_bits: int,
    rng: np.random.Generator,
    p_eve_range: Tuple[float, float] = (0.0, 0.8),
    p_chan_range: Tuple[float, float] = (0.0, 0.08),
    eve_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate a labeled dataset of BB84 sessions.

    Label rule (ground truth):
      y=1 (Eve) if p_eve >= eve_threshold else 0 (benign)

    Returns:
      X: (n_sessions, 7) features
      y: (n_sessions,) labels
      meta: dict containing p_eve, p_chan for analysis
    """
    n_sessions = int(n_sessions)
    X = np.zeros((n_sessions, 7), dtype=float)
    y = np.zeros((n_sessions,), dtype=int)
    p_eves = np.zeros((n_sessions,), dtype=float)
    p_chans = np.zeros((n_sessions,), dtype=float)

    for i in range(n_sessions):
        p_eve = float(rng.uniform(*p_eve_range))
        p_chan = float(rng.uniform(*p_chan_range))
        feats = simulate_bb84_session(n_bits=n_bits, p_eve=p_eve, p_chan=p_chan, rng=rng)
        X[i] = feats.as_vector()
        y[i] = 1 if p_eve >= eve_threshold else 0
        p_eves[i] = p_eve
        p_chans[i] = p_chan

    meta = {"p_eve": p_eves, "p_chan": p_chans}
    return X, y, meta
