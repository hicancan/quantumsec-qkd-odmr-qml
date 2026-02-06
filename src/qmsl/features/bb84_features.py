from __future__ import annotations
import numpy as np

BB84_FEATURE_NAMES = [
    "qber", "qber_z", "qber_x",
    "sift_rate", "basis_imbalance",
    "burst", "secret_key_rate"
]

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sig, mu, sig
