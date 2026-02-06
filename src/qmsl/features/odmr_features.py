from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from sklearn.decomposition import PCA

@dataclass
class ODMRFeatureConfig:
    n_components: int = 4

def odmr_to_features(X_spectrum: np.ndarray, cfg: ODMRFeatureConfig) -> Tuple[np.ndarray, PCA]:
    """Reduce high-dimensional spectra to a compact feature vector (default PCA->4 dims)."""
    pca = PCA(n_components=int(cfg.n_components), random_state=0)
    X_red = pca.fit_transform(X_spectrum)
    return X_red.astype(float), pca
