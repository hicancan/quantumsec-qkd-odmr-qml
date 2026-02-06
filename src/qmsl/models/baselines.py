from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

from sklearn.metrics import roc_auc_score

@dataclass(frozen=True)
class ThresholdModel:
    feature_index: int = 0  # by default qber
    threshold: float = 0.11

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = X[:, self.feature_index]
        # proba of class 1
        p1 = (s >= self.threshold).astype(float)
        # return Nx2
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.feature_index] >= self.threshold).astype(int)

def fit_threshold_by_auc(X: np.ndarray, y: np.ndarray, feature_index: int = 0) -> ThresholdModel:
    # brute search thresholds
    s = X[:, feature_index]
    ts = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    best_auc = -1.0
    best_t = float(np.median(s))
    for t in ts:
        p = (s >= t).astype(float)
        auc = roc_auc_score(y, p)
        if auc > best_auc:
            best_auc = auc
            best_t = float(t)
    return ThresholdModel(feature_index=feature_index, threshold=best_t)
