from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve
)

@dataclass(frozen=True)
class EvalResult:
    auc: float
    acc: float
    cm: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thr: np.ndarray

def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> EvalResult:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return EvalResult(auc=auc, acc=acc, cm=cm, fpr=fpr, tpr=tpr, thr=thr)

def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # find minimal fpr where tpr >= target
    idx = np.where(tpr >= target_tpr)[0]
    if idx.size == 0:
        return float("nan")
    return float(np.min(fpr[idx]))
