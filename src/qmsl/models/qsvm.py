from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from sklearn.svm import SVC

from qmsl.kernels.backend_kernel_interface import kernel_matrix, KernelConfig, BackendName

@dataclass
class QSVMResult:
    backend: str
    cfg: KernelConfig
    clf: SVC

def train_qsvm_precomputed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    backend: BackendName,
    cfg: KernelConfig,
    C: float = 2.0,
) -> tuple[QSVMResult, np.ndarray]:
    """Train QSVM with a precomputed kernel. Returns (model, proba_test)."""
    K_train = kernel_matrix(X_train, X_train, backend=backend, cfg=cfg)
    K_test = kernel_matrix(X_test, X_train, backend=backend, cfg=cfg)

    clf = SVC(kernel="precomputed", C=float(C), probability=True, class_weight="balanced", random_state=int(cfg.seed))
    clf.fit(K_train, y_train)
    proba_test = clf.predict_proba(K_test)[:, 1]
    return QSVMResult(backend=str(backend), cfg=cfg, clf=clf), proba_test
