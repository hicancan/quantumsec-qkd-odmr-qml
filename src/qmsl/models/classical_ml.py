from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

@dataclass
class ClassicalResult:
    name: str
    model: Any

def train_classical_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, ClassicalResult]:
    models: Dict[str, ClassicalResult] = {}
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train, y_train)
    models["logreg"] = ClassicalResult("logreg", lr)

    rbf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced")
    rbf.fit(X_train, y_train)
    models["svm_rbf"] = ClassicalResult("svm_rbf", rbf)

    rf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced_subsample")
    rf.fit(X_train, y_train)
    models["rf"] = ClassicalResult("rf", rf)

    return models
