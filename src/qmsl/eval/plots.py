from __future__ import annotations
from typing import Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

from .metrics import EvalResult

def save_confusion_matrix(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def save_roc_curve(res: EvalResult, out_path: str, title: str = "ROC Curve") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(res.fpr, res.tpr, label=f"AUC={res.auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def save_scatter(x: np.ndarray, y: np.ndarray, out_path: str, title: str, xlabel: str, ylabel: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def save_line(x: np.ndarray, y: np.ndarray, out_path: str, title: str, xlabel: str, ylabel: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
