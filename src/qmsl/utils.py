from __future__ import annotations
import json, os, hashlib
from dataclasses import asdict
from typing import Any, Dict, Optional
import numpy as np

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def set_global_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def sha1_of_array(x: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(x).view(np.uint8))
    return h.hexdigest()

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def merge_dicts(*ds: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in ds:
        out.update(d)
    return out
