from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GlobalConfig:
    seed: int = 0
    # BB84 defaults
    bb84_sessions: int = 1500
    bb84_bits_per_session: int = 400
    bb84_p_eve_min: float = 0.0
    bb84_p_eve_max: float = 0.8
    bb84_p_chan_min: float = 0.0
    bb84_p_chan_max: float = 0.08

    # ODMR defaults
    odmr_samples_per_class: int = 400
    odmr_n_freq: int = 80
    odmr_f0: float = 2.80e9
    odmr_span: float = 20e6
    odmr_linewidth: float = 2.5e6

    # Kernel defaults
    kernel_n_qubits: int = 4
    kernel_entangle: bool = True
    kernel_shots_cudaq: int = 4096
