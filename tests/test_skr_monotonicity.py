import numpy as np
from qmsl.datasets.bb84_numpy import h2

def test_skr_key_fraction_monotonic():
    # key_fraction = max(0, 1 - 2H2(Q))
    qs = np.linspace(1e-6, 0.49, 200)
    key_frac = np.maximum(0.0, 1.0 - 2.0*np.array([h2(q) for q in qs]))
    # should be non-increasing
    dif = np.diff(key_frac)
    assert np.all(dif <= 1e-6)
