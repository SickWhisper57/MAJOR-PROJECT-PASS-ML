"""Sanity checks: python test_sanity.py"""
import numpy as np
from pass_sim import P, new_trial, random_init, zf_ao, psi, zf_beamformer, zf_power, sinr, is_feasible, is_los
from pass_ml import ASSIGNMENTS, features, FEATURE_NAMES, refine_assignment, nearest_assignment

rng = np.random.default_rng(1)
tr = new_trial(rng)
X = zf_ao(tr, random_init(rng))
Ps = psi(X, tr); W = zf_beamformer(Ps)
assert is_feasible(X), "positions violate spacing/bounds"
assert np.allclose(10 * np.log10(sinr(Ps, W)), P.gamma_dB, atol=1e-6), "ZF must hit the SINR target exactly"
assert np.isclose((abs(W) ** 2).sum(), zf_power(Ps)), "power formula mismatch"
assert np.allclose(psi(X, tr), psi(X, tr)), "channel must be repeatable within a trial (frozen fading)"
assert not is_los(5.0, 5.0, np.array([15.0]), 5.0)[0] and is_los(12.0, 20.0, np.array([15.0]), 20.0)[0]
assert len(ASSIGNMENTS) == 240 and features(tr).shape == (240, len(FEATURE_NAMES))
p, Xr = refine_assignment(tr, nearest_assignment(tr))
assert is_feasible(Xr) and np.isfinite(p)
print("all sanity checks passed")
