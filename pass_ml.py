"""
pass_ml.py - Stage 5: ML-assisted pinching-antenna placement.

Idea ("learn coarse, search fine")
----------------------------------
1. Coarse decision: every waveguide clusters its M antennas above one
   user.  An assignment a = (a_1..a_N), a_n in {user indices}, with every
   user served by at least one waveguide (240 options for N=5, K=4).
2. A regression model (gradient boosting / random forest) predicts the
   transmit power [dBm] an assignment will need, from geometry and the
   large-scale link gains.
3. At run time all assignments are scored by the model, the best `top`
   are refined with a short local grid search (phase alignment), and the
   one with the lowest true power is kept.
"""
import itertools
import numpy as np
from pass_sim import P, project, sweep, psi, zf_power, dBm, is_los

ASSIGNMENTS = np.array([a for a in itertools.product(range(P.K), repeat=P.N) if len(set(a)) == P.K])
_OFFSETS = (np.arange(P.M) - (P.M - 1) / 2) * P.dx
FINE_WINDOW, FINE_Q = 0.05, 20          # +/- 5 cm, 20 grid points


def cluster_positions(tr, a):
    """All M antennas of waveguide n packed (spacing dx) above user a[n]."""
    return project(np.stack([tr.users[a[n], 0] + _OFFSETS for n in range(P.N)], 1))


def fine_search(tr, X, sweeps=1, counter=None):
    return sweep(X, tr, FINE_Q, window=FINE_WINDOW, sweeps=sweeps, counter=counter)


def refine_assignment(tr, a, counter=None):
    """Cluster + one fine sweep. Returns (power_dBm, X)."""
    X = fine_search(tr, cluster_positions(tr, a), counter=counter)
    return float(dBm(zf_power(psi(X, tr)))), X


def geometry(tr):
    """dy[n,k]: horizontal offset waveguide n - user k;  L[n,k]: 1 if the
    point of waveguide n directly above user k's x has line of sight."""
    u = tr.users
    dy = np.abs(P.wg_y[:, None] - u[None, :, 1])
    L = np.array([[is_los(u[k, 0], u[k, 1], np.array([u[k, 0]]), P.wg_y[n])[0]
                   for k in range(P.K)] for n in range(P.N)], float)
    return dy, L


FEATURE_NAMES = (
    [f"dy_wg{n+1}_to_its_user" for n in range(P.N)] +
    [f"los_wg{n+1}_to_its_user" for n in range(P.N)] +
    [f"n_wg_serving_user{k+1}" for k in range(P.K)] +
    [f"best_dy_user{k+1}" for k in range(P.K)] +
    [f"any_los_user{k+1}" for k in range(P.K)] +
    [f"sorted_best_dy_{k+1}" for k in range(P.K)] +
    [f"dx_wg{n+1}_to_nearest_other_user" for n in range(P.N)] +
    ["n_users_with_los", "sum_best_dy", "max_best_dy"] +
    [f"gain_dB_wg{n+1}_to_its_user" for n in range(P.N)] +
    [f"best_gain_dB_user{k+1}" for k in range(P.K)] +
    [f"sorted_best_gain_dB_{k+1}" for k in range(P.K)] +
    ["worst_user_gain_dB", "sum_best_gain_dB"]
)


def link_gain_dB(tr, L):
    """Large-scale link gain [dB] of waveguide n towards user k (blockage
    loss and the frozen fading amplitude), i.e. what the base station
    would measure from pilots.  Shape N x K."""
    amp = np.where(L > 0, np.abs(tr.f_los.T), P.wall_amp * np.abs(tr.f_nlos.T))
    return 20 * np.log10(amp)


def features(tr, assignments=ASSIGNMENTS, geo=None):
    """Feature matrix, one row per candidate assignment."""
    dy, L = geo if geo is not None else geometry(tr)
    gdb = link_gain_dB(tr, L)
    u = tr.users
    rows, ar = [], np.arange(P.N)
    for a in assignments:
        d_as, l_as = dy[ar, a], L[ar, a]
        cnt = np.bincount(a, minlength=P.K)
        best = np.array([d_as[a == k].min() for k in range(P.K)])
        anyl = np.array([l_as[a == k].max() for k in range(P.K)])
        dxo = np.abs(u[a, 0][:, None] - u[None, :, 0])
        dxo[ar, a] = 99.0
        g_as = gdb[ar, a]
        gbest = np.array([g_as[a == k].max() for k in range(P.K)])
        rows.append(np.r_[d_as, l_as, cnt, best, anyl, np.sort(best), dxo.min(1),
                          anyl.sum(), best.sum(), best.max(),
                          g_as, gbest, np.sort(gbest), gbest.min(), gbest.sum()])
    return np.array(rows)


def nearest_assignment(tr, geo=None):
    """Non-ML rule: each waveguide serves the user closest in y, then
    repair so that every user is served."""
    dy = (geo or geometry(tr))[0]
    a = dy.argmin(1)
    for k in range(P.K):
        if k not in a:
            cnt = np.bincount(a, minlength=P.K)
            cand = [n for n in range(P.N) if cnt[a[n]] > 1]
            a[min(cand, key=lambda n: dy[n, k])] = k
    return a


def ml_placement(tr, model, top=3, counter=None):
    """Stage 5. Returns (X, power_dBm, chosen assignment)."""
    order = np.argsort(model.predict(features(tr)))[:top]
    res = [refine_assignment(tr, ASSIGNMENTS[i], counter) + (ASSIGNMENTS[i],) for i in order]
    p, X, a = min(res, key=lambda r: r[0])
    return X, p, a
