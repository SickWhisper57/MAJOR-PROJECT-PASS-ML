"""
pass_sim.py - Pinching-Antenna System (PASS) indoor simulator, Python port.

Vectorised NumPy port of the MATLAB minor-project code (PASS_Parameters.m,
PASS_Environment.m, PASS_ChannelModel.m, PASS_ZFBeamformer.m, Stage 3/4
ZF-AO, MIMO benchmarks).  Equations follow Wang et al. 2025.

Change with respect to the MATLAB version
-----------------------------------------
Small-scale fading is drawn ONCE per trial (one Rician and one Rayleigh
coefficient per user-waveguide link) and then kept fixed.  The MATLAB
code redrew fading on every call of PASS_ChannelModel, so the position
search compared candidates under different fading and the final SINR was
measured on yet another draw.
"""
import numpy as np


class P:
    """System parameters (same values as PASS_Parameters.m)."""
    fc = 15e9
    lam = 3e8 / fc
    beta0 = 2 * np.pi / lam                 # free-space propagation constant
    ng = 1.4
    betag = 2 * np.pi * ng / lam            # in-waveguide propagation constant
    alpha_g = 1e-3                          # waveguide attenuation [Np/m]
    room_x, room_y = 30.0, 30.0
    N, M, K = 5, 6, 4                       # waveguides, PAs per waveguide, users
    wg_z = 2.8
    wg_y = np.linspace(5, 25, 5)
    xmax, dx = 28.0, 0.1                    # usable length, min PA spacing
    sigma2 = 10 ** (-80 / 10) * 1e-3        # noise power [W]
    gamma_dB = 10.0
    gamma = 10 ** (gamma_dB / 10)           # SINR target
    eta = lam / (4 * np.pi)
    alpha = np.sqrt(0.9 / M)                # equal-power radiation coefficient
    wall_loss_dB = 10.0
    wall_amp = 10 ** (-wall_loss_dB / 20)
    K_rician = 10.0
    walls = np.array([[10, 0, 10, 12], [20, 18, 20, 30],
                      [0, 15, 15, 15], [22, 8, 30, 8]], float)
    obstacles = np.array([[5, 5, 3, 2], [5, 25, 3, 2], [25, 5, 3, 2],
                          [25, 25, 3, 2], [15, 15, 4, 4]], float)


def _blocking_segments():
    segs = [P.walls]
    for cx, cy, w, d in P.obstacles:
        c = np.array([[cx - w / 2, cy - d / 2], [cx + w / 2, cy - d / 2],
                      [cx + w / 2, cy + d / 2], [cx - w / 2, cy + d / 2]])
        segs.append(np.hstack([c, np.roll(c, -1, axis=0)]))
    return np.vstack(segs)


SEGS = _blocking_segments()


def is_los(ux, uy, tx, ty):
    """2-D ray-cast (same test as check_LoS_2D in PASS_Environment.m).
    tx may be an array; returns a boolean array of the same shape."""
    tx = np.asarray(tx, float)[..., None]
    x3, y3, x4, y4 = SEGS.T
    d1x, d1y = tx - ux, ty - uy
    d2x, d2y = x4 - x3, y4 - y3
    den = d1x * d2y - d1y * d2x
    ok = np.abs(den) > 1e-10
    den = np.where(ok, den, 1.0)
    t = ((x3 - ux) * d2y - (y3 - uy) * d2x) / den
    u = ((x3 - ux) * d1y - (y3 - uy) * d1x) / den
    hit = ok & (t > 0) & (t < 1) & (u > 0) & (u < 1)
    return ~hit.any(-1)


class Trial:
    """One Monte-Carlo trial: user positions plus frozen fading."""

    def __init__(self, users, rng=None, fading=True):
        self.users = np.asarray(users, float)
        K = len(self.users)
        if fading and rng is not None:
            mu = np.sqrt(P.K_rician / (P.K_rician + 1))
            sig = np.sqrt(1 / (2 * (P.K_rician + 1)))
            self.f_los = mu + sig * (rng.standard_normal((K, P.N)) + 1j * rng.standard_normal((K, P.N)))
            self.f_nlos = (rng.standard_normal((K, P.N)) + 1j * rng.standard_normal((K, P.N))) / np.sqrt(2)
        else:
            self.f_los = np.ones((K, P.N), complex)
            self.f_nlos = np.ones((K, P.N), complex)


def random_users(rng, K=None):
    K = K or P.K
    u = np.column_stack([2 + (P.room_x - 4) * rng.random(K), 2 + (P.room_y - 4) * rng.random(K)])
    return u[np.argsort(u[:, 0])]           # canonical order: sorted by x


def new_trial(rng, fading=True):
    return Trial(random_users(rng), rng, fading)


# --------------------------------------------------------------- channel
def contrib(x, n, tr):
    """Effective-channel contribution of PAs at positions x (shape S) on
    waveguide n to every user: conj(g_nm) * h_k,nm  ->  S x K."""
    x = np.asarray(x, float)[:, None]
    u = tr.users
    r = np.sqrt((x - u[None, :, 0]) ** 2 + (P.wg_y[n] - u[None, :, 1]) ** 2 + P.wg_z ** 2)
    r = np.maximum(r, 0.1)
    L = np.stack([is_los(u[k, 0], u[k, 1], x[:, 0], P.wg_y[n]) for k in range(len(u))], 1)
    fade = np.where(L, tr.f_los[None, :, n], P.wall_amp * tr.f_nlos[None, :, n])
    h = P.eta * np.exp(-1j * P.beta0 * r) / r * fade                 # Wang eq. 17
    g = P.alpha * np.exp(-P.alpha_g * x) * np.exp(-1j * P.betag * x)  # Wang eq. 25
    return np.conj(g) * h


def psi(X, tr):
    """Effective channel Psi = G^H H  (N x K) for PA positions X (M x N)."""
    return np.stack([contrib(X[:, n], n, tr).sum(0) for n in range(P.N)], 0)


# ----------------------------------------------------------- beamforming
def zf_power(Psi):
    """Minimum ZF transmit power [W]: gamma*sigma2*tr((Psi^H Psi)^-1)
    (Wang eq. 63-64).  Works on a batch of Psi matrices."""
    A = np.swapaxes(Psi.conj(), -1, -2) @ Psi
    A = A + 1e-30 * np.eye(A.shape[-1])
    return P.gamma * P.sigma2 * np.real(np.trace(np.linalg.inv(A), axis1=-2, axis2=-1))


def zf_beamformer(Psi):
    """W = Psi (Psi^H Psi)^-1 P^(1/2)  (Wang eq. 58)."""
    A = Psi.conj().T @ Psi
    return Psi @ np.linalg.inv(A) * np.sqrt(P.gamma * P.sigma2)


def mf_beamformer(Psi, total_power):
    """Matched filter, equal power per user, given total power."""
    W = Psi / np.linalg.norm(Psi, axis=0, keepdims=True)
    return W * np.sqrt(total_power / Psi.shape[1])


def sinr(Psi, W):
    """Per-user SINR (Wang eq. 33)."""
    G = np.abs(Psi.conj().T @ W) ** 2                # G[k,i] = |h_k^H w_i|^2
    sig = np.diag(G)
    return sig / (G.sum(1) - sig + P.sigma2)


def dBm(p):
    return 10 * np.log10(np.asarray(p) * 1e3)


# ------------------------------------------------------------- positions
def project(X):
    """Make an M x N position matrix feasible (sorted, spacing, bounds)."""
    X = np.sort(np.clip(np.array(X, float), 0, P.xmax), 0)
    for m in range(1, P.M):
        X[m] = np.maximum(X[m], X[m - 1] + P.dx)
    return X - np.maximum(X[-1] - P.xmax, 0)[None, :]


def is_feasible(X, tol=1e-9):
    return bool(X.min() >= -tol and X.max() <= P.xmax + tol and (np.diff(X, axis=0) >= P.dx - tol).all())


def random_init(rng):
    X = np.sort(rng.random((P.M, P.N)), 0) * (P.xmax - P.M * P.dx)
    return X + (np.arange(P.M) * P.dx)[:, None]


def smart_init(tr):
    """Stage-4 heuristic of the minor project: PAs at user x-projections."""
    t = np.sort(tr.users[:, 0])
    return project(np.stack([t[np.arange(P.M) % P.K]] * P.N, 1))


def sweep(X, tr, Q, window=None, sweeps=1, counter=None, trace=None):
    """Element-wise 1-D grid search over PA positions (Wang Alg. 3).
    window=None : search the whole feasible interval (baseline ZF-AO)
    window=w    : search only +/- w metres around the current position
    counter[0] accumulates the number of cost evaluations."""
    X = X.copy()
    Psi = psi(X, tr)
    for _ in range(sweeps):
        for n in range(P.N):
            for m in range(P.M):
                lo = 0.0 if m == 0 else X[m - 1, n] + P.dx
                hi = P.xmax if m == P.M - 1 else X[m + 1, n] - P.dx
                if window is not None:
                    lo, hi = max(lo, X[m, n] - window), min(hi, X[m, n] + window)
                if lo >= hi:
                    continue
                grid = np.append(np.linspace(lo, hi, Q), X[m, n])
                base = Psi[n] - contrib(X[m:m + 1, n], n, tr)[0]
                rows = base[None, :] + contrib(grid, n, tr)
                cand = np.repeat(Psi[None], len(grid), 0)
                cand[:, n, :] = rows
                best = int(np.argmin(zf_power(cand)))
                X[m, n], Psi[n] = grid[best], rows[best]
                if counter is not None:
                    counter[0] += len(grid)
        if trace is not None:
            trace.append((counter[0] if counter else 0, float(zf_power(Psi))))
    return X


def zf_ao(tr, X0, Q=50, max_iter=10, tol=1e-3, counter=None, trace=None):
    """Full ZF-AO position optimisation, run to convergence."""
    X, prev = X0, None
    for _ in range(max_iter):
        X = sweep(X, tr, Q, counter=counter, trace=trace)
        cur = float(zf_power(psi(X, tr)))
        if prev is not None and abs(prev - cur) / prev < tol:
            break
        prev = cur
    return X


# ------------------------------------------------------- MIMO benchmarks
def _mimo_channel(n_ant, tr):
    bs = np.array([1.0, P.room_y / 2, 3.0])
    ax = bs[0] + np.arange(n_ant) * P.lam / 2
    u = tr.users
    r = np.sqrt((ax[:, None] - u[None, :, 0]) ** 2 + (bs[1] - u[None, :, 1]) ** 2 + bs[2] ** 2)
    H = P.eta * np.exp(-1j * P.beta0 * r) / np.maximum(r, 0.1)
    L = np.array([is_los(u[k, 0], u[k, 1], np.array([bs[0]]), bs[1])[0] for k in range(len(u))])
    return H * np.where(L, tr.f_los[:, 0], P.wall_amp * tr.f_nlos[:, 0])[None, :]


def conventional_mimo_power(tr):
    """Fully-digital MIMO, N antennas, ZF (same walls and fading model)."""
    return float(zf_power(_mimo_channel(P.N, tr)))


def massive_mimo_power(tr):
    """Sub-connected hybrid massive MIMO, N*M antennas, N RF chains."""
    H = _mimo_channel(P.N * P.M, tr)
    W_rf = np.zeros((P.N * P.M, P.N), complex)
    for n in range(P.N):
        idx = slice(n * P.M, (n + 1) * P.M)
        k = int(np.argmax((np.abs(H[idx]) ** 2).sum(0)))
        W_rf[idx, n] = np.exp(1j * np.angle(H[idx, k])) / np.sqrt(P.M)
    return float(zf_power(W_rf.conj().T @ H))
