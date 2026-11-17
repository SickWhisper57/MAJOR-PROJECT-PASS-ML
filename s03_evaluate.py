"""Step 3 - compare every scheme on the unseen test drops.

Stage 1  random positions + matched filter
Stage 2  random positions + ZF
Stage 3  ZF-AO position search from a random start      (Wang Alg. 3)
Stage 4  ZF-AO position search from the smart start      (minor project)
Rule     nearest-waveguide rule + fine search            (no ML)
Stage 5  ML-chosen placement + fine search               (proposed)
Stage 5+ Stage 5 followed by ZF-AO sweeps (warm start)
"""
import os
# One numerical thread per worker process. The layouts are already spread over
# all processor cores, so letting every worker start its own thread pool for the
# model prediction makes the processes compete and inflates the measured time.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
import joblib
from multiprocessing import Pool, cpu_count
from pass_sim import (P, psi, zf_power, zf_beamformer, mf_beamformer, sinr, dBm, random_init, smart_init,
                      zf_ao, sweep, is_feasible)
from pass_ml import ml_placement, nearest_assignment, refine_assignment, fine_search
from s01_generate_dataset import make_trial

WARM_SWEEPS = 2
TRACE_SWEEPS = 6
MODEL = None


def run(seed):
    global MODEL
    if MODEL is None:
        MODEL = joblib.load("results/model.joblib")
        MODEL.predict(np.zeros((1, MODEL.n_features_in_)))
    tr = make_trial(seed)
    rng = np.random.default_rng(seed + 1)
    out = {}

    def rec(name, fn):
        c = [0]
        t0 = time.perf_counter()
        X = fn(c)
        dt = time.perf_counter() - t0
        Ps = psi(X, tr)
        assert is_feasible(X), name
        out[name] = (float(zf_power(Ps)), c[0], dt)
        return X

    X0 = random_init(rng)
    Ps0 = psi(X0, tr)
    out["sinr_s1"] = 10 * np.log10(sinr(Ps0, mf_beamformer(Ps0, zf_power(Ps0))))
    out["s2"] = (float(zf_power(Ps0)), 0, 0.0)
    X3 = rec("s3", lambda c: zf_ao(tr, X0, counter=c))
    rec("s4", lambda c: zf_ao(tr, smart_init(tr), counter=c))
    rec("rule", lambda c: refine_assignment(tr, nearest_assignment(tr), c)[1])
    X5 = rec("s5", lambda c: ml_placement(tr, MODEL, top=3, counter=c)[0])
    rec("s5p", lambda c: sweep(ml_placement(tr, MODEL, top=3, counter=c)[0], tr, 50, sweeps=WARM_SWEEPS, counter=c))
    out["sinr_s5"] = 10 * np.log10(sinr(psi(X5, tr), zf_beamformer(psi(X5, tr))))

    # convergence traces: power after each full ZF-AO sweep
    tr_out = {}
    for name, Xi, c0 in [("random", X0, 0), ("smart", smart_init(tr), 0), ("ml", X5, out["s5"][1])]:
        c, t = [c0], [(c0, float(zf_power(psi(Xi, tr))))]
        sweep(Xi, tr, 50, sweeps=TRACE_SWEEPS, counter=c, trace=t)
        tr_out[name] = np.array(t)
    out["trace"] = tr_out
    out["example"] = (tr.users, X3, X5)
    return out


if __name__ == "__main__":
    seeds = [int(s) for s in np.load("data/dataset.npz")["test_seeds"]]
    with Pool(cpu_count()) as pool:
        R = pool.map(run, seeds, chunksize=4)
    keys = ["s2", "s3", "s4", "rule", "s5", "s5p"]
    res = {k: np.array([r[k] for r in R]) for k in keys}          # columns: power W, evals, time s
    save = {k: res[k] for k in keys}
    save.update(sinr_s1=np.array([r["sinr_s1"] for r in R]), sinr_s5=np.array([r["sinr_s5"] for r in R]),
                ex_users=R[0]["example"][0], ex_X3=R[0]["example"][1], ex_X5=R[0]["example"][2])
    for name in ["random", "smart", "ml"]:
        save["trace_" + name] = np.array([r["trace"][name] for r in R])
    np.savez("results/evaluation.npz", **save)

    labels = dict(s2="Stage 2  random + ZF", s3="Stage 3  ZF-AO (random start)", s4="Stage 4  ZF-AO (smart start)",
                  rule="Rule     nearest waveguide + fine", s5="Stage 5  ML + fine search",
                  s5p=f"Stage 5+ ML + {WARM_SWEEPS} ZF-AO sweeps")
    lines = [f"Test drops: {len(seeds)} | SINR target {P.gamma_dB:.0f} dB | power = mean of dBm values",
             f"{'scheme':36s}{'mean dBm':>9s}{'median':>8s}{'evals':>8s}{'time ms':>9s}"]
    for k in keys:
        p = dBm(res[k][:, 0])
        lines.append(f"{labels[k]:36s}{p.mean():9.2f}{np.median(p):8.2f}{res[k][:, 1].mean():8.0f}{1e3 * res[k][:, 2].mean():9.1f}")
    lines.append(f"Stage 1 (random + MF) mean per-user SINR: {save['sinr_s1'].mean():.2f} dB")
    lines.append(f"Stage 5 per-user SINR min/max: {save['sinr_s5'].min():.3f} / {save['sinr_s5'].max():.3f} dB")
    open("results/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))