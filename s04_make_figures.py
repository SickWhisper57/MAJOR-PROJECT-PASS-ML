"""Step 4 - figures for the report (saved to figures/ as PNG, 300 dpi)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pass_sim import P, dBm

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
GRAY, DARK, ACC = "#9a9a9a", "#3c3c3c", "#1f4e9c"      # baselines grey, proposed blue
E = np.load("results/evaluation.npz")
T = np.load("results/train_report.npz", allow_pickle=True)
J = json.load(open("results/train_report.json"))
D = np.load("data/dataset.npz")
mean_dBm = lambda k: dBm(E[k][:, 0]).mean()


def room(ax, users):
    for x1, y1, x2, y2 in P.walls:
        ax.plot([x1, x2], [y1, y2], color="black", lw=3)
    for cx, cy, w, d in P.obstacles:
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - d / 2), w, d, color="#cfcfcf"))
    for y in P.wg_y:
        ax.plot([0, P.xmax], [y, y], color=GRAY, lw=1)
    ax.scatter(users[:, 0], users[:, 1], marker="*", s=140, color="black", zorder=5, label="users")
    ax.set(xlim=(0, 30), ylim=(0, 30), aspect="equal", xlabel="x [m]", ylabel="y [m]")
    ax.grid(False)


# Fig 1 - example placement: ZF-AO vs ML
fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
for ax, X, title, col in [(axs[0], E["ex_X3"], "Stage 3: ZF-AO from a random start", DARK),
                          (axs[1], E["ex_X5"], "Stage 5: ML-chosen placement", ACC)]:
    room(ax, E["ex_users"])
    for n in range(P.N):
        ax.scatter(X[:, n], np.full(P.M, P.wg_y[n]), s=28, color=col, zorder=4, label="pinching antennas" if n == 0 else None)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
fig.savefig("figures/fig1_placement_example.png"); plt.close(fig)

# Fig 2 - transmit power per scheme
names = ["Stage 2\nrandom+ZF", "Stage 3\nZF-AO\n(random)", "Stage 4\nZF-AO\n(smart)",
         "Nearest\nrule", "Stage 5\nML", "Stage 5+\nML+ZF-AO"]
vals = [mean_dBm(k) for k in ["s2", "s3", "s4", "rule", "s5", "s5p"]]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(names, vals, color=[GRAY] + [DARK] * 3 + [ACC] * 2)
ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylim(-4.5, 18)
ax.set_ylabel("Transmit power for 10 dB SINR [dBm]")
ax.set_title(f"Average transmit power over {len(E['s3'])} unseen user drops")
fig.savefig("figures/fig2_power_per_scheme.png"); plt.close(fig)

# Fig 3 - CDF of transmit power
fig, ax = plt.subplots(figsize=(7, 4.2))
for k, lab, col, ls in [("s2", "Stage 2 random + ZF", GRAY, ":"), ("s3", "Stage 3 ZF-AO (random start)", DARK, "--"),
                        ("s4", "Stage 4 ZF-AO (smart start)", DARK, "-"), ("rule", "Nearest rule + fine search", GRAY, "-"),
                        ("s5", "Stage 5 ML + fine search", ACC, "-"), ("s5p", "Stage 5+ ML + ZF-AO", ACC, "--")]:
    v = np.sort(dBm(E[k][:, 0]))
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=1.8, label=lab)
ax.set(xlabel="Transmit power [dBm]", ylabel="CDF over user drops", title="Distribution of required transmit power")
ax.legend(fontsize=8)
fig.savefig("figures/fig3_power_cdf.png"); plt.close(fig)

# Fig 4 - convergence: power vs number of cost evaluations
fig, ax = plt.subplots(figsize=(7, 4.2))
for k, lab, col, ls in [("random", "ZF-AO from random start", GRAY, "-"), ("smart", "ZF-AO from smart start", DARK, "-"),
                        ("ml", "ZF-AO from ML placement", ACC, "-")]:
    t = E["trace_" + k]
    ax.plot(t[:, :, 0].mean(0), dBm(t[:, :, 1]).mean(0), marker="o", color=col, ls=ls, lw=1.8, ms=4, label=lab)
ax.set(xlabel="Cost evaluations (candidate positions tested)", ylabel="Mean transmit power [dBm]",
       title="Search cost: each marker is one full ZF-AO sweep")
ax.legend(fontsize=8)
fig.savefig("figures/fig4_convergence.png"); plt.close(fig)

# Fig 5 - predicted vs actual
fig, ax = plt.subplots(figsize=(4.8, 4.6))
i = np.random.default_rng(0).choice(len(T["yva"]), 4000, replace=False)
ax.scatter(T["yva"][i], T["pva"][i], s=4, alpha=0.35, color=ACC)
lim = [T["yva"].min(), T["yva"].max()]
ax.plot(lim, lim, color="black", lw=1)
sel = J["selected"]
ax.set(xlabel="Simulated power [dBm]", ylabel="Predicted power [dBm]",
       title=f"{sel}: R$^2$ = {J['models'][sel]['R2']:.2f}, MAE = {J['models'][sel]['MAE_dB']:.2f} dB")
fig.savefig("figures/fig5_predicted_vs_actual.png"); plt.close(fig)

# Fig 6 - feature importance
o = np.argsort(T["importance"])[-12:]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.barh([str(s) for s in T["names"][o]], T["importance"][o], color=DARK)
ax.set(xlabel="Permutation importance (increase in MSE)", title="What the model relies on (top 12 features)")
fig.savefig("figures/fig6_feature_importance.png"); plt.close(fig)

# Fig 7 - quality of the coarse choice + learning curve
Pt = D["test_power"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
m = J["models"][sel]
labs = ["Random\nassignment", "Nearest\nrule", "ML\ntop-1", "ML\ntop-3", "Best of\nall 240"]
v = [Pt.mean(), dBm(E["rule"][:, 0]).mean(), m["top1_dBm"], m["top3_dBm"], Pt.min(1).mean()]
b = axs[0].bar(labs, v, color=[GRAY, DARK, ACC, ACC, GRAY])
axs[0].bar_label(b, fmt="%.2f", padding=2, fontsize=9)
axs[0].axhline(0, color="black", lw=0.8)
axs[0].set(ylabel="Mean transmit power [dBm]", title="Who picks the coarse placement?")
c = T["curve"]
axs[1].plot(c[:, 0], c[:, 1], "o-", color=ACC, label="ML top-1")
axs[1].plot(c[:, 0], c[:, 2], "s--", color=ACC, label="ML top-3")
axs[1].axhline(v[1], color=DARK, ls=":", label="nearest rule")
axs[1].axhline(v[4], color=GRAY, ls=":", label="best of all 240")
axs[1].set(xscale="log", xlabel="Training user drops", ylabel="Mean transmit power [dBm]", title="Learning curve")
axs[1].legend(fontsize=8)
fig.savefig("figures/fig7_choice_quality.png"); plt.close(fig)

# Fig 8 - model comparison
fig, ax = plt.subplots(figsize=(7, 3.8))
mn = list(J["models"])
x = np.arange(len(mn))
b1 = ax.bar(x - 0.2, [J["models"][k]["top1_dBm"] for k in mn], 0.4, color=DARK, label="top-1 pick")
b2 = ax.bar(x + 0.2, [J["models"][k]["top3_dBm"] for k in mn], 0.4, color=ACC, label="best of top-3")
ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2); ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, [f"{k}\nR$^2$={J['models'][k]['R2']:.2f}" for k in mn])
ax.set(ylabel="Mean transmit power [dBm]", title="Comparison of ML models (lower is better)", ylim=(-1.6, 0.35))
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.savefig("figures/fig8_model_comparison.png"); plt.close(fig)
print("8 figures written to figures/")
