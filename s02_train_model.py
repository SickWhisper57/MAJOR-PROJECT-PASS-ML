"""Step 2 - train and compare go-to regressors that predict the transmit
power [dBm] of a candidate assignment from geometry features."""
import json
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from pass_ml import FEATURE_NAMES, features
from s01_generate_dataset import make_trial

d = np.load("data/dataset.npz")
X, y, g = d["X_train"], d["y_train"], d["groups"]
n_drops = g.max() + 1
val = g >= int(0.8 * n_drops)                      # hold out whole drops
Xtr, ytr, Xva, yva = X[~val], y[~val], X[val], y[val]

# test-drop features (for "which assignment does the model pick?")
Ft = [features(make_trial(int(s))) for s in d["test_seeds"]]
Pt = d["test_power"]


def pick_quality(model, top=1):
    """Mean true power [dBm] of the model's choice on the test drops."""
    return float(np.mean([Pt[i, np.argsort(model.predict(Ft[i]))[:top]].min() for i in range(len(Ft))]))


models = {
    "Ridge regression": make_pipeline(StandardScaler(), Ridge(1.0)),
    "k-NN (k=15)": make_pipeline(StandardScaler(), KNeighborsRegressor(15)),
    "Random forest": RandomForestRegressor(150, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "Gradient boosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08, random_state=0),
}
report = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xva)
    report[name] = dict(R2=r2_score(yva, p), MAE_dB=mean_absolute_error(yva, p),
                        top1_dBm=pick_quality(m, 1), top3_dBm=pick_quality(m, 3))
    print(f"{name:18s} R2={report[name]['R2']:.3f}  MAE={report[name]['MAE_dB']:.2f} dB  "
          f"pick top-1={report[name]['top1_dBm']:.2f} dBm  top-3={report[name]['top3_dBm']:.2f} dBm")

best_name = min(report, key=lambda k: report[k]["MAE_dB"])     # chosen on validation drops only
best = models[best_name]
print("selected:", best_name)

# learning curve of the selected model type
curve = []
for nd in [50, 100, 200, 400, 800, int(0.8 * n_drops), n_drops]:
    m = clone(best).fit(X[g < nd], y[g < nd])
    curve.append((nd, pick_quality(m, 1), pick_quality(m, 3)))
    print("  drops", nd, "top-1 %.2f  top-3 %.2f" % curve[-1][1:])

imp = permutation_importance(best, Xva[:20000], yva[:20000], n_repeats=3, random_state=0, n_jobs=-1)
final = clone(best).fit(X, y)       # refit on all training drops
joblib.dump(final, "results/model.joblib")
np.savez("results/train_report.npz", yva=yva, pva=best.predict(Xva), importance=imp.importances_mean,
         names=np.array(FEATURE_NAMES), curve=np.array(curve))
json.dump(dict(models=report, selected=best_name, n_train_drops=int(n_drops), n_rows=int(len(y)),
               reference=dict(oracle_dBm=float(Pt.min(1).mean()), random_dBm=float(Pt.mean()))),
          open("results/train_report.json", "w"), indent=2)
