"""Table 4 and Supplementary Table S6: prediction-model interventions on ACS race at n = 5000.

Same finite-frame design as Table 3 (reference sample drawn from the frame, target = frame
subgroup rate, Eq. 5 intervals), 600 replicates for each of 5 predictor-fitting seeds. Within a
seed every arm receives the identical reference samples (common random numbers).

  pooled, base predictor        HGB fit on 15,000 records
  pooled, 4x fitting data       HGB fit on 60,000 records (the Table 3 predictor when seed = 0)
  pooled, MLP                   multilayer perceptron with race indicators as inputs
  pooled, post-hoc offset       base predictor + per-race mean residual from a 2,000-record audit
                                sample; the offset's own estimation error is not propagated
  subgroup PPI, base predictor  Eq. 2
  reference-only                no predictor
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.neural_network import MLPClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppi, data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
N_GOLD, REPLICATES, ALPHA, SEEDS = 5000, 600, 0.05, (0, 1, 2, 3, 4)

X, y, g, perm = data.acs_arrays()
small, big, frame = perm[:data.ACS_N_FIT_SMALL], perm[:data.ACS_N_FIT], perm[data.ACS_N_FIT:]
yp, gp = y[frame], g[frame]
keys = ppi.retained_groups(gp)


def auc(f):
    r = rankdata(f); n1 = (yp == 1).sum(); n0 = (yp == 0).sum()
    return (r[yp == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def overall_coverage(sim):
    """Overall-rate PPI interval (all records) under Eq. 5; identical for every arm sharing a predictor."""
    se = ppi.design_se(sim["sRp"], sim["n_gold"], sim["N"], fpc=True)
    crit = float(ppi.critical_value(ALPHA, sim["n_gold"], "t"))
    return float((np.abs(sim["fbar_overall"] + sim["mRp"] - sim["theta_overall"]) <= crit * se).mean())


rows = []
for seed in SEEDS:
    t0 = time.time()
    f0 = data.acs_predictor(X, y, small, seed).predict_proba(X[frame])[:, 1]
    f1 = data.acs_predictor(X, y, big, seed).predict_proba(X[frame])[:, 1]
    cats = sorted(set(g[small].tolist()))
    onehot = lambda gg: pd.get_dummies(pd.Categorical(gg, categories=cats)).values.astype(float)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=60, random_state=seed)
    mlp.fit(np.column_stack([X[small], onehot(g[small])]), y[small])
    f2 = mlp.predict_proba(np.column_stack([X[frame], onehot(gp)]))[:, 1]
    audit = np.random.default_rng(300 + seed).choice(len(yp), 2000, replace=False)
    f3 = f0.copy()
    for k in keys:
        ma = audit[gp[audit] == k]
        if len(ma) >= 3:
            f3[gp == k] += (yp[ma] - f0[ma]).mean()
    arms = [("pooled correction, base predictor", f0, "pooled"),
            ("pooled correction, 4x fitting data", f1, "pooled"),
            ("pooled correction, MLP predictor", f2, "pooled"),
            ("pooled correction, post-hoc offset", f3, "pooled"),
            ("subgroup PPI, base predictor", f0, "ppi"),
            ("reference-only", f0, "ref")]
    for label, f, arm in arms:
        sim = ppi.simulate(yp, f, gp, N_GOLD, REPLICATES, seed=seed)
        s = ppi.summarize(sim, arm, ALPHA, "t", True)
        sb = ppi.summarize(sim, arm, ALPHA, "t", True, bonferroni=True)
        rows.append(dict(arm=label, seed=seed, auc=np.nan if arm == "ref" else auc(f),
                         overall=np.nan if arm == "ref" else overall_coverage(sim),
                         mean=s["marginal"].mean(), worst=s["marginal"].min(),
                         min_available=s["p_available"].min(), simul_bonf=sb["simultaneous"],
                         mean_width=float(np.nanmean(s["mean_width"]))))
    print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)

D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "ablation_by_seed.csv"), index=False)
order = list(dict.fromkeys(D.arm))
A = (D.groupby("arm", sort=False)
       .agg(auc=("auc", "mean"), overall=("overall", "mean"), mean=("mean", "mean"),
            worst=("worst", "mean"), worst_min=("worst", "min"), worst_max=("worst", "max"),
            min_available=("min_available", "mean"), simul_bonf=("simul_bonf", "mean"),
            mean_width=("mean_width", "mean"))
       .reindex(order).reset_index())
A.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
print(f"\nACS race, K={len(keys)}, n={N_GOLD}, {len(SEEDS)} seeds x {REPLICATES} replicates, Eq. 5 intervals")
print(A.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
