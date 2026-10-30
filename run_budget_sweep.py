"""Figure 3 and Supplementary Table S5: ACS race across reference budgets.

The Table 3 ACS predictor and frame; 600 replicates per budget (seed 0), Eq. 5 intervals. Also
reports the count-planning budget of Supplementary bound S1.4 for m0 = 3 and beta = 0.05.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppi, data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BUDGETS, REPLICATES, ALPHA = (500, 1000, 2000, 5000, 10000, 20000), 600, 0.05

y, f, p = data.acs(); g = p["race"]
keys = ppi.retained_groups(g)
Na = [int((g == k).sum()) for k in keys]
n_plan = ppi.count_planning_budget(Na, len(g), beta=0.05)
rows = []
for n in BUDGETS:
    sim = ppi.simulate(y, f, g, n, REPLICATES, seed=0)
    row = dict(n=n, shortfall_bound=min(1.0, ppi.count_shortfall_probability(Na, len(g), n)))
    for arm in ["pooled", "ppi", "ref"]:
        s = ppi.summarize(sim, arm, ALPHA, "t", True)
        sb = ppi.summarize(sim, arm, ALPHA, "t", True, bonferroni=True)
        row.update({f"worst_{arm}": s["marginal"].min(), f"mean_{arm}": s["marginal"].mean(),
                    f"min_available_{arm}": s["p_available"].min(),
                    f"worst_cond_{arm}": np.nanmin(s["conditional"]),
                    f"simul_bonf_{arm}": sb["simultaneous"],
                    f"width_{arm}": float(np.nanmean(s["mean_width"]))})
    w = ppi.summarize(sim, "ppi")["mean_width"] / ppi.summarize(sim, "ref")["mean_width"]
    row["ratio_ppi_ref_median"] = float(np.nanmedian(w))
    rows.append(row); print(f"n={n} done", flush=True)

S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "acs_budget_sweep.csv"), index=False)
with open(os.path.join(OUT, "acs_count_planning.txt"), "w") as fh:
    fh.write(f"ACS race: K={len(keys)}, N={len(g)}, smallest N_a={min(Na)} (pi={min(Na)/len(g):.5f})\n")
    fh.write(f"expected-count floor m0/pi_min = {3*len(g)/min(Na):.0f}\n")
    fh.write(f"count-planning budget (bound S1.4, m0=3, beta=0.05) = {n_plan}\n")
print(open(os.path.join(OUT, "acs_count_planning.txt")).read())
print(S[["n", "shortfall_bound", "worst_pooled", "worst_ppi", "worst_ref", "min_available_ppi",
         "worst_cond_ppi", "simul_bonf_ppi", "simul_bonf_ref", "ratio_ppi_ref_median"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
