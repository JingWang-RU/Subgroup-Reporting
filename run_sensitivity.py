"""Supplementary Table S4: the three subgroups excluded by the 100-record retention rule.

Re-runs the two affected partitions with every recorded category retained (SUPPORT2 race adds
Asian and missing; ACS race adds Alaska Native alone), same seed and Eq. 5 intervals.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppi, data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
REPLICATES = 600
y2, f2, p2 = data.support2()
ya, fa, pa = data.acs()
rows = []
for name, y, f, g, budgets in [("SUPPORT2|race", y2, f2, p2["race"], (1000,)),
                               ("ACS|race", ya, fa, pa["race"], (1000, 10000))]:
    for n in budgets:
        sim = ppi.simulate(y, f, g, n, REPLICATES, seed=0, min_group=1)
        S = {arm: ppi.summarize(sim, arm) for arm in ["pooled", "ppi", "ref"]}
        Sb = ppi.summarize(sim, "ppi", bonferroni=True)
        for j, k in enumerate(sim["keys"]):
            rows.append(dict(name=name, n=n, K_all=len(sim["keys"]), group=str(k), N_a=int(sim["Na"][j]),
                             retained=bool(sim["Na"][j] >= ppi.MIN_GROUP),
                             expected_n_a=n * sim["Na"][j] / sim["N"], p_available=S["ppi"]["p_available"][j],
                             cov_pooled=S["pooled"]["marginal"][j], cov_ppi=S["ppi"]["marginal"][j],
                             cov_ref=S["ref"]["marginal"][j], cond_ppi=S["ppi"]["conditional"][j],
                             simul_ppi_bonf_all=Sb["simultaneous"]))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "sensitivity_all_groups.csv"), index=False)
print(D[~D.retained].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(D.groupby(["name", "n"]).agg(worst_ppi=("cov_ppi", "min"), simul=("simul_ppi_bonf_all", "first")))
