"""Supplementary Tables S7-S8: permutation placebo for the precondition of Equation 3, ACS race.

Twenty relabelings of the ACS race partition (subgroup sizes fixed), each evaluated on the same
600 samples of 1000 reference labels as Table 3, with Eq. 5 intervals.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppi, data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
N_SEEDS, N_GOLD, REPLICATES = 20, 1000, 600

y, f, p = data.acs()
seeds, groups = ppi.placebo_sweep(y, f, p["race"], N_SEEDS, N_GOLD, REPLICATES)
pd.DataFrame(seeds).to_csv(os.path.join(OUT, "placebo_acs_race.csv"), index=False)
pd.DataFrame(groups).to_csv(os.path.join(OUT, "placebo_acs_race_groups.csv"), index=False)
summary = ppi.summarize_placebo(seeds)
json.dump(summary, open(os.path.join(OUT, "placebo_acs_race_summary.json"), "w"), indent=1)
print(pd.DataFrame(seeds).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(json.dumps(summary, indent=1))
