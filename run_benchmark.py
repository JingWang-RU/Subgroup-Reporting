"""Table 3, Figure 2 and Supplementary Tables S1-S3: the 11-partition benchmark at n = 1000.

For every partition: 600 overall simple random samples of 1000 reference labels (seed 0), the
three estimators on identical labels, and two interval rules reduced from the same draws:
  eq5       Studentized t_{n_a-1} with the finite-population correction (Eq. 5)  -> manuscript
  original  fixed z = 1.959964, no FPC (the ML4H-era analysis)                -> reconciliation
The "original" reduction of the pooled and subgroup arms is asserted equal to the previously
reported numbers (PPI_treatment/results/main_experiment.csv) before anything is written.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppi, data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RAW = os.path.join(OUT, "replicates"); os.makedirs(RAW, exist_ok=True)
N_GOLD, REPLICATES, ALPHA = 1000, 600, 0.05
PREVIOUS = os.environ.get("PREVIOUS_MAIN",
    "/*/results/main_experiment.csv")


def split_half_stability(y, f, g, keys):
    """Split-half correlation of Delta_a (the Table 3 asterisk); identical to the original analysis."""
    ok = np.isfinite(y) & np.isfinite(f); y, f, g = y[ok], f[ok], g[ok]
    R = y - f; N = len(y)
    pm = np.random.default_rng(5).permutation(N); h1, h2 = pm[:N // 2], pm[N // 2:]
    D1 = np.array([R[h1][g[h1] == k].mean() - R[h1].mean() for k in keys])
    D2 = np.array([R[h2][g[h2] == k].mean() - R[h2].mean() for k in keys])
    return float(np.corrcoef(D1, D2)[0, 1]) if len(keys) > 2 else float(np.sign(D1[0] * D2[0]))


prows, grows, rrows = [], [], []
for name, y, f, g in data.partitions():
    t0 = time.time()
    sim = ppi.simulate(y, f, g, N_GOLD, REPLICATES, seed=0)
    np.savez_compressed(os.path.join(RAW, f"{name.replace('|', '__')}.npz"), **sim)
    keys = list(sim["keys"]); K = len(keys)
    S = {(arm, rule, bonf): ppi.summarize(sim, arm, ALPHA, dist, fpc, bonf)
         for arm in ["pooled", "ppi", "ref"]
         for rule, dist, fpc in [("eq5", "t", True), ("original", "z_original", False), ("t_nofpc", "t", False)]
         for bonf in [False, True]}
    e = lambda arm, bonf=False: S[(arm, "eq5", bonf)]
    o = lambda arm: S[(arm, "original", False)]
    ratio_g = e("ppi")["mean_width"] / e("ref")["mean_width"]
    resource, attr = name.split("|")
    prows.append(dict(
        name=name, resource=resource, partition=attr, N=sim["N"], K=K, n_gold=N_GOLD,
        stability=split_half_stability(np.asarray(y, float), np.asarray(f, float), np.asarray(g), keys),
        worst_pooled=e("pooled")["marginal"].min(), worst_ppi=e("ppi")["marginal"].min(),
        worst_ref=e("ref")["marginal"].min(),
        mean_pooled=e("pooled")["marginal"].mean(), mean_ppi=e("ppi")["marginal"].mean(),
        mean_ref=e("ref")["marginal"].mean(),
        min_available=e("ppi")["p_available"].min(),
        worst_cond_ppi=np.nanmin(e("ppi")["conditional"]), worst_cond_ref=np.nanmin(e("ref")["conditional"]),
        simul_pooled=e("pooled")["simultaneous"], simul_pooled_bonf=e("pooled", True)["simultaneous"],
        simul_ppi=e("ppi")["simultaneous"], simul_ppi_bonf=e("ppi", True)["simultaneous"],
        simul_ref=e("ref")["simultaneous"], simul_ref_bonf=e("ref", True)["simultaneous"],
        bonf_width_cost_ppi=float(np.nanmean(e("ppi", True)["mean_width"]) / np.nanmean(e("ppi")["mean_width"])),
        ratio_ppi_ref_median=float(np.nanmedian(ratio_g)), ratio_ppi_ref_min=float(np.nanmin(ratio_g)),
        ratio_ppi_ref_max=float(np.nanmax(ratio_g)),
        ratio_ppi_pooled=float(np.nanmean(e("ppi")["mean_width"]) / e("pooled")["mean_width"][0]),
        # the ML4H-era reduction of the same draws, for the reconciliation table
        orig_worst_pooled=o("pooled")["marginal"].min(), orig_worst_ppi=o("ppi")["marginal"].min(),
        orig_mean_pooled=o("pooled")["marginal"].mean(), orig_mean_ppi=o("ppi")["marginal"].mean(),
        orig_ratio_ppi_pooled=float(np.nanmean(o("ppi")["mean_width"]) / o("pooled")["mean_width"][0]),
        worst_ppi_t_nofpc=S[("ppi", "t_nofpc", False)]["marginal"].min(),
        fpc_width_factor=float(np.nanmean(e("ppi")["mean_width"]) / np.nanmean(S[("ppi", "t_nofpc", False)]["mean_width"])),
    ))
    for j, k in enumerate(keys):
        grows.append(dict(
            name=name, group=str(k), N_a=int(sim["Na"][j]), pi=sim["Na"][j] / sim["N"],
            expected_n_a=N_GOLD * sim["Na"][j] / sim["N"], theta=sim["truth"][j], Delta=sim["Delta"][j],
            rho=sim["rho"][j], pop_ratio=sim["S_R"][j] / sim["S_Y"][j],
            p_available=e("ppi")["p_available"][j],
            cov_pooled=e("pooled")["marginal"][j], cov_ppi=e("ppi")["marginal"][j], cov_ref=e("ref")["marginal"][j],
            cond_ppi=e("ppi")["conditional"][j], cond_ref=e("ref")["conditional"][j],
            w_pooled=e("pooled")["mean_width"][j], w_ppi=e("ppi")["mean_width"][j], w_ref=e("ref")["mean_width"][j],
            ratio_ppi_ref=ratio_g[j],
            zero_width_ppi=e("ppi")["zero_width"][j], zero_width_ref=e("ref")["zero_width"][j],
            orig_cov_pooled=o("pooled")["marginal"][j], orig_cov_ppi=o("ppi")["marginal"][j],
        ))
    print(f"{name:18s} K={K} done in {time.time() - t0:.1f}s", flush=True)

P = pd.DataFrame(prows); G = pd.DataFrame(grows)

# ---- reproduction guard: the original rule must reproduce the previously reported numbers exactly
prev = pd.read_csv(PREVIOUS).set_index("name")
for _, r in P.iterrows():
    q = prev.loc[r["name"]]
    for a, b in [("orig_worst_pooled", "worst_pooled"), ("orig_worst_ppi", "worst_group"),
                 ("orig_mean_pooled", "mean_pooled"), ("orig_mean_ppi", "mean_group"),
                 ("orig_ratio_ppi_pooled", "width_ratio"), ("stability", "delta_stability")]:
        assert abs(r[a] - q[b]) < 1e-9, (r["name"], a, r[a], q[b])
print("reproduction guard passed: original-rule reduction equals the previously reported table")

P.to_csv(os.path.join(OUT, "benchmark_partitions.csv"), index=False)
G.to_csv(os.path.join(OUT, "benchmark_groups.csv"), index=False)
cols = ["name", "K", "worst_pooled", "worst_ppi", "worst_ref", "ratio_ppi_ref_median",
        "simul_pooled_bonf", "simul_ppi", "simul_ppi_bonf", "min_available", "worst_cond_ppi"]
print(P[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("wrote results/benchmark_partitions.csv, results/benchmark_groups.csv, results/replicates/*.npz")
