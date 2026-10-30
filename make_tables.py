"""Write every manuscript and supplementary table, and results/numbers.json, from results/*.csv.

No number in a table is typed by hand. Run after the four run_*.py scripts.
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MT = os.path.join(ROOT, "manuscript", "tables"); ST = os.path.join(ROOT, "supplement", "tables")
os.makedirs(MT, exist_ok=True); os.makedirs(ST, exist_ok=True)
PREV_ABLATION = os.environ.get("PREVIOUS_ABLATION",
    "/*/results/ablation.csv")

P = pd.read_csv(os.path.join(RES, "benchmark_partitions.csv"))
G = pd.read_csv(os.path.join(RES, "benchmark_groups.csv"))
A = pd.read_csv(os.path.join(RES, "ablation.csv"))
AS = pd.read_csv(os.path.join(RES, "ablation_by_seed.csv"))
SW = pd.read_csv(os.path.join(RES, "acs_budget_sweep.csv"))
SE = pd.read_csv(os.path.join(RES, "sensitivity_all_groups.csv"))
PL = pd.read_csv(os.path.join(RES, "placebo_acs_race.csv"))
PG = pd.read_csv(os.path.join(RES, "placebo_acs_race_groups.csv"))
ORDER = ["ACS|race", "CheXpert|device", "CheXpert|sex", "RHC|income", "RHC|ninsclas", "RHC|race",
         "RHC|sex", "SUPPORT2|dzgroup", "SUPPORT2|income", "SUPPORT2|race", "SUPPORT2|sex"]
P = P.set_index("name").loc[ORDER].reset_index()
ACS_RACE = {"1": "White alone", "2": "Black alone", "3": "American Indian alone", "4": "Alaska Native alone",
            "5": "AIAN, tribe specified", "6": "Asian alone", "7": "NHPI alone", "8": "Some other race",
            "9": "Two or more races"}
PART = {"dzgroup": "disease group", "ninsclas": "insurance class", "device": "support devices"}
f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"


def split(name):
    r, a = name.split("|")
    return r, PART.get(a, a)


def esc(s):
    return str(s).replace("&", r"\&").replace("$", r"\$").replace("%", r"\%").replace(">", r"$>$").replace("<", r"$<$")


def label(name):
    r, a = split(name)
    star = r"$^{\ast}$" if P.set_index("name").loc[name, "stability"] < 0.3 else ""
    return r, a + star


def group_label(name, g):
    return ACS_RACE.get(str(g), str(g)) if name.startswith("ACS") else esc(g)


def write(path, s):
    open(path, "w").write(s.strip() + "\n")


# ------------------------------------------------------------------ Table 3 (main)
lines = []
for _, r in P.iterrows():
    res, part = label(r["name"])
    lines.append(f"\t{res} & {part} & {r.K} & {f3(r.worst_pooled)} & \\textbf{{{f3(r.worst_ppi)}}} & {f3(r.worst_ref)} "
                 f"& {f2(r.ratio_ppi_ref_median)} & {f2(r.ratio_ppi_ref_min)} & {f2(r.ratio_ppi_ref_max)} "
                 f"& {f3(r.simul_pooled_bonf)} & {f3(r.simul_ppi_bonf)} \\\\")
write(os.path.join(MT, "tab_main.tex"), r"""
\begin{tabular}{llrcccccccc}
	\toprule
	& & & \multicolumn{3}{c}{Worst-subgroup coverage} & \multicolumn{3}{c}{\begin{tabular}[b]{@{}c@{}}Width ratio,\\ PPI/reference-only\end{tabular}} & \multicolumn{2}{c}{\begin{tabular}[b]{@{}c@{}}Simultaneous coverage,\\ Bonferroni\end{tabular}} \\
	\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}
	Resource & Partition & $K$ & Pooled & Subgroup PPI & Reference-only & Median & Min & Max & Pooled & Subgroup PPI \\
	\midrule
""" + "\n".join(lines) + r"""
	\bottomrule
\end{tabular}""")

# ------------------------------------------------------------------ Table 4 (ablation)
names = {"pooled correction, base predictor": "Pooled correction, base predictor",
         "pooled correction, 4x fitting data": r"\quad 4$\times$ predictor fitting data",
         "pooled correction, MLP predictor": r"\quad MLP predictor with race inputs",
         "pooled correction, post-hoc offset": r"\quad post-hoc subgroup offset",
         "subgroup PPI, base predictor": r"\textbf{Subgroup PPI}, base predictor",
         "reference-only": "Reference-only (no predictor)"}
variant = {"pooled correction, base predictor": ("Pooled correction", "Base predictor"),
           "pooled correction, 4x fitting data": ("Pooled correction", r"4$\times$ predictor fitting data"),
           "pooled correction, MLP predictor": ("Pooled correction", "MLP predictor with race inputs"),
           "pooled correction, post-hoc offset": ("Pooled correction", "Post-hoc subgroup offset"),
           "subgroup PPI, base predictor": (r"\textbf{Subgroup PPI}", "Base predictor"),
           "reference-only": ("Reference-only", "No predictor")}
lines = []
for _, r in A.iterrows():
    auc = "--" if np.isnan(r.auc) else f3(r.auc)
    ov = "--" if np.isnan(r.overall) else f3(r.overall)
    est, var = variant[r.arm]
    lines.append(f"\t{est} & {var} & {auc} & {ov} & {f3(r['mean'])} & {f3(r.worst)} & {f3(r.mean_width)} \\\\")
write(os.path.join(MT, "tab_ablation.tex"), r"""
\begin{tabular}{llccccc}
	\toprule
	& & & \multicolumn{3}{c}{Coverage of nominal 95\% intervals} & Mean subgroup \\
	\cmidrule(lr){4-6}
	Estimator & Predictor variant & AUC & Overall rate & Mean subgroup & Worst subgroup & interval width \\
	\midrule
""" + "\n".join(lines) + r"""
	\bottomrule
\end{tabular}""")

# ---------------------------------------------------------------- S1: per-group results (longtable)
lines = []
for name in ORDER:
    d = G[G.name == name]
    res, part = split(name)
    lines.append(r"\multicolumn{11}{l}{\textit{" + f"{res}, {part}" + r"}}\\")
    for _, r in d.iterrows():
        lines.append(f"\\quad {group_label(name, r.group)} & {r.N_a:,} & {r.expected_n_a:.1f} & {f3(r.theta)} & "
                     f"{r.Delta:+.3f} & {f3(r.p_available)} & {f3(r.cov_pooled)} & {f3(r.cov_ppi)} & {f3(r.cov_ref)} & "
                     f"{f3(r.cond_ppi)} & {f3(r.ratio_ppi_ref)} \\\\")
write(os.path.join(ST, "tabS_groups.tex"), r"""
\begin{longtable}{lrrrrrrrrrr}
\caption{Every retained subgroup at $n=1000$, 600 repeated samples, Equation 5 intervals. $N_a$: frame size; $n\pi_a$: expected reference count; $\theta_a$: frame rate; $\Delta_a$: residual-mean difference; $\Pr(V_a)$: interval availability ($n_a\ge3$); coverage columns include unavailable intervals as misses; ``cond.'' is subgroup-PPI coverage given availability; ratio is mean subgroup-PPI width over mean reference-only width.}\label{tab:s-groups}\\
\toprule
Subgroup & $N_a$ & $n\pi_a$ & $\theta_a$ & $\Delta_a$ & $\Pr(V_a)$ & Pooled & PPI & Ref.-only & PPI cond. & Ratio\\
\midrule\endfirsthead
\toprule
Subgroup & $N_a$ & $n\pi_a$ & $\theta_a$ & $\Delta_a$ & $\Pr(V_a)$ & Pooled & PPI & Ref.-only & PPI cond. & Ratio\\
\midrule\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}""")

# ---------------------------------------------------------------- S2: reconciliation with the previous table
lines = []
for _, r in P.iterrows():
    res, part = split(r["name"])
    lines.append(f"{res} & {part} & {f3(r.orig_worst_pooled)} & {f3(r.orig_worst_ppi)} & {r.orig_ratio_ppi_pooled:.1f} & "
                 f"{f3(r.worst_ppi_t_nofpc)} & {f3(r.worst_pooled)} & {f3(r.worst_ppi)} & {r.ratio_ppi_pooled:.1f} & "
                 f"{f3(r.fpc_width_factor)} \\\\")
write(os.path.join(ST, "tabS_reconcile.tex"), r"""
\begin{tabular}{llcccccccc}
\toprule
& & \multicolumn{3}{c}{Previous rule: $z$, no FPC} & $t$, no FPC & \multicolumn{3}{c}{Equation 5: $t$ with FPC} & FPC width \\
\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}
Resource & Partition & Pooled & PPI & Width $\times$ & PPI & Pooled & PPI & Width $\times$ & factor \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------------------------------------------------------- S3: simultaneous coverage
lines = []
for _, r in P.iterrows():
    res, part = split(r["name"])
    lines.append(f"{res} & {part} & {r.K} & {f3(r.simul_pooled)} & {f3(r.simul_pooled_bonf)} & {f3(r.simul_ppi)} & "
                 f"{f3(r.simul_ppi_bonf)} & {f3(r.simul_ref)} & {f3(r.simul_ref_bonf)} & {r.bonf_width_cost_ppi:.2f} \\\\")
write(os.path.join(ST, "tabS_simul.tex"), r"""
\begin{tabular}{llrccccccc}
\toprule
& & & \multicolumn{2}{c}{Pooled} & \multicolumn{2}{c}{Subgroup PPI} & \multicolumn{2}{c}{Reference-only} & Bonferroni \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Resource & Partition & $K$ & 0.05 & $0.05/K$ & 0.05 & $0.05/K$ & 0.05 & $0.05/K$ & width cost \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------------------------------------------------------- S4: excluded-subgroup sensitivity
lines = []
for _, r in SE[~SE.retained].iterrows():
    res, part = split(r["name"])
    cond = "--" if np.isnan(r.cond_ppi) else f3(r.cond_ppi)
    lines.append(f"{res} {part} & {group_label(r['name'], r.group)} & {r.N_a} & {r.n:,} & {r.expected_n_a:.2f} & "
                 f"{f3(r.p_available)} & {f3(r.cov_pooled)} & {f3(r.cov_ppi)} & {f3(r.cov_ref)} & {cond} \\\\")
write(os.path.join(ST, "tabS_sensitivity.tex"), r"""
\begin{tabular}{llrrrccccc}
\toprule
Partition & Subgroup & $N_a$ & $n$ & $n\pi_a$ & $\Pr(V_a)$ & Pooled & PPI & Ref.-only & PPI cond. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------------------------------------------------------- S5: ACS budget sweep
lines = []
for _, r in SW.iterrows():
    lines.append(f"{int(r.n):,} & {f3(r.shortfall_bound)} & {f3(r.worst_pooled)} & {f3(r.worst_ppi)} & {f3(r.worst_ref)} & "
                 f"{f3(r.min_available_ppi)} & {f3(r.worst_cond_ppi)} & {f3(r.simul_bonf_ppi)} & {f3(r.simul_bonf_ref)} & "
                 f"{f2(r.ratio_ppi_ref_median)} \\\\")
write(os.path.join(ST, "tabS_sweep.tex"), r"""
\begin{tabular}{rccccccccc}
\toprule
& Shortfall & \multicolumn{3}{c}{Worst-subgroup coverage} & PPI & PPI & \multicolumn{2}{c}{Simultaneous, Bonf.} & Width \\
\cmidrule(lr){3-5}\cmidrule(lr){8-9}
$n$ & bound & Pooled & PPI & Ref.-only & $\min\Pr(V_a)$ & min cond. & PPI & Ref.-only & ratio \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------------------------------------------------------- S6: ablation, by seed and vs previous
prev = pd.read_csv(PREV_ABLATION)
prev_map = {"A0 pooled rectifier (status quo)": "pooled correction, base predictor",
            "A1 better labeler (4x train data)": "pooled correction, 4x fitting data",
            "A2 sufficiency-regularized labeler": "pooled correction, MLP predictor",
            "A3 post-hoc multiaccuracy repair": "pooled correction, post-hoc offset",
            "A4 group-conditional rectifier (ours)": "subgroup PPI, base predictor"}
prev["arm"] = prev.arm.map(prev_map); prev = prev.set_index("arm")
lines = []
for _, r in A.iterrows():
    p = prev.loc[r.arm] if r.arm in prev.index else None
    old = (f"{f3(p.marginal)} & {f3(p['mean'])} & {f3(p.worst)}" if p is not None else "-- & -- & --")
    lines.append(f"{names[r.arm]} & {old} & {f3(r.overall) if not np.isnan(r.overall) else '--'} & {f3(r['mean'])} & "
                 f"{f3(r.worst)} & {f3(r.worst_min)}--{f3(r.worst_max)} & {f3(r.simul_bonf)} \\\\")
write(os.path.join(ST, "tabS_ablation.tex"), r"""
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{Previously reported} & \multicolumn{5}{c}{Regenerated (finite-frame design, Equation 5)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-9}
Estimator and predictor & Overall & Mean & Worst & Overall & Mean & Worst & Worst, seed range & Simult., Bonf. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------------------------------------------------------- S7: permutation placebo, by relabeling
acsP = P.set_index("name").loc["ACS|race"]
acs_real_maxd = float(G[G.name == "ACS|race"].Delta.abs().max())
pl_row = lambda lab, d, wp, wg, wr, mp, mg, sp, sg, rr, rp: (
    f"{lab} & {f3(d)} & {f3(wp)} & {f3(wg)} & {f3(wr)} & {f3(mp)} & {f3(mg)} & {f3(sp)} & {f3(sg)} & {f2(rr)} & {rp:.1f} \\\\")
lines = [pl_row(int(r.seed), r.max_abs_delta, r.worst_pooled, r.worst_ppi, r.worst_ref, r.mean_pooled, r.mean_ppi,
                r.simul_pooled_bonf, r.simul_ppi_bonf, r.ratio_ppi_ref_median, r.ratio_ppi_pooled) for _, r in PL.iterrows()]
lines.append(r"\midrule")
m = PL.mean(numeric_only=True)
lines.append(pl_row("Mean", m.max_abs_delta, m.worst_pooled, m.worst_ppi, m.worst_ref, m.mean_pooled, m.mean_ppi,
                    m.simul_pooled_bonf, m.simul_ppi_bonf, m.ratio_ppi_ref_median, m.ratio_ppi_pooled))
lines.append(pl_row("True labels", acs_real_maxd, acsP.worst_pooled, acsP.worst_ppi, acsP.worst_ref, acsP.mean_pooled,
                    acsP.mean_ppi, acsP.simul_pooled_bonf, acsP.simul_ppi_bonf, acsP.ratio_ppi_ref_median,
                    acsP.ratio_ppi_pooled))
write(os.path.join(ST, "tabS_placebo.tex"), r"""
\begin{tabular}{lcccccccccc}
\toprule
& & \multicolumn{3}{c}{Worst-subgroup coverage} & \multicolumn{2}{c}{Mean subgroup coverage} & \multicolumn{2}{c}{Simultaneous, Bonf.} & \multicolumn{2}{c}{Width ratio} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Relabeling & $\max_a|\Delta_a|$ & Pooled & PPI & Ref.-only & Pooled & PPI & Pooled & PPI & PPI/ref. & PPI/pooled \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------------------------------------------------------- S8: permutation placebo, by subgroup size
N_acs, n_pl = int(acsP.N), int(acsP.n_gold)
c_pl = stats.t.ppf(0.975, n_pl - 1)
PG["approx"] = stats.norm.cdf(c_pl - PG.zeta.abs()) - stats.norm.cdf(-c_pl - PG.zeta.abs())
PG["w_ratio"] = PG.w_ppi / PG.w_pooled
GA = G[G.name == "ACS|race"].assign(group=lambda d: d.group.astype(str)).set_index("group")
PG["group"] = PG.group.astype(str)
pgs = PG.groupby("group").agg(N_a=("N_a", "first"), absd=("Delta", lambda v: v.abs().mean()),
                              cov=("cov_pooled", "mean"), cov_min=("cov_pooled", "min"), cov_max=("cov_pooled", "max"),
                              approx=("approx", "mean"), cov_ppi=("cov_ppi", "mean"), w_ratio=("w_ratio", "mean"))
pgs["sd_ratio"] = np.sqrt((1 - pgs.N_a / N_acs) / pgs.N_a) / np.sqrt((1 - n_pl / N_acs) / n_pl)
pgs = pgs.sort_values("N_a")
lines = []
for g, r in pgs.iterrows():
    q = GA.loc[g]
    lines.append(f"{group_label('ACS|race', g)} & {int(r.N_a):,} & {q.Delta:+.3f} & {f3(q.cov_pooled)} & {f3(r.absd)} & "
                 f"{f2(r.sd_ratio)} & {f3(r['cov'])} & {f3(r.cov_min)}--{f3(r.cov_max)} & {f3(r.approx)} & {f3(r.cov_ppi)} & "
                 f"{r.w_ratio:.1f} \\\\")
write(os.path.join(ST, "tabS_placebo_groups.tex"), r"""
\begin{tabular}{lrcccccccccc}
\toprule
& & \multicolumn{2}{c}{True labels} & \multicolumn{7}{c}{Placebo relabelings (20)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-11}
Subgroup size held & $N_a$ & $\Delta_a$ & Pooled & Mean $|\Delta_a|$ & SD ratio & Pooled & Pooled range & Approx. & PPI & Width $\times$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ------------------------------------------------------------------ numbers quoted in the text
ten = P[P.name != "ACS|race"]; acs = P.set_index("name").loc["ACS|race"]
acsg = G[G.name == "ACS|race"]
row = lambda s: A.set_index("arm").loc[s]
sw = SW.set_index("n")
nums = dict(
    pooled_below_020=int((P.worst_pooled < 0.20).sum()),
    ppi_ten_range=[round(ten.worst_ppi.min(), 3), round(ten.worst_ppi.max(), 3)],
    ref_ten_range=[round(ten.worst_ref.min(), 3), round(ten.worst_ref.max(), 3)],
    acs_pooled=round(acs.worst_pooled, 3), acs_ppi=round(acs.worst_ppi, 3), acs_ref=round(acs.worst_ref, 3),
    acs_min_available=round(acs.min_available, 3), acs_worst_cond_ppi=round(acs.worst_cond_ppi, 3),
    acs_small_groups_available=sorted(round(v, 3) for v in acsg.p_available if v < 1),
    acs_small_groups_cond=sorted(round(v, 3) for v in acsg.cond_ppi[acsg.p_available < 1]),
    ratio_groups_below_1=int((G.ratio_ppi_ref < 1).sum()), n_groups=len(G),
    ratio_range=[round(G.ratio_ppi_ref.min(), 3), round(G.ratio_ppi_ref.max(), 3)],
    ratio_median=round(G.ratio_ppi_ref.median(), 3),
    ratio_partition_median_range=[round(P.ratio_ppi_ref_median.min(), 2), round(P.ratio_ppi_ref_median.max(), 2)],
    max_abs_ratio_vs_pop_large=round(float((G[G.expected_n_a >= 30].ratio_ppi_ref - G[G.expected_n_a >= 30].pop_ratio).abs().max()), 4),
    worst_ratio_group=G.loc[G.ratio_ppi_ref.idxmax(), ["name", "group", "expected_n_a", "ratio_ppi_ref", "pop_ratio"]].to_dict(),
    simul_ppi_unadj_ten_range=[round(ten.simul_ppi.min(), 3), round(ten.simul_ppi.max(), 3)],
    simul_ppi_bonf_ten_range=[round(ten.simul_ppi_bonf.min(), 3), round(ten.simul_ppi_bonf.max(), 3)],
    simul_ref_bonf_ten_range=[round(ten.simul_ref_bonf.min(), 3), round(ten.simul_ref_bonf.max(), 3)],
    simul_pooled_bonf_range=[round(P.simul_pooled_bonf.min(), 3), round(P.simul_pooled_bonf.max(), 3)],
    simul_pooled_bonf_zero_or_below_011=int((P.simul_pooled_bonf <= 0.11).sum()),
    acs_simul_ppi_bonf=round(acs.simul_ppi_bonf, 3),
    bonf_width_cost_range_ten=[round(ten.bonf_width_cost_ppi.min(), 2), round(ten.bonf_width_cost_ppi.max(), 2)],
    fpc_width_factor={k: round(v, 3) for k, v in zip(P.name, P.fpc_width_factor)},
    previous_ppi_ten_range=[round(ten.orig_worst_ppi.min(), 3), round(ten.orig_worst_ppi.max(), 3)],
    ablation={k: {c: (None if pd.isna(row(k)[c]) else round(float(row(k)[c]), 3))
                  for c in ["auc", "overall", "mean", "worst", "worst_min", "worst_max", "mean_width", "simul_bonf"]}
              for k in A.arm},
    sweep_ppi_worst={int(n): round(sw.loc[n, "worst_ppi"], 3) for n in sw.index},
    sweep_ref_worst={int(n): round(sw.loc[n, "worst_ref"], 3) for n in sw.index},
    sweep_min_available={int(n): round(sw.loc[n, "min_available_ppi"], 3) for n in sw.index},
    sweep_cond={int(n): round(sw.loc[n, "worst_cond_ppi"], 3) for n in sw.index},
    count_planning=open(os.path.join(RES, "acs_count_planning.txt")).read().strip().splitlines(),
    placebo=dict(
        n_seeds=len(PL), real_max_abs_delta=round(acs_real_maxd, 3),
        **{f"{c}_{s}": round(float(getattr(PL[c], s)()), 3)
           for c in ["max_abs_delta", "worst_pooled", "worst_ppi", "worst_ref", "mean_pooled", "simul_pooled_bonf",
                     "simul_ppi_bonf", "ratio_ppi_ref_median", "ratio_ppi_pooled"]
           for s in ["mean", "min", "max"]},
        worst_pooled_median=round(float(PL.worst_pooled.median()), 3),
        spearman_maxdelta_worst_pooled=round(float(stats.spearmanr(PL.max_abs_delta, PL.worst_pooled)[0]), 3),
        large_groups_min_N_a=int(PG[PG.N_a >= 10000].N_a.min()),
        large_groups_pooled_cov=[round(float(PG[PG.N_a >= 10000].cov_pooled.min()), 3),
                                 round(float(PG[PG.N_a >= 10000].cov_pooled.max()), 3)],
        large_groups_pooled_cov_mean=round(float(PG[PG.N_a >= 10000].cov_pooled.mean()), 3),
        large_groups_width_ppi_over_pooled=[round(float(pgs[pgs.N_a >= 10000].w_ratio.min()), 2),
                                            round(float(pgs[pgs.N_a >= 10000].w_ratio.max()), 2)],
        worst_row_N_a_counts={int(k): int(v) for k, v in
                              PG.loc[PG.groupby("seed").cov_pooled.idxmin(), "N_a"].value_counts().items()},
        small_groups={int(r.N_a): dict(sd_ratio=round(float(r.sd_ratio), 2), cov_mean=round(float(r["cov"]), 3),
                                       cov_min=round(float(r.cov_min), 3), approx_mean=round(float(r.approx), 3),
                                       cov_ppi=round(float(r.cov_ppi), 3))
                      for _, r in pgs[pgs.N_a < 10000].iterrows()},
        approx_corr=round(float(np.corrcoef(PG.approx, PG.cov_pooled)[0, 1]), 3),
        approx_max_abs_diff=round(float((PG.approx - PG.cov_pooled).abs().max()), 3),
    ),
)
json.dump(nums, open(os.path.join(RES, "numbers.json"), "w"), indent=1, default=str)
print(json.dumps(nums, indent=1, default=str))
