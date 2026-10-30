"""Figures 2 and 3 from results/*.csv (Figure 1 is the TikZ schematic in manuscript/figures)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated all-pairs (light surface)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
# SUPPORT2 and RHC share one hue (RHC derives from SUPPORT); marker shape separates them
STYLE = {"SUPPORT2": (BLUE, "o"), "RHC": (BLUE, "s"), "ACS": (ORANGE, "^"), "CheXpert": (AQUA, "D")}


def finish(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2: efficiency vs reference-only
G = pd.read_csv(os.path.join(RES, "benchmark_groups.csv"))
G["resource"] = G.name.str.split("|").str[0]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax = axes[0]
for res, (c, m) in STYLE.items():
    d = G[G.resource == res]
    ax.scatter(d.expected_n_a, d.ratio_ppi_ref, s=26, marker=m, facecolor=c, edgecolor="white",
               linewidth=0.8, label=res, zorder=3)
ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
ax.set_xscale("log"); ax.set_ylim(0.72, 1.06)
ax.set_xlabel("expected reference labels in subgroup, $n\\pi_a$")
ax.set_ylabel("interval width, subgroup PPI / reference-only")
t = G.loc[G.ratio_ppi_ref.idxmax()]
ax.annotate(f"ACS race {t.group}\n$n\\pi_a$={t.expected_n_a:.1f}, ratio {t.ratio_ppi_ref:.3f}",
            (t.expected_n_a, t.ratio_ppi_ref), xytext=(4.5, 1.035), fontsize=7.5, color=MUTED,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.text(0.02, 0.02, "below 1: predictions narrow the interval", transform=ax.transAxes,
        fontsize=7.5, color=MUTED)
ax.set_title("A. Width ratio by subgroup size", loc="left", fontsize=9, color=INK, pad=10)
finish(ax)
ax = axes[1]
for res, (c, m) in STYLE.items():
    d = G[G.resource == res]
    ax.scatter(d.pop_ratio, d.ratio_ppi_ref, s=26, marker=m, facecolor=c, edgecolor="white",
               linewidth=0.8, label=res, zorder=3)
lim = [0.72, 1.06]
ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (4, 3)))
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("frame value $S_{R,a}/S_{Y,a}$ (Equation 4)")
ax.set_ylabel("observed width ratio")
ax.set_title("B. Observed ratio vs. frame prediction", loc="left", fontsize=9, color=INK)
ax.legend(frameon=False, loc="lower right", fontsize=7.5, handletextpad=0.3)
finish(ax)
fig.tight_layout(w_pad=2.0)
save(fig, "figure2_efficiency")

# ------------------------------------------------------------------ Figure 3: ACS race budget sweep
S = pd.read_csv(os.path.join(RES, "acs_budget_sweep.csv"))
plan = int(open(os.path.join(RES, "acs_count_planning.txt")).read().split("= ")[-1])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
ax = axes[0]
for col, c, m, lab in [("worst_pooled", ORANGE, "^", "pooled correction"),
                       ("worst_ppi", BLUE, "o", "subgroup PPI"),
                       ("worst_ref", AQUA, "s", "reference-only")]:
    ax.plot(S.n, S[col], color=c, lw=2, marker=m, ms=5, mec="white", mew=0.8, label=lab)
ax.set_title("A. Reported worst-subgroup coverage", loc="left", fontsize=9, color=INK)
ax.set_ylabel("coverage (600 repeated samples)")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.90), fontsize=7.5)
ax = axes[1]
for col, c, m, lab in [("min_available_ppi", BLUE, "o", "interval available"),
                       ("worst_cond_ppi", BLUE, "s", "covers when available")]:
    ax.plot(S.n, S[col], color=c, lw=2, marker=m, ms=5, mec="white", mew=0.8, label=lab,
            ls="-" if "available" in col else (0, (3, 1.5)))
ax.set_title("B. Subgroup PPI: availability vs. calibration", loc="left", fontsize=9, color=INK)
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.40), fontsize=7.5)
for ax in axes:
    ax.axhline(0.95, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.axvline(plan, color=MUTED, lw=0.8, ls=(0, (1, 1.6)))
    ax.text(plan * 1.08, 0.60, f"count-planning\nbudget {plan:,}", fontsize=7.5, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks(S.n); ax.set_xticklabels([f"{v:,}" for v in S.n], fontsize=7.5)
    ax.set_xlabel("reference labels, $n$"); ax.set_ylim(-0.03, 1.03)
    finish(ax)
fig.tight_layout(w_pad=1.5)
save(fig, "figure3_acs_budget")
print("wrote manuscript/figures/figure2_efficiency.{pdf,png}, figure3_acs_budget.{pdf,png}")
