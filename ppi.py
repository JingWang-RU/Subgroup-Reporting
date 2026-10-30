"""Subgroup outcome-rate estimation with prediction-powered inference (PPI).

Implements exactly the estimators and intervals of the JAMIA manuscript
"Prediction-Powered Inference for Clinical Subgroup Reporting" and nothing else.
Equation numbers refer to the manuscript; "S1.x" refers to Supplementary Section S1.

Notation: a fixed frame of N records with predictions f for every record, reference outcomes
Y revealed on a simple random sample S without replacement, residual R = Y - f, and a
prespecified partition g with subgroup sizes N_a and realized reference counts n_a.

Three estimators of the subgroup rate theta_a (Eq. 1), all computed from the same sample S:
    subgroup PPI       fbar_a + mean(R over S_a)          Eq. 2
    pooled comparator  fbar_a + mean(R over S)            Eq. 3 (deliberately mis-targeted)
    reference-only     mean(Y over S_a)                   Section "Inputs, target, and estimators"
Intervals are Studentized with the finite-population correction of Eq. 5.
"""
import numpy as np
from scipy import stats

MIN_GOLD = 3        # reporting convention: an interval is "available" (V_a = 1) when n_a >= 3
MIN_GROUP = 100     # subgroups with fewer frame records are not retained (Table 2)
Z_ORIGINAL = 1.959964   # the fixed normal quantile used by the original ML4H-era analysis


# ----------------------------------------------------------------------------- estimators
def subgroup_ppi(f_group, R_sample_group):
    """Eq. 2: mean prediction over all subgroup records plus the subgroup residual mean."""
    return float(np.mean(f_group)) + float(np.mean(R_sample_group))


def pooled_comparator(f_group, R_sample):
    """Eq. 3 comparator: subgroup mean prediction plus the overall residual mean (bias -Delta_a)."""
    return float(np.mean(f_group)) + float(np.mean(R_sample))


def reference_only(y_sample_group):
    """Reference-only comparator theta_a^ref: the mean reference outcome in S_a."""
    return float(np.mean(y_sample_group))


# ----------------------------------------------------------------------------- intervals
def design_se(sd, n, N, fpc=True):
    """Square root of Eq. 5: sqrt((1 - n/N) s^2 / n). With fpc=False the (1 - n/N) factor is dropped."""
    sd = np.asarray(sd, float); n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = sd / np.sqrt(n)
        if fpc:
            se = se * np.sqrt(np.clip(1.0 - n / N, 0.0, 1.0))
    return se


def critical_value(alpha, n, dist="t"):
    """t_{n-1, 1-alpha/2} as in Eq. 5 (dist="t"), or the fixed normal quantile (dist="z_original")."""
    n = np.asarray(n, float)
    if dist == "t":
        with np.errstate(invalid="ignore"):
            return np.where(n >= 2, stats.t.ppf(1 - alpha / 2, np.maximum(n - 1, 1)), np.nan)
    if dist == "z_original":
        if alpha != 0.05:
            return np.full(n.shape, stats.norm.ppf(1 - alpha / 2))
        return np.full(n.shape, Z_ORIGINAL)
    raise ValueError(dist)


def studentized_interval(center, sd, n, N, alpha=0.05, min_count=MIN_GOLD, dist="t", fpc=True):
    """Eq. 5 interval center +/- t_{n-1,1-alpha/2} sqrt(V_hat).

    Returns (lower, upper, available). available is V_a in the manuscript: False when n < min_count,
    in which case the interval is not reported (lower = upper = nan)."""
    if n < min_count:
        return np.nan, np.nan, False
    h = float(critical_value(alpha, n, dist)) * float(design_se(sd, n, N, fpc))
    return center - h, center + h, True


# ----------------------------------------------------------------------------- planning
def count_shortfall_probability(group_sizes, N, n, m0=MIN_GOLD):
    """Supplementary bound S1.4: sum over subgroups of Pr(n_a < m0), n_a ~ Hypergeometric(N, N_a, n).

    The sum upper-bounds the probability that at least one subgroup has fewer than m0 references."""
    return float(sum(stats.hypergeom.cdf(m0 - 1, N, Na, n) for Na in group_sizes))


def count_planning_budget(group_sizes, N, beta=0.05, m0=MIN_GOLD):
    """Smallest overall SRS size n with count_shortfall_probability <= beta (bound S1.4)."""
    lo, hi = 1, N
    while lo < hi:
        mid = (lo + hi) // 2
        if count_shortfall_probability(group_sizes, N, mid, m0) <= beta:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ----------------------------------------------------------------------------- simulation
def retained_groups(g, min_group=MIN_GROUP):
    """Prespecified subgroups with at least min_group frame records, in sorted order."""
    return [k for k in sorted(set(g.tolist())) if (g == k).sum() >= min_group]


def simulate(y, f, g, n_gold, replicates=600, seed=0, min_group=MIN_GROUP):
    """Draw `replicates` overall simple random samples of size n_gold without replacement.

    Logs per replicate and subgroup the sufficient statistics of all three estimators, so every
    interval rule (t or z, with or without FPC, any alpha) is a post-hoc reduction of the same
    draws. The pooled and subgroup arms always receive identical labels."""
    y = np.asarray(y, float); f = np.asarray(f, float); g = np.asarray(g)
    ok = np.isfinite(y) & np.isfinite(f); y, f, g = y[ok], f[ok], g[ok]
    R = y - f; N = len(y)
    keys = retained_groups(g, min_group)
    K = len(keys)
    idx = [np.flatnonzero(g == k) for k in keys]
    Na = np.array([len(i) for i in idx])
    out = dict(
        keys=np.array(keys, dtype=object), N=N, Na=Na, n_gold=n_gold,
        truth=np.array([y[i].mean() for i in idx]),          # Eq. 1, the finite-frame target
        fbar=np.array([f[i].mean() for i in idx]),
        Delta=np.array([R[i].mean() - R.mean() for i in idx]),
        theta_overall=y.mean(), fbar_overall=f.mean(),
        S_Y=np.array([y[i].std(ddof=1) for i in idx]),
        S_R=np.array([R[i].std(ddof=1) for i in idx]),
        rho=np.array([np.corrcoef(y[i], f[i])[0, 1] for i in idx]),
    )
    na = np.zeros((replicates, K), int)
    mR = np.full((replicates, K), np.nan); sR = np.full((replicates, K), np.nan)
    mY = np.full((replicates, K), np.nan); sY = np.full((replicates, K), np.nan)
    mRp = np.zeros(replicates); sRp = np.zeros(replicates)
    rng = np.random.default_rng(seed)
    for r in range(replicates):
        s = rng.choice(N, n_gold, replace=False)
        mRp[r] = R[s].mean(); sRp[r] = R[s].std(ddof=1)
        gs = g[s]
        for j, k in enumerate(keys):
            sk = s[gs == k]; na[r, j] = len(sk)
            if len(sk) >= 1:
                mR[r, j] = R[sk].mean(); mY[r, j] = y[sk].mean()
            if len(sk) >= 2:
                sR[r, j] = R[sk].std(ddof=1); sY[r, j] = y[sk].std(ddof=1)
    out.update(na=na, mR=mR, sR=sR, mY=mY, sY=sY, mRp=mRp, sRp=sRp)
    return out


def intervals(sim, arm, alpha=0.05, dist="t", fpc=True, min_count=MIN_GOLD):
    """Centers, half-widths and availability (replicates x K) for one arm under one interval rule.

    arm: "pooled" (Eq. 3 comparator), "ppi" (Eq. 2) or "ref" (reference-only)."""
    na = sim["na"]; K = na.shape[1]; N = sim["N"]
    if arm == "pooled":
        n = sim["n_gold"]
        center = sim["fbar"][None, :] + sim["mRp"][:, None]
        se = np.repeat(design_se(sim["sRp"], n, N, fpc)[:, None], K, axis=1)
        crit = float(critical_value(alpha, n, dist))
        avail = np.ones_like(na, bool)
        return center, crit * se, avail
    if arm == "ppi":
        center = sim["fbar"][None, :] + sim["mR"]; sd = sim["sR"]
    elif arm == "ref":
        center = sim["mY"]; sd = sim["sY"]
    else:
        raise ValueError(arm)
    avail = na >= min_count
    se = design_se(sd, na, sim["Na"][None, :], fpc)
    half = critical_value(alpha, na, dist) * se
    return np.where(avail, center, np.nan), np.where(avail, half, np.nan), avail


def coverage(sim, arm, alpha=0.05, dist="t", fpc=True, min_count=MIN_GOLD):
    """Per-replicate coverage indicators and availability (replicates x K).

    covered is the joint event {V_a = 1 and theta_a in C_a}; unavailable intervals are misses."""
    center, half, avail = intervals(sim, arm, alpha, dist, fpc, min_count)
    with np.errstate(invalid="ignore"):
        hit = np.abs(center - sim["truth"][None, :]) <= half
    return avail & hit, avail, np.where(avail, 2 * half, np.nan)


def summarize(sim, arm, alpha=0.05, dist="t", fpc=True, bonferroni=False, min_count=MIN_GOLD):
    """Marginal, availability-conditional and simultaneous coverage plus widths for one arm.

    bonferroni=True uses alpha/K per subgroup (Eq. 5, alpha_a = 0.05/K)."""
    K = len(sim["keys"])
    a = alpha / K if bonferroni else alpha
    cov, avail, width = coverage(sim, arm, a, dist, fpc, min_count)
    p_avail = avail.mean(0)
    with np.errstate(invalid="ignore"):
        cond = np.where(avail.sum(0) > 0, cov.sum(0) / np.maximum(avail.sum(0), 1), np.nan)
        mean_width = np.nanmean(np.where(avail, width, np.nan), axis=0)
    return dict(
        marginal=cov.mean(0),                  # Pr(V_a = 1 and theta_a in C_a): the reported metric
        p_available=p_avail,                   # Pr(V_a = 1)
        conditional=cond,                      # Pr(theta_a in C_a | V_a = 1)
        simultaneous=float(cov.all(1).mean()), # every subgroup available and covering in one replicate
        mean_width=mean_width,                 # per subgroup, over replicates with V_a = 1
        zero_width=np.nanmean(np.where(avail, width == 0, np.nan), axis=0),
    )


# ----------------------------------------------------------------------------- permutation placebo
def make_placebo_partition(g, seed=0):
    """Randomly permute subgroup labels over the frame, keeping every subgroup size N_a fixed.

    Membership is then independent of R by design, so E[Delta_a] = 0 over relabelings; the
    Delta_a realized by one relabeling is a frame quantity and is not exactly zero."""
    return np.random.default_rng(seed).permutation(np.asarray(g))


def placebo_sweep(y, f, g, n_seeds=20, n_gold=1000, replicates=600, alpha=0.05, sample_seed=0):
    """Repeat the benchmark of one partition on n_seeds placebo relabelings (Eq. 3 precondition).

    Every relabeling reuses the reference samples of the real-partition run (sample_seed), so the
    comparison with Table 3 differs only in subgroup membership. Returns (per-seed rows,
    per-subgroup rows); zeta is Delta_a over the design SE of the overall residual mean (S1.5)."""
    y = np.asarray(y, float); f = np.asarray(f, float); g = np.asarray(g)
    ok = np.isfinite(y) & np.isfinite(f); y, f, g = y[ok], f[ok], g[ok]
    R = y - f; N = len(y)
    sigma_n = float(design_se(R.std(ddof=1), n_gold, N))
    seeds, groups = [], []
    for seed in range(n_seeds):
        sim = simulate(y, f, make_placebo_partition(g, seed), n_gold, replicates, seed=sample_seed)
        S = {arm: summarize(sim, arm, alpha) for arm in ["pooled", "ppi", "ref"]}
        Sb = {arm: summarize(sim, arm, alpha, bonferroni=True) for arm in ["pooled", "ppi"]}
        ratio = S["ppi"]["mean_width"] / S["ref"]["mean_width"]
        seeds.append(dict(
            seed=seed, max_abs_delta=float(np.abs(sim["Delta"]).max()),
            worst_pooled=S["pooled"]["marginal"].min(), worst_ppi=S["ppi"]["marginal"].min(),
            worst_ref=S["ref"]["marginal"].min(),
            mean_pooled=S["pooled"]["marginal"].mean(), mean_ppi=S["ppi"]["marginal"].mean(),
            simul_pooled_bonf=Sb["pooled"]["simultaneous"], simul_ppi_bonf=Sb["ppi"]["simultaneous"],
            ratio_ppi_ref_median=float(np.nanmedian(ratio)),
            ratio_ppi_pooled=float(np.nanmean(S["ppi"]["mean_width"]) / S["pooled"]["mean_width"][0])))
        for j, k in enumerate(sim["keys"]):
            groups.append(dict(
                seed=seed, group=str(k), N_a=int(sim["Na"][j]), Delta=float(sim["Delta"][j]),
                zeta=float(sim["Delta"][j]) / sigma_n,
                cov_pooled=S["pooled"]["marginal"][j], cov_ppi=S["ppi"]["marginal"][j],
                p_available=S["ppi"]["p_available"][j], cond_ppi=S["ppi"]["conditional"][j],
                w_pooled=S["pooled"]["mean_width"][j], w_ppi=S["ppi"]["mean_width"][j]))
    return seeds, groups


def summarize_placebo(seeds):
    """Means and ranges across relabelings of the per-seed rows returned by placebo_sweep."""
    out = dict(n_seeds=len(seeds))
    for c in [c for c in seeds[0] if c != "seed"]:
        v = np.array([s[c] for s in seeds], float)
        out[c] = dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    return out
