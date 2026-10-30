"""Loaders for the 11 dataset-attribute partitions of Table 2.

Each loader returns (y, f, groups) over the fixed evaluation frame: y is the binary reference
outcome, f the frozen prediction, and groups one prespecified partition. Paths are set by
environment variables (defaults are the authors' machine); see README.md for sources.

  SUPPORT2_CSV     SUPPORT2 public release, 9,105 records                 (public)
  RHC_CSV          Right heart catheterization data, 5,735 records         (public)
  ACS_NPZ          folktables ACSPublicCoverage, 2018 1-Year, six states   (public; built by build_acs())
  CHEXPERT_ROOT    directory holding CheXpert-v1.0-small train.csv         (Stanford AIMI data-use agreement)
  CHEXPERT_SCORES  frame predictions written by chexpert_scores.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = "/*/data_local"
SUPPORT2_CSV = os.environ.get("SUPPORT2_CSV", os.path.join(_LOCAL, "support2.csv"))
RHC_CSV = os.environ.get("RHC_CSV", os.path.join(_LOCAL, "rhc.csv"))
ACS_NPZ = os.environ.get("ACS_NPZ", "/*/ppi_fair/acs_pubcov.npz")
CHEXPERT_ROOT = os.environ.get("CHEXPERT_ROOT", "/home/jieshen/jing/cvpr2026")
CHEXPERT_SCORES = os.environ.get("CHEXPERT_SCORES", "/data/jing/ppi_fair/chexpert_scores.npz")

ACS_STATES = ["CA", "TX", "NY", "FL", "GA", "MS"]
ACS_N_FIT = 60000        # records used to fit the ACS predictor, disjoint from the frame
ACS_N_FIT_SMALL = 15000  # smaller fitting set used by the ablation (Table 4)


def support2():
    """Y = death within 60 days of study entry; f = 1 - surv2m (released SUPPORT 2-month survival estimate)."""
    s = pd.read_csv(SUPPORT2_CSV)
    y = ((s["death"] == 1) & (s["d.time"] <= 60)).astype(float).values
    f = (1.0 - s["surv2m"]).values
    parts = {}
    for col in ["race", "income", "dzgroup", "sex"]:
        gg = s[col].astype(str).values
        parts[col] = np.where(pd.isna(s[col]).values, "missing", gg)
    return y, f, parts


def rhc():
    """Y = death within 30 days (dth30); f = 1 - surv2md1 (released SUPPORT 2-month survival estimate, day 1)."""
    rh = pd.read_csv(RHC_CSV)
    y = (rh["dth30"].astype(str).str.strip().str.lower() == "yes").astype(float).values
    f = (1.0 - pd.to_numeric(rh["surv2md1"], errors="coerce")).values
    parts = {col: rh[col].astype(str).values for col in ["race", "income", "ninsclas", "sex"]}
    return y, f, parts


def build_acs(path=ACS_NPZ, root_dir=None):
    """Download ACS PUMS through folktables and cache the ACSPublicCoverage task arrays."""
    from folktables import ACSDataSource, ACSPublicCoverage
    ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person",
                       root_dir=root_dir or os.path.join(os.path.dirname(path), "data"))
    d = ds.get_data(states=ACS_STATES, download=True)
    X, y, g = ACSPublicCoverage.df_to_numpy(d)
    np.savez_compressed(path, X=X, y=y, g=g)


def acs_arrays():
    if not os.path.exists(ACS_NPZ):
        build_acs()
    Z = np.load(ACS_NPZ, allow_pickle=True)
    X, y, g = Z["X"], Z["y"].astype(float), Z["g"].astype(str)
    perm = np.random.default_rng(0).permutation(len(y))
    return X, y, g, perm


def acs_predictor(X, y, fit_idx, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=120, random_state=seed).fit(X[fit_idx], y[fit_idx])


def acs():
    """Y = public insurance coverage; f = gradient-boosted classifier fit on 60,000 separate records."""
    X, y, g, perm = acs_arrays()
    fit, frame = perm[:ACS_N_FIT], perm[ACS_N_FIT:]
    f = acs_predictor(X, y, fit).predict_proba(X[frame])[:, 1]
    return y[frame], f, {"race": g[frame]}


def chexpert_frame():
    """The 60,000-image frontal frame from a 40% patient-level split of CheXpert train (see chexpert_scores.py)."""
    tr = pd.read_csv(os.path.join(CHEXPERT_ROOT, "train.csv"))
    tr["pid"] = tr.Path.str.extract(r"(patient\d+)")[0]
    tr["local"] = tr.Path.str.replace("CheXpert-v1.0-small/", "", regex=False)
    uzero = lambda c: (c.fillna(0.0).replace(-1.0, 0.0) > 0.5).astype(int)   # U-Zeros policy
    tr["Y"] = uzero(tr["Pleural Effusion"]); tr["DEV"] = uzero(tr["Support Devices"])
    tr["SEXF"] = (tr.Sex == "Female").astype(int)
    trf = tr[tr["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    pids = np.array(sorted(trf.pid.unique())); np.random.default_rng(0).shuffle(pids)
    cut = int(0.6 * len(pids))
    fit = trf[trf.pid.isin(set(pids[:cut]))]
    frame = trf[trf.pid.isin(set(pids[cut:]))]
    return fit, frame.sample(60000, random_state=1).reset_index(drop=True)


def chexpert():
    """Y = pleural effusion from the CheXpert report labeler (U-Zeros); f = cached DenseNet-121 score."""
    _, frame = chexpert_frame()
    Z = np.load(CHEXPERT_SCORES, allow_pickle=True)
    assert list(frame.local.values) == list(Z["pool_local"]), "CheXpert frame does not match cached scores"
    y = frame.Y.values.astype(float); f = Z["f_pool"]
    parts = {"device": np.where(frame.DEV.values == 1, "Device", "NoDevice"),
             "sex": np.where(frame.SEXF.values == 1, "Female", "Male")}
    return y, f, parts


def partitions(include_chexpert=True):
    """Yield (name, y, f, groups) for all 11 partitions, in Table 3 order of resources."""
    y, f, p = support2()
    for col in ["race", "income", "dzgroup", "sex"]:
        yield f"SUPPORT2|{col}", y, f, p[col]
    y, f, p = rhc()
    for col in ["race", "income", "ninsclas", "sex"]:
        yield f"RHC|{col}", y, f, p[col]
    y, f, p = acs()
    yield "ACS|race", y, f, p["race"]
    if include_chexpert:
        y, f, p = chexpert()
        for col in ["device", "sex"]:
            yield f"CheXpert|{col}", y, f, p[col]
