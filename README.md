# Code: Prediction-Powered Inference for Clinical Subgroup Reporting (JAMIA)

Regenerates every number, table and figure in the manuscript and supplement.

```bash
pip install -r requirements.txt     # Python 3.10.12
./run_all.sh                        # < 1 minute on a 64-core workstation once inputs are present
```

## Inputs

Set these environment variables, or edit the defaults at the top of `data.py`.

| variable | content | access |
|---|---|---|
| `SUPPORT2_CSV` | SUPPORT2 release, 9,105 records (UCI dataset 880) | public |
| `RHC_CSV` | right heart catheterization data, 5,735 records (hbiostat.org/data/repo/rhc.html) | public |
| `ACS_NPZ` | ACS Public Coverage arrays; built on first use by `data.build_acs()` through folktables (2018 1-year, CA TX NY FL GA MS) | public |
| `CHEXPERT_ROOT` | directory with CheXpert-v1.0-small `train.csv` | Stanford AIMI research use agreement |
| `CHEXPERT_SCORES` | frame predictions; produce with `chexpert_scores.py` (GPU) | derived |

MD5 checksums of the inputs behind the published numbers are listed in Supplementary Section S2.

## Files

| file | produces | manuscript object |
|---|---|---|
| `ppi.py` | estimators, Eq. 5 intervals, availability, simultaneous coverage, count-planning bound, permutation placebo | Eqs. 1–5, Supplementary S1.4, S9 |
| `data.py` | the 11 dataset–attribute partitions | Table 2 |
| `chexpert_scores.py` | retrains the CheXpert DenseNet-121 predictor | Table 2, Supplementary S2 |
| `run_benchmark.py` | `results/benchmark_*.csv`, `results/replicates/*.npz` | Table 3, Figure 2, Tables S1–S3 |
| `run_budget_sweep.py` | `results/acs_budget_sweep.csv`, `results/acs_count_planning.txt` | Figure 3, Table S5 |
| `run_ablation.py` | `results/ablation*.csv` | Table 4, Table S6 |
| `run_sensitivity.py` | `results/sensitivity_all_groups.csv` | Table S4 |
| `run_placebo.py` | `results/placebo_acs_race*.csv`, `results/placebo_acs_race_summary.json` | Results 4.6, Tables S7–S8 |
| `make_tables.py` | `manuscript/tables/*.tex`, `supplement/tables/*.tex`, `results/numbers.json` | all tables |
| `make_figures.py` | `manuscript/figures/figure2_*`, `figure3_*` | Figures 2–3 |

Figure 1 is a schematic (`manuscript/figures/figure1_workflow.tex`).

## Design, in one paragraph

For each partition, `ppi.simulate` draws 600 overall simple random samples of reference labels
(seed 0) and logs the sufficient statistics of three estimators on identical labels: subgroup PPI
(Eq. 2), the pooled comparator (Eq. 3) and reference-only estimation. `ppi.summarize` then reduces
the same draws under any interval rule. The manuscript uses Eq. 5: a Studentized `t_{n_a-1}`
interval with the finite-population correction, at alpha = 0.05 or 0.05/K (Bonferroni). An interval
is available (V_a = 1) when n_a >= 3; the code reports Pr(V_a = 1), coverage given availability,
their joint event (the reported coverage) and simultaneous coverage across all K subgroups.

## Reproduction check

`run_benchmark.py` also reduces the draws under the earlier analysis's rule (fixed z = 1.959964, no
finite-population correction) and asserts that this reproduces the previously reported coverage,
width ratios and split-half stability to 1e-9 (`PREVIOUS_MAIN`, default
`../../PPI_treatment/results/main_experiment.csv`). Supplementary Table S2 shows the effect of each
change to the interval rule.
