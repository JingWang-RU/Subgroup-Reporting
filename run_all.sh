#!/usr/bin/env bash
# Regenerate every result, table and figure of the manuscript and supplement (< 1 minute on 64 cores).
set -euo pipefail
cd "$(dirname "$0")"
python3 run_benchmark.py      # Table 3, Figure 2, Tables S1-S3; asserts reproduction of the earlier table
python3 run_budget_sweep.py   # Figure 3, Table S5
python3 run_ablation.py       # Table 4, Table S6
python3 run_sensitivity.py    # Table S4
python3 run_placebo.py        # Tables S7-S8 (permutation placebo, ACS race)
python3 make_tables.py        # writes manuscript/tables, supplement/tables, results/numbers.json
python3 make_figures.py       # writes manuscript/figures/figure2_*, figure3_*
