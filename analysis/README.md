# Analysis

| file | produces |
|---|---|
| `main_figures_and_tables.ipynb` | Figures 1 and 2; Tables 1, 2, 3 (main paper). Also emits the per-model safe-yield CSV that the supplementary notebook reads. Run end-to-end with *Run All*. |
| `supplementary_tables.ipynb` | Supplementary Tables 10 (aggregate calibration), 11 (per-model calibration), 12 (per-model safe yield). Tables 13–15 live in CSVs produced by `evaluation/statistical_analysis.py` and are loaded for display. |
| `helpers.py` | Shared metric implementations (ECE, Brier, AUROC, AUCPR), tier configs, model lists, and the supplementary-table tabulators. |

Outputs land in `results/figures/` (PDF + PNG) and `results/tables/` (one CSV per supplementary table).
