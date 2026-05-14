# Evaluation

Each script reads trained checkpoints + representations and emits per-record confidence-score JSONs into `data/confidence_scores/<model>/<method>/CORTEX_<tier>.json`. The notebooks in `analysis/` consume those JSONs directly.

| file | reads | writes |
|---|---|---|
| `pik_eval.py` | `representations/PIK_<tier>/<model>/test/*.npy` + `data/trained_models/pik/<model>/best/` | per-record P(IK) scores |
| `saplma_eval.py` | `representations/SAPLMA_<tier>/<model>/test/<layer>/*.npy` + `data/trained_models/saplma/<model>/<layer>/best/` | per-record SAPLMA scores (one folder per layer) |
| `ccps_eval.py` | `features/OrigPert_<tier>/<model>/test/*.pickle` + `data/trained_models/ccps/{contrastive,classifier}/<model>/` | per-record CCPS scores |
| `statistical_analysis.py` | `data/confidence_scores/` | `results/tables/supp_table_{13,14,15}*.csv` — bootstrap CIs (B=10,000), Wilcoxon signed-rank with Holm-Bonferroni, 5-fold × 20-repeat CV safe yield |

P(True) is computed online by `models/ptrue.py` and writes directly into `data/confidence_scores/<model>/ptrue/`.
