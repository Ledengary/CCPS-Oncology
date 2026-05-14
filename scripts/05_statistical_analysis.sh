#!/bin/bash
# Produce Supplementary Tables 13, 14, 15 (bootstrap CIs, Wilcoxon paired tests
# with Holm-Bonferroni correction, and 5-fold × 20-repeat CV safe yield).
#
# Runtime: ~10-15 min on CPU at B = 10,000 bootstraps.

set -euo pipefail
cd "$(dirname "$0")/.."

python evaluation/statistical_analysis.py \
    --confidence-scores-dir data/confidence_scores \
    --output-dir           results/tables \
    --n-bootstrap          10000 \
    --n-cv-folds            5 \
    --n-cv-repeats         20 \
    --seed                 23
