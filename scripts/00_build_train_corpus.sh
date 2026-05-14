#!/bin/bash
# Reconstruct the EHRNoteQA × MIMIC-IV training corpus from credentialed
# PhysioNet downloads. Outputs data/train/ehrnoteqa_train_mcqa.jsonl (1659 rows).
#
# Requires two PhysioNet downloads (you must complete CITI training and the
# data use agreement to access these):
#   • EHRNoteQA  (https://doi.org/10.13026/ACGA-HT95)
#   • MIMIC-IV-Note discharge table (https://physionet.org/content/mimic-iv-note/)
#
# Set EHRNOTEQA_CSV and MIMIC_DISCHARGE_CSV to your local copies and run.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${EHRNOTEQA_CSV:?Set EHRNOTEQA_CSV to the EHRNoteQA MCQ CSV}"
: "${MIMIC_DISCHARGE_CSV:?Set MIMIC_DISCHARGE_CSV to MIMIC-IV-Note's discharge.csv(.gz)}"

python preprocessing/build_train_corpus.py \
    --ehrnoteqa-csv  "$EHRNOTEQA_CSV" \
    --mimic-discharge "$MIMIC_DISCHARGE_CSV" \
    --out             data/train/ehrnoteqa_train_mcqa.jsonl
