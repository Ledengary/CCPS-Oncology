# Training corpus (not redistributed)

The 1,659-row EHRNoteQA × MIMIC-IV training corpus used to train the confidence estimators is not shipped here, because both source datasets are already on PhysioNet under the same credentialed framework. Rebuilding it locally takes seconds once you have the source files.

```bash
EHRNOTEQA_CSV=/path/to/ehrnoteqa.csv \
MIMIC_DISCHARGE_CSV=/path/to/discharge.csv \
bash ../../scripts/00_build_train_corpus.sh
```

Produces `data/train/ehrnoteqa_train_mcqa.jsonl`. Required by `scripts/01_run_llm_inference.sh` (so the LLMs can be answered on the training set) and `scripts/02_extract_representations.sh` (so PIK and SAPLMA training hidden states can be extracted).

Source datasets:
- EHRNoteQA  — https://doi.org/10.13026/ACGA-HT95
- MIMIC-IV-Note — https://physionet.org/content/mimic-iv-note/
