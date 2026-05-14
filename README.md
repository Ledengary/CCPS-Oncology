# CCPS-Oncology

Code for the manuscript *Confidence estimation determines safe automation of oncology chart review across clinical reasoning complexity tiers* (Khanmohammadi et al., 2026). The repository contains the question-generation pipeline, four confidence estimation methods (P(True), P(IK), SAPLMA, CCPS), and the analysis layer that reproduces every numeric value in the main paper and the supplementary appendix.

The benchmark itself, **CORTEX** (*CORAL Tiered EXamination*) — three CORAL-derived oncology QA tiers (Contextual n=1,458, Synthesis n=764, Clinical Inference n=612) — is released **on PhysioNet, not in this repository**, under the Credentialed Health Data License 1.5.0. The source progress notes come from CORAL, which is itself credentialed; CORTEX inherits that access barrier. Per-LLM predictions, correctness labels, raw confidence scores, and trained PIK / SAPLMA classifier weights are shipped here under `data/` — they are numeric only, with no question or note text, so they do not leak the benchmark. To reproduce, download CORTEX from PhysioNet and drop its `source/` directory into `data/`; the rest of the analysis layer runs unchanged. The training corpus is rebuilt locally from EHRNoteQA and MIMIC-IV via `preprocessing/build_train_corpus.py` (both sources are themselves credentialed PhysioNet datasets).

## Quickstart — reproduce the paper's tables and figures without a GPU

```bash
# 1. clone this repo + create the environment
git clone https://github.com/Ledengary/CCPS-Oncology
cd CCPS-Oncology
conda env create -f environment.yml
conda activate ccpsonc

# 2. download the PhysioNet release for CCPS-Oncology (credentialed access)
#    and extract it under data/  — see data/README.md for the expected layout

# 3. main figures + tables
jupyter nbconvert --to notebook --execute analysis/main_figures_and_tables.ipynb \
        --output executed.ipynb

# 4. supplementary tables (10, 11, 12)
jupyter nbconvert --to notebook --execute analysis/supplementary_tables.ipynb \
        --output executed.ipynb

# 5. supplementary tables 13, 14, 15  (~10 min CPU)
bash scripts/05_statistical_analysis.sh
```

Outputs land in `results/figures/` (PDFs + PNGs) and `results/tables/` (CSVs, one per supplementary table).

## What produces what

| Paper artifact | Produced by |
|---|---|
| Figure 1, Figure 2, Table 1, Table 2, Table 3 | `analysis/main_figures_and_tables.ipynb` |
| Supp Tables 1–3 (generation prompts) | `preprocessing/prompts/{contextual,synthesis,clinical_inference}.json` |
| Supp Tables 4, 5 (inference prompts) | `utils/general.py` (`TRAIN_PROMPT_TEMPLATE`, `TEST_PROMPT_TEMPLATE`) |
| Supp Table 6 (dataset statistics) | `python preprocessing/aggregate_dataset_stats.py` |
| Supp Table 7 (per-model accuracy) | same script |
| Supp Table 8 (P(True) prompt) | `utils/general.py` (`PTRUE_PROMPT`) |
| Supp Table 9 (CCPS feature list) | `models/ccps/extract_features.py` |
| Supp Tables 10, 11, 12 | `analysis/supplementary_tables.ipynb` |
| Supp Tables 13, 14, 15 | `evaluation/statistical_analysis.py` |

## Reproducing from scratch (GPU required)

The five scripts under `scripts/` mirror the pipeline described in *Methods*:

1. `scripts/00_build_train_corpus.sh` — rebuilds the 1,659-row training corpus from EHRNoteQA × MIMIC-IV.
2. `scripts/01_run_llm_inference.sh` — zero-shot answers from each of the five LLMs at T=0.
3. `scripts/02_extract_representations.sh` — pre- and post-answer hidden states; CCPS perturbation features (PEI radius 20, 5 steps).
4. `scripts/03_train_confidence_estimators.sh` — MLP probes (PIK, SAPLMA-{F,UM,M}); two-stage CCPS (contrastive encoder → classifier).
5. `scripts/04_run_confidence_evaluation.sh` — scores every test record under every method and writes `data/confidence_scores/`.
6. `scripts/05_statistical_analysis.sh` — bootstrap CIs, Wilcoxon tests, CV safe yield.

Seeds are pinned to 23 (`utils/general.py:seed_everything`). Non-determinism in CUDA matmul kernels can shift bootstrap-aggregated metrics by ≲ 0.005; the seed and Wilcoxon p-values reproduce exactly.

## Layout

```
preprocessing/   four-stage GPT-5.1 test set pipeline, vLLM answering,
                 training-corpus reconstruction, dataset statistics
models/          PTRUE, PIK, SAPLMA, CCPS (extraction + training)
evaluation/      per-method scoring scripts + statistical_analysis.py
analysis/        two notebooks; helpers.py with shared metric code
scripts/         numbered .sh launchers covering the full pipeline
utils/           seeding, prompt templates, calibration metrics
data/            PhysioNet release (test sets + predictions + scores)
results/         figures and tables written by the analysis layer
```

## Citation

If you use this code or the released datasets, please cite:

> Khanmohammadi R, Ghanem AI, Jee Y, Bhatnagar A, Siddiqui S, Bagher-Ebadian H, Movsas B, Ghassemi MM, Thind K. *Confidence estimation determines safe automation of oncology chart review across clinical reasoning complexity tiers.* 2026.

The source CORAL dataset must also be cited: Sushil M et al., *CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference*, NEJM AI 2024.

## License

Code: MIT (see `LICENSE`). Data: PhysioNet Credentialed Health Data License 1.5.0 (see `data/LICENSE`).
