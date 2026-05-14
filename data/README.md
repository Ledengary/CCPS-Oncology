# Data — CORTEX

The benchmark distributed with this work is named **CORTEX** (*CORAL Tiered EXamination*). It consists of three CORAL-derived oncology MCQ tiers:

- `CORTEX_contextual.jsonl`         — n = 1,458 (single-passage retrieval)
- `CORTEX_synthesis.jsonl`          — n = 764   (multi-hop reasoning)
- `CORTEX_clinical_inference.jsonl` — n = 612   (expert-level inference)

This directory holds two kinds of artifacts:

1. **Numeric outputs that ship with the repo**: per-LLM `predictions/`, per-record `confidence_scores/`, and `trained_models/` weights. These contain no MCQ text — only model IDs and numbers — so they live on GitHub without leaking the benchmark.
2. **CORTEX itself**: the three MCQ CSVs in `source/`. Because the source notes come from CORAL — itself credentialed — CORTEX is released **only on PhysioNet** under the Credentialed Health Data License 1.5.0. It is not in this repo at any commit.

To reproduce the analysis, download the PhysioNet release and drop `source/` into this directory.

## Getting CORTEX

1. Create a PhysioNet account at https://physionet.org and complete CITI human subjects research training (good for three years).
2. Sign the data use agreement for the **CORTEX** project on its PhysioNet page.
3. Download the release archive and unpack it so the final layout is:

```
data/
├── README.md   (this file, in the repo)
├── LICENSE     (CHDL 1.5.0, in the repo)
├── predictions/<model_id>/CORTEX_<tier>_labeled.jsonl         in the repo  (numeric only)
├── confidence_scores/<model_id>/<method>/CORTEX_<tier>.json in the repo  (numeric only)
├── trained_models/{pik,saplma}/<model_id>/                in the repo
└── source/                                                from PhysioNet (CORTEX)
    ├── CORTEX_contextual.jsonl          n = 1,458
    ├── CORTEX_synthesis.jsonl           n = 764
    └── CORTEX_clinical_inference.jsonl  n = 612
```

The five LLMs are Qwen2.5-{0.5B, 1.5B, 3B}-Instruct and Llama-3.2-{1B, 3B}-Instruct. The four confidence methods are P(True), P(IK), SAPLMA (three layer variants F / UM / M), and CCPS.

## File schema

`source/CORTEX_*.jsonl` (PhysioNet-only)

| column | meaning |
|---|---|
| `sidx` | stable identifier; matches the `sidx` column in `predictions/` and the `record_id` index in `confidence_scores/` |
| `doc_idx`, `cancer_type`, `section_name` | CORAL provenance fields |
| `task`, `logic_type` | reasoning category (Synthesis and Clinical Inference only) |
| `EHR` | de-identified oncology progress note |
| `question`, `option_A` … `option_D` | the MCQ |
| `correct_option` | gold answer letter |
| `llm_input` | the test prompt rendered with the Supp Table 5 template |
| `system_prompt` | empty for zero-shot inference |

`predictions/<model>/*_labeled.jsonl` (in the repo — text-free)

Columns: `sidx, doc_idx, cancer_type, section_name, task, logic_type, llm_output, correctness`. The MCQ text columns are deliberately omitted; rejoin against `source/` on `sidx` if you need them.

`confidence_scores/<model>/<method>/CORTEX_<tier>.json` (in the repo)

```json
{
  "record_id":              "0",            // 0-indexed row of the source file
  "dataset":                "CORAL-MCQA",
  "ground_truth_correctness": 1,             // = correctness from predictions/
  "confidence_score":       0.873,           // in [0, 1]; what to use downstream
  "original_result":        { ... }          // method-specific numeric metadata
}
```

Records share `record_id` across methods for the same `(model, tier)`, so confidence scores can be joined directly.

`trained_models/{pik,saplma}/<model>/<...>/best/model.pth` (in the repo) — PyTorch state dicts for the MLP probes used to produce `confidence_scores/`. CCPS weights are not shipped; regenerate via `scripts/03_train_confidence_estimators.sh`.

## Training corpus

The training corpus (n=1,659, EHRNoteQA × MIMIC-IV) is not redistributed because both sources are already on PhysioNet under the same credentialed framework. Rebuild it locally with:

```bash
EHRNOTEQA_CSV=/path/to/ehrnoteqa.csv \
MIMIC_DISCHARGE_CSV=/path/to/discharge.csv \
bash scripts/00_build_train_corpus.sh
```

## License and citation

License: PhysioNet Credentialed Health Data License 1.5.0 for the PhysioNet-only artifacts (see `LICENSE`). Non-commercial research only; re-identification is prohibited.

Cite both:

- Khanmohammadi R et al. *Confidence estimation determines safe automation of oncology chart review across clinical reasoning complexity tiers.* 2026.
- Sushil M et al. *CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference.* NEJM AI, 2024.
