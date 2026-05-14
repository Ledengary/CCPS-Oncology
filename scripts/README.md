# Scripts

Numbered launchers; each one corresponds to a step of the methods section.

| script | purpose | runtime |
|---|---|---|
| `00_build_train_corpus.sh` | Reconstruct the n=1,659 EHRNoteQA × MIMIC-IV training set. Set `EHRNOTEQA_CSV` and `MIMIC_DISCHARGE_CSV` to your credentialed PhysioNet copies before running. | seconds |
| `01_run_llm_inference.sh` | Zero-shot vLLM inference (T = 0) on all 4 QA datasets across the 5 LLMs. | ~3 h on 5 × A100-class GPUs (parallel) |
| `02_extract_representations.sh` | Pre-answer (PIK), post-answer (SAPLMA, 3 layers), and CCPS perturbation features. | ~6 h on 5 GPUs |
| `03_train_confidence_estimators.sh` | MLP probes (PIK, SAPLMA) and the two-stage CCPS encoder + classifier. | ~30 min on 5 GPUs |
| `04_run_confidence_evaluation.sh` | Score every test record under every method; emits `data/confidence_scores/`. | ~1 h on 5 GPUs |
| `05_statistical_analysis.sh` | Bootstrap CIs, Wilcoxon tests, CV safe yield (Supp Tables 13–15). | ~10 min on CPU |

All scripts use round-robin GPU assignment across the five LLMs; edit the `gpu_index` block if you have fewer GPUs available.
