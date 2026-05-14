# Preprocessing

| file | role |
|---|---|
| `generate_test_set.py` | Unified four-stage GPT-5.1 pipeline. Pass `--tier {contextual,synthesis,clinical_inference}` to swap prompt bundles loaded from `prompts/`. |
| `prompts/{contextual,synthesis,clinical_inference}.json` | Verbatim system prompts for each pipeline stage; identical to Supplementary Tables 1–3. |
| `answer_with_vllm.py` | Zero-shot vLLM inference at T = 0. Emits `<csv>_ans.jsonl` and `<csv>_answered.csv`. |
| `label_predictions.py` | Parses `llm_output` → predicted letter, writes the binary `correctness` column. |
| `build_train_corpus.py` | Reconstructs the n=1,659 EHRNoteQA × MIMIC-IV training corpus from the credentialed PhysioNet sources. Replaces the absent `data/train/` file. |
| `aggregate_dataset_stats.py` | Emits Supplementary Tables 6 (record counts + lengths) and 7 (per-model accuracy) as CSVs under `results/tables/`. |
| `_coral_task_prompts.py` | Vendored from CORAL — used only by `generate_test_set.py` to inject the task-specific prompt into the fact-extraction stage. |
