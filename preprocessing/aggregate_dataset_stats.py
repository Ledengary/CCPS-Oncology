#!/usr/bin/env python3
"""
Compute Supplementary Tables 6 (dataset statistics: record counts, length
quartiles, answer-option distribution) and 7 (per-model accuracy) from the
shipped QA tables (JSONL) and the labeled prediction tables.

Outputs CSVs into results/tables/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_io import read_table  # noqa: E402


TIERS = [
    ("train",                "ehrnoteqa_train_mcqa.jsonl",         "data/train"),
    ("test_contextual",      "CORTEX_contextual.jsonl",            "data/source"),
    ("test_synthesis",       "CORTEX_synthesis.jsonl",             "data/source"),
    ("test_inference",       "CORTEX_clinical_inference.jsonl",    "data/source"),
]

MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
]


def _length_stats(series: pd.Series, prefix: str) -> dict:
    words = series.astype(str).str.split().str.len()
    chars = series.astype(str).str.len()
    return {
        f"{prefix}_word_mean":   round(float(words.mean()),   1),
        f"{prefix}_word_median": round(float(words.median()), 1),
        f"{prefix}_word_q25":    round(float(words.quantile(0.25)), 1),
        f"{prefix}_word_q75":    round(float(words.quantile(0.75)), 1),
        f"{prefix}_char_mean":   round(float(chars.mean()),   1),
        f"{prefix}_char_median": round(float(chars.median()), 1),
        f"{prefix}_char_q25":    round(float(chars.quantile(0.25)), 1),
        f"{prefix}_char_q75":    round(float(chars.quantile(0.75)), 1),
    }


def supp_table_06(data_root: Path) -> pd.DataFrame:
    rows = {}
    for tier, fname, subdir in TIERS:
        csv = data_root / subdir.split("/", 1)[1] / fname
        if not csv.exists():
            print(f"[WARN] missing {csv}")
            continue
        df = read_table(csv)
        ehr_col      = "EHR" if "EHR" in df.columns else "ehr"
        opt_cols     = [c for c in df.columns if c.startswith("option_")]
        options_text = df[opt_cols].astype(str).agg("\n".join, axis=1)
        full_text    = (df[ehr_col].astype(str) + " " + df["question"].astype(str) + " " + options_text)
        row = {"n_records": len(df)}
        row.update(_length_stats(df[ehr_col],      "ehr"))
        row.update(_length_stats(df["question"],   "question"))
        row.update(_length_stats(options_text,     "options"))
        row.update(_length_stats(full_text,        "full"))
        # answer-option distribution
        ans_counts = df["correct_option"].astype(str).str.strip().str.upper().value_counts()
        for letter in ["A", "B", "C", "D", "E"]:
            row[f"ans_{letter}"] = int(ans_counts.get(letter, 0)) or 0
        rows[tier] = row
    if "test_contextual" in rows and "test_synthesis" in rows and "test_inference" in rows:
        merged = {k: 0 for k in next(iter(rows.values()))}
        # Aggregate by re-reading the three test CSVs concatenated
        frames = []
        for tier, fname, subdir in TIERS:
            if tier == "train":
                continue
            p = data_root / subdir.split("/", 1)[1] / fname
            if p.exists():
                frames.append(read_table(p))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            ehr_col = "EHR" if "EHR" in df.columns else "ehr"
            opt_cols = [c for c in df.columns if c.startswith("option_")]
            options_text = df[opt_cols].astype(str).agg("\n".join, axis=1)
            full_text = df[ehr_col].astype(str) + " " + df["question"].astype(str) + " " + options_text
            merged = {"n_records": len(df)}
            merged.update(_length_stats(df[ehr_col],    "ehr"))
            merged.update(_length_stats(df["question"], "question"))
            merged.update(_length_stats(options_text,   "options"))
            merged.update(_length_stats(full_text,      "full"))
            ans_counts = df["correct_option"].astype(str).str.strip().str.upper().value_counts()
            for letter in ["A", "B", "C", "D", "E"]:
                merged[f"ans_{letter}"] = int(ans_counts.get(letter, 0)) or 0
            rows["test_all"] = merged
    return pd.DataFrame(rows).reindex(columns=["train", "test_contextual", "test_synthesis", "test_inference", "test_all"])


def supp_table_07(data_root: Path) -> pd.DataFrame:
    """Per-model accuracy from data/predictions/<model>/CORTEX_<tier>_labeled.jsonl."""
    out = []
    pred_root = data_root / "predictions"
    for model in MODELS:
        for tier_label, file_stem in [
            ("train",           "train_labeled.jsonl"),
            ("test_contextual", "CORTEX_contextual_labeled.jsonl"),
            ("test_synthesis",  "CORTEX_synthesis_labeled.jsonl"),
            ("test_inference",  "CORTEX_clinical_inference_labeled.jsonl"),
        ]:
            p = pred_root / model / file_stem
            if not p.exists():
                continue
            df = read_table(p)
            n = len(df)
            n_correct = int(df["correctness"].sum())
            out.append({
                "model": model,
                "dataset": tier_label,
                "n": n,
                "n_correct": n_correct,
                "pct_correct": round(100 * n_correct / n, 2),
                "n_incorrect": n - n_correct,
                "pct_incorrect": round(100 * (n - n_correct) / n, 2),
            })
        # combined test rows
        test_rows = [r for r in out if r["model"] == model and r["dataset"].startswith("test_")
                     and r["dataset"] != "test_all"]
        if len(test_rows) == 3:
            n = sum(r["n"] for r in test_rows)
            n_correct = sum(r["n_correct"] for r in test_rows)
            out.append({
                "model": model, "dataset": "test_all", "n": n,
                "n_correct": n_correct, "pct_correct": round(100 * n_correct / n, 2),
                "n_incorrect": n - n_correct, "pct_incorrect": round(100 * (n - n_correct) / n, 2),
            })
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data", help="Root of data/ directory")
    p.add_argument("--out-dir", default="results/tables", help="Output directory for CSVs")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t6 = supp_table_06(data_root)
    t6.to_csv(out_dir / "supp_table_06_dataset_stats.csv")
    print(f"Wrote {out_dir/'supp_table_06_dataset_stats.csv'}")
    print(t6)

    t7 = supp_table_07(data_root)
    t7.to_csv(out_dir / "supp_table_07_per_model_accuracy.csv", index=False)
    print(f"Wrote {out_dir/'supp_table_07_per_model_accuracy.csv'}")
    print(t7)


if __name__ == "__main__":
    main()
