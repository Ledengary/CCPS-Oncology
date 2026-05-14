#!/usr/bin/env python3
"""
Convert an answered JSONL/CSV (from answer_with_vllm.py) into a labeled file by
extracting the predicted letter from `llm_output` and comparing it against
`correct_option`. Adds a binary `correctness` column.

Used as the supervision signal for all confidence estimation methods.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_io import read_table, write_table  # noqa: E402

logging.basicConfig(level=logging.INFO)


def add_correctness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pred = df["llm_output"].astype(str).str.strip().str.upper().str[0]
    gold = df["correct_option"].astype(str).str.strip().str.upper()
    df["correctness"] = (pred == gold).astype(int)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in",  dest="in_path",  required=True, help="Path to *_answered.jsonl|.csv")
    p.add_argument("--out", dest="out_path", required=True, help="Output path (.jsonl|.csv)")
    args = p.parse_args()

    df = read_table(args.in_path)
    if "llm_output" not in df.columns or "correct_option" not in df.columns:
        raise ValueError("Input must contain 'llm_output' and 'correct_option' columns")
    labeled = add_correctness(df)
    write_table(labeled, args.out_path)
    n_correct = int(labeled["correctness"].sum())
    n = len(labeled)
    logging.info(f"{args.out_path}: {n_correct}/{n} = {n_correct/n:.4f} accuracy")


if __name__ == "__main__":
    main()
