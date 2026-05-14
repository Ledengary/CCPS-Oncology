#!/usr/bin/env python3
"""
Reconstruct the EHRNoteQA × MIMIC-IV training corpus used by the confidence
estimators.

This script does not ship the training set itself: both MIMIC-IV and EHRNoteQA
are credentialed PhysioNet resources, and reproducing the linkage in code is
the licensed path. After downloading both, point this script at the local
copies.

Pipeline
--------
1. Load EHRNoteQA's 962 clinician-reviewed MCQ items.
2. Load MIMIC-IV's discharge.csv (table of discharge summaries, one row per
   admission).
3. For each EHRNoteQA item, find the patient's discharge summaries, concatenate
   them chronologically when there are multiple admissions, and emit one row
   per (QA item, admission). This produces 1,659 training rows from 962 source
   QA pairs.
4. Render each row with the 5-option training prompt template (Supp Table 4)
   into the `llm_input` column and write to
   data/train/ehrnoteqa_train_mcqa.jsonl.

Inputs
------
--ehrnoteqa-csv   : EHRNoteQA MCQ CSV from PhysioNet
                    (https://doi.org/10.13026/ACGA-HT95).
--mimic-discharge : MIMIC-IV-Note discharge.csv (or .csv.gz).
--out             : output CSV path, default data/train/ehrnoteqa_train_mcqa.jsonl.

Expected output schema matches what was used to train the confidence estimators:
sidx, patient_id, EHR, question, option_A..E, correct_option, llm_input,
system_prompt.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_io import write_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from general import TRAIN_PROMPT_TEMPLATE  # noqa: E402


SYSTEM_PROMPT = ""  # zero-shot; system prompt is empty for the training corpus


def link_ehrnoteqa_to_mimic(ehrnoteqa: pd.DataFrame, discharge: pd.DataFrame) -> pd.DataFrame:
    """Expand each EHRNoteQA item into one row per patient discharge summary.

    EHRNoteQA exposes ``patient_id`` (subject_id in MIMIC) plus a question and
    five answer options. MIMIC-IV-Note's ``discharge`` table provides the
    full note text. We sort each patient's notes by ``charttime`` (or
    ``chartdate``) and emit one row per (QA, admission), assigning a stable
    sidx as ``{subject_id}-DS-{note_idx}``.
    """
    needed = {"patient_id", "question", "option_A", "option_B", "option_C",
              "option_D", "option_E", "correct_option"}
    missing = needed - set(ehrnoteqa.columns)
    if missing:
        raise ValueError(f"EHRNoteQA CSV is missing columns: {sorted(missing)}")

    # MIMIC-IV-Note discharge schema: note_id, subject_id, hadm_id, note_type,
    # note_seq, charttime, storetime, text
    sort_col = "charttime" if "charttime" in discharge.columns else "chartdate"
    discharge = discharge.sort_values(["subject_id", sort_col]).reset_index(drop=True)
    note_idx = discharge.groupby("subject_id").cumcount()
    discharge = discharge.assign(note_idx=note_idx)
    discharge["sidx"] = (
        discharge["subject_id"].astype(str) + "-DS-" + (discharge["note_idx"] + 1).astype(str)
    )

    expanded = ehrnoteqa.merge(
        discharge[["subject_id", "sidx", "text"]],
        left_on="patient_id", right_on="subject_id",
        how="inner",
    ).rename(columns={"text": "EHR"})

    expanded = expanded[
        ["sidx", "patient_id", "EHR", "question",
         "option_A", "option_B", "option_C", "option_D", "option_E",
         "correct_option"]
    ].reset_index(drop=True)
    return expanded


def render_prompts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["llm_input"] = [
        TRAIN_PROMPT_TEMPLATE.format(
            note=r.EHR, question=r.question,
            a=r.option_A, b=r.option_B, c=r.option_C, d=r.option_D, e=r.option_E,
        )
        for r in df.itertuples(index=False)
    ]
    df["system_prompt"] = SYSTEM_PROMPT
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ehrnoteqa-csv", required=True, help="EHRNoteQA QA CSV from PhysioNet")
    p.add_argument("--mimic-discharge", required=True, help="MIMIC-IV-Note discharge.csv(.gz)")
    p.add_argument("--out", default="data/train/ehrnoteqa_train_mcqa.jsonl")
    args = p.parse_args()

    print(f"Loading EHRNoteQA   ← {args.ehrnoteqa_csv}")
    ehrnoteqa = pd.read_csv(args.ehrnoteqa_csv)
    print(f"  {len(ehrnoteqa)} MCQ items")

    print(f"Loading MIMIC notes ← {args.mimic_discharge}")
    discharge = pd.read_csv(args.mimic_discharge)
    print(f"  {len(discharge)} discharge summaries")

    linked = link_ehrnoteqa_to_mimic(ehrnoteqa, discharge)
    linked = render_prompts(linked)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_table(linked, out)
    print(f"Wrote {len(linked)} rows → {out}")


if __name__ == "__main__":
    main()
