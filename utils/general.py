"""
Shared utilities: deterministic seeding, dtype helpers, JSONL I/O, and the
prompt templates referenced by Supplementary Tables 4, 5, and 8.

The constants below are the exact templates used during inference and during
P(True) elicitation; the supplementary tables in the paper are rendered from
them, so changing the text here will change the manuscript.
"""

import json
import os
import random

import jsonlines
import numpy as np
import torch


SEED = 23


def seed_everything(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_visible_cudas(gpu_ids: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def save_jsonl(data, file_path) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def load_jsonl(file_path):
    try:
        with jsonlines.open(file_path, "r") as reader:
            return [obj for obj in reader]
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def get_dtype(dtype: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


# Supp Table 4: EHRNoteQA training prompt (5 options A-E).
TRAIN_PROMPT_TEMPLATE = (
    "Discharge summary: {note}\n\n"
    "Question: {question}\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}\nE) {e}\n\n"
    "Return only the letter of the correct answer choice."
)

# Supp Table 5: CORAL test prompt (4 options A-D).
TEST_PROMPT_TEMPLATE = (
    "Clinical note: {note}\n\n"
    "Question: {question}\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
    "Return only the letter of the correct answer choice."
)

# Supp Table 8: P(True) self-evaluation prompt.
PTRUE_PROMPT = (
    "Is the proposed answer correct?\n"
    "(a) no\n(b) yes\n"
    "Reply with (a) or (b) only.\n"
    "Answer: "
)


def get_uncertainty_query() -> str:
    return PTRUE_PROMPT
