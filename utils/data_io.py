"""
Lightweight wrappers around pandas for the dataset files used in this
repository. CORTEX, predictions, and the reconstructed training corpus are
distributed as JSONL (one JSON object per line) so that embedded clinical
text doesn't trip RFC 4180 edge cases. These helpers transparently fall back
to CSV when the path ends in `.csv`, which keeps upstream CORAL/MIMIC source
files working unchanged.
"""

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def read_table(path: PathLike) -> pd.DataFrame:
    """Read a JSONL or CSV table into a DataFrame, dispatching on extension."""
    p = str(path)
    if p.endswith(".jsonl") or p.endswith(".json"):
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    return pd.read_csv(p)


def write_table(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to JSONL or CSV, dispatching on extension."""
    p = str(path)
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    if p.endswith(".jsonl"):
        df.to_json(p, orient="records", lines=True, force_ascii=False)
    elif p.endswith(".json"):
        df.to_json(p, orient="records", force_ascii=False)
    else:
        df.to_csv(p, index=False)
