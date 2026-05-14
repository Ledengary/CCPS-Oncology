"""
Shared helpers for the analysis notebooks.

These read per-record confidence scores from data/confidence_scores/{model}/{method}/CORTEX_{tier}.json
and compute the calibration / discrimination metrics reported in Supplementary
Tables 10, 11, 12.
"""

from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc


# Display-friendly model labels used in Supp Tables 7, 11, 12.
LLM_DISPLAY = {
    "Qwen2.5-0.5B-Instruct": "Qwen-0.5B",
    "Llama-3.2-1B-Instruct": "Llama-1B",
    "Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
    "Llama-3.2-3B-Instruct": "Llama-3B",
    "Qwen2.5-3B-Instruct":   "Qwen-3B",
}

LLM_IDS = list(LLM_DISPLAY.keys())

# The 3 models that clear the minimum competence floor (Methods, paragraph 21).
CAPABLE_LLMS = ["Qwen2.5-1.5B-Instruct", "Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct"]

METHOD_INFO = {
    "PTRUE":     {"subdir": "ptrue",     "display": "P(True)"},
    "SAPLMA-M":  {"subdir": "saplma_m",  "display": "SAPLMA-M"},
    "SAPLMA-UM": {"subdir": "saplma_um", "display": "SAPLMA-UM"},
    "SAPLMA-F":  {"subdir": "saplma_f",  "display": "SAPLMA-F"},
    "PIK":       {"subdir": "pik",       "display": "P(IK)"},
    "CCPS":      {"subdir": "ccps",      "display": "CCPS (Ours)"},
}

TIER_INFO = {
    "contextual":         {"display": "Contextual",         "n": 1458},
    "synthesis":          {"display": "Synthesis",          "n": 764},
    "clinical_inference": {"display": "Clinical Inference", "n": 612},
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ece(y_true: np.ndarray, y_conf: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error using `n_bins` equally-spaced bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_conf > lo) & (y_conf <= hi)
        if m.any():
            total += abs(y_conf[m].mean() - y_true[m].mean()) * m.mean()
    return float(total)


def brier(y_true, y_conf):
    return float(brier_score_loss(y_true, y_conf))


def auroc(y_true, y_conf):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_conf))


def aucpr(y_true, y_conf):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    p, r, _ = precision_recall_curve(y_true, y_conf)
    return float(auc(r, p))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scores(confidence_scores_dir: Path) -> Dict:
    """Return data[method][tier][model] = list of records."""
    out = {m: {t: {} for t in TIER_INFO} for m in METHOD_INFO}
    missing = []
    for method, mcfg in METHOD_INFO.items():
        for tier in TIER_INFO:
            for model in LLM_IDS:
                p = confidence_scores_dir / model / mcfg["subdir"] / f"CORTEX_{tier}.json"
                if p.exists():
                    out[method][tier][model] = json.load(open(p))
                else:
                    missing.append(p)
                    out[method][tier][model] = []
    if missing:
        print(f"  ! {len(missing)} confidence-score files not found, first = {missing[0]}")
    return out


def arrays_for(records: List[dict]):
    y_true = np.array([r["ground_truth_correctness"] for r in records], dtype=int)
    y_conf = np.array([r["confidence_score"] for r in records], dtype=float)
    return y_true, y_conf


# ---------------------------------------------------------------------------
# Tabulation
# ---------------------------------------------------------------------------

def supp_table_11(data) -> pd.DataFrame:
    """Per-model calibration breakdown: one row per (tier, model, method)."""
    rows = []
    for tier, tcfg in TIER_INFO.items():
        for model in LLM_IDS:
            for method, mcfg in METHOD_INFO.items():
                recs = data[method][tier].get(model, [])
                if not recs:
                    continue
                yt, yc = arrays_for(recs)
                rows.append({
                    "Dataset": tcfg["display"],
                    "LLM":     LLM_DISPLAY[model],
                    "Method":  mcfg["display"],
                    "ECE":     round(ece(yt, yc),    3),
                    "Brier":   round(brier(yt, yc),  3),
                    "AUCPR":   round(aucpr(yt, yc),  3),
                    "AUROC":   round(auroc(yt, yc),  3),
                })
    return pd.DataFrame(rows)


def supp_table_10(per_model: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Supp Table 11 to mean ± std across 5 LLMs per (tier, method, metric)."""
    if per_model.empty:
        return per_model
    long = per_model.melt(id_vars=["Dataset", "LLM", "Method"],
                          value_vars=["ECE", "Brier", "AUCPR", "AUROC"],
                          var_name="Metric", value_name="value")
    agg = (long.groupby(["Dataset", "Metric", "Method"])["value"]
                .agg(["mean", "std"]).reset_index())
    agg["mean_std"] = agg.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    wide = (agg.pivot_table(index=["Dataset", "Metric"], columns="Method",
                            values="mean_std", aggfunc="first").reset_index())
    # Also compute the "All Datasets" row
    long_all = long.copy()
    long_all["Dataset"] = "All Datasets"
    agg_all = (long_all.groupby(["Dataset", "Metric", "Method"])["value"]
                       .agg(["mean", "std"]).reset_index())
    agg_all["mean_std"] = agg_all.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    wide_all = (agg_all.pivot_table(index=["Dataset", "Metric"], columns="Method",
                                    values="mean_std", aggfunc="first").reset_index())
    return pd.concat([wide, wide_all], ignore_index=True)


def find_safe_threshold(y_true: np.ndarray, y_conf: np.ndarray,
                        target_accuracy: float = 0.95, min_samples: int = 10):
    """Lowest tau s.t. acc(y_true[y_conf >= tau]) >= target_accuracy. Returns dict or None."""
    order = np.argsort(-y_conf)
    yt, yc = y_true[order], y_conf[order]
    for i in range(min_samples, len(yt) + 1):
        acc = yt[:i].mean()
        if acc < target_accuracy:
            if i == min_samples:
                return None
            tau = yc[i - 2]
            return {"tau": float(tau), "yield": (i - 1) / len(y_true), "accuracy": float(yt[:i-1].mean())}
    tau = yc[-1]
    return {"tau": float(tau), "yield": 1.0, "accuracy": float(yt.mean())}


def supp_table_12(data) -> pd.DataFrame:
    """Per-model safe yield operating points (capable models only)."""
    rows = []
    methods = ["PTRUE", "SAPLMA-F", "PIK", "CCPS"]
    for tier, tcfg in TIER_INFO.items():
        for model in CAPABLE_LLMS:
            for method in methods:
                recs = data[method][tier].get(model, [])
                if not recs:
                    continue
                yt, yc = arrays_for(recs)
                op = find_safe_threshold(yt, yc, target_accuracy=0.95)
                if op is None:
                    rows.append({"Dataset": tcfg["display"], "N": tcfg["n"],
                                 "LLM": LLM_DISPLAY[model],
                                 "Method": METHOD_INFO[method]["display"],
                                 "Meets_5pct": False, "Tau": None,
                                 "%Yield": 0.0, "Accuracy": None, "AUROC": None})
                else:
                    auto = yc >= op["tau"]
                    if auto.sum() > 1 and len(np.unique(yt[auto])) > 1:
                        a = round(auroc(yt[auto], yc[auto]), 3)
                    else:
                        a = None
                    rows.append({
                        "Dataset": tcfg["display"], "N": tcfg["n"],
                        "LLM": LLM_DISPLAY[model],
                        "Method": METHOD_INFO[method]["display"],
                        "Meets_5pct": True,
                        "Tau":      round(op["tau"],      2),
                        "%Yield":   round(100 * op["yield"], 1),
                        "Accuracy": round(op["accuracy"], 3),
                        "AUROC":    a,
                    })
    return pd.DataFrame(rows)
