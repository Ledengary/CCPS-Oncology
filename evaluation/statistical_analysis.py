#!/usr/bin/env python3
"""
Statistical validation layer for CCPS-ONC.
Produces bootstrap CIs, paired significance tests, and CV-validated safe yield.
All outputs go to the supplementary appendix -- main manuscript is untouched.

Usage:
    python evaluation/statistical_analysis.py \
        --results-dir results \
        --output-dir results/overall_results/statistical \
        --n-bootstrap 10000 \
        --n-cv-folds 5 \
        --n-cv-repeats 20 \
        --seed 23
"""

import argparse
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.eval import calculate_ece, bootstrap_ci, paired_bootstrap_test, wilcoxon_paired_test


# ==============================================================================
# Configuration -- mirrors results_2.ipynb method2path
# ==============================================================================

LLM_IDS = [
    "Qwen2.5-0.5B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-3B-Instruct",
]

CAPABLE_LLMS = [
    "Qwen2.5-1.5B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-3B-Instruct",
]

METHOD_CONFIG = {
    "PTRUE":    {"display": "P(True)",     "subdir": "ptrue"},
    "SAPLMA-F": {"display": "SAPLMA-F",    "subdir": "saplma_f"},
    "PIK":      {"display": "P(IK)",       "subdir": "pik"},
    "CCPS":     {"display": "CCPS (Ours)", "subdir": "ccps"},
}

TIER_CONFIG = {
    "contextual":         {"display": "Contextual"},
    "synthesis":          {"display": "Synthesis"},
    "clinical_inference": {"display": "Clinical Inference"},
}

METRICS_CONFIG = {
    "ece": {"fn_name": "ece", "higher_better": False},
    "brier": {"fn_name": "brier", "higher_better": False},
    "aucpr": {"fn_name": "aucpr", "higher_better": True},
    "auroc": {"fn_name": "auroc", "higher_better": True},
}


# ==============================================================================
# Metric functions (matching utils/eval.py exactly)
# ==============================================================================

def metric_auroc(y_true, y_conf):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_conf))


def metric_ece(y_true, y_conf):
    return calculate_ece(y_true, y_conf)


def metric_brier(y_true, y_conf):
    return float(brier_score_loss(y_true, y_conf))


def metric_aucpr(y_true, y_conf):
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    precision, recall, _ = precision_recall_curve(y_true, y_conf)
    return float(auc(recall, precision))


METRIC_FNS = {
    "auroc": metric_auroc,
    "ece": metric_ece,
    "brier": metric_brier,
    "aucpr": metric_aucpr,
}


# ==============================================================================
# A. Data Loading
# ==============================================================================

def load_all_results(confidence_scores_dir: Path):
    """Load every (model, method, tier) JSON under data/confidence_scores/.

    Layout: confidence_scores/{model_id}/{method_subdir}/CORTEX_{tier}.json
    Returns: data[method][tier][model] = list of record dicts
    """
    data = {}
    missing = []

    for method, mcfg in METHOD_CONFIG.items():
        data[method] = {}
        for tier in TIER_CONFIG:
            data[method][tier] = {}
            for model_id in LLM_IDS:
                path = confidence_scores_dir / model_id / mcfg["subdir"] / f"CORTEX_{tier}.json"
                if path.exists():
                    data[method][tier][model_id] = json.load(open(path))
                else:
                    missing.append(str(path))
                    data[method][tier][model_id] = []

    if missing:
        print(f"WARNING: {len(missing)} files not found:")
        for p in missing[:5]:
            print(f"  {p}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    return data


def get_shared_record_ids(data, tier, model_id):
    """Get record_ids shared across all methods for a given (tier, model)."""
    id_sets = []
    for method in METHOD_CONFIG:
        records = data[method][tier].get(model_id, [])
        id_sets.append(set(r["record_id"] for r in records))
    if not id_sets:
        return set()
    return set.intersection(*id_sets)


def extract_aligned_arrays(data, method, tier, model_id, shared_ids):
    """Extract y_true, y_conf aligned by sorted shared record_ids."""
    records = data[method][tier].get(model_id, [])
    rec_map = {r["record_id"]: r for r in records}
    sorted_ids = sorted(shared_ids)

    y_true = np.array([rec_map[rid]["ground_truth_correctness"] for rid in sorted_ids])
    y_conf = np.array([rec_map[rid]["confidence_score"] for rid in sorted_ids])
    return y_true, y_conf


# ==============================================================================
# B. Bootstrap CIs
# ==============================================================================

def compute_bootstrap_cis(data, B=10000, alpha=0.05, seed=23):
    """
    Compute bootstrap CIs for each (method, model, tier, metric).
    Also computes aggregated CIs (mean across models).
    """
    per_model = {}
    aggregated = {}

    for tier in TIER_CONFIG:
        for method in METHOD_CONFIG:
            for model_id in LLM_IDS:
                shared_ids = get_shared_record_ids(data, tier, model_id)
                if len(shared_ids) < 20:
                    continue

                y_true, y_conf = extract_aligned_arrays(data, method, tier, model_id, shared_ids)

                for metric_name, metric_fn in METRIC_FNS.items():
                    key = (method, tier, model_id, metric_name)
                    result = bootstrap_ci(y_true, y_conf, metric_fn,
                                          B=B, alpha=alpha, seed=seed,
                                          method='percentile')
                    per_model[key] = result

        # Aggregated CIs: common bootstrap index across all models
        for method in METHOD_CONFIG:
            for metric_name, metric_fn in METRIC_FNS.items():
                agg_result = _aggregated_bootstrap_ci(
                    data, method, tier, metric_name, metric_fn,
                    LLM_IDS, B=B, alpha=alpha, seed=seed
                )
                if agg_result is not None:
                    aggregated[(method, tier, metric_name)] = agg_result

    return per_model, aggregated


def _aggregated_bootstrap_ci(data, method, tier, metric_name, metric_fn,
                              model_ids, B=10000, alpha=0.05, seed=23):
    """Bootstrap CI on mean metric across models, using common resample indices."""
    # Collect aligned arrays for all models
    model_data = {}
    n = None
    for mid in model_ids:
        shared_ids = get_shared_record_ids(data, tier, mid)
        if len(shared_ids) < 20:
            continue
        y_true, y_conf = extract_aligned_arrays(data, method, tier, mid, shared_ids)
        model_data[mid] = (y_true, y_conf)
        if n is None:
            n = len(y_true)

    if len(model_data) < 2:
        return None

    # Point estimate: mean metric across models
    model_points = []
    for mid, (yt, yc) in model_data.items():
        model_points.append(metric_fn(yt, yc))
    point = float(np.mean(model_points))

    # Bootstrap with common index per model's own n
    rng = np.random.RandomState(seed)
    boot_means = np.empty(B)

    for b in range(B):
        model_metrics = []
        for mid, (yt, yc) in model_data.items():
            n_m = len(yt)
            idx = rng.randint(0, n_m, size=n_m)
            try:
                model_metrics.append(metric_fn(yt[idx], yc[idx]))
            except Exception:
                model_metrics.append(np.nan)
        boot_means[b] = np.nanmean(model_metrics)

    boot_means = boot_means[~np.isnan(boot_means)]
    if len(boot_means) < B * 0.5:
        return None

    se = float(np.std(boot_means, ddof=1))
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {'point': point, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'se': se}


# ==============================================================================
# C. Paired Significance Tests
# ==============================================================================

def compute_paired_tests(data, B=10000, seed=23):
    """
    For each (tier, metric): Wilcoxon signed-rank test of CCPS vs each baseline.
    Per-question squared error differences pooled across models.
    Also computes aggregate metric deltas via bootstrap for CIs.
    """
    per_model = {}
    aggregated = {}
    baselines = ["PTRUE", "SAPLMA-F", "PIK"]

    for tier in TIER_CONFIG:
        # Per-model Wilcoxon tests
        for model_id in LLM_IDS:
            shared_ids = get_shared_record_ids(data, tier, model_id)
            if len(shared_ids) < 20:
                continue

            y_true, y_conf_ccps = extract_aligned_arrays(
                data, "CCPS", tier, model_id, shared_ids)

            for baseline in baselines:
                _, y_conf_base = extract_aligned_arrays(
                    data, baseline, tier, model_id, shared_ids)

                # Wilcoxon on per-question Brier differences
                wilcox = wilcoxon_paired_test(y_true, y_conf_ccps, y_conf_base)

                # Also compute metric-level deltas for reporting
                metric_deltas = {}
                for metric_name, metric_fn in METRIC_FNS.items():
                    ma = metric_fn(y_true, y_conf_ccps)
                    mb = metric_fn(y_true, y_conf_base)
                    metric_deltas[metric_name] = ma - mb

                for metric_name in METRIC_FNS:
                    key = (tier, model_id, baseline, metric_name)
                    per_model[key] = {
                        'delta': metric_deltas[metric_name],
                        'p_value': wilcox['p_value'],
                        'wilcoxon_statistic': wilcox['statistic'],
                        'n_nonzero': wilcox['n_nonzero'],
                    }

        # Aggregated: pool per-question differences across all models
        for baseline in baselines:
            agg = _aggregated_wilcoxon_test(
                data, tier, baseline, LLM_IDS)
            if agg is not None:
                for metric_name in METRIC_FNS:
                    aggregated[(tier, baseline, metric_name)] = {
                        'delta': agg['metric_deltas'][metric_name],
                        'p_value': agg['p_value'],
                        'p_value_corrected': agg['p_value'],  # corrected below
                        'wilcoxon_statistic': agg['statistic'],
                        'n_nonzero': agg['n_nonzero'],
                    }

    # Apply Holm-Bonferroni correction
    aggregated = _holm_bonferroni(aggregated)

    return per_model, aggregated


def _aggregated_wilcoxon_test(data, tier, baseline, model_ids):
    """Wilcoxon signed-rank on per-question Brier differences pooled across models."""
    all_y_true = []
    all_conf_ccps = []
    all_conf_base = []

    for mid in model_ids:
        shared_ids = get_shared_record_ids(data, tier, mid)
        if len(shared_ids) < 20:
            continue
        y_true, y_conf_ccps = extract_aligned_arrays(data, "CCPS", tier, mid, shared_ids)
        _, y_conf_base = extract_aligned_arrays(data, baseline, tier, mid, shared_ids)
        all_y_true.append(y_true)
        all_conf_ccps.append(y_conf_ccps)
        all_conf_base.append(y_conf_base)

    if len(all_y_true) < 2:
        return None

    # Pool across models
    y_true_pooled = np.concatenate(all_y_true)
    conf_ccps_pooled = np.concatenate(all_conf_ccps)
    conf_base_pooled = np.concatenate(all_conf_base)

    # Wilcoxon on pooled per-question differences
    wilcox = wilcoxon_paired_test(y_true_pooled, conf_ccps_pooled, conf_base_pooled)

    # Metric-level deltas (mean across models)
    metric_deltas = {}
    for metric_name, metric_fn in METRIC_FNS.items():
        model_deltas = []
        for yt, yc_c, yc_b in zip(all_y_true, all_conf_ccps, all_conf_base):
            model_deltas.append(metric_fn(yt, yc_c) - metric_fn(yt, yc_b))
        metric_deltas[metric_name] = float(np.mean(model_deltas))

    return {
        'p_value': wilcox['p_value'],
        'statistic': wilcox['statistic'],
        'n_nonzero': wilcox['n_nonzero'],
        'mean_brier_diff': wilcox['mean_diff'],
        'metric_deltas': metric_deltas,
    }


def _holm_bonferroni(aggregated):
    """Apply Holm-Bonferroni correction within each (tier, metric) group."""
    groups = defaultdict(list)
    for key, val in aggregated.items():
        tier, baseline, metric_name = key
        groups[(tier, metric_name)].append((key, val))

    for group_key, items in groups.items():
        # Sort by raw p-value
        items.sort(key=lambda x: x[1]['p_value'])
        m = len(items)
        for rank, (key, val) in enumerate(items):
            corrected = min(val['p_value'] * (m - rank), 1.0)
            aggregated[key]['p_value_corrected'] = corrected

    return aggregated


# ==============================================================================
# D. Cross-Validated Safe Yield
# ==============================================================================

def compute_cv_safe_yield(data, n_folds=5, n_repeats=20, seed=23):
    """
    Repeated stratified K-fold CV for safe yield threshold selection.
    Select τ on K-1 folds, evaluate yield on held-out fold.
    """
    results = {}
    rng = np.random.RandomState(seed)

    for tier in TIER_CONFIG:
        for method in METHOD_CONFIG:
            for model_id in CAPABLE_LLMS:
                shared_ids = get_shared_record_ids(data, tier, model_id)
                if len(shared_ids) < 20:
                    continue

                y_true, y_conf = extract_aligned_arrays(
                    data, method, tier, model_id, shared_ids)

                cv_result = _run_cv_safe_yield(
                    y_true, y_conf, n_folds, n_repeats, rng)

                key = (method, tier, model_id)
                results[key] = cv_result

    return results


def _run_cv_safe_yield(y_true, y_conf, n_folds, n_repeats, rng,
                        target_accuracy=0.95, min_samples=10):
    """Run repeated stratified K-fold for safe yield estimation."""
    n = len(y_true)
    fold_yields = []
    fold_accuracies = []
    fold_meets = []
    fold_taus = []

    for rep in range(n_repeats):
        # Stratified shuffle
        indices = np.arange(n)
        rng.shuffle(indices)

        # Create stratified folds
        pos_idx = indices[y_true[indices] == 1]
        neg_idx = indices[y_true[indices] == 0]

        folds = [[] for _ in range(n_folds)]
        for class_idx in [pos_idx, neg_idx]:
            for i, idx in enumerate(class_idx):
                folds[i % n_folds].append(idx)

        for k in range(n_folds):
            test_idx = np.array(folds[k])
            train_idx = np.concatenate([np.array(folds[j])
                                        for j in range(n_folds) if j != k])

            if len(test_idx) < 5 or len(train_idx) < 20:
                continue

            # Select τ on train folds
            tau = _find_safe_threshold(
                y_true[train_idx], y_conf[train_idx],
                target_accuracy, min_samples)

            # Evaluate on test fold
            if tau is not None:
                above_tau = y_conf[test_idx] >= tau
                n_above = np.sum(above_tau)
                if n_above > 0:
                    yield_k = float(np.mean(above_tau))
                    acc_k = float(np.mean(y_true[test_idx][above_tau]))
                    meets_k = acc_k >= target_accuracy
                else:
                    yield_k, acc_k, meets_k = 0.0, 0.0, False
            else:
                yield_k, acc_k, meets_k = 0.0, 0.0, False
                tau = None

            fold_yields.append(yield_k)
            fold_accuracies.append(acc_k)
            fold_meets.append(meets_k)
            fold_taus.append(tau)

    if not fold_yields:
        return {
            'cv_yield_mean': 0.0, 'cv_yield_std': 0.0,
            'cv_accuracy_mean': 0.0, 'cv_accuracy_std': 0.0,
            'cv_meets_rate': 0.0, 'n_folds_evaluated': 0,
            'cv_yield_ci_lo': 0.0, 'cv_yield_ci_hi': 0.0,
        }

    yields_arr = np.array(fold_yields)
    accs_arr = np.array(fold_accuracies)
    meets_arr = np.array(fold_meets)
    taus_arr = [t for t in fold_taus if t is not None]

    return {
        'cv_yield_mean': float(np.mean(yields_arr) * 100),
        'cv_yield_std': float(np.std(yields_arr) * 100),
        'cv_accuracy_mean': float(np.mean(accs_arr)),
        'cv_accuracy_std': float(np.std(accs_arr)),
        'cv_meets_rate': float(np.mean(meets_arr)),
        'cv_tau_mean': float(np.mean(taus_arr)) if taus_arr else None,
        'cv_tau_std': float(np.std(taus_arr)) if taus_arr else None,
        'n_folds_evaluated': len(fold_yields),
        'cv_yield_ci_lo': float(np.percentile(yields_arr * 100, 2.5)),
        'cv_yield_ci_hi': float(np.percentile(yields_arr * 100, 97.5)),
    }


def _find_safe_threshold(y_true, y_conf, target_accuracy=0.95, min_samples=10):
    """Find lowest threshold where accuracy >= target among above-threshold samples."""
    thresholds = np.arange(0.0, 1.0, 0.01)

    for tau in sorted(thresholds):
        above = y_conf >= tau
        n_above = np.sum(above)
        if n_above < min_samples:
            continue
        acc = np.mean(y_true[above])
        if acc >= target_accuracy:
            return float(tau)

    return None


# ==============================================================================
# E. Output Formatting
# ==============================================================================

def format_bootstrap_cis_table(per_model_cis, aggregated_cis):
    """Format CIs into supplementary table structure."""
    rows = []

    # Aggregated table (Table S-X: main summary with CIs)
    for tier, tcfg in TIER_CONFIG.items():
        for metric_name in METRICS_CONFIG:
            row = {'Dataset': tcfg['display'], 'Metric': metric_name.upper()}
            for method, mcfg in METHOD_CONFIG.items():
                key = (method, tier, metric_name)
                if key in aggregated_cis:
                    r = aggregated_cis[key]
                    row[mcfg['display']] = f"{r['point']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                else:
                    row[mcfg['display']] = "-"
            rows.append(row)

    # Add "All Datasets" (merged) -- computed as weighted average
    for metric_name in METRICS_CONFIG:
        row = {'Dataset': 'All Datasets', 'Metric': metric_name.upper()}
        for method, mcfg in METHOD_CONFIG.items():
            tier_results = []
            for tier in TIER_CONFIG:
                key = (method, tier, metric_name)
                if key in aggregated_cis:
                    tier_results.append(aggregated_cis[key])
            if tier_results:
                mean_point = np.mean([r['point'] for r in tier_results])
                mean_lo = np.mean([r['ci_lo'] for r in tier_results])
                mean_hi = np.mean([r['ci_hi'] for r in tier_results])
                row[mcfg['display']] = f"{mean_point:.3f} [{mean_lo:.3f}, {mean_hi:.3f}]"
            else:
                row[mcfg['display']] = "-"
        rows.append(row)

    return rows


def format_paired_tests_table(aggregated_tests):
    """Format paired test results into supplementary table."""
    rows = []
    baselines = ["PTRUE", "SAPLMA-F", "PIK"]

    for tier, tcfg in TIER_CONFIG.items():
        for metric_name in METRICS_CONFIG:
            for baseline in baselines:
                key = (tier, baseline, metric_name)
                if key not in aggregated_tests:
                    continue
                r = aggregated_tests[key]
                stars = _significance_stars(r['p_value_corrected'])

                # Format p-value with proper scientific notation
                p_raw = _format_pvalue(r['p_value'])
                p_corr = _format_pvalue(r['p_value_corrected'])

                rows.append({
                    'Dataset': tcfg['display'],
                    'Metric': metric_name.upper(),
                    'Comparison': f"CCPS vs {METHOD_CONFIG[baseline]['display']}",
                    'Delta': f"{r['delta']:+.3f}",
                    'p_raw': p_raw,
                    'p_corrected': p_corr,
                    'Significance': stars,
                    'n': r.get('n_nonzero', '-'),
                })

    return rows


def format_cv_safe_yield_table(cv_results, original_table_path=None):
    """Format CV safe yield alongside original numbers."""
    rows = []
    methods_order = ["PTRUE", "SAPLMA-F", "PIK", "CCPS"]

    for tier, tcfg in TIER_CONFIG.items():
        for method in methods_order:
            model_yields = []
            model_cv_meets = []

            for model_id in CAPABLE_LLMS:
                key = (method, tier, model_id)
                if key in cv_results:
                    r = cv_results[key]
                    model_yields.append(r['cv_yield_mean'])
                    model_cv_meets.append(r['cv_meets_rate'])

            if model_yields:
                mean_yield = np.mean(model_yields)
                std_yield = np.std(model_yields)
                mean_meets = np.mean(model_cv_meets)
                ci_lo = np.mean([cv_results[(method, tier, m)]['cv_yield_ci_lo']
                                 for m in CAPABLE_LLMS
                                 if (method, tier, m) in cv_results])
                ci_hi = np.mean([cv_results[(method, tier, m)]['cv_yield_ci_hi']
                                 for m in CAPABLE_LLMS
                                 if (method, tier, m) in cv_results])
            else:
                mean_yield, std_yield, mean_meets = 0.0, 0.0, 0.0
                ci_lo, ci_hi = 0.0, 0.0

            rows.append({
                'Dataset': tcfg['display'],
                'Method': METHOD_CONFIG[method]['display'],
                'CV_Safety_Rate': f"{mean_meets:.2f}",
                'CV_Yield_pct': f"{mean_yield:.1f} ± {std_yield:.1f}",
                'CV_Yield_CI': f"[{ci_lo:.1f}, {ci_hi:.1f}]",
            })

    return rows


def _significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def _format_pvalue(p):
    """Format p-value for publication: show exact value in scientific notation."""
    if p == 0.0:
        return "<1e-300"
    elif p < 0.0001:
        return f"{p:.2e}"
    elif p < 0.001:
        return f"{p:.4f}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical validation for CCPS-ONC supplementary appendix")
    parser.add_argument("--confidence-scores-dir", type=str,
                        default="data/confidence_scores",
                        help="Directory containing per-record confidence-score JSONs")
    parser.add_argument("--output-dir", type=str,
                        default="results/tables",
                        help="Output directory for statistical analysis tables")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap resamples")
    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--n-cv-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    confidence_scores_dir = Path(args.confidence_scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CCPS-ONC Statistical Validation")
    print("=" * 70)

    # A. Load data
    print("\n[A] Loading results...")
    data = load_all_results(confidence_scores_dir)

    # Quick summary
    for tier in TIER_CONFIG:
        for model_id in LLM_IDS[:1]:
            shared = get_shared_record_ids(data, tier, model_id)
            print(f"  {tier}: {len(shared)} shared records ({model_id})")

    # B. Bootstrap CIs
    print(f"\n[B] Computing bootstrap CIs (B={args.n_bootstrap})...")
    per_model_cis, aggregated_cis = compute_bootstrap_cis(
        data, B=args.n_bootstrap, alpha=args.alpha, seed=args.seed)
    print(f"  Per-model CIs: {len(per_model_cis)} entries")
    print(f"  Aggregated CIs: {len(aggregated_cis)} entries")

    # C. Paired significance tests
    print(f"\n[C] Computing paired significance tests (B={args.n_bootstrap})...")
    per_model_tests, aggregated_tests = compute_paired_tests(
        data, B=args.n_bootstrap, seed=args.seed)
    print(f"  Per-model tests: {len(per_model_tests)} entries")
    print(f"  Aggregated tests: {len(aggregated_tests)} entries")

    # D. Cross-validated safe yield
    print(f"\n[D] Computing CV safe yield ({args.n_cv_folds}-fold × {args.n_cv_repeats} repeats)...")
    cv_results = compute_cv_safe_yield(
        data, n_folds=args.n_cv_folds, n_repeats=args.n_cv_repeats, seed=args.seed)
    print(f"  CV results: {len(cv_results)} entries")

    # E. Save outputs
    print("\n[E] Saving outputs...")

    # Raw JSON outputs
    def serialize_key(d):
        """Convert tuple keys to string for JSON."""
        return {"|".join(str(x) for x in k): v for k, v in d.items()}

    with open(output_dir / "bootstrap_cis_per_model.json", "w") as f:
        json.dump(serialize_key(per_model_cis), f, indent=2)

    with open(output_dir / "bootstrap_cis_aggregated.json", "w") as f:
        json.dump(serialize_key(aggregated_cis), f, indent=2)

    with open(output_dir / "paired_tests_per_model.json", "w") as f:
        json.dump(serialize_key(per_model_tests), f, indent=2)

    with open(output_dir / "paired_tests_aggregated.json", "w") as f:
        json.dump(serialize_key(aggregated_tests), f, indent=2)

    with open(output_dir / "cv_safe_yield.json", "w") as f:
        json.dump(serialize_key(cv_results), f, indent=2)

    # Formatted CSV tables
    import csv

    # Supp Table 13: Bootstrap CIs
    ci_rows = format_bootstrap_cis_table(per_model_cis, aggregated_cis)
    if ci_rows:
        with open(output_dir / "supp_table_13_bootstrap_ci.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ci_rows[0].keys())
            writer.writeheader()
            writer.writerows(ci_rows)
        print(f"  Saved supp_table_13_bootstrap_ci.csv ({len(ci_rows)} rows)")

    # Supp Table 14: Paired Wilcoxon tests
    test_rows = format_paired_tests_table(aggregated_tests)
    if test_rows:
        with open(output_dir / "supp_table_14_wilcoxon.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=test_rows[0].keys())
            writer.writeheader()
            writer.writerows(test_rows)
        print(f"  Saved supp_table_14_wilcoxon.csv ({len(test_rows)} rows)")

    # Supp Table 15: CV safe yield
    cv_rows = format_cv_safe_yield_table(cv_results)
    if cv_rows:
        with open(output_dir / "supp_table_15_cv_yield.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cv_rows[0].keys())
            writer.writeheader()
            writer.writerows(cv_rows)
        print(f"  Saved supp_table_15_cv_yield.csv ({len(cv_rows)} rows)")

    # Print key results summary
    print("\n" + "=" * 70)
    print("KEY RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Aggregated Bootstrap CIs (mean across 5 LLMs) ---")
    for tier, tcfg in TIER_CONFIG.items():
        print(f"\n  {tcfg['display']}:")
        for metric_name in ["auroc", "ece"]:
            line = f"    {metric_name.upper():6s}  "
            for method in ["CCPS", "PIK", "PTRUE"]:
                key = (method, tier, metric_name)
                if key in aggregated_cis:
                    r = aggregated_cis[key]
                    line += f"{METHOD_CONFIG[method]['display']:12s} {r['point']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
            print(line)

    print("\n--- Wilcoxon Paired Tests: CCPS vs baselines (Holm-corrected) ---")
    for tier, tcfg in TIER_CONFIG.items():
        print(f"\n  {tcfg['display']}:")
        for baseline in ["PIK", "PTRUE", "SAPLMA-F"]:
            for metric_name in ["auroc"]:
                key = (tier, baseline, metric_name)
                if key in aggregated_tests:
                    r = aggregated_tests[key]
                    stars = _significance_stars(r['p_value_corrected'])
                    p_str = _format_pvalue(r['p_value_corrected'])
                    print(f"    AUROC CCPS vs {METHOD_CONFIG[baseline]['display']:10s}: "
                          f"Δ={r['delta']:+.3f}  p={p_str} {stars}  (n={r.get('n_nonzero', '?')})")

    print("\n--- CV Safe Yield (capable models only) ---")
    for tier, tcfg in TIER_CONFIG.items():
        print(f"\n  {tcfg['display']}:")
        for method in ["CCPS", "PIK"]:
            yields = []
            for mid in CAPABLE_LLMS:
                key = (method, tier, mid)
                if key in cv_results:
                    yields.append(cv_results[key]['cv_yield_mean'])
            if yields:
                print(f"    {METHOD_CONFIG[method]['display']:12s}: "
                      f"{np.mean(yields):.1f}% ± {np.std(yields):.1f}%")

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
