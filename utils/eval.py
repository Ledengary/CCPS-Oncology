#!/usr/bin/env python3
"""
Comprehensive evaluation functions for confidence estimation methods.
Calculates all core metrics with proper zero-division handling.
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)
from typing import Dict, Any, List, Callable, Optional
import json


def calculate_ece(y_true: np.ndarray, y_conf: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if len(y_true) == 0:
        return 1.0  # Worst possible ECE
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_conf[in_bin].mean()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def calculate_all_metrics(y_true: np.ndarray, y_conf: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate all evaluation metrics with proper zero-division handling.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_conf: Confidence scores between 0 and 1
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing all metrics
    """
    if len(y_true) == 0:
        return {
            'n_samples': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'ece': 1.0,
            'brier': 1.0,
            'auroc': 0.5,
            'aucpr': 0.0
        }
    
    # Convert confidence to binary predictions
    y_pred = (y_conf >= threshold).astype(int)
    
    # Basic metrics
    metrics = {
        'n_samples': len(y_true),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'ece': calculate_ece(y_true, y_conf),
        'brier': float(brier_score_loss(y_true, y_conf))
    }
    
    # Handle edge cases for classification metrics
    if len(np.unique(y_true)) < 2:
        # All samples have the same label
        if np.all(y_true == y_pred):
            metrics.update({
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'sensitivity': 1.0,
                'specificity': 1.0
            })
        else:
            metrics.update({
                'precision': 0.0,  # Worst case
                'recall': 0.0,     # Worst case
                'f1': 0.0,         # Worst case
                'sensitivity': 0.0, # Worst case
                'specificity': 0.0  # Worst case
            })
    else:
        # Calculate with proper zero division handling
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0.0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0.0))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0.0))
        
        # Calculate sensitivity and specificity
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        except ValueError:
            metrics['sensitivity'] = 0.0
            metrics['specificity'] = 0.0
    
    # AUC metrics (handle edge cases)
    if len(np.unique(y_true)) < 2:
        metrics['auroc'] = 0.5  # Random performance
        metrics['aucpr'] = float(np.mean(y_true))  # Baseline AUCPR
    else:
        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_conf))
            precision, recall, _ = precision_recall_curve(y_true, y_conf)
            metrics['aucpr'] = float(auc(recall, precision))
        except Exception:
            metrics['auroc'] = 0.5
            metrics['aucpr'] = float(np.mean(y_true))
    
    return metrics


def evaluate_by_groups(y_true: np.ndarray, y_conf: np.ndarray, 
                      datasets: np.ndarray, categories: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate metrics across all data, by dataset, and by dataset-category combinations.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_conf: Confidence scores between 0 and 1
        datasets: Dataset labels for each sample
        categories: Category labels for each sample
        
    Returns:
        Dictionary containing results for all groups
    """
    results = {}
    
    # Overall results
    print("Calculating overall metrics...")
    results['overall'] = calculate_all_metrics(y_true, y_conf)
    
    # Results by dataset
    print("Calculating metrics by dataset...")
    results['by_dataset'] = {}
    unique_datasets = np.unique(datasets)
    
    for dataset in unique_datasets:
        dataset_mask = datasets == dataset
        dataset_results = calculate_all_metrics(
            y_true[dataset_mask], 
            y_conf[dataset_mask]
        )
        results['by_dataset'][str(dataset)] = dataset_results
    
    # Results by dataset and category
    print("Calculating metrics by dataset and category...")
    results['by_dataset_category'] = {}
    
    for dataset in unique_datasets:
        dataset_mask = datasets == dataset
        dataset_categories = categories[dataset_mask]
        unique_categories = np.unique(dataset_categories)
        
        results['by_dataset_category'][str(dataset)] = {}
        
        for category in unique_categories:
            category_mask = dataset_categories == category
            full_mask = dataset_mask.copy()
            full_mask[dataset_mask] = category_mask
            
            category_results = calculate_all_metrics(
                y_true[full_mask],
                y_conf[full_mask]
            )
            results['by_dataset_category'][str(dataset)][str(category)] = category_results
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to {output_path}")


# ==============================================================================
# Bootstrap confidence intervals and paired significance tests
# ==============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_conf: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 23,
    method: str = 'bca'
) -> Dict[str, float]:
    """
    Compute point estimate and bootstrap confidence interval for a metric.

    Args:
        y_true: Binary ground truth labels
        y_conf: Confidence scores [0, 1]
        metric_fn: Function(y_true, y_conf) -> float
        B: Number of bootstrap resamples
        alpha: Significance level (0.05 for 95% CI)
        seed: Random seed
        method: 'bca' for bias-corrected accelerated, 'percentile' for basic

    Returns:
        Dictionary with point, ci_lo, ci_hi, se
    """
    n = len(y_true)
    rng = np.random.RandomState(seed)

    point = metric_fn(y_true, y_conf)

    # Bootstrap resamples
    boot_stats = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        try:
            boot_stats[b] = metric_fn(y_true[idx], y_conf[idx])
        except Exception:
            boot_stats[b] = np.nan

    boot_stats = boot_stats[~np.isnan(boot_stats)]
    if len(boot_stats) < B * 0.5:
        return {'point': point, 'ci_lo': np.nan, 'ci_hi': np.nan, 'se': np.nan}

    se = float(np.std(boot_stats, ddof=1))

    if method == 'bca':
        ci_lo, ci_hi = _bca_interval(y_true, y_conf, metric_fn, point, boot_stats, alpha)
    else:
        ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {'point': float(point), 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'se': se}


def _bca_interval(
    y_true: np.ndarray,
    y_conf: np.ndarray,
    metric_fn: Callable,
    point: float,
    boot_stats: np.ndarray,
    alpha: float
) -> tuple:
    """Compute BCa (bias-corrected and accelerated) confidence interval."""
    B_eff = len(boot_stats)

    # Bias correction factor z0
    prop_less = np.mean(boot_stats < point)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_less)

    # Acceleration factor via jackknife
    n = len(y_true)
    jack_stats = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            jack_stats[i] = metric_fn(y_true[mask], y_conf[mask])
        except Exception:
            jack_stats[i] = np.nan

    jack_stats = jack_stats[~np.isnan(jack_stats)]
    if len(jack_stats) < n * 0.5:
        # Fallback to percentile if jackknife fails
        return (
            float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        )

    jack_mean = np.mean(jack_stats)
    diff = jack_mean - jack_stats
    a = np.sum(diff ** 3) / (6.0 * (np.sum(diff ** 2) ** 1.5 + 1e-12))

    # Adjusted percentiles
    z_alpha_lo = norm.ppf(alpha / 2)
    z_alpha_hi = norm.ppf(1 - alpha / 2)

    def adjusted_percentile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - a * num
        if abs(denom) < 1e-12:
            return norm.cdf(z0 + z_alpha)
        return norm.cdf(z0 + num / denom)

    p_lo = adjusted_percentile(z_alpha_lo)
    p_hi = adjusted_percentile(z_alpha_hi)

    p_lo = np.clip(p_lo, 0.5 / B_eff, 1 - 0.5 / B_eff)
    p_hi = np.clip(p_hi, 0.5 / B_eff, 1 - 0.5 / B_eff)

    ci_lo = float(np.percentile(boot_stats, 100 * p_lo))
    ci_hi = float(np.percentile(boot_stats, 100 * p_hi))

    return ci_lo, ci_hi


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_conf_a: np.ndarray,
    y_conf_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 10000,
    seed: int = 23
) -> Dict[str, float]:
    """
    Paired bootstrap test for H0: metric(A) = metric(B).
    Uses same resample indices for both methods (paired design).

    Args:
        y_true: Shared binary ground truth labels
        y_conf_a: Confidence scores from method A
        y_conf_b: Confidence scores from method B
        metric_fn: Function(y_true, y_conf) -> float
        B: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dictionary with delta (A-B), p_value (two-sided), ci_lo, ci_hi
    """
    n = len(y_true)
    rng = np.random.RandomState(seed)

    point_a = metric_fn(y_true, y_conf_a)
    point_b = metric_fn(y_true, y_conf_b)
    observed_delta = point_a - point_b

    deltas = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        try:
            ma = metric_fn(y_true[idx], y_conf_a[idx])
            mb = metric_fn(y_true[idx], y_conf_b[idx])
            deltas[b] = ma - mb
        except Exception:
            deltas[b] = np.nan

    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return {'delta': float(observed_delta), 'p_value': 1.0,
                'ci_lo': np.nan, 'ci_hi': np.nan}

    # Two-sided p-value
    p_greater = np.mean(deltas >= 0)
    p_less = np.mean(deltas <= 0)
    p_value = 2.0 * min(p_greater, p_less)
    p_value = min(p_value, 1.0)

    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))

    return {
        'delta': float(observed_delta),
        'p_value': float(p_value),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi
    }


def wilcoxon_paired_test(
    y_true: np.ndarray,
    y_conf_a: np.ndarray,
    y_conf_b: np.ndarray,
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test on per-question squared error differences.

    For each question i: d_i = (conf_b_i - y_true_i)^2 - (conf_a_i - y_true_i)^2
    Positive d_i means method A has lower error (better).
    Tests H0: median(d) = 0.

    Args:
        y_true: Shared binary ground truth labels
        y_conf_a: Confidence scores from method A (the one we claim is better)
        y_conf_b: Confidence scores from method B (baseline)

    Returns:
        Dictionary with statistic, p_value, mean_diff, median_diff
    """
    from scipy.stats import wilcoxon as scipy_wilcoxon

    # Per-question squared errors
    se_a = (y_conf_a - y_true) ** 2
    se_b = (y_conf_b - y_true) ** 2

    # Difference: positive means A is better (lower error)
    d = se_b - se_a

    # Remove exact zeros (Wilcoxon requirement)
    nonzero_mask = d != 0
    d_nonzero = d[nonzero_mask]

    if len(d_nonzero) < 10:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'mean_diff': float(np.mean(d)),
            'median_diff': float(np.median(d)),
            'n_nonzero': int(len(d_nonzero)),
        }

    stat, p_value = scipy_wilcoxon(d_nonzero, alternative='two-sided')

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'mean_diff': float(np.mean(d)),
        'median_diff': float(np.median(d)),
        'n_nonzero': int(len(d_nonzero)),
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(23)
    
    # Generate test data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.6, n_samples)
    y_conf = np.random.uniform(0.2, 0.9, n_samples)
    datasets = np.random.choice(['dataset1', 'dataset2'], n_samples)
    categories = np.random.choice(['cat1', 'cat2', 'cat3'], n_samples)
    
    # Evaluate
    results = evaluate_by_groups(y_true, y_conf, datasets, categories)
    
    # Print overall results
    print("Overall Results:")
    for metric, value in results['overall'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    save_evaluation_results(results, 'test_evaluation_results.json')