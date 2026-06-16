"""Evaluation metrics with bootstrap 95% confidence intervals.

Metrics follow the manuscript "Statistical analysis and performance evaluation":
sensitivity, specificity, precision, F1, accuracy, and AUC. 95% CIs are obtained
by bootstrap resampling (1,000 iterations).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from config import MULTICLASS_TASKS, BINARY_TASKS, BOOTSTRAP_N, SEED


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return dict(sensitivity=sens, specificity=spec, precision=prec,
                f1=f1, accuracy=acc, auc=auc)


def bootstrap_ci(y_true, y_prob, metric="auc", n=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(binary_metrics(y_true[idx], y_prob[idx])[metric])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def multiclass_accuracy(y_true, y_prob):
    return accuracy_score(np.asarray(y_true).astype(int), np.asarray(y_prob).argmax(axis=1))


def evaluate_all(y_true_dict, preds):
    """Compute metrics for every task. Returns a nested dict."""
    results = {}
    for t in BINARY_TASKS:
        m = binary_metrics(y_true_dict[t], preds[t])
        m["auc_95ci"] = bootstrap_ci(y_true_dict[t], preds[t], "auc")
        results[t] = m
    for t in MULTICLASS_TASKS:
        results[t] = dict(accuracy=multiclass_accuracy(y_true_dict[t], preds[t]))
    return results
