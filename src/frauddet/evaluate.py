from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from frauddet.config import METRICS_PATH, REPORTS_DIR


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose the score cutoff that maximises F1 on a labelled validation split."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    if thresholds.size == 0:
        return float(np.quantile(scores, 0.998))
    # precision_recall_curve returns one extra precision/recall point vs thresholds
    best = int(np.nanargmax(f1[:-1] if f1.size == thresholds.size + 1 else f1[: thresholds.size]))
    return float(thresholds[min(best, thresholds.size - 1)])


def classification_report_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "flagged": int(y_pred.sum()),
        "fraud_support": int(y_true.sum()),
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "prevalence": float(np.mean(y_true)),
    }


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return {**ranking_metrics(y_true, scores), **classification_report_at_threshold(y_true, scores, threshold)}


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def save_pr_curve_points(y_true: np.ndarray, scores: np.ndarray, name: str) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    path = REPORTS_DIR / f"pr_curve_{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, precision=precision, recall=recall)
    return path
