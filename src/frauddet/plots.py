from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from frauddet.config import REPORTS_DIR, TARGET


sns.set_theme(style="whitegrid", context="talk")


def plot_class_balance(df: pd.DataFrame, dest: Path | None = None) -> Path:
    dest = dest or (REPORTS_DIR / "class_balance.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#3b82f6", "#ef4444"])
    ax.set_ylabel("Transactions")
    ax.set_title("Extreme class imbalance")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.3%})", ha="center", va="bottom", fontsize=12)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def plot_amount_by_class(df: pd.DataFrame, dest: Path | None = None) -> Path:
    dest = dest or (REPORTS_DIR / "amount_by_class.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(n=min(len(df), 40_000), random_state=42).copy()
    sample["log_amount"] = np.log1p(sample["Amount"].clip(lower=0))
    sns.kdeplot(
        data=sample,
        x="log_amount",
        hue=TARGET,
        common_norm=False,
        ax=ax,
        warn_singular=False,
    )
    ax.set_xlabel("log1p(Amount)")
    ax.set_title("Transaction amount by class")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def plot_pr_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], dest: Path | None = None) -> Path:
    dest = dest or (REPORTS_DIR / "pr_curves.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, scores) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=name, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall (fraud = positive)")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def plot_roc_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], dest: Path | None = None) -> Path:
    dest = dest or (REPORTS_DIR / "roc_curves.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, scores) in results.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest
