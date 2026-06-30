import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(y_true, y_scores, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }


def plot_confusion_matrix(y_true, y_scores, threshold: float = 0.5, ax=None):
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay(matrix, display_labels=["Not Fraud", "Fraud"]).plot(
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix at Threshold {threshold:.2f}")
    return ax


def plot_precision_recall_curve(y_true, y_scores, ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.3)
    return ax


def plot_roc_curve(y_true, y_scores, ax=None):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_scores)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(false_positive_rate, true_positive_rate)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.3)
    return ax
