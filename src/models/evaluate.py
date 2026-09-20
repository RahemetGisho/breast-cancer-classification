"""Evaluation metrics for the classifier.

Accuracy is reported for completeness but is explicitly NOT the metric
used to select or gate models. In this dataset, a model that always
predicts "benign" would score ~63% accuracy while missing every single
malignant case — recall on the malignant (positive) class is what
actually matters for a screening tool, because a false negative here
means a missed cancer diagnosis, while a false positive just costs an
extra confirmatory test.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    """Compute the full metric set for a set of predictions.

    y_proba should be the predicted probability of the positive
    (malignant, target=1) class.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malignant": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_malignant": float(recall_score(y_true, y_pred, pos_label=1)),
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(
            fn
        ),  # missed cancer cases — the number that matters most
    }


def format_metrics_report(model_name: str, metrics: dict[str, Any]) -> str:
    return (
        f"\n=== {model_name} ===\n"
        f"  Accuracy:              {metrics['accuracy']:.4f}\n"
        f"  Recall (malignant):    {metrics['recall_malignant']:.4f}  <- primary metric\n"
        f"  Precision (malignant): {metrics['precision_malignant']:.4f}\n"
        f"  F1 (malignant):        {metrics['f1_malignant']:.4f}\n"
        f"  ROC-AUC:               {metrics['roc_auc']:.4f}\n"
        f"  Confusion matrix:      TP={metrics['true_positives']} FP={metrics['false_positives']} "
        f"TN={metrics['true_negatives']} FN={metrics['false_negatives']}"
        f"  (FN = missed malignant cases)\n"
    )
