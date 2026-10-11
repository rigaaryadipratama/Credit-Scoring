import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve,
)

from credit_scoring.config import FN_COST, FP_COST


def calculate_metrics(y, probability, threshold: float) -> dict:
    y = np.asarray(y)
    probability = np.asarray(probability)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, probability)
    return {
        "rows": int(len(y)), "default_rate": float(y.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "ks": float(np.max(tpr - fpr)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "review_rate": float(prediction.mean()),
        "illustrative_cost_per_1000": float((FN_COST * fn + FP_COST * fp) / len(y) * 1000),
    }


def threshold_table(y, probability) -> list[dict]:
    return [calculate_metrics(y, probability, float(threshold))
            for threshold in np.linspace(0.01, 0.99, 99)]


def choose_threshold(y, probability) -> tuple[float, list[dict]]:
    table = threshold_table(y, probability)
    best = min(table, key=lambda row: (
        row["illustrative_cost_per_1000"], -row["recall"], row["review_rate"],
    ))
    return best["threshold"], table
