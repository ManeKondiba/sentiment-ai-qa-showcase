"""Dependency-light classification metrics."""
from collections import Counter
from .predictor import LABELS


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    if not y_true:
        raise ValueError("At least one labeled example is required")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    invalid = (set(y_true) | set(y_pred)) - set(LABELS)
    if invalid:
        raise ValueError(f"Invalid labels: {sorted(invalid)}")

    total = len(y_true)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    per_class = {}
    f1_values = []
    matrix = {actual: {pred: 0 for pred in LABELS} for actual in LABELS}

    for actual, pred in zip(y_true, y_pred):
        matrix[actual][pred] += 1

    for label in LABELS:
        tp = sum(a == label and p == label for a, p in zip(y_true, y_pred))
        fp = sum(a != label and p == label for a, p in zip(y_true, y_pred))
        fn = sum(a == label and p != label for a, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        support = sum(a == label for a in y_true)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_values.append(f1)

    return {
        "sample_count": total,
        "accuracy": round(correct / total, 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "label_order": list(LABELS),
    }
