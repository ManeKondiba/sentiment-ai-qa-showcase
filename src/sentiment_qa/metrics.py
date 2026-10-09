"""Dependency-light classification metrics and Quality Gate threshold validation."""
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


def verify_quality_gate(
    metrics: dict,
    benchmark: dict | None = None,
    min_accuracy: float = 0.80,
    min_macro_f1: float = 0.75,
    max_p95_latency_ms: float = 1000.0,
) -> dict:
    """Validate model evaluation metrics against configurable SLA Quality Gate thresholds."""
    acc = metrics.get("accuracy", 0.0)
    f1 = metrics.get("macro_f1", 0.0)
    p95_lat = benchmark.get("p95_latency_ms", 0.0) if benchmark else 0.0

    acc_pass = acc >= min_accuracy
    f1_pass = f1 >= min_macro_f1
    lat_pass = (p95_lat <= max_p95_latency_ms) if benchmark else True

    overall_pass = acc_pass and f1_pass and lat_pass

    checks = [
        {
            "metric": "accuracy",
            "actual": acc,
            "threshold": min_accuracy,
            "passed": acc_pass,
            "rule": f"accuracy ({acc}) >= min_accuracy ({min_accuracy})",
        },
        {
            "metric": "macro_f1",
            "actual": f1,
            "threshold": min_macro_f1,
            "passed": f1_pass,
            "rule": f"macro_f1 ({f1}) >= min_macro_f1 ({min_macro_f1})",
        },
    ]

    if benchmark:
        checks.append({
            "metric": "p95_latency_ms",
            "actual": p95_lat,
            "threshold": max_p95_latency_ms,
            "passed": lat_pass,
            "rule": f"p95_latency_ms ({p95_lat}) <= max_p95_latency_ms ({max_p95_latency_ms})",
        })

    return {
        "quality_gate_passed": overall_pass,
        "checks": checks,
    }
