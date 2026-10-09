import pytest
from sentiment_qa.metrics import evaluate

def test_metrics_perfect_predictions():
    result = evaluate(
        ["positive", "negative", "neutral"],
        ["positive", "negative", "neutral"],
    )
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["sample_count"] == 3

def test_metrics_detect_misclassification():
    result = evaluate(
        ["positive", "negative", "neutral"],
        ["positive", "positive", "neutral"],
    )
    assert result["accuracy"] == pytest.approx(2 / 3, abs=0.0001)
    assert result["confusion_matrix"]["negative"]["positive"] == 1

def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate(["positive"], [])

def test_metrics_reject_invalid_label():
    with pytest.raises(ValueError, match="Invalid labels"):
        evaluate(["mixed"], ["positive"])

def test_metrics_reject_empty_dataset():
    with pytest.raises(ValueError, match="At least one"):
        evaluate([], [])
