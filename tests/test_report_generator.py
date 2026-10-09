from pathlib import Path
import tempfile
from sentiment_qa.report_generator import generate_html_report


def test_generate_html_report():
    dummy_report = {
        "generated_at_utc": "2026-10-09T18:00:00Z",
        "predictor": "TestPredictor",
        "metrics": {
            "sample_count": 3,
            "accuracy": 0.6667,
            "macro_f1": 0.6667,
            "per_class": {
                "positive": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
                "negative": {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1},
                "neutral": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
            },
            "confusion_matrix": {
                "positive": {"positive": 1, "negative": 0, "neutral": 0},
                "negative": {"positive": 0, "negative": 1, "neutral": 0},
                "neutral": {"positive": 0, "negative": 1, "neutral": 0},
            },
            "label_order": ["positive", "negative", "neutral"],
        },
        "cases": [
            {"id": "S-1", "category": "clear", "expected": "positive", "predicted": "positive", "passed": True, "text": "Good"},
            {"id": "S-2", "category": "clear", "expected": "negative", "predicted": "negative", "passed": True, "text": "Bad"},
            {"id": "S-3", "category": "negation", "expected": "neutral", "predicted": "negative", "passed": False, "text": "Not bad"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.html"
        res_path = generate_html_report(dummy_report, out_path)
        assert res_path.exists()
        content = res_path.read_text(encoding="utf-8")
        assert "Sentiment AI QA Report" in content
        assert "TestPredictor" in content
        assert "S-1" in content
        assert "Not bad" in content
