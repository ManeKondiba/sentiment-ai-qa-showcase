import json
import pytest
from pathlib import Path

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric
from deepeval import assert_test

from sentiment_qa.predictor import DemoLexiconPredictor
from sentiment_qa.deepeval_adapter import create_deepeval_test_cases, run_deepeval_suite


@pytest.fixture
def sample_dataset(tmp_path):
    cases = [
        {"id": "c1", "text": "I love this wonderful software!", "expected": "positive", "category": "functional"},
        {"id": "c2", "text": "This service is terrible and awful.", "expected": "negative", "category": "functional"},
        {"id": "c3", "text": "Meeting at 3 PM.", "expected": "neutral", "category": "functional"},
    ]
    file_path = tmp_path / "test_cases.json"
    file_path.write_text(json.dumps(cases), encoding="utf-8")
    return file_path


def test_create_deepeval_test_cases(sample_dataset):
    predictor = DemoLexiconPredictor()
    test_cases = create_deepeval_test_cases(sample_dataset, predictor)

    assert len(test_cases) == 3
    for tc in test_cases:
        assert isinstance(tc, LLMTestCase)
        assert tc.input is not None
        assert tc.actual_output in ("positive", "negative", "neutral")
        assert tc.expected_output in ("positive", "negative", "neutral")


def test_deepeval_exact_match_metric():
    test_case = LLMTestCase(
        input="I love this product!",
        actual_output="positive",
        expected_output="positive",
    )
    metric = ExactMatchMetric(threshold=1.0)
    metric.measure(test_case)

    assert metric.is_successful()
    assert metric.score == 1.0


def test_run_deepeval_suite(sample_dataset):
    predictor = DemoLexiconPredictor()
    summary = run_deepeval_suite(sample_dataset, predictor, "DemoLexiconPredictor")

    assert summary["framework"] == "DeepEval"
    assert summary["total_test_cases"] == 3
    assert summary["passed_cases"] == 3
    assert summary["pass_rate"] == 1.0
    assert len(summary["results_summary"]) == 3
