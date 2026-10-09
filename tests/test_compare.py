from pathlib import Path
import tempfile
from sentiment_qa.compare import compare_predictors, generate_comparison_html
from sentiment_qa.predictor import DemoLexiconPredictor
from sentiment_qa.llm_judge import RuleBasedJudge


def test_compare_predictors():
    dataset_file = Path("datasets/sentiment_cases.json")
    predictors = {
        "ModelA": DemoLexiconPredictor(),
        "ModelB": DemoLexiconPredictor(),
    }
    report = compare_predictors(predictors, dataset_file, judge=RuleBasedJudge())

    assert "compared_models" in report
    assert report["compared_models"] == ["ModelA", "ModelB"]
    assert "pairwise_agreement" in report
    assert report["pairwise_agreement"]["ModelA"]["ModelB"] == 1.0


def test_generate_comparison_html():
    dataset_file = Path("datasets/sentiment_cases.json")
    predictors = {"ModelA": DemoLexiconPredictor()}
    report = compare_predictors(predictors, dataset_file, judge=RuleBasedJudge())

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "comparison.html"
        res_path = generate_comparison_html(report, out_path)
        assert res_path.exists()
        content = res_path.read_text(encoding="utf-8")
        assert "Side-by-Side Comparison" in content
        assert "ModelA" in content
