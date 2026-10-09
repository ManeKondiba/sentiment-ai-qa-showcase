import pytest
from sentiment_qa.benchmark import benchmark_predictor, BenchmarkResult
from sentiment_qa.predictor import DemoLexiconPredictor


def test_benchmark_predictor():
    predictor = DemoLexiconPredictor()
    texts = ["I love this product!", "This is terrible.", "Meeting at 5 PM."]
    result = benchmark_predictor(predictor, texts, warmup=1)

    assert isinstance(result, BenchmarkResult)
    assert result.sample_count == 3
    assert result.total_time_ms > 0
    assert result.avg_latency_ms >= 0
    assert result.p50_latency_ms >= 0
    assert result.p95_latency_ms >= 0

    d = result.to_dict()
    assert "avg_latency_ms" in d
    assert "p95_latency_ms" in d
    assert "throughput_qps" in d


def test_benchmark_empty_texts_raises():
    predictor = DemoLexiconPredictor()
    with pytest.raises(ValueError, match="At least one text sample"):
        benchmark_predictor(predictor, [])
