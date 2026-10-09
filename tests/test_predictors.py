from io import BytesIO
import json
import pytest
from unittest.mock import MagicMock, patch

from sentiment_qa.predictor import (
    DemoLexiconPredictor,
    HttpPredictor,
    OpenAIPredictor,
    HuggingFacePredictor,
    normalize_label,
)
from sentiment_qa.evaluate import get_predictor_instance


def test_normalize_label_variants():
    assert normalize_label("positive") == "positive"
    assert normalize_label("POS") == "positive"
    assert normalize_label("LABEL_2") == "positive"
    assert normalize_label("5 stars") == "positive"

    assert normalize_label("negative") == "negative"
    assert normalize_label("NEG") == "negative"
    assert normalize_label("LABEL_0") == "negative"

    assert normalize_label("neutral") == "neutral"
    assert normalize_label("LABEL_1") == "neutral"

    with pytest.raises(ValueError, match="Unable to normalize"):
        normalize_label("unknown_label_123")


def test_http_predictor_success():
    predictor = HttpPredictor(endpoint="http://fake-api/predict")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"label": "positive"}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        res = predictor.predict("This service was wonderful.")
        assert res == "positive"


def test_openai_predictor_success():
    predictor = OpenAIPredictor(api_key="test-key", model="gpt-4o-mini")

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": "negative"}}]
    }).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        res = predictor.predict("The product arrived broken.")
        assert res == "negative"


def test_huggingface_predictor_success():
    predictor = HuggingFacePredictor(api_key="hf_test", model_id="test-model")

    mock_response = MagicMock()
    # Typical HF inference payload
    mock_response.read.return_value = json.dumps([
        [
            {"label": "positive", "score": 0.98},
            {"label": "neutral", "score": 0.01},
            {"label": "negative", "score": 0.01},
        ]
    ]).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        res = predictor.predict("Loved it!")
        assert res == "positive"


def test_predictor_factory():
    p_demo, desc_demo = get_predictor_instance("demo")
    assert isinstance(p_demo, DemoLexiconPredictor)
    assert "baseline" in desc_demo

    p_http, _ = get_predictor_instance("http", endpoint="http://localhost:8000/predict")
    assert isinstance(p_http, HttpPredictor)

    p_openai, _ = get_predictor_instance("openai", api_key="test-key")
    assert isinstance(p_openai, OpenAIPredictor)

    p_hf, _ = get_predictor_instance("huggingface", api_key="hf_test")
    assert isinstance(p_hf, HuggingFacePredictor)
