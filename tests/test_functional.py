import pytest
from sentiment_qa.predictor import DemoLexiconPredictor, LABELS

@pytest.fixture
def predictor():
    return DemoLexiconPredictor()

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I love this excellent product", "positive"),
        ("This is a terrible and awful experience", "negative"),
        ("The appointment is at 10 AM", "neutral"),
    ],
)
def test_clear_sentiment_examples(predictor, text, expected):
    assert predictor.predict(text) == expected

def test_prediction_is_one_of_supported_labels(predictor):
    result = predictor.predict("The product arrived today.")
    assert result in LABELS

@pytest.mark.parametrize("bad_text", ["", " ", "\n\t"])
def test_empty_text_is_rejected(predictor, bad_text):
    with pytest.raises(ValueError, match="must not be empty"):
        predictor.predict(bad_text)

def test_non_string_input_is_rejected(predictor):
    with pytest.raises(TypeError, match="must be a string"):
        predictor.predict(None)
