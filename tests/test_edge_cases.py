from sentiment_qa.predictor import DemoLexiconPredictor, LABELS

def test_case_insensitive():
    model = DemoLexiconPredictor()
    assert model.predict("I LOVE this product") == "positive"

def test_punctuation_does_not_break_prediction():
    model = DemoLexiconPredictor()
    assert model.predict("Excellent!!!") == "positive"

def test_neutral_input_returns_supported_label():
    model = DemoLexiconPredictor()
    assert model.predict("The report is on the table.") in LABELS

def test_mixed_sentiment_has_documented_baseline_behavior():
    # This baseline uses a simple word count. It is not expected to understand context.
    model = DemoLexiconPredictor()
    result = model.predict("good but terrible")
    assert result == "neutral"
