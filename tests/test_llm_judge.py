from sentiment_qa.llm_judge import LLMJudge, RuleBasedJudge, JudgeVerdict


def test_rule_based_judge_agree():
    judge = RuleBasedJudge()
    verdict = judge.judge_case("Great service!", "positive", "positive")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.judge_agrees_with_predicted is True
    assert verdict.score == 1.0


def test_rule_based_judge_disagree():
    judge = RuleBasedJudge()
    verdict = judge.judge_case("Terrible experience.", "negative", "positive")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.judge_agrees_with_predicted is False
    assert verdict.score == 0.0


def test_llm_judge_fallback():
    # LLMJudge should fallback gracefully to RuleBasedJudge when API key is missing
    judge = LLMJudge(api_key="")
    verdict = judge.judge_case("Good job!", "positive", "positive")
    assert verdict.judge_agrees_with_predicted is True
    assert verdict.score == 1.0
