from sentiment_qa.red_team import run_red_team_audit, ENCODED_PAYLOADS
from sentiment_qa.predictor import DemoLexiconPredictor


def test_red_team_audit_execution():
    predictor = DemoLexiconPredictor()
    audit = run_red_team_audit(predictor)

    assert "overall_robustness_score" in audit
    assert "vector_breakdown" in audit
    assert audit["total_attacks_tested"] > 0
    assert audit["overall_robustness_score"] >= 0.0

    # Ensure all 6 vectors are audited
    vector_names = [v["vector_name"] for v in audit["vector_breakdown"]]
    assert "prompt_injection_jailbreaks" in vector_names
    assert "secret_and_prompt_leakage" in vector_names
    assert "code_and_sql_injection" in vector_names
    assert "homoglyphs_and_leetspeak" in vector_names
    assert "token_exhaustion_buffer_overflow" in vector_names
    assert "out_of_distribution_noise" in vector_names
