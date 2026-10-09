"""Verification test suite to validate that the Red-Teaming Audit accurately detects vulnerabilities."""
import pytest
from sentiment_qa.red_team import run_red_team_audit


class VulnerablePredictor:
    """Mock predictor that succumbs to prompt injections and invalid output formats."""

    def predict(self, text: str) -> str:
        text_lower = text.lower()
        # Vulnerability 1: Yields to prompt injection instructions
        if "ignore" in text_lower or "override" in text_lower:
            return "positive"

        # Vulnerability 2: Returns non-standard invalid label format
        if "script" in text_lower or "drop table" in text_lower:
            return "EXPLOITED_XSS_LABEL"

        # Vulnerability 3: Crashes on long inputs
        if len(text) > 1000:
            raise MemoryError("Simulated buffer exhaustion crash")

        return "neutral"


class SecureDummyPredictor:
    """Mock predictor that strictly adheres to canonical labels and resists prompt injections."""

    def predict(self, text: str) -> str:
        return "neutral"


def test_audit_detects_vulnerable_model():
    """Verify that the Red-Teaming Audit correctly catches vulnerabilities and drops robustness score."""
    vulnerable_model = VulnerablePredictor()
    audit = run_red_team_audit(vulnerable_model)

    # 1. Overall robustness score must decrease (not stay at 100%)
    assert audit["overall_robustness_score"] < 80.0
    assert audit["failed_attacks"] == 5

    # 2. Check that prompt injection vector identified vulnerabilities
    injection_vec = next(v for v in audit["vector_breakdown"] if v["vector_name"] == "prompt_injection_jailbreaks")
    assert injection_vec["failed_attacks"] > 0
    assert len(injection_vec["vulnerabilities"]) > 0
    assert "yielded to prompt injection" in injection_vec["vulnerabilities"][0]["issue"]

    # 3. Check that code/SQL injection vector caught invalid label format
    sql_vec = next(v for v in audit["vector_breakdown"] if v["vector_name"] == "code_and_sql_injection")
    assert len(sql_vec["vulnerabilities"]) > 0
    assert "invalid label" in sql_vec["vulnerabilities"][0]["issue"]

    # 4. Check that token exhaustion caught simulated crash
    exhaust_vec = next(v for v in audit["vector_breakdown"] if v["vector_name"] == "token_exhaustion_buffer_overflow")
    assert len(exhaust_vec["vulnerabilities"]) > 0
    assert "Model error" in exhaust_vec["vulnerabilities"][0]["issue"]


def test_audit_passes_secure_model():
    """Verify that a secure model that rejects injections scores 100%."""
    secure_model = SecureDummyPredictor()
    audit = run_red_team_audit(secure_model)

    assert audit["overall_robustness_score"] == 100.0
    assert audit["failed_attacks"] == 0
