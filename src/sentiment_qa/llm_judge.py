"""LLM-as-a-Judge module for evaluating and auditing sentiment predictions."""
from dataclasses import dataclass
import json
import os
import urllib.request
import urllib.error

from .predictor import normalize_label, LABELS


@dataclass
class JudgeVerdict:
    text: str
    expected: str
    predicted: str
    judge_label: str
    judge_agrees_with_predicted: bool
    judge_agrees_with_expected: bool
    score: float
    explanation: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "expected": self.expected,
            "predicted": self.predicted,
            "judge_label": self.judge_label,
            "judge_agrees_with_predicted": self.judge_agrees_with_predicted,
            "judge_agrees_with_expected": self.judge_agrees_with_expected,
            "score": round(self.score, 2),
            "explanation": self.explanation,
        }


class RuleBasedJudge:
    """Deterministic offline judge used for automated testing without API credentials."""

    def judge_case(self, text: str, expected: str, predicted: str) -> JudgeVerdict:
        is_pred_correct = (expected.lower() == predicted.lower())
        score = 1.0 if is_pred_correct else 0.0
        explanation = (
            f"Prediction '{predicted}' matches ground truth '{expected}'."
            if is_pred_correct
            else f"Prediction '{predicted}' deviates from ground truth '{expected}'."
        )
        return JudgeVerdict(
            text=text,
            expected=expected,
            predicted=predicted,
            judge_label=expected.lower(),
            judge_agrees_with_predicted=is_pred_correct,
            judge_agrees_with_expected=True,
            score=score,
            explanation=explanation,
        )


class LLMJudge:
    """LLM-as-a-Judge using OpenAI Chat Completions API (or compatible local LLMs)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 15.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._fallback_judge = RuleBasedJudge()

    def judge_case(self, text: str, expected: str, predicted: str) -> JudgeVerdict:
        if not self.api_key and "openai.com" in self.api_base:
            # Fallback to rule-based evaluation if API key is not provided
            return self._fallback_judge.judge_case(text, expected, predicted)

        endpoint = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        system_prompt = (
            "You are an expert NLP Quality Assurance Judge. Your job is to evaluate a sentiment classification case.\n"
            "Analyze the input text, the ground truth expected sentiment, and the model's predicted sentiment.\n"
            "Return a JSON object with EXACTLY these keys:\n"
            "- judge_label: your independent sentiment assessment ('positive', 'negative', 'neutral')\n"
            "- score: rating between 0.0 (completely wrong prediction) and 1.0 (perfect prediction)\n"
            "- explanation: concise reasoning for your judgment\n"
            "Output JSON ONLY."
        )

        user_content = json.dumps({
            "input_text": text,
            "ground_truth_expected": expected,
            "model_prediction": predicted,
        })

        payload = json.dumps({
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                content_str = body["choices"][0]["message"]["content"]
                result = json.loads(content_str)

                raw_judge_label = result.get("judge_label", expected)
                judge_label = normalize_label(raw_judge_label)
                score = float(result.get("score", 1.0 if judge_label == predicted else 0.0))
                explanation = str(result.get("explanation", "No explanation provided."))

                return JudgeVerdict(
                    text=text,
                    expected=expected,
                    predicted=predicted,
                    judge_label=judge_label,
                    judge_agrees_with_predicted=(judge_label == predicted),
                    judge_agrees_with_expected=(judge_label == expected),
                    score=score,
                    explanation=explanation,
                )
        except Exception as exc:
            # On network/API error, fallback gracefully
            fallback = self._fallback_judge.judge_case(text, expected, predicted)
            fallback.explanation += f" (LLM judge fallback due to: {exc})"
            return fallback
