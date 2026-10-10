"""Predictor adapters for baseline and real AI models/APIs.

Supported predictors:
- DemoLexiconPredictor: Deterministic word-count baseline for demo purposes.
- HttpPredictor: Adapter for custom HTTP REST endpoints.
- OpenAIPredictor: Adapter for OpenAI Chat API and compatible endpoints (Ollama, vLLM, Groq, etc.).
- HuggingFacePredictor: Adapter for Hugging Face Inference API sentiment models.
"""
from dataclasses import dataclass
import json
import os
import re
import time
import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LABELS = ("positive", "negative", "neutral")

POSITIVE_WORDS = {
    "amazing", "excellent", "good", "great", "happy", "helpful",
    "love", "loved", "perfect", "wonderful", "awesome", "enjoy",
}
NEGATIVE_WORDS = {
    "awful", "bad", "broken", "disappointed", "hate", "hated",
    "poor", "terrible", "worst", "unhelpful", "angry", "slow",
}


@dataclass
class Prediction:
    label: str
    positive_hits: int = 0
    negative_hits: int = 0


def normalize_label(raw_label: str) -> str:
    """Normalize raw API model output to canonical ('positive', 'negative', 'neutral')."""
    cleaned = str(raw_label).strip().lower().strip(".\"'`")
    if cleaned in ("pos", "positive", "label_2", "star 5", "star 4", "5 stars", "4 stars"):
        return "positive"
    if cleaned in ("neg", "negative", "label_0", "star 1", "star 2", "1 star", "2 stars"):
        return "negative"
    if cleaned in ("neu", "neutral", "label_1", "star 3", "3 stars"):
        return "neutral"

    # Fallback substring search if model returned full sentence
    if "positive" in cleaned:
        return "positive"
    if "negative" in cleaned:
        return "negative"
    if "neutral" in cleaned:
        return "neutral"

    raise ValueError(f"Unable to normalize model prediction '{raw_label}' to valid label {LABELS}")


def execute_request_with_retry(req: urllib.request.Request, timeout: float = 15.0, max_retries: int = 3, initial_delay: float = 0.5) -> bytes:
    """Execute HTTP request with exponential backoff retry for HTTP 429 and transient 5xx errors."""
    delay = initial_delay
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            # Retry on Rate Limit (429) or Server Errors (500, 502, 503, 504)
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2.0
                continue
            raise RuntimeError(f"HTTP request failed with status code {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2.0
                continue
            raise RuntimeError(f"Network error during request: {exc.reason}") from exc

    raise RuntimeError(f"Request failed after {max_retries} retries: {last_exc}")


class DemoLexiconPredictor:
    """Simple word-count baseline used only to demonstrate the test harness."""

    def predict(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        tokens = set(re.findall(r"[a-z']+", text.lower()))
        positive = len(tokens & POSITIVE_WORDS)
        negative = len(tokens & NEGATIVE_WORDS)

        if positive > negative:
            return "positive"
        if negative > positive:
            return "negative"
        return "neutral"


class HttpPredictor:
    """Adapter for a custom HTTP REST API endpoint with retry handling."""

    def __init__(
        self,
        endpoint: str,
        headers: dict | None = None,
        timeout: float = 15.0,
        text_key: str = "text",
        response_key: str = "label",
        max_retries: int = 3,
    ):
        if not endpoint:
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.text_key = text_key
        self.response_key = response_key
        self.max_retries = max_retries

    def predict(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        payload = json.dumps({self.text_key: text}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers=self.headers,
            method="POST",
        )

        resp_bytes = execute_request_with_retry(req, timeout=self.timeout, max_retries=self.max_retries)
        body = json.loads(resp_bytes.decode("utf-8"))
        raw_label = body.get(self.response_key, "")
        return normalize_label(raw_label)


class OpenAIPredictor:
    """Adapter for OpenAI Chat Completions API or OpenAI-compatible server (vLLM, Ollama, Groq)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 15.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key and "openai.com" in self.api_base:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required")

        # Auto-detect Groq and OpenRouter keys if api_base is standard OpenAI
        if self.api_key.startswith("gsk_") and "openai.com" in self.api_base:
            self.api_base = "https://api.groq.com/openai/v1"
            if self.model == "gpt-4o-mini":
                self.model = "llama-3.1-8b-instant"
        elif self.api_key.startswith("sk-or-v1-") and "openai.com" in self.api_base:
            self.api_base = "https://openrouter.ai/api/v1"
            if self.model == "gpt-4o-mini":
                self.model = "meta-llama/llama-3.2-1b-instruct:free"

    def predict(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        endpoint = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = json.dumps({
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict sentiment classifier. Analyze the user text and output EXACTLY "
                        "one word: 'positive', 'negative', or 'neutral'. Do not output any punctuation or explanation."
                    ),
                },
                {"role": "user", "content": text},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        resp_bytes = execute_request_with_retry(req, timeout=self.timeout, max_retries=self.max_retries)
        body = json.loads(resp_bytes.decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return normalize_label(content)


class HuggingFacePredictor:
    """Adapter for Hugging Face Inference API sentiment models."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        timeout: float = 15.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_TOKEN", "")
        self.model_id = model_id
        self.timeout = timeout
        self.max_retries = max_retries

    def predict(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

        endpoint = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = json.dumps({"inputs": text}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")

        resp_bytes = execute_request_with_retry(req, timeout=self.timeout, max_retries=self.max_retries)
        body = json.loads(resp_bytes.decode("utf-8"))
        if isinstance(body, list) and body and isinstance(body[0], list):
            top_candidate = max(body[0], key=lambda x: x.get("score", 0.0))
            raw_label = top_candidate.get("label", "")
        elif isinstance(body, dict) and "label" in body:
            raw_label = body["label"]
        else:
            raw_label = str(body)
        return normalize_label(raw_label)
