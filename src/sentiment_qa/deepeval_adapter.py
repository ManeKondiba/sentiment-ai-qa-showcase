"""DeepEval integration adapter for Sentiment AI QA & Evaluation Framework.

Provides seamless DeepEval evaluation, converting sentiment test cases into
DeepEval LLMTestCases and evaluating them using DeepEval metrics (ExactMatchMetric, G-Eval, etc.).
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric

from .predictor import DemoLexiconPredictor
from .evaluate import get_predictor_instance


def create_deepeval_test_cases(dataset_path: Path, predictor=None) -> list[LLMTestCase]:
    """Convert dataset cases into DeepEval LLMTestCase objects with actual model outputs."""
    if not dataset_path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt_dataset = repo_root / dataset_path
        if alt_dataset.exists():
            dataset_path = alt_dataset

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    test_cases = []

    for case in cases:
        text = case["text"]
        expected = case["expected"]
        predicted = predictor.predict(text)

        test_case = LLMTestCase(
            input=text,
            actual_output=predicted,
            expected_output=expected,
            additional_metadata={"id": case.get("id"), "category": case.get("category")},
        )
        test_cases.append(test_case)

    return test_cases


def run_deepeval_suite(
    dataset_path: Path,
    predictor=None,
    predictor_name: str = "DemoLexiconPredictor",
) -> dict[str, Any]:
    """Run a full DeepEval evaluation suite over the target dataset and predictor."""
    if predictor is None:
        predictor = DemoLexiconPredictor()

    test_cases = create_deepeval_test_cases(dataset_path, predictor)

    print(f"[DeepEval] Evaluating {len(test_cases)} cases with DeepEval metrics...")
    start_time = time.perf_counter()

    results_summary = []
    passed_cases = 0

    for tc in test_cases:
        metric = ExactMatchMetric(threshold=1.0)
        metric.measure(tc)
        success = metric.is_successful()

        if success:
            passed_cases += 1

        results_summary.append({
            "input": tc.input,
            "actual_output": tc.actual_output,
            "expected_output": tc.expected_output,
            "metric_score": metric.score,
            "metric_reason": metric.reason if hasattr(metric, "reason") else None,
            "success": success,
        })

    elapsed_sec = time.perf_counter() - start_time
    total_cases = len(test_cases)
    pass_rate = round(passed_cases / total_cases, 4) if total_cases > 0 else 0.0

    summary = {
        "framework": "DeepEval",
        "predictor": predictor_name,
        "total_test_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "pass_rate": pass_rate,
        "elapsed_seconds": round(elapsed_sec, 2),
        "results_summary": results_summary,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run DeepEval test suite on sentiment dataset.")
    parser.add_argument("--dataset", default="datasets/sentiment_cases.json", help="Path to sentiment JSON dataset")
    parser.add_argument("--predictor", default="demo", choices=["demo", "http", "openai", "huggingface"], help="Predictor to evaluate")
    parser.add_argument("--endpoint", default=None, help="HTTP API Endpoint URL")
    parser.add_argument("--model", default=None, help="Model ID/Name")
    parser.add_argument("--api-key", default=None, help="API key override")
    parser.add_argument("--api-base", default=None, help="Custom API Base URL (for Groq, OpenRouter, Ollama, vLLM)")

    args = parser.parse_args()

    predictor, description = get_predictor_instance(
        type_name=args.predictor,
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )

    summary = run_deepeval_suite(Path(args.dataset), predictor, description)

    print("\n==================================================")
    print("           DEEPEVAL EVALUATION SUMMARY            ")
    print("==================================================")
    print(f"Predictor       : {summary['predictor']}")
    print(f"Total Cases     : {summary['total_test_cases']}")
    print(f"Passed Cases    : {summary['passed_cases']}")
    print(f"Failed Cases    : {summary['failed_cases']}")
    print(f"Pass Rate       : {summary['pass_rate'] * 100:.1f}%")
    print(f"Elapsed Time    : {summary['elapsed_seconds']}s")
    print("==================================================\n")


if __name__ == "__main__":
    main()
