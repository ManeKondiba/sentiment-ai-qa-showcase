"""Run sentiment evaluation on a labeled JSON dataset and save JSON & HTML reports."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import evaluate, verify_quality_gate
from .predictor import (
    DemoLexiconPredictor,
    HttpPredictor,
    OpenAIPredictor,
    HuggingFacePredictor,
)
from .report_generator import generate_html_report
from .benchmark import benchmark_predictor


def get_predictor_instance(type_name: str, endpoint: str | None = None, model: str | None = None, api_key: str | None = None, api_base: str | None = None):
    """Factory function to build a predictor from CLI options."""
    name = type_name.lower().strip()
    if name == "demo":
        return DemoLexiconPredictor(), "DemoLexiconPredictor (illustrative word-count baseline)"

    if name == "http":
        if not endpoint:
            raise ValueError("--endpoint URL is required when using --predictor http")
        return HttpPredictor(endpoint=endpoint), f"HttpPredictor ({endpoint})"

    if name == "openai":
        model_name = model or "gpt-4o-mini"
        base_url = api_base or "https://api.openai.com/v1"
        predictor = OpenAIPredictor(api_key=api_key, model=model_name, api_base=base_url)
        return predictor, f"OpenAIPredictor ({predictor.model} @ {predictor.api_base})"

    if name == "huggingface":
        model_name = model or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        return HuggingFacePredictor(api_key=api_key, model_id=model_name), f"HuggingFacePredictor ({model_name})"

    raise ValueError(f"Unknown predictor type: {type_name}. Choose from demo, http, openai, huggingface.")


def run_evaluation(
    dataset_path: Path,
    report_path: Path,
    html_report_path: Path | None = None,
    predictor=None,
    predictor_name: str = "DemoLexiconPredictor",
    min_accuracy: float = 0.80,
    min_macro_f1: float = 0.75,
    max_p95_latency_ms: float = 1000.0,
) -> dict:
    if not dataset_path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt_dataset = repo_root / dataset_path
        if alt_dataset.exists():
            dataset_path = alt_dataset
        else:
            raise FileNotFoundError(f"Dataset file not found at '{dataset_path}' or '{alt_dataset}'")

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    y_true, y_pred, rows = [], [], []

    for case in cases:
        t0 = time.perf_counter()
        prediction = predictor.predict(case["text"])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        y_true.append(case["expected"])
        y_pred.append(prediction)
        rows.append({
            "id": case["id"],
            "category": case.get("category", "unspecified"),
            "expected": case["expected"],
            "predicted": prediction,
            "passed": case["expected"] == prediction,
            "latency_ms": round(latency_ms, 2),
            "text": case["text"],
        })

    texts = [c["text"] for c in cases]
    benchmark = benchmark_predictor(predictor, texts)
    metrics_result = evaluate(y_true, y_pred)
    quality_gate = verify_quality_gate(
        metrics_result,
        benchmark.to_dict(),
        min_accuracy=min_accuracy,
        min_macro_f1=min_macro_f1,
        max_p95_latency_ms=max_p95_latency_ms,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictor": predictor_name,
        "metrics": metrics_result,
        "benchmark": benchmark.to_dict(),
        "quality_gate": quality_gate,
        "cases": rows,
        "limitations": [
            "Sample dataset size; review larger representative dataset for production claims.",
            "Verify label definitions match evaluation taxonomy.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if html_report_path is None:
        html_report_path = report_path.with_suffix(".html")

    generate_html_report(report, html_report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment model on JSON dataset.")
    parser.add_argument("--dataset", default="datasets/sentiment_cases.json", help="Path to evaluation JSON dataset")
    parser.add_argument("--report", default="reports/evaluation.json", help="Path to output JSON report")
    parser.add_argument("--html-report", default=None, help="Path to output HTML report (defaults to matching --report path)")
    parser.add_argument("--predictor", default="demo", choices=["demo", "http", "openai", "huggingface"], help="Predictor adapter to evaluate")
    parser.add_argument("--endpoint", default=None, help="HTTP API Endpoint URL (for --predictor http)")
    parser.add_argument("--model", default=None, help="Model ID/Name (for --predictor openai or huggingface)")
    parser.add_argument("--api-key", default=None, help="API key override (or use environment variables)")
    parser.add_argument("--api-base", default=None, help="Custom API Base URL (for Groq, OpenRouter, Ollama, vLLM)")
    parser.add_argument("--min-accuracy", type=float, default=0.80, help="Minimum accuracy SLA threshold")
    parser.add_argument("--min-f1", type=float, default=0.75, help="Minimum Macro-F1 SLA threshold")
    parser.add_argument("--max-latency", type=float, default=1000.0, help="Maximum P95 latency (ms) SLA threshold")

    args = parser.parse_args()

    try:
        predictor, description = get_predictor_instance(
            type_name=args.predictor,
            endpoint=args.endpoint,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
        )
    except ValueError as exc:
        print(f"\n[ERROR] Predictor Configuration Failed: {exc}")
        print("\nTo evaluate using AI models (OpenAI / Groq / HuggingFace), please provide your API key:")
        print("  1. PowerShell: $env:OPENAI_API_KEY=\"gsk_your_key_here\"")
        print("  2. CLI Flag:   python -m sentiment_qa.evaluate --predictor openai --api-key \"gsk_your_key_here\"")
        print("  3. File:       Create a '.env' file containing: OPENAI_API_KEY=gsk_your_key_here\n")
        return

    report_path = Path(args.report)
    html_path = Path(args.html_report) if args.html_report else report_path.with_suffix(".html")

    report = run_evaluation(
        dataset_path=Path(args.dataset),
        report_path=report_path,
        html_report_path=html_path,
        predictor=predictor,
        predictor_name=description,
        min_accuracy=args.min_accuracy,
        min_macro_f1=args.min_f1,
        max_p95_latency_ms=args.max_latency,
    )

    print(f"--- Sentiment QA Report: {description} ---")
    print(json.dumps(report["metrics"], indent=2))
    print(f"\n--- Quality Gate SLA Status ---")
    print(f"Passed: {report['quality_gate']['quality_gate_passed']}")
    print(json.dumps(report["quality_gate"]["checks"], indent=2))
    print(f"\nJSON Report written to: {report_path}")
    print(f"HTML Dashboard written to: {html_path}")


if __name__ == "__main__":
    main()
