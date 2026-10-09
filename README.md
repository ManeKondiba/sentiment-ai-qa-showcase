# Sentiment AI QA Showcase

A beginner-friendly portfolio project demonstrating how to test, benchmark, evaluate, and compare sentiment classifiers with Python and Pytest.

> **Important:** The included `DemoLexiconPredictor` is a small, deterministic demo baseline. Production AI evaluations can be run against real AI models using `HttpPredictor`, `OpenAIPredictor`, or `HuggingFacePredictor`.

## What this project demonstrates

- Functional checks for `positive`, `negative`, and `neutral` labels
- Input validation and edge-case tests
- Labeled-dataset evaluation: accuracy, per-class precision/recall/F1, macro-F1, confusion matrix
- Performance & Latency Benchmarking (Avg, P50, P95, P99 latency, Throughput QPS)
- **Adversarial Safety & Red-Teaming Audit**: 6-vector security audit (`src/sentiment_qa/red_team.py`)
- **Interactive Web App & Live Dashboard**: Zero-dependency Web Server (`app.py`) for live prediction testing & 315-case benchmarks
- **LLM-as-a-Judge Evaluation Pipeline**: Independent AI model auditing & qualitative reasoning (`src/sentiment_qa/llm_judge.py`)
- **Multi-Model Side-by-Side Comparison**: Inter-model agreement matrix & side-by-side comparison dashboard (`src/sentiment_qa/compare.py`)
- Support for real AI model adapters (`OpenAI`, `HuggingFace`, `HTTP REST APIs`)
- Robust label normalization across model prediction formats
- Interactive HTML Dashboard generation (`reports/evaluation.html` & `reports/comparison.html`)
- GitHub Actions CI/CD pipeline (`.github/workflows/eval.yml`)

## Requirements

- Python 3.10+
- standard library (zero required external dependencies for baseline execution)
- `pytest` (for running unit tests)

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -v
python app.py
```

Open your browser at: `http://localhost:8080`

## CLI Evaluation, Security Audit & Comparison

```powershell
# Run 315-case evaluation
python -m sentiment_qa.evaluate --dataset datasets/sentiment_cases.json --report reports/evaluation.json

# Run 6-vector Adversarial Safety Red-Teaming Audit
python -m sentiment_qa.red_team

# Run Multi-Model Side-by-Side Comparison
python -m sentiment_qa.compare --dataset datasets/sentiment_cases.json --report reports/comparison.json
```

## Side-by-Side Model Comparison & LLM-as-a-Judge

Run comparative audits across multiple models side-by-side:

```powershell
python -m sentiment_qa.compare --dataset datasets/sentiment_cases.json --report reports/comparison.json --html-report reports/comparison.html
```

Outputs:
- **`reports/comparison.json`**: Pairwise agreement matrices, per-model classification metrics, and LLM judge qualitative explanations.
- **`reports/comparison.html`**: Interactive visual dashboard featuring model cards, consensus badges, and detailed disagreement matrices.

## Evaluating Real AI Models

The evaluation pipeline includes adapters for real AI endpoints:

### 1. Custom HTTP REST API
```powershell
python -m sentiment_qa.evaluate --predictor http --endpoint http://localhost:8000/api/predict
```

### 2. OpenAI (or OpenAI-Compatible API like Ollama, vLLM, Groq)
```powershell
$env:OPENAI_API_KEY="your-api-key"
python -m sentiment_qa.evaluate --predictor openai --model gpt-4o-mini
```

### 3. Hugging Face Inference API
```powershell
$env:HF_API_KEY="your-hf-key"
python -m sentiment_qa.evaluate --predictor huggingface --model cardiffnlp/twitter-roberta-base-sentiment-latest
```

## Project structure

```text
sentiment-ai-qa-showcase/
├── .github/workflows/
│   └── eval.yml
├── datasets/
│   └── sentiment_cases.json
├── reports/
│   ├── evaluation.json
│   ├── evaluation.html
│   ├── comparison.json
│   └── comparison.html
├── src/sentiment_qa/
│   ├── __init__.py
│   ├── predictor.py
│   ├── metrics.py
│   ├── benchmark.py
│   ├── report_generator.py
│   ├── llm_judge.py
│   ├── compare.py
│   ├── red_team.py
│   └── evaluate.py
├── tests/
│   ├── test_app.py
│   ├── test_functional.py
│   ├── test_edge_cases.py
│   ├── test_metrics.py
│   ├── test_predictors.py
│   ├── test_benchmark.py
│   ├── test_report_generator.py
│   ├── test_robustness.py
│   ├── test_llm_judge.py
│   ├── test_compare.py
│   └── test_red_team.py
├── app.py
├── requirements.txt
├── pyproject.toml
└── README.md
```
