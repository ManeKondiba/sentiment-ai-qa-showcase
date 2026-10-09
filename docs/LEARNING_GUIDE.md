# Sentiment AI QA & Evaluation Framework — Comprehensive Learning Guide

Welcome to the **Sentiment AI QA & Evaluation Framework** learning guide! This document is designed to teach you how automated testing, benchmarking, and evaluation work for Sentiment Analysis and NLP AI models.

---

## 1. Tech Stack & Tooling Breakdown

Here are the languages, frameworks, libraries, and tools used to build this framework:

| Category | Tool / Language | Purpose / Role |
|---|---|---|
| **Primary Language** | **Python 3.10+** (Python 3.13) | Core framework, adapters, metrics calculation, and evaluation pipeline. |
| **Testing Framework** | **Pytest 8.x** | Automated unit testing, edge-case testing, and robustness verification. |
| **Standard Library** | `urllib.request`, `json`, `dataclasses`, `statistics`, `argparse`, `re` | Standard library components ensuring zero required third-party dependencies for baseline execution. |
| **CI/CD Automation** | **GitHub Actions** | Continuous integration pipeline executing unit tests and evaluation reports on push/PR (`.github/workflows/eval.yml`). |
| **Reporting UI** | **HTML5, Bootstrap 5, Vanilla JavaScript** | Standalone interactive visual dashboards (`reports/evaluation.html` & `reports/comparison.html`). |
| **LLM Integration** | **OpenAI API Protocol** (REST API) | LLM-as-a-Judge evaluation and prompt-injection safety auditing. |

---

## 2. System Architecture & Design Patterns

The codebase adheres to clean software engineering design patterns to isolate the test framework from specific model implementations:

```text
                               ┌──────────────────────────────────┐
                               │     datasets/sentiment_cases.json│
                               └────────────────┬─────────────────┘
                                                │ (Test Examples)
                                                ▼
┌─────────────────────────┐            ┌─────────────────┐
│     Predictor Adapters   │──────────► │  evaluate.py    │
│  - DemoLexiconPredictor │            │  - Inference    │
│  - HttpPredictor        │            │  - Benchmarking │
│  - OpenAIPredictor      │            └────────┬────────┘
│  - HuggingFacePredictor │                     │
└─────────────────────────┘                     ▼
                               ┌──────────────────────────────────┐
                               │          metrics.py              │
                               │ Accuracy, F1, Confusion Matrix   │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │      report_generator.py         │
                               │ JSON & Interactive HTML Reports  │
                               └──────────────────────────────────┘
```

### Key Design Patterns:
1. **Adapter Pattern** ([predictor.py](file:///c:/Users/HP/Downloads/sentiment-ai-qa-showcase/src/sentiment_qa/predictor.py)):
   - Defines a unified contract: `predictor.predict(text: str) -> str`.
   - Allows switching between local rule baselines (`DemoLexiconPredictor`), REST microservices (`HttpPredictor`), and cloud LLMs (`OpenAIPredictor`) without altering test code.

2. **Factory Pattern** (`get_predictor_instance` in [evaluate.py](file:///c:/Users/HP/Downloads/sentiment-ai-qa-showcase/src/sentiment_qa/evaluate.py)):
   - Instantiates the target model dynamically based on command-line flags (`--predictor demo|http|openai|huggingface`).

3. **Resilience Pattern (Exponential Backoff Retry)**:
   - Function `execute_request_with_retry()` automatically handles HTTP 429 (Rate Limit) and transient 5xx server errors with progressive delay retries.

4. **Taxonomy Normalization**:
   - `normalize_label()` maps varied API response strings (`"LABEL_0"`, `"pos"`, `"5 stars"`) into canonical labels: `"positive"`, `"negative"`, or `"neutral"`.

---

## 3. Core NLP Classification Metrics

Evaluating a machine learning classification model requires measuring precision, recall, and class balance rather than just raw accuracy.

### Mathematical Definitions

1. **Accuracy**: The ratio of correct predictions to total samples.
   $$Accuracy = \frac{Correct\ Predictions}{Total\ Samples}$$

2. **Precision**: The accuracy of positive predictions for a given class.
   $$Precision = \frac{True\ Positives (TP)}{True\ Positives (TP) + False\ Positives (FP)}$$

3. **Recall**: The ability of the model to find all positive instances of a class.
   $$Recall = \frac{True\ Positives (TP)}{True\ Positives (TP) + False\ Negatives (FN)}$$

4. **F1-Score**: Harmonic mean of Precision and Recall.
   $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

5. **Macro-F1**: The unweighted average of F1 scores across all classes (`positive`, `negative`, `neutral`). Ensures fair scoring when datasets have imbalanced class counts.
   $$Macro\ F1 = \frac{F1_{positive} + F1_{negative} + F1_{neutral}}{3}$$

6. **Confusion Matrix**: A $3 \times 3$ grid comparing actual true labels against model predicted labels to identify exact misclassification patterns.

---

## 4. LLM-as-a-Judge & Multi-Model Comparison

### Why use an LLM as a Judge?
While ground-truth dataset metrics tell you *how many* cases failed, an **LLM Judge** ([llm_judge.py](file:///c:/Users/HP/Downloads/sentiment-ai-qa-showcase/src/sentiment_qa/llm_judge.py)) provides *qualitative reasoning* for *why* a model failed.

### Dual Evaluation Paradigm:
- **Ground Truth Evaluation**: Compares model output directly against human-annotated labels.
- **LLM Judge Audit**: Asks a high-capacity model (e.g., GPT-4) to read the input text, ground truth, and prediction to provide a confidence score ($0.0 - 1.0$) and textual explanation.
- **Inter-Model Agreement Rate**: Measures consensus percentage between multiple candidate models ([compare.py](file:///c:/Users/HP/Downloads/sentiment-ai-qa-showcase/src/sentiment_qa/compare.py)).

---

## 5. Deployment Readiness Guide

Before deploying this evaluation framework to production or continuous integration, follow this checklist:

### Pre-Deployment Checklist
- [x] All unit tests pass cleanly (`pytest -v`).
- [x] Environment variables configured (`OPENAI_API_KEY`, `HF_API_KEY`, or REST API endpoints).
- [x] Dataset expanded to include representative domain traffic (300+ annotated examples recommended for production claims).
- [x] GitHub Actions workflow `.github/workflows/eval.yml` committed to repository.

### How to Run in Production CI/CD
In your automated deployment pipeline, execute:
```bash
# 1. Run unit tests
pytest -v

# 2. Execute evaluation & generate reports
python -m sentiment_qa.evaluate --dataset datasets/sentiment_cases.json --report reports/evaluation.json

# 3. Compare candidate model against baseline
python -m sentiment_qa.compare --dataset datasets/sentiment_cases.json --report reports/comparison.json
```
