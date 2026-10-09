"""Multi-model comparison pipeline and side-by-side evaluation runner."""
import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .metrics import evaluate
from .predictor import DemoLexiconPredictor, LABELS
from .llm_judge import LLMJudge, RuleBasedJudge


def compare_predictors(
    predictors: dict,
    dataset_path: Path,
    judge=None,
) -> dict:
    """Compare multiple predictor models side-by-side on a labeled dataset."""
    if not predictors:
        raise ValueError("At least one predictor must be provided for comparison.")

    if judge is None:
        judge = RuleBasedJudge()

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    model_names = list(predictors.keys())

    # Data structure to hold per-case comparison
    comparison_cases = []
    model_predictions = {name: [] for name in model_names}
    y_true = [c["expected"] for c in cases]

    for case in cases:
        text = case["text"]
        expected = case["expected"]
        case_id = case["id"]
        category = case.get("category", "unspecified")

        preds_by_model = {}
        for name, pred_obj in predictors.items():
            try:
                pred = pred_obj.predict(text)
            except Exception as exc:
                pred = f"ERROR: {exc}"
            preds_by_model[name] = pred
            model_predictions[name].append(pred)

        # Inter-model agreement check: do all models agree?
        valid_preds = [p for p in preds_by_model.values() if not p.startswith("ERROR")]
        all_agree = len(set(valid_preds)) == 1 if valid_preds else False

        # Audit with Judge
        first_pred = valid_preds[0] if valid_preds else "neutral"
        verdict = judge.judge_case(text, expected, first_pred)

        comparison_cases.append({
            "id": case_id,
            "category": category,
            "text": text,
            "expected": expected,
            "predictions": preds_by_model,
            "all_agree": all_agree,
            "judge_label": verdict.judge_label,
            "judge_explanation": verdict.explanation,
        })

    # Calculate individual model metrics
    model_metrics = {}
    for name, y_pred in model_predictions.items():
        try:
            model_metrics[name] = evaluate(y_true, y_pred)
        except Exception as exc:
            model_metrics[name] = {"error": str(exc)}

    # Calculate pairwise agreement rates
    agreement_matrix = {}
    for m1 in model_names:
        agreement_matrix[m1] = {}
        for m2 in model_names:
            p1 = model_predictions[m1]
            p2 = model_predictions[m2]
            agreed = sum(a == b for a, b in zip(p1, p2))
            rate = agreed / len(p1) if p1 else 0.0
            agreement_matrix[m1][m2] = round(rate, 4)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "compared_models": model_names,
        "sample_count": len(cases),
        "model_metrics": model_metrics,
        "pairwise_agreement": agreement_matrix,
        "cases": comparison_cases,
    }
    return report


def generate_comparison_html(report: dict, output_path: Path) -> Path:
    """Generate an interactive side-by-side HTML comparison dashboard."""
    models = report.get("compared_models", [])
    metrics = report.get("model_metrics", {})
    agreement = report.get("pairwise_agreement", {})
    cases = report.get("cases", [])
    generated_at = report.get("generated_at_utc", "")

    # Header metric cards for each model
    cards_html = ""
    for name in models:
        m = metrics.get(name, {})
        acc = m.get("accuracy", 0.0) * 100
        f1 = m.get("macro_f1", 0.0)
        cards_html += f"""
        <div class="col">
            <div class="card h-100 shadow-sm border-0">
                <div class="card-body text-center">
                    <h6 class="text-muted fw-bold text-uppercase">{html.escape(name)}</h6>
                    <div class="display-6 fw-bold text-primary">{acc:.1f}%</div>
                    <div class="text-muted small">Macro F1: <strong>{f1:.3f}</strong></div>
                </div>
            </div>
        </div>
        """

    # Model comparison table headers
    table_headers = "".join([f'<th class="text-capitalize">{html.escape(m)}</th>' for m in models])

    # Table rows
    rows_html = ""
    for c in cases:
        preds = c.get("predictions", {})
        expected = c.get("expected", "")
        all_agree = c.get("all_agree", False)
        
        row_bg = "" if all_agree else "table-warning-subtle"

        pred_cells = ""
        for m in models:
            val = preds.get(m, "")
            is_correct = (val == expected)
            badge_class = "bg-success" if is_correct else "bg-danger"
            pred_cells += f'<td><span class="badge {badge_class} text-capitalize">{html.escape(val)}</span></td>'

        agree_badge = '<span class="badge bg-success">AGREE</span>' if all_agree else '<span class="badge bg-warning text-dark">DISAGREE</span>'

        rows_html += f"""
        <tr class="{row_bg}">
            <td><code>{html.escape(c.get("id", ""))}</code></td>
            <td><span class="badge bg-secondary">{html.escape(c.get("category", ""))}</span></td>
            <td class="fw-bold text-capitalize">{html.escape(expected)}</td>
            {pred_cells}
            <td>{agree_badge}</td>
            <td><small>{html.escape(c.get("judge_explanation", ""))}</small></td>
            <td class="text-wrap">{html.escape(c.get("text", ""))}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Models Side-by-Side Comparison</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .table-wrap {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
    </style>
</head>
<body class="py-4">
    <div class="container-fluid px-4">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h2 class="fw-bold mb-1">Sentiment Models Side-by-Side Comparison</h2>
                <p class="text-muted mb-0">LLM-as-a-Judge Evaluation Audit | Generated: {html.escape(generated_at)}</p>
            </div>
        </div>

        <!-- Model Metric Cards -->
        <div class="row row-cols-1 row-cols-md-{min(len(models), 4)} g-3 mb-4">
            {cards_html}
        </div>

        <!-- Detailed Comparison Table -->
        <div class="table-wrap">
            <h5 class="fw-bold mb-3">Model Prediction & Disagreement Matrix</h5>
            <div class="table-responsive">
                <table class="table table-hover align-middle">
                    <thead>
                        <tr class="table-light">
                            <th>ID</th>
                            <th>Category</th>
                            <th>Ground Truth</th>
                            {table_headers}
                            <th>Consensus</th>
                            <th>Judge Explanation</th>
                            <th>Input Text</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run side-by-side predictor comparison and LLM judge evaluation.")
    parser.add_argument("--dataset", default="datasets/sentiment_cases.json", help="Path to evaluation dataset")
    parser.add_argument("--report", default="reports/comparison.json", help="Path to JSON comparison report")
    parser.add_argument("--html-report", default="reports/comparison.html", help="Path to HTML comparison report")
    args = parser.parse_args()

    # Compare Demo Baseline against RuleBasedJudge / LLMJudge
    predictors = {
        "DemoLexiconPredictor": DemoLexiconPredictor(),
    }

    judge = LLMJudge()
    dataset_path = Path(args.dataset)
    json_path = Path(args.report)
    html_path = Path(args.html_report)

    report = compare_predictors(predictors, dataset_path, judge=judge)
    
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    generate_comparison_html(report, html_path)

    print(f"--- Side-by-Side Model Comparison ---")
    print(json.dumps(report["model_metrics"], indent=2))
    print(f"\nComparison JSON written to: {json_path}")
    print(f"Comparison HTML written to: {html_path}")


if __name__ == "__main__":
    main()
