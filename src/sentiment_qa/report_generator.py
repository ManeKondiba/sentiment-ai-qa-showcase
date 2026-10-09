"""HTML Dashboard Report Generator for Sentiment AI QA Evaluations."""
import html
import json
from pathlib import Path


def generate_html_report(report_data: dict, output_path: Path) -> Path:
    """Generate a responsive, standalone HTML report for evaluation results."""
    metrics = report_data.get("metrics", {})
    benchmark = report_data.get("benchmark", {})
    cases = report_data.get("cases", [])
    predictor = report_data.get("predictor", "Unknown Predictor")
    generated_at = report_data.get("generated_at_utc", "")
    
    accuracy = metrics.get("accuracy", 0.0) * 100
    macro_f1 = metrics.get("macro_f1", 0.0)
    sample_count = metrics.get("sample_count", 0)
    passed_count = sum(1 for c in cases if c.get("passed"))
    failed_count = sample_count - passed_count
    
    per_class = metrics.get("per_class", {})
    matrix = metrics.get("confusion_matrix", {})
    labels = metrics.get("label_order", ["positive", "negative", "neutral"])

    avg_latency = benchmark.get("avg_latency_ms", 0.0)
    p95_latency = benchmark.get("p95_latency_ms", 0.0)
    throughput = benchmark.get("throughput_qps", 0.0)

    # Generate Confusion Matrix Rows
    cm_rows_html = ""
    for actual in labels:
        row_cells = f"<td class='fw-bold text-capitalize bg-light'>{html.escape(actual)}</td>"
        for pred in labels:
            count = matrix.get(actual, {}).get(pred, 0)
            if actual == pred:
                bg_style = "background-color: #d1e7dd; color: #0f5132; font-weight: bold;" if count > 0 else "background-color: #f8f9fa;"
            else:
                bg_style = "background-color: #f8d7da; color: #842029; font-weight: bold;" if count > 0 else "background-color: #f8f9fa;"
            row_cells += f"<td style='{bg_style}' class='text-center'>{count}</td>"
        cm_rows_html += f"<tr>{row_cells}</tr>"

    # Generate Per-Class Metrics Rows
    per_class_html = ""
    for label, vals in per_class.items():
        per_class_html += f"""
        <tr>
            <td class="text-capitalize fw-bold">{html.escape(label)}</td>
            <td>{vals.get('precision', 0.0):.4f}</td>
            <td>{vals.get('recall', 0.0):.4f}</td>
            <td>{vals.get('f1', 0.0):.4f}</td>
            <td>{vals.get('support', 0)}</td>
        </tr>
        """

    # Generate Case Details Rows
    case_rows_html = ""
    for c in cases:
        passed = c.get("passed", False)
        status_badge = '<span class="badge bg-success">PASS</span>' if passed else '<span class="badge bg-danger">FAIL</span>'
        latency_val = c.get("latency_ms", 0.0)
        text_content = html.escape(str(c.get("text", "")))
        
        case_rows_html += f"""
        <tr class="case-row" data-category="{html.escape(c.get('category', ''))}" data-passed="{str(passed).lower()}">
            <td><code>{html.escape(c.get('id', ''))}</code></td>
            <td><span class="badge bg-secondary">{html.escape(c.get('category', ''))}</span></td>
            <td class="text-capitalize">{html.escape(c.get('expected', ''))}</td>
            <td class="text-capitalize">{html.escape(c.get('predicted', ''))}</td>
            <td>{status_badge}</td>
            <td><small class="text-muted">{latency_val} ms</small></td>
            <td class="text-wrap">{text_content}</td>
        </tr>
        """

    categories = sorted(list({c.get("category", "") for c in cases}))
    category_options = "".join([f'<option value="{html.escape(cat)}">{html.escape(cat)}</option>' for cat in categories])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment AI QA Evaluation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .metric-card {{ border-radius: 12px; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        .table-wrap {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        .cm-table th, .cm-table td {{ vertical-align: middle; }}
    </style>
</head>
<body class="py-4">
    <div class="container">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h2 class="fw-bold mb-1">Sentiment AI QA Report</h2>
                <p class="text-muted mb-0">Model: <strong>{html.escape(predictor)}</strong> | Generated: {html.escape(generated_at)}</p>
            </div>
            <div>
                <span class="badge bg-primary fs-6 p-2">Total Cases: {sample_count}</span>
            </div>
        </div>

        <!-- Accuracy & Quality Cards -->
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card metric-card bg-white p-3 text-center">
                    <div class="text-muted small fw-bold">ACCURACY</div>
                    <div class="display-6 fw-bold text-primary">{accuracy:.1f}%</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card bg-white p-3 text-center">
                    <div class="text-muted small fw-bold">MACRO F1</div>
                    <div class="display-6 fw-bold text-info">{macro_f1:.3f}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card bg-white p-3 text-center">
                    <div class="text-muted small fw-bold">AVG LATENCY</div>
                    <div class="display-6 fw-bold text-secondary">{avg_latency:.1f}<small class="fs-6 text-muted">ms</small></div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card bg-white p-3 text-center">
                    <div class="text-muted small fw-bold">P95 LATENCY</div>
                    <div class="display-6 fw-bold text-warning">{p95_latency:.1f}<small class="fs-6 text-muted">ms</small></div>
                </div>
            </div>
        </div>

        <!-- Confusion Matrix & Class Metrics -->
        <div class="row g-4 mb-4">
            <div class="col-lg-6">
                <div class="table-wrap h-100">
                    <h5 class="fw-bold mb-3">Confusion Matrix</h5>
                    <table class="table table-bordered cm-table">
                        <thead>
                            <tr class="table-light text-center">
                                <th>Actual \\ Pred</th>
                                {"".join(f'<th class="text-capitalize">{html.escape(l)}</th>' for l in labels)}
                            </tr>
                        </thead>
                        <tbody>
                            {cm_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="table-wrap h-100">
                    <h5 class="fw-bold mb-3">Per-Class Metrics</h5>
                    <table class="table table-hover align-middle">
                        <thead>
                            <tr class="table-light">
                                <th>Class</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
                            {per_class_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Cases Table -->
        <div class="table-wrap">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="fw-bold mb-0">Detailed Test Cases</h5>
                <div class="d-flex gap-2">
                    <select id="categoryFilter" class="form-select form-select-sm" style="width: 200px;">
                        <option value="ALL">All Categories</option>
                        {category_options}
                    </select>
                    <select id="statusFilter" class="form-select form-select-sm" style="width: 150px;">
                        <option value="ALL">All Status</option>
                        <option value="true">Passed Only</option>
                        <option value="false">Failed Only</option>
                    </select>
                </div>
            </div>
            <div class="table-responsive">
                <table class="table table-hover align-middle" id="casesTable">
                    <thead>
                        <tr class="table-light">
                            <th>ID</th>
                            <th>Category</th>
                            <th>Expected</th>
                            <th>Predicted</th>
                            <th>Result</th>
                            <th>Latency</th>
                            <th>Input Text</th>
                        </tr>
                    </thead>
                    <tbody>
                        {case_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const categoryFilter = document.getElementById('categoryFilter');
        const statusFilter = document.getElementById('statusFilter');
        const rows = document.querySelectorAll('.case-row');

        function filterRows() {{
            const cat = categoryFilter.value;
            const status = statusFilter.value;

            rows.forEach(row => {{
                const rowCat = row.getAttribute('data-category');
                const rowPassed = row.getAttribute('data-passed');
                const catMatch = (cat === 'ALL' || rowCat === cat);
                const statusMatch = (status === 'ALL' || rowPassed === status);

                if (catMatch && statusMatch) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}

        categoryFilter.addEventListener('change', filterRows);
        statusFilter.addEventListener('change', filterRows);
    </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
