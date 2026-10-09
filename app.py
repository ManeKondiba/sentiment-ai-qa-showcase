"""Interactive Web Application and Dashboard Server for Sentiment AI QA.

Zero external dependencies required. Launch with:
python app.py
"""
import http.server
import json
from pathlib import Path
import urllib.parse

import sys

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_qa.predictor import DemoLexiconPredictor, LABELS
from sentiment_qa.evaluate import run_evaluation
from sentiment_qa.compare import compare_predictors, generate_comparison_html
from sentiment_qa.llm_judge import RuleBasedJudge

PORT = 8080
DATASET_PATH = Path("datasets/sentiment_cases.json")
REPORT_PATH = Path("reports/evaluation.json")
HTML_REPORT_PATH = Path("reports/evaluation.html")


class SentimentQADashboardHandler(http.server.SimpleHTTPRequestHandler):
    predictor = DemoLexiconPredictor()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.render_dashboard_html().encode("utf-8"))
            return

        if path == "/api/evaluate":
            report = run_evaluation(
                dataset_path=DATASET_PATH,
                report_path=REPORT_PATH,
                html_report_path=HTML_REPORT_PATH,
                predictor=self.predictor,
                predictor_name="DemoLexiconPredictor (Baseline)",
            )
            self.send_json(report)
            return

        if path == "/reports/evaluation.html" and HTML_REPORT_PATH.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_REPORT_PATH.read_bytes())
            return

        if path == "/reports/comparison.html":
            comp_path = Path("reports/comparison.html")
            if not comp_path.exists():
                comp_report = compare_predictors(
                    {"DemoLexiconPredictor": self.predictor},
                    DATASET_PATH,
                    judge=RuleBasedJudge(),
                )
                generate_comparison_html(comp_report, comp_path)

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(comp_path.read_bytes())
            return

        return super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                text = data.get("text", "")
                if not text:
                    self.send_error(400, "Field 'text' is required")
                    return
                prediction = self.predictor.predict(text)
                self.send_json({"text": text, "prediction": prediction})
            except Exception as exc:
                self.send_error(500, str(exc))
            return

        self.send_error(404, "Endpoint not found")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def render_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment AI QA Live Web Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .card-custom { border-radius: 12px; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    </style>
</head>
<body class="py-4">
    <div class="container">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h2 class="fw-bold mb-1">Sentiment AI QA Interactive Dashboard</h2>
                <p class="text-muted mb-0">Real-Time Evaluation, Benchmarking & Model Quality Harness</p>
            </div>
            <div class="d-flex gap-2">
                <a href="/reports/evaluation.html" class="btn btn-outline-primary" target="_blank">View Evaluation Dashboard</a>
                <a href="/reports/comparison.html" class="btn btn-outline-secondary" target="_blank">View Comparison Dashboard</a>
            </div>
        </div>

        <!-- Live Prediction Box -->
        <div class="card card-custom p-4 mb-4 bg-white">
            <h5 class="fw-bold mb-3">Live Interactive Sentiment Prediction</h5>
            <div class="input-group mb-3">
                <input type="text" id="inputText" class="form-control form-control-lg" placeholder="Type any review, customer comment, or prompt..." value="I love this product, it is excellent!">
                <button class="btn btn-primary px-4" id="btnPredict">Analyze Sentiment</button>
            </div>
            <div id="predictionResult" class="p-3 bg-light border rounded d-none">
                <strong>Result:</strong> <span id="labelResult" class="badge fs-6 text-capitalize"></span>
            </div>
        </div>

        <!-- Run Full Evaluation Button -->
        <div class="card card-custom p-4 bg-white mb-4">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <div>
                    <h5 class="fw-bold mb-1">Run 315-Case Benchmark Evaluation</h5>
                    <p class="text-muted mb-0">Executes the 315-case test suite against the predictor and computes NLP metrics.</p>
                </div>
                <button class="btn btn-success btn-lg px-4" id="btnRunEval">Run 315-Case Benchmark</button>
            </div>
            <div id="evalMetrics" class="d-none">
                <div class="row g-3 text-center mt-2">
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded">
                            <div class="text-muted small fw-bold">SAMPLE COUNT</div>
                            <div id="mSamples" class="h3 fw-bold text-dark">-</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded">
                            <div class="text-muted small fw-bold">ACCURACY</div>
                            <div id="mAccuracy" class="h3 fw-bold text-primary">-</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded">
                            <div class="text-muted small fw-bold">MACRO F1</div>
                            <div id="mF1" class="h3 fw-bold text-info">-</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-light rounded">
                            <div class="text-muted small fw-bold">THROUGHPUT</div>
                            <div id="mQPS" class="h3 fw-bold text-success">-</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('btnPredict').addEventListener('click', async () => {
            const text = document.getElementById('inputText').value;
            if (!text.trim()) return;

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();
            const box = document.getElementById('predictionResult');
            const badge = document.getElementById('labelResult');

            box.classList.remove('d-none');
            badge.textContent = data.prediction;
            badge.className = 'badge fs-6 text-capitalize ' + 
                (data.prediction === 'positive' ? 'bg-success' : data.prediction === 'negative' ? 'bg-danger' : 'bg-secondary');
        });

        document.getElementById('btnRunEval').addEventListener('click', async () => {
            const btn = document.getElementById('btnRunEval');
            btn.disabled = true;
            btn.textContent = 'Running Benchmark...';

            const res = await fetch('/api/evaluate');
            const data = await res.json();
            btn.disabled = false;
            btn.textContent = 'Run 315-Case Benchmark';

            const metrics = data.metrics || {};
            const benchmark = data.benchmark || {};

            document.getElementById('evalMetrics').classList.remove('d-none');
            document.getElementById('mSamples').textContent = metrics.sample_count || 315;
            document.getElementById('mAccuracy').textContent = ((metrics.accuracy || 0) * 100).toFixed(1) + '%';
            document.getElementById('mF1').textContent = (metrics.macro_f1 || 0).toFixed(3);
            document.getElementById('mQPS').textContent = Math.round(benchmark.throughput_qps || 0) + ' QPS';
        });
    </script>
</body>
</html>
"""


def main():
    server = http.server.HTTPServer(("0.0.0.0", PORT), SentimentQADashboardHandler)
    print(f"--- Sentiment AI QA Interactive Dashboard Server Running ---")
    print(f"Open in browser: http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
