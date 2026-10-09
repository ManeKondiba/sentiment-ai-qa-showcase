import sys
from pathlib import Path

# Add root directory to sys.path so app.py can be imported directly in tests
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import SentimentQADashboardHandler


def test_dashboard_handler_class():
    handler = SentimentQADashboardHandler
    assert hasattr(handler, "render_dashboard_html")
    html = handler.render_dashboard_html(None)
    assert "Sentiment AI QA Interactive Dashboard" in html
    assert "Run 315-Case Benchmark" in html
