from sentiment_qa.metrics import verify_quality_gate


def test_quality_gate_passed():
    dummy_metrics = {"accuracy": 0.85, "macro_f1": 0.82}
    dummy_benchmark = {"p95_latency_ms": 12.5}

    gate = verify_quality_gate(
        dummy_metrics,
        dummy_benchmark,
        min_accuracy=0.80,
        min_macro_f1=0.75,
        max_p95_latency_ms=100.0,
    )

    assert gate["quality_gate_passed"] is True
    assert len(gate["checks"]) == 3
    assert gate["checks"][0]["passed"] is True


def test_quality_gate_failed_accuracy():
    dummy_metrics = {"accuracy": 0.50, "macro_f1": 0.45}

    gate = verify_quality_gate(
        dummy_metrics,
        min_accuracy=0.80,
        min_macro_f1=0.75,
    )

    assert gate["quality_gate_passed"] is False
    assert gate["checks"][0]["passed"] is False
