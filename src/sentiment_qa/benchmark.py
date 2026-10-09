"""Performance and latency benchmarking module for sentiment predictors."""
import time
from dataclasses import dataclass
import statistics


@dataclass
class BenchmarkResult:
    sample_count: int
    total_time_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "throughput_qps": round(self.throughput_qps, 2),
        }


def benchmark_predictor(predictor, texts: list[str], warmup: int = 2) -> BenchmarkResult:
    """Benchmark prediction latency and throughput across input texts."""
    if not texts:
        raise ValueError("At least one text sample is required for benchmarking.")

    # Warmup runs (discarded)
    for text in texts[:warmup]:
        try:
            predictor.predict(text)
        except Exception:
            pass

    latencies_ms = []
    start_total = time.perf_counter()

    for text in texts:
        t0 = time.perf_counter()
        predictor.predict(text)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    total_time_ms = (time.perf_counter() - start_total) * 1000.0
    sorted_latencies = sorted(latencies_ms)
    count = len(sorted_latencies)

    def percentile(p: float) -> float:
        idx = int(len(sorted_latencies) * p)
        return sorted_latencies[min(idx, count - 1)]

    avg_ms = statistics.mean(sorted_latencies)
    min_ms = sorted_latencies[0]
    max_ms = sorted_latencies[-1]
    p50_ms = percentile(0.50)
    p95_ms = percentile(0.95)
    p99_ms = percentile(0.99)
    throughput = (count / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    return BenchmarkResult(
        sample_count=count,
        total_time_ms=total_time_ms,
        avg_latency_ms=avg_ms,
        min_latency_ms=min_ms,
        max_latency_ms=max_ms,
        p50_latency_ms=p50_ms,
        p95_latency_ms=p95_ms,
        p99_latency_ms=p99_ms,
        throughput_qps=throughput,
    )
