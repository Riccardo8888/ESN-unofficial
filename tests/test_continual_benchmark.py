"""TDD: ContinualBenchmark runs a task sequence -> R accuracy matrix -> CL metrics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))


def _domain(centers, labels, n=80, seed=0, jitter=0.4):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c, l in zip(centers, labels):
        X.append(rng.standard_normal((n, 2)) * jitter + np.asarray(c, float))
        y += [l] * n
    return np.vstack(X), np.array(y)


def test_run_builds_R_and_metrics_for_identical_tasks():
    from reservoirs.learning.online import OnlineReadout
    from reservoirs.learning.benchmark import ContinualBenchmark
    X, y = _domain([[-2, 0], [2, 0]], [0, 1], seed=1)
    task = ((np.tile(X, (3, 1)), np.tile(y, 3)), (X, y))   # (train repeated, test)
    bench = ContinualBenchmark(OnlineReadout(forgetting=1.0)).run([task, task], classes=[0, 1])
    assert bench.R_.shape == (2, 2)
    assert {"acc", "bwt", "forgetting"}.issubset(bench.metrics_)
    assert bench.R_[0, 0] > 0.9 and bench.R_[1, 1] > 0.9
    assert abs(bench.metrics_["bwt"]) < 0.05     # identical tasks -> ~no forgetting


def test_benchmark_detects_catastrophic_forgetting():
    from reservoirs.learning.online import OnlineReadout
    from reservoirs.learning.benchmark import ContinualBenchmark
    X, y0 = _domain([[-2, 0], [2, 0]], [0, 1], seed=2)
    y1 = 1 - y0                                  # same inputs, flipped labels: conflicting domains
    t0 = ((np.tile(X, (5, 1)), np.tile(y0, 5)), (X, y0))
    t1 = ((np.tile(X, (5, 1)), np.tile(y1, 5)), (X, y1))
    bench = ContinualBenchmark(OnlineReadout(forgetting=0.8)).run([t0, t1], classes=[0, 1])
    R = bench.R_
    assert R[0, 0] > 0.9        # learned task 0
    assert R[1, 1] > 0.9        # overwrote with task 1
    assert R[1, 0] < 0.5        # forgot task 0
    assert bench.metrics_["bwt"] < -0.3
