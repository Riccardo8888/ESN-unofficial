"""TDD: ContinualBenchmark runs a task sequence, builds the R accuracy matrix, then the CL metrics."""
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


# The conceptor method, driven through the benchmark (AUDIT.md F3).
# Proves ConceptorReadout conforms to the partial_fit/predict protocol AND that the prior
# "per-class bank is forgetting-free by construction, so BWT is trivially ~0" claim is FALSE:
# the representation is forgetting-free but the multi-class DECISIONS forget on overlapping streams.

def _subspace(dims, scale, n, N=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N)) * 0.05
    X[:, list(dims)] += rng.standard_normal((n, len(list(dims)))) * scale
    return X


def test_conceptor_readout_runs_through_benchmark_disjoint_no_forgetting():
    from reservoirs.learning.continual import ConceptorReadout
    from reservoirs.learning.benchmark import ContinualBenchmark
    X0 = np.vstack([_subspace([0, 1], 2.0, 60, seed=1), _subspace([2, 3], 2.0, 60, seed=2)])
    X1 = np.vstack([_subspace([0, 1], 2.0, 60, seed=3), _subspace([2, 3], 2.0, 60, seed=4)])
    y = np.array([0] * 60 + [1] * 60)
    bench = ContinualBenchmark(ConceptorReadout(aperture=4.0)).run(
        [((X0, y), (X0, y)), ((X1, y), (X1, y))], classes=[0, 1])
    assert bench.R_.shape == (2, 2)
    assert bench.R_[0, 0] > 0.9 and bench.R_[1, 1] > 0.9
    assert abs(bench.metrics_["bwt"]) < 0.05      # disjoint subspaces -> the genuinely trivial case


def test_conceptor_readout_benchmark_reveals_class_incremental_forgetting():
    from reservoirs.learning.continual import ConceptorReadout
    from reservoirs.learning.benchmark import ContinualBenchmark
    # task 0: class 0 (dims 0,1) + class 1 (dims 2,3). task 1 adds a HIGH-energy class 2 that
    # overlaps class 0's subspace -> its positive evidence out-competes class 0 on class-0 samples.
    X0 = np.vstack([_subspace([0, 1], 1.0, 60, seed=1), _subspace([2, 3], 1.0, 60, seed=2)])
    y0 = np.array([0] * 60 + [1] * 60)
    X2 = _subspace([0, 1], 4.0, 80, seed=5)
    y2 = np.array([2] * 80)
    bench = ContinualBenchmark(ConceptorReadout(aperture=4.0)).run(
        [((X0, y0), (X0, y0)), ((X2, y2), (X2, y2))], classes=[0, 1, 2])
    R = bench.R_
    assert R[0, 0] > 0.9 and R[1, 1] > 0.9        # both stages learn their own classes
    assert R[1, 0] < 0.7                          # class 0 is FORGOTTEN after class 2 arrives
    assert bench.metrics_["bwt"] < -0.3           # NOT trivially ~0, so the deferral's claim was wrong
