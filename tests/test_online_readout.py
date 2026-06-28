"""TDD: online readouts that learn per-sample via partial_fit (RLS / LMS / NLMS)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def test_rls_recovers_linear_map():
    from reservoirs.learning.online import RLSReadout
    rng = np.random.default_rng(0)
    f, o, n = 5, 2, 200
    Wtrue = rng.standard_normal((f, o))
    X = rng.standard_normal((n, f)); Y = X @ Wtrue
    r = RLSReadout(f, o, forgetting=1.0, ridge=1e-6)
    out = r.partial_fit(X, Y)
    assert out is r                       # returns self
    assert _mse(r.predict(X), Y) < 1e-4   # RLS = exact recursive LS


def test_rls_equals_batch_ridge():
    from reservoirs.learning.online import RLSReadout
    rng = np.random.default_rng(1)
    f, o, n, ridge = 4, 3, 150, 1.0
    X = rng.standard_normal((n, f)); Y = rng.standard_normal((n, o))
    r = RLSReadout(f, o, forgetting=1.0, ridge=ridge); r.partial_fit(X, Y)
    W_batch = np.linalg.solve(X.T @ X + ridge * np.eye(f), X.T @ Y)
    np.testing.assert_allclose(r.W_, W_batch, atol=1e-4)


def test_P_stays_exactly_symmetric():
    # The explicit re-symmetrization (online.py) makes P bit-exactly symmetric every step;
    # an EXACT check actually guards that line (without it, roundoff drifts P by ~1e-14).
    from reservoirs.learning.online import RLSReadout
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 4)); Y = rng.standard_normal((100, 1))
    r = RLSReadout(4, 1); r.partial_fit(X, Y)
    np.testing.assert_array_equal(r.P_, r.P_.T)


def test_forgetting_tracks_nonstationary_target():
    from reservoirs.learning.online import RLSReadout
    rng = np.random.default_rng(3)
    f, o, n = 4, 2, 300
    W1, W2 = rng.standard_normal((f, o)), rng.standard_normal((f, o))
    X1 = rng.standard_normal((n, f)); X2 = rng.standard_normal((n, f))
    r = RLSReadout(f, o, forgetting=0.95, ridge=1e-3)
    r.partial_fit(X1, X1 @ W1)            # regime 1
    r.partial_fit(X2, X2 @ W2)            # regime 2 (target changed)
    assert _mse(r.predict(X2), X2 @ W2) < 0.05   # adapted to the new target


def test_nlms_reduces_error_over_epochs():
    from reservoirs.learning.online import RLSReadout
    rng = np.random.default_rng(4)
    f, o, n = 3, 1, 100
    Wtrue = rng.standard_normal((f, o))
    X = rng.standard_normal((n, f)); Y = X @ Wtrue
    r = RLSReadout(f, o, method="nlms", mu=0.5)
    first = _mse(np.zeros_like(Y), Y)
    for _ in range(20):
        r.partial_fit(X, Y)
    assert _mse(r.predict(X), Y) < 0.1 * first   # converged well below the start


def test_lms_reduces_error_over_epochs():
    # actually exercises the method='lms' branch (small step size for stability)
    from reservoirs.learning.online import RLSReadout
    rng = np.random.default_rng(4)
    f, o, n = 3, 1, 100
    Wtrue = rng.standard_normal((f, o))
    X = rng.standard_normal((n, f)); Y = X @ Wtrue
    r = RLSReadout(f, o, method="lms", mu=0.02)
    first = _mse(np.zeros_like(Y), Y)
    for _ in range(80):
        r.partial_fit(X, Y)
    assert _mse(r.predict(X), Y) < 0.3 * first   # LMS converges (slower than RLS/NLMS)


def test_classifier_separates_blobs_and_requires_classes():
    from reservoirs.learning.online import OnlineReadout
    rng = np.random.default_rng(5)
    centers = np.array([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0]], float)
    X = np.vstack([rng.standard_normal((60, 4)) * 0.5 + c for c in centers])
    y = np.repeat([0, 1, 2], 60)

    clf = OnlineReadout()
    with pytest.raises(Exception):
        clf.partial_fit(X, y)             # must declare classes on first call
    clf.partial_fit(X, y, classes=[0, 1, 2])
    assert list(clf.classes_) == [0, 1, 2]
    pred = clf.predict(X)
    assert set(np.unique(pred)).issubset({0, 1, 2})
    assert float(np.mean(pred == y)) > 0.9
