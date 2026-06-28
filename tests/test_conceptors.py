"""TDD: conceptors (Jaeger 2014) + a forgetting-free conceptor classifier."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))

N = 10


def _gen(class_dims, n=40, T=20, seed=0):
    rng = np.random.default_rng(seed)
    S = np.zeros((n, T, N))
    for i in range(n):
        s = rng.standard_normal((T, N)) * 0.2
        s[:, list(class_dims)] += rng.standard_normal((T, len(list(class_dims)))) * 2.0
        S[i] = s
    return S


def test_conceptor_eigenvalues_in_unit_interval():
    from reservoirs.learning.continual import conceptor_from_states
    C = conceptor_from_states(_gen(range(0, 5)), aperture=4.0)
    eig = np.linalg.eigvalsh(C)
    assert eig.min() >= -1e-9 and eig.max() < 1.0


def test_not_is_involution():
    from reservoirs.learning.continual import conceptor_from_states, conceptor_not
    C = conceptor_from_states(_gen(range(0, 5)), aperture=4.0)
    np.testing.assert_allclose(conceptor_not(conceptor_not(C)), C, atol=1e-8)


def test_larger_aperture_passes_more_variance():
    from reservoirs.learning.continual import conceptor_from_states
    S = _gen(range(0, 5))
    lo = np.trace(conceptor_from_states(S, aperture=0.5))
    hi = np.trace(conceptor_from_states(S, aperture=50.0))
    assert hi > lo   # aperture -> infinity makes C -> I (trace -> N)


def test_conceptor_and_or_valid_and_ordered():
    """AND/OR produce valid conceptors and obey intersection/union semantics (closes the dead-code gap)."""
    from reservoirs.learning.continual import conceptor_from_states, conceptor_and, conceptor_or
    A = conceptor_from_states(_gen(range(0, 6), seed=1), aperture=4.0)
    B = conceptor_from_states(_gen(range(4, 10), seed=2), aperture=4.0)
    AND, OR = conceptor_and(A, B), conceptor_or(A, B)
    for C in (AND, OR):
        e = np.linalg.eigvalsh(0.5 * (C + C.T))
        assert e.min() >= -1e-6 and e.max() <= 1 + 1e-6     # still a valid conceptor
    assert np.trace(AND) <= min(np.trace(A), np.trace(B)) + 1e-6   # AND shrinks (intersection)
    assert np.trace(OR) >= max(np.trace(A), np.trace(B)) - 1e-6    # OR grows (union)


def test_classifier_separates_two_signal_classes():
    from reservoirs.learning.continual import ConceptorClassifier
    A, B = _gen(range(0, 5), seed=1), _gen(range(5, 10), seed=2)
    clf = ConceptorClassifier(aperture=4.0)
    clf.learn_class(A[:30], "A"); clf.learn_class(B[:30], "B")
    preds = clf.predict(np.concatenate([A[30:], B[30:]]))
    truth = ["A"] * 10 + ["B"] * 10
    assert float(np.mean([p == t for p, t in zip(preds, truth)])) > 0.9


def test_continual_learning_is_forgetting_free():
    """The conceptor representation is the forgetting-free part: each class's conceptor and its
    evidence are a pure function of that class's data, so learning a LATER class cannot alter
    them (unlike a shared readout, which would drift). (Multi-class *decisions* can still shift
    as competitors are added, which is the separate class-incremental difficulty.)"""
    from reservoirs.learning.continual import ConceptorClassifier
    A, B = _gen(range(0, 5), seed=1), _gen(range(5, 10), seed=2)
    clf = ConceptorClassifier(aperture=4.0)
    clf.learn_class(A[:30], "A")
    clf.learn_class(B[:30], "B")
    CA_before = clf.conceptors_["A"].copy()
    evA_before = clf.evidence(A[30:])["A"].copy()
    clf.learn_class(_gen(range(2, 7), seed=3)[:30], "C")  # a later, overlapping task
    np.testing.assert_array_equal(clf.conceptors_["A"], CA_before)        # conceptor untouched
    np.testing.assert_allclose(clf.evidence(A[30:])["A"], evA_before)     # A-evidence unchanged
