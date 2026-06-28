"""TDD: continual-learning metrics are pure functions of the R accuracy matrix.

R[i, j] = test accuracy on task j after training through task i.
Definitions: GEM (Lopez-Paz & Ranzato 2017) ACC/BWT/FWT; RWalk (Chaudhry 2018) Forgetting/Intransigence.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))

# A hand-computed 3-task example (values chosen so each metric is non-trivial).
R = np.array([
    [0.90, 0.30, 0.00],   # after task 0
    [0.80, 0.85, 0.20],   # after task 1
    [0.70, 0.75, 0.95],   # after task 2 (final)
])
BASELINES = [0.10, 0.10, 0.10]   # random-init accuracy per task (for FWT)
JOINT = [0.92, 0.90, 0.97]       # jointly-trained reference per task (for intransigence)


def test_average_accuracy():
    from reservoirs.learning.metrics import cl_metrics
    m = cl_metrics(R)
    assert m["acc"] == pytest.approx(0.80)   # mean of final row [0.70,0.75,0.95]


def test_backward_transfer():
    from reservoirs.learning.metrics import cl_metrics
    m = cl_metrics(R)
    # mean over i<T-1 of (R[-1,i] - R[i,i]) = mean([0.70-0.90, 0.75-0.85])
    assert m["bwt"] == pytest.approx(-0.15)


def test_forward_transfer_requires_baselines():
    from reservoirs.learning.metrics import cl_metrics
    m = cl_metrics(R, baselines=BASELINES)
    # mean over i>=1 of (R[i-1,i] - baseline[i]) = mean([0.30-0.10, 0.20-0.10])
    assert m["fwt"] == pytest.approx(0.15)
    # without baselines, fwt is not reported
    assert "fwt" not in cl_metrics(R)


def test_forgetting_rwalk():
    from reservoirs.learning.metrics import cl_metrics
    m = cl_metrics(R)
    # mean over j<T-1 of max_{l<T-1}(R[l,j]) - R[T-1,j]
    #   f_0 = max(0.90,0.80)-0.70 = 0.20 ; f_1 = 0.85-0.75 = 0.10
    assert m["forgetting"] == pytest.approx(0.15)


def test_intransigence_requires_joint():
    from reservoirs.learning.metrics import cl_metrics
    m = cl_metrics(R, joint=JOINT)
    # mean over k of (joint[k] - R[k,k]) = mean([0.02, 0.05, 0.02])
    assert m["intransigence"] == pytest.approx(0.03)
    assert "intransigence" not in cl_metrics(R)


def test_perfect_retention_has_zero_forgetting():
    from reservoirs.learning.metrics import cl_metrics
    perfect = np.array([[0.9, 0.0], [0.9, 0.9]])  # task 0 never degrades
    m = cl_metrics(perfect)
    assert m["bwt"] == pytest.approx(0.0)
    assert m["forgetting"] == pytest.approx(0.0)
