"""ContinualBenchmark: run a sequence of tasks through a readout and build the R matrix.

R[i, j] = accuracy on task j's test set after training through task i. After the run, the
GEM/RWalk metrics (acc, bwt, fwt, forgetting, intransigence) are computed via `cl_metrics`.

The readout only needs a scikit-learn-style incremental interface: `partial_fit(X, y[, classes])`
and `predict(X)`, for example `reservoirs.learning.online.OnlineReadout`.
"""
import numpy as np

from .metrics import cl_metrics


class ContinualBenchmark:
    def __init__(self, readout):
        self.readout = readout

    def _fit(self, X, y, classes):
        if classes is not None:
            self.readout.partial_fit(X, y, classes=classes)
        else:
            self.readout.partial_fit(X, y)

    def run(self, tasks, classes=None, baselines=None, joint=None):
        """tasks: ordered list of ((X_train, y_train), (X_test, y_test))."""
        T = len(tasks)
        R = np.zeros((T, T))
        for i, ((Xtr, ytr), _) in enumerate(tasks):
            self._fit(Xtr, ytr, classes if i == 0 else None)
            for j, (_, (Xte, yte)) in enumerate(tasks):
                pred = np.asarray(self.readout.predict(Xte))
                R[i, j] = float(np.mean(pred == np.asarray(yte)))
        self.R_ = R
        self.metrics_ = cl_metrics(R, baselines=baselines, joint=joint)
        return self
