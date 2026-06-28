"""reservoirs.learning — the continuous-learning layer.

Online (per-sample) and continual (task-sequence, no forgetting) readouts that sit on top of a
frozen reservoir substrate, plus a benchmark + metrics. See ../../docs/CONTINUOUS_LEARNING_DESIGN.md.

  online.py    : RLSReadout (rls/lms/nlms) + OnlineReadout (classifier)
  continual.py : conceptor algebra + ConceptorClassifier (forgetting-free)
  benchmark.py : ContinualBenchmark (task sequence -> R matrix)
  metrics.py   : cl_metrics (ACC/BWT/FWT + Forgetting/Intransigence)
"""
from .metrics import cl_metrics
from .online import RLSReadout, OnlineReadout
from .continual import (
    ConceptorClassifier, conceptor_from_states,
    conceptor_not, conceptor_and, conceptor_or,
)
from .benchmark import ContinualBenchmark

__all__ = [
    "cl_metrics",
    "RLSReadout", "OnlineReadout",
    "ConceptorClassifier", "conceptor_from_states",
    "conceptor_not", "conceptor_and", "conceptor_or",
    "ContinualBenchmark",
]
