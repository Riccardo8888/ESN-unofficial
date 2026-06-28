"""reservoirs.learning: the continuous-learning layer.

Online (per-sample) and continual (task-sequence, no forgetting) readouts that sit on top of a
frozen reservoir substrate, plus a benchmark and metrics. See ../../docs/CONTINUOUS_LEARNING_DESIGN.md.

  online.py    : RLSReadout (rls/lms/nlms) and OnlineReadout (classifier)
  continual.py : conceptor algebra and ConceptorClassifier (forgetting-free representation),
                 plus ConceptorReadout (a partial_fit adapter so it runs through ContinualBenchmark)
  benchmark.py : ContinualBenchmark (turns a task sequence into the R matrix)
  metrics.py   : cl_metrics (ACC/BWT/FWT plus Forgetting/Intransigence)
"""
from .metrics import cl_metrics
from .online import RLSReadout, OnlineReadout
from .continual import (
    ConceptorClassifier, ConceptorReadout,
    conceptor_from_states, conceptor_from_correlation,
    conceptor_not, conceptor_and, conceptor_or,
)
from .benchmark import ContinualBenchmark

__all__ = [
    "cl_metrics",
    "RLSReadout", "OnlineReadout",
    "ConceptorClassifier", "ConceptorReadout",
    "conceptor_from_states", "conceptor_from_correlation",
    "conceptor_not", "conceptor_and", "conceptor_or",
    "ContinualBenchmark",
]
