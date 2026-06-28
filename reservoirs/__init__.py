"""
reservoirs: frozen reservoir-computing substrates for the continuous-learning package.

Engines (recurrent weights are fixed; only a downstream readout learns):
  - random.py     : Reservoir / RingReservoir / GaussianReservoir / ErdosRenyiReservoir
  - connectome.py : ConnectomeReservoir (brain-connectome-derived; canonical engine)

The continuous-learning readouts live under `reservoirs.learning` (Phase 2b).
See ../docs/CONTINUOUS_LEARNING_DESIGN.md and ../CLEANUP_PLAN.md.
"""
from .connectome import ConnectomeReservoir
from .random import (
    Reservoir,
    Reservoir2,
    Reservoir3,
    RingReservoir,
    GaussianReservoir,
    ErdosRenyiReservoir,
)

__all__ = [
    "ConnectomeReservoir",
    "Reservoir",
    "Reservoir2",
    "Reservoir3",
    "RingReservoir",
    "GaussianReservoir",
    "ErdosRenyiReservoir",
]
