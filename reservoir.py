"""
Compatibility shim for the top-level ``reservoir`` module.

Historically this file contained three duplicate ad-hoc classes
(``Reservoir``, ``Reservoir2``, ``Reservoir3``) that mirrored the proper
implementations in :mod:`reservoirs`.  They were known to be broken and the
author explicitly recommended using the package classes instead.

This module now simply re-exports the canonical classes from the
``reservoirs`` package so that legacy callers (older notebooks, the Slither
training scripts, the hyper-parameter searches) keep working with no code
changes:

    from reservoir import Reservoir          # ErdosRenyiReservoir
    from reservoir import Reservoir2         # FullyConnectedReservoir (paper "fixed-matrix")
                                             #   NOTE: was previously a deterministic
                                             #   ±0.5 ring; now an actual all-to-all
                                             #   uniform-weight network with state reset.
    from reservoir import Reservoir3         # GaussianReservoir

For new code please import from :mod:`reservoirs` directly:

    from reservoirs.ErdosRenyi import ErdosRenyiReservoir
    from reservoirs.FullyConnected import FullyConnectedReservoir
    from reservoirs.Gaussian import GaussianReservoir
    from reservoirs.NonResettingFullyConnected import NonResettingFullyConnectedReservoir
    from reservoirs.brain_connectome_reservoir import ConnectomeReservoir
"""

from reservoirs.ErdosRenyi import ErdosRenyiReservoir as Reservoir
from reservoirs.FullyConnected import FullyConnectedReservoir as Reservoir2
from reservoirs.Gaussian import GaussianReservoir as Reservoir3

__all__ = ["Reservoir", "Reservoir2", "Reservoir3"]
