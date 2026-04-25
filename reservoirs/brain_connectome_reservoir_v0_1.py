"""
Compatibility shim.

The canonical implementation is now in
:mod:`reservoirs.brain_connectome_reservoir`.  This module re-exports it so
that any existing imports of the form

    from reservoirs.brain_connectome_reservoir_v0_1 import ConnectomeReservoir

continue to work.  New code should import from the canonical location:

    from reservoirs.brain_connectome_reservoir import ConnectomeReservoir
"""

from reservoirs.brain_connectome_reservoir import ConnectomeReservoir  # noqa: F401

__all__ = ["ConnectomeReservoir"]
