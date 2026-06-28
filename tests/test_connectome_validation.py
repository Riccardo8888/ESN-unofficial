"""TDD: ConnectomeReservoir constructor validation guards (review flagged 0 tests on 9+ raise paths)."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
GRAPH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)),
                     "generated_artifacts", "graphs")


def _cr(**kw):
    from reservoirs.connectome import ConnectomeReservoir
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConnectomeReservoir(4, **kw)


def test_requires_graph_or_adjacency():
    with pytest.raises(ValueError, match="graph_dir"):
        _cr()


def test_bad_combine_rejected():
    with pytest.raises(ValueError, match="combine"):
        _cr(graph_dir=GRAPH, spectral_radius=0.9, combine="avg")


def test_nonpositive_spectral_radius_rejected():
    with pytest.raises(ValueError, match="spectral_radius"):
        _cr(graph_dir=GRAPH, spectral_radius=0.0)


def test_bad_leak_range_rejected():
    with pytest.raises(ValueError, match="leak_range"):
        _cr(graph_dir=GRAPH, spectral_radius=0.9, leak_range=(0.3, 0.1))


def test_bad_resize_method_rejected():
    with pytest.raises(ValueError, match="resize_method"):
        _cr(graph_dir=GRAPH, spectral_radius=0.9, resize_method="grow")


def test_missing_graph_dir_rejected():
    with pytest.raises(FileNotFoundError):
        _cr(graph_dir=os.path.join(GRAPH, "does_not_exist"), spectral_radius=0.9)


def test_non_square_adjacency_rejected():
    with pytest.raises(ValueError, match="square"):
        _cr(adjacency=np.zeros((3, 4)), spectral_radius=0.9)


def test_rhow_alias_deprecation_warns():
    from reservoirs.connectome import ConnectomeReservoir
    with pytest.warns(DeprecationWarning):
        ConnectomeReservoir(4, graph_dir=GRAPH, rhow=0.9)
