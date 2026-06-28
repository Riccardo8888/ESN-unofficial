"""slither — the slither.io gameplay-prediction pipeline (data + metrics + mock helpers).

Pairs with the `reservoirs` package: features -> ConnectomeReservoir states -> ridge readout
-> angle/boost prediction. See examples/03_slither_pipeline.ipynb.
"""
from . import config
from .data import (
    discover_sessions, load_session, normalize_grids, prepare_features,
    convert_to_angle_bins, load_all_sessions, make_windows, train_test_split_windows,
)
from .metrics import angle_accuracy, boost_accuracy, compute_wout
from .mock import ensure_mock_graph, ensure_mock_data

__all__ = [
    "config",
    "discover_sessions", "load_session", "normalize_grids", "prepare_features",
    "convert_to_angle_bins", "load_all_sessions", "make_windows", "train_test_split_windows",
    "angle_accuracy", "boost_accuracy", "compute_wout",
    "ensure_mock_graph", "ensure_mock_data",
]
