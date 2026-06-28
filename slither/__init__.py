"""slither: the slither.io gameplay-prediction pipeline (data, metrics, and mock helpers).

Works alongside the `reservoirs` package. Features feed a ConnectomeReservoir, whose states
pass through a ridge readout to predict the steering angle and boost. See the slither section
of examples/combined_examples.ipynb.
"""
from . import config
from .data import (
    discover_sessions, load_session, normalize_grids, prepare_features,
    convert_to_angle_bins, load_all_sessions, make_windows, train_test_split_windows,
)
from .metrics import angle_accuracy, boost_accuracy, compute_wout, leaked_feature_baseline
from .mock import ensure_mock_graph, ensure_mock_data

__all__ = [
    "config",
    "discover_sessions", "load_session", "normalize_grids", "prepare_features",
    "convert_to_angle_bins", "load_all_sessions", "make_windows", "train_test_split_windows",
    "angle_accuracy", "boost_accuracy", "compute_wout", "leaked_feature_baseline",
    "ensure_mock_graph", "ensure_mock_data",
]
