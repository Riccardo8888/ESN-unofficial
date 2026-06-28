"""End-to-end smoke test of the slither.io pipeline on the committed mock fixtures.

Replaces the self-contained `humand_data_1015_slither_copy.ipynb` smoke notebook: runs the
whole pipeline (load -> features -> windows -> connectome reservoir -> ridge -> metrics) on
`data/mock_user_*` + `generated_artifacts/graphs/mock_connectome.graphml`, with NO private data.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, REPO)


def test_slither_pipeline_end_to_end():
    import slither
    from reservoirs.connectome import ConnectomeReservoir
    from slither.config import LEAK_RANGE, INPUT_SCALE, T_WASHOUT, ALPHA, WINDOW_LEN, WINDOW_STRIDE, OUTPUT_DIM

    data, graph = Path(REPO) / "data", Path(REPO) / "generated_artifacts" / "graphs"
    slither.ensure_mock_graph(graph)
    slither.ensure_mock_data(data)

    X_list, y_list, names = slither.load_all_sessions(data)
    assert len(names) >= 1, "no sessions discovered under data/"
    u, y, sid = slither.make_windows(X_list, y_list, WINDOW_LEN, WINDOW_STRIDE)
    assert u.ndim == 3 and y.shape[-1] == OUTPUT_DIM

    u_tr, y_tr, u_te, y_te, _, _ = slither.train_test_split_windows(u, y, sid, seed=7, group_by_session=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ConnectomeReservoir(n_inputs=u_tr.shape[2], graph_dir=str(graph), edge_attr="weight",
                                  combine="mean", rhow=1.05, leak_range=LEAK_RANGE, symmetric=True,
                                  seed=7, input_scale=INPUT_SCALE)
    Xtr, Xte = res.collect_states_batch(u_tr), res.collect_states_batch(u_te)  # Phase-5 fast path
    assert np.isfinite(Xtr).all() and Xtr.shape[-1] == res.n_neurons

    wout = slither.compute_wout(Xtr, y_tr, washout=T_WASHOUT, alpha=ALPHA)
    assert wout.shape == (res.n_neurons, OUTPUT_DIM)

    pred = Xte @ wout
    aa, ba = slither.angle_accuracy(pred, y_te), slither.boost_accuracy(pred, y_te)
    assert 0.0 <= aa <= 1.0 and 0.0 <= ba <= 1.0
    assert np.isfinite(np.mean((pred - y_te) ** 2))
