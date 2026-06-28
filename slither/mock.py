"""Generate schema-compatible mock slither.io data + a mock connectome, so the
pipeline runs end-to-end on a fresh clone without the private scraper data.

The committed `data/mock_user_*/session_*` and `generated_artifacts/graphs/mock_connectome.graphml`
were produced by these functions; they are idempotent (no-ops if data already exists).
"""
import json
from pathlib import Path

import numpy as np
import networkx as nx

from .data import discover_sessions


def ensure_mock_graph(graph_dir, seed: int = 7):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    files = list(graph_dir.glob("*.graphml"))
    if files:
        return files
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(60, 0.08, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.2, 1.0))
    out = graph_dir / "mock_connectome.graphml"
    nx.write_graphml(G, out)
    return [out]


def ensure_mock_data(data_dir, n_sessions: int = 3, n_frames: int = 240, seed: int = 7):
    data_dir = Path(data_dir)
    if discover_sessions(data_dir):
        return False
    rng = np.random.default_rng(seed)
    for user_id in range(1, n_sessions + 1):
        session_path = data_dir / f"mock_user_{user_id}" / f"session_{1000 + user_id}"
        session_path.mkdir(parents=True, exist_ok=True)

        headings = np.cumsum(rng.normal(0.0, 0.12, size=n_frames)).astype(np.float32)
        velocity = np.clip(90 + 30 * np.sin(np.linspace(0, 4 * np.pi, n_frames))
                           + rng.normal(0, 7, n_frames), 20, 180).astype(np.float32)
        distance = np.clip(12000 + 4000 * np.sin(np.linspace(0, 2 * np.pi, n_frames) + user_id)
                           + rng.normal(0, 300, n_frames), 1500, 21500).astype(np.float32)
        boost = (velocity > 110).astype(np.float32)

        mx, my = np.cos(headings), np.sin(headings)
        player_inputs = np.stack([mx, my, boost], axis=1).astype(np.float32)

        grids = np.zeros((n_frames, 64, 24, 4), dtype=np.float32)
        for t in range(n_frames):
            ang_idx = int(((np.degrees(headings[t]) + 180) / 360.0) * 64) % 64
            rad_idx = min(23, max(0, int((velocity[t] / 180.0) * 23)))
            grids[t, ang_idx, max(0, rad_idx - 1):min(24, rad_idx + 2), 0] = 1.0
            grids[t, (ang_idx + 3) % 64, :, 1] = np.linspace(0.2, 0.8, 24)
            grids[t, ang_idx, : max(2, rad_idx), 2] = 0.5
            grids[t, (ang_idx + 8) % 64, min(23, rad_idx + 2), 3] = 1.0
            grids[t] += rng.normal(0.0, 0.03, size=grids[t].shape)
        grids = np.clip(grids, 0.0, 1.0)
        timestamps = np.arange(n_frames, dtype=np.float32) * 0.1

        np.save(session_path / "grids.npy", grids)
        np.save(session_path / "player_inputs.npy", player_inputs)
        np.save(session_path / "timestamps.npy", timestamps)
        np.save(session_path / "headings.npy", headings)
        np.save(session_path / "velocities.npy", velocity)
        np.save(session_path / "distances_to_border.npy", distance)
        np.save(session_path / "boost_states.npy", boost)
        (session_path / "metadata.json").write_text(
            json.dumps({"username": f"mock_user_{user_id}", "source": "generated_demo"}))
    return True
