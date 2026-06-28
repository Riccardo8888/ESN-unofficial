"""Slither.io data loading, feature engineering, windowing and splitting.

Extracted verbatim (Phase 3) from the canonical slither notebook, repointed to live in
the `slither` package. Reads the lightweight committed `.npy` demo sessions under
`data/<user>/session_*/` (and real scraper sessions via optional zarr).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import (
    ANGLE_MIN, ANGLE_MAX, ANGLE_RESOLUTION, NUM_ANGLE_BINS,
    WINDOW_LEN, WINDOW_STRIDE, MIN_FRAMES_PER_SESSION, TEST_RATIO,
)

_ARRAYS = ["grids", "player_inputs", "timestamps", "headings",
           "velocities", "distances_to_border", "boost_states"]


def _maybe_import_zarr():
    try:
        import zarr
        return zarr
    except Exception:
        return None


def discover_sessions(data_path) -> List[Path]:
    data_path = Path(data_path)
    sessions: List[Path] = []
    if not data_path.exists():
        return sessions
    for user_dir in sorted(data_path.iterdir()):
        if not user_dir.is_dir():
            continue
        for item in sorted(user_dir.iterdir()):
            if not item.is_dir():
                continue
            if item.name.startswith("session_") or item.name.startswith("game_"):
                if (item / "grids.npy").exists() or (item / "grids").exists():
                    sessions.append(item)
    return sessions


def load_session(session_path) -> Optional[Dict[str, np.ndarray]]:
    session_path = Path(session_path)
    if (session_path / "grids.npy").exists():
        data = {name: np.load(session_path / f"{name}.npy") for name in _ARRAYS}
    else:
        zarr = _maybe_import_zarr()
        if zarr is None:
            raise RuntimeError(
                "Real scraper sessions need `zarr` installed, or use the committed .npy demo data."
            )
        try:
            try:
                root = zarr.open_group(str(session_path), mode="r")
            except Exception:
                root = zarr.group(store=zarr.DirectoryStore(str(session_path)))
            data = {name: np.array(root[name]) for name in _ARRAYS}
        except Exception as e:  # pragma: no cover
            print(f"Skipping {session_path.name}: {e}")
            return None

    meta_file = session_path / "metadata.json"
    data["metadata"] = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    data["session_path"] = session_path

    n_frames = int(data["grids"].shape[0])
    if n_frames < MIN_FRAMES_PER_SESSION:
        print(f"Skipping {session_path.name}: only {n_frames} frames")
        return None
    return data


def normalize_grids(grids: np.ndarray) -> np.ndarray:
    return np.clip(grids.astype(np.float32), 0.0, 1.0)


def prepare_features(session_data: Dict[str, np.ndarray]) -> np.ndarray:
    n_frames = session_data["grids"].shape[0]
    grids_flat = normalize_grids(session_data["grids"]).reshape(n_frames, -1)

    velocity = session_data["velocities"].reshape(-1, 1).astype(np.float32) / 200.0
    distance = session_data["distances_to_border"].reshape(-1, 1).astype(np.float32) / 21600.0

    headings = session_data["headings"].astype(np.float32)
    angular_vel = np.diff(headings, prepend=headings[0])
    angular_vel = np.arctan2(np.sin(angular_vel), np.cos(angular_vel)) / np.pi
    angular_vel = angular_vel.reshape(-1, 1)

    mx = session_data["player_inputs"][:, 0].astype(np.float32)
    my = session_data["player_inputs"][:, 1].astype(np.float32)
    prev_angle = np.arctan2(my, mx)
    prev_angle = np.roll(prev_angle, 1)
    prev_angle[0] = 0.0
    prev_sin = np.sin(prev_angle).reshape(-1, 1)
    prev_cos = np.cos(prev_angle).reshape(-1, 1)

    X = np.concatenate([grids_flat, velocity, distance, angular_vel, prev_sin, prev_cos], axis=1)
    return X.astype(np.float32)


def convert_to_angle_bins(player_inputs: np.ndarray) -> np.ndarray:
    mx, my, boost = player_inputs[:, 0], player_inputs[:, 1], player_inputs[:, 2]
    angles_deg = np.degrees(np.arctan2(my, mx))
    angles_deg = np.clip(angles_deg, ANGLE_MIN, ANGLE_MAX)
    positions = np.clip((angles_deg - ANGLE_MIN) / ANGLE_RESOLUTION, 0, NUM_ANGLE_BINS - 1)

    y_angle = np.zeros((len(player_inputs), NUM_ANGLE_BINS), dtype=np.float32)
    for i, pos in enumerate(positions):
        lo = int(np.clip(np.floor(pos), 0, NUM_ANGLE_BINS - 1))
        hi = int(np.clip(np.ceil(pos), 0, NUM_ANGLE_BINS - 1))
        if lo == hi:
            y_angle[i, lo] = 1.0
        else:
            w_hi = pos - lo
            y_angle[i, lo] = 1.0 - w_hi
            y_angle[i, hi] = w_hi
    return np.concatenate([y_angle, boost.reshape(-1, 1).astype(np.float32)], axis=1)


def load_all_sessions(data_dir):
    X_list, y_list, names = [], [], []
    for session_path in discover_sessions(data_dir):
        session = load_session(session_path)
        if session is None:
            continue
        X_list.append(prepare_features(session))
        y_list.append(convert_to_angle_bins(session["player_inputs"]))
        names.append(session_path.name)
    return X_list, y_list, names


def make_windows(X_list, y_list, window_len=WINDOW_LEN, stride=WINDOW_STRIDE):
    u, y, session_ids = [], [], []
    for sid, (X, Y) in enumerate(zip(X_list, y_list)):
        if len(X) < window_len:
            continue
        for start in range(0, len(X) - window_len + 1, stride):
            u.append(X[start:start + window_len])
            y.append(Y[start:start + window_len])
            session_ids.append(sid)
    if not u:
        raise ValueError(
            f"make_windows produced 0 windows: every session is shorter than window_len={window_len}."
        )
    return (np.array(u, dtype=np.float32), np.array(y, dtype=np.float32), np.array(session_ids))


def train_test_split_windows(u, y, session_ids, test_ratio=TEST_RATIO, seed=7,
                             group_by_session=True):
    """Split windows into train and test sets.

    With group_by_session=True (the default since Phase 4), whole sessions are held out, which
    keeps the split leakage-free. With group_by_session=False you get the original behavior:
    individual windows are shuffled, so overlapping windows (when stride < window_len) leak
    across the split and bias the test metrics optimistically.
    """
    rng_local = np.random.default_rng(seed)
    if group_by_session:
        uniq = np.unique(session_ids)
        if len(uniq) < 2:
            raise ValueError(
                f"group_by_session=True needs >=2 distinct sessions to hold one out; got {len(uniq)}. "
                "Use group_by_session=False (with the leakage caveat) for a single-session dataset."
            )
        rng_local.shuffle(uniq)
        n_test = max(1, int(len(uniq) * test_ratio))
        effective = n_test / len(uniq)
        if abs(effective - test_ratio) > 0.1:
            import warnings
            warnings.warn(
                f"group_by_session split holds out {n_test}/{len(uniq)} sessions "
                f"(~{effective:.0%}), which differs materially from test_ratio={test_ratio:.0%} "
                f"because whole sessions are indivisible (e.g. 2 sessions always split 50/50). "
                f"Add more sessions for a split closer to test_ratio.",
                UserWarning, stacklevel=2,
            )
        test_sessions = set(uniq[:n_test].tolist())
        test_mask = np.array([s in test_sessions for s in session_ids])
        train_idx = np.sort(np.where(~test_mask)[0])
        test_idx = np.sort(np.where(test_mask)[0])
    else:
        idx = np.arange(len(u))
        rng_local.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx = np.sort(idx[:n_test])
        train_idx = np.sort(idx[n_test:])
    return (u[train_idx], y[train_idx], u[test_idx], y[test_idx],
            session_ids[train_idx], session_ids[test_idx])
