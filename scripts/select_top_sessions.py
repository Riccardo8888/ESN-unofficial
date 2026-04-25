#!/usr/bin/env python3
"""
Select the top X% of Slither.io sessions by frame count.

Walks the data tree, reads the player_inputs Zarr to count frames per
session, ranks sessions by length, and emits a JSON manifest listing the
top-X% session paths plus per-user breakdown.

Usage:
    python scripts/select_top_sessions.py --data vnicktest/scripts/data --top 0.30
    python scripts/select_top_sessions.py --data vnicktest/scripts/data --top 0.30 --exclude AI_bot
"""

import argparse, json, sys
from pathlib import Path
import numpy as np


def count_frames(session_path: Path) -> int:
    """Read player_inputs to determine session length."""
    pi = session_path / "player_inputs"
    if not pi.is_dir():
        return 0
    # Zarr v2 stores .zarray with shape; read directly to avoid full-load.
    zarray = pi / ".zarray"
    if zarray.exists():
        meta = json.loads(zarray.read_text())
        shape = meta.get("shape", [])
        return int(shape[0]) if shape else 0
    # Fallback: actually load
    try:
        import zarr
        return int(zarr.open(str(pi), mode="r").shape[0])
    except Exception:
        return 0


def discover(data_root: Path):
    sessions = []
    for user_dir in sorted(data_root.iterdir()):
        if not user_dir.is_dir():
            continue
        for sess in sorted(user_dir.iterdir()):
            if not sess.is_dir():
                continue
            if sess.name.startswith(("session_", "game_")):
                if (sess / "grids").exists():
                    sessions.append(sess)
    return sessions


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--top", type=float, default=0.30,
                   help="Fraction (0-1) of sessions to keep, ranked by frame count.")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Usernames to skip entirely.")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    if not (0 < args.top <= 1.0):
        raise SystemExit("--top must be in (0, 1]")

    sessions = discover(args.data)
    if args.exclude:
        sessions = [s for s in sessions if s.parent.name not in args.exclude]
    if not sessions:
        raise SystemExit(f"No sessions found under {args.data}")

    print(f"Found {len(sessions)} sessions across "
          f"{len({s.parent.name for s in sessions})} users")

    rows = []
    for s in sessions:
        n = count_frames(s)
        rows.append({"user": s.parent.name, "session": s.name,
                     "path": str(s), "frames": n})
    rows.sort(key=lambda r: r["frames"], reverse=True)

    k = max(1, int(round(args.top * len(rows))))
    top = rows[:k]

    print(f"\nTop {args.top:.0%} = {k} sessions:")
    print(f"  total frames: {sum(r['frames'] for r in top):,}")
    print(f"  median frames per session: {int(np.median([r['frames'] for r in top]))}")
    by_user = {}
    for r in top:
        by_user.setdefault(r["user"], 0)
        by_user[r["user"]] += 1
    print("  per-user counts:")
    for u, n in sorted(by_user.items(), key=lambda kv: -kv[1]):
        print(f"    {u:<25s} {n:>4d} sessions")

    out = args.output or Path("scripts/results/top_sessions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "top": args.top,
        "n_sessions": k,
        "total_frames": int(sum(r["frames"] for r in top)),
        "sessions": top,
        "by_user": by_user,
    }, indent=2))
    print(f"\nManifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
