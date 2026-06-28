"""
Regenerate the golden snapshots from the current engine code.

Run this once against the pre-refactor code to establish the baseline:
    python tests/regen_goldens.py

After a refactor, don't re-run it; that would defeat the purpose. Instead run
`python tests/test_characterization.py` (or `pytest tests/`) to verify the refactored
code still reproduces these goldens. Only re-run this when an intended behavior change
has been reviewed and the new numbers deliberately accepted.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _characterization_common import snapshot  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    snaps = snapshot()
    for name, arr in snaps.items():
        np.save(os.path.join(GOLDEN_DIR, name + ".npy"), np.asarray(arr))
    import numpy as _np
    import networkx as _nx
    meta = {
        "seed": 7,
        "numpy": _np.__version__,
        "networkx": _nx.__version__,
        "n_goldens": len(snaps),
        "keys": sorted(snaps.keys()),
        "note": "Baseline pinned against pre-refactor code. See tests/README.md.",
    }
    with open(os.path.join(GOLDEN_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(snaps)} goldens to {GOLDEN_DIR}")
    for k in sorted(snaps):
        a = np.asarray(snaps[k])
        print(f"  {k:28s} shape={a.shape} dtype={a.dtype}")


if __name__ == "__main__":
    main()
