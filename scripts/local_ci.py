"""Local CI gate: run the same checks GitHub Actions runs, but on your own machine.

Usage (from the repo root):
    python scripts/local_ci.py

Cloud CI (.github/workflows/ci.yml) stays dormant on purpose until the repo has a GitHub
remote and the HCP connectome data-licensing question is settled (see README "License & data").
Until then this script is the gate to run before pushing. It uses the standard library only,
with no extra deps.
"""
import os
import subprocess
import sys

# Pin BLAS threads for deterministic golden trajectories (matches ci.yml).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

STEPS = [
    ("fresh-install smoke check",
     [sys.executable, "-c",
      "import reservoirs, slither, reservoirs.learning, reservoirs.baselines, "
      "reservoirs.tasks, reservoirs.tuning; print('imports OK')"]),
    ("portable suite (-m 'not golden')",
     [sys.executable, "-m", "pytest", "tests/", "-m", "not golden", "-q"]),
    ("golden characterization (-m golden)",
     [sys.executable, "-m", "pytest", "tests/", "-m", "golden", "-q"]),
    ("example notebook(s) (nbmake)",
     [sys.executable, "-m", "pytest", "--nbmake", "examples/", "-q", "--nbmake-timeout=900"]),
]


def main():
    failed = []
    for name, cmd in STEPS:
        print(f"\n===== {name} =====", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    print("\n" + ("ALL GREEN" if not failed else "FAILED: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
