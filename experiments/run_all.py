"""Regenerate every result file and figure, then refresh the README results tables.

Usage: ``python -m experiments.run_all [--workers N]``
"""

from __future__ import annotations

import argparse
import json
import time

from experiments import (
    e1_selection_inflation,
    e2_calibration,
    e3_power,
    e4_pbo_sanity,
    e5_real_data,
    report,
)
from experiments.common import RESULTS_DIR, default_workers

EXPERIMENTS = (
    ("E1 selection inflation", e1_selection_inflation.run),
    ("E2 calibration", e2_calibration.run),
    ("E3 power", e3_power.run),
    ("E4 PBO sanity", e4_pbo_sanity.run),
    ("E5 real data", e5_real_data.run),
)


def main(workers: int) -> None:
    """Run all experiments in order and record their wall-clock times."""
    timings = {}
    total_start = time.perf_counter()
    for name, run in EXPERIMENTS:
        start = time.perf_counter()
        run(workers)
        timings[name] = round(time.perf_counter() - start, 1)
        print(f"{name}: {timings[name]:.1f} s", flush=True)
    total = round(time.perf_counter() - total_start, 1)
    (RESULTS_DIR / "runtime.json").write_text(
        json.dumps({"workers": workers, "seconds": timings, "total_seconds": total}, indent=2)
        + "\n"
    )
    report.write_readme()
    print(f"Total runtime: {total:.1f} s ({total / 60:.1f} min) with {workers} workers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=default_workers())
    main(parser.parse_args().workers)
