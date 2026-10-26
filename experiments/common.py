"""Shared helpers for the experiments: parallel runs, result files, intervals, figure style."""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

T = TypeVar("T")
R = TypeVar("R")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

# Categorical slots in fixed order, plus chart ink (light surface).
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "red": "#e34948",
}
INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def binomial_ci(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for a binomial proportion."""
    ci = stats.binomtest(int(successes), int(trials)).proportion_ci(level, method="exact")
    return float(ci.low), float(ci.high)


def rate_summary(successes: int, trials: int) -> dict[str, float | int]:
    """Rate with its 95% Clopper-Pearson interval, as stored in result files."""
    low, high = binomial_ci(successes, trials)
    return {
        "count": int(successes),
        "n": int(trials),
        "rate": successes / trials,
        "ci_low": low,
        "ci_high": high,
    }


def write_result(name: str, payload: dict[str, Any], runtime_seconds: float) -> Path:
    """Write ``results/<name>.json`` with run metadata."""
    RESULTS_DIR.mkdir(exist_ok=True)
    meta = {
        "experiment": name,
        "runtime_seconds": round(runtime_seconds, 2),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps({"meta": meta, **payload}, indent=2, default=_to_builtin) + "\n")
    return path


def load_result(name: str) -> dict[str, Any]:
    """Read ``results/<name>.json``."""
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)}")


def default_workers() -> int:
    """Worker processes for simulation studies (at most 4)."""
    return max(1, min(4, os.cpu_count() or 1))


def parallel_map(func: Callable[[T], R], tasks: Sequence[T], workers: int) -> list[R]:
    """Ordered map over ``tasks``, in worker processes when ``workers > 1``.

    Each task carries its own seed, so results do not depend on the number of workers.
    Child processes are started with single-threaded BLAS to avoid oversubscription.
    """
    if workers <= 1:
        return [func(task) for task in tasks]
    saved = {var: os.environ.get(var) for var in _THREAD_VARS}
    os.environ.update({var: "1" for var in _THREAD_VARS})
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            return list(pool.map(func, tasks))
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def chunked(items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Split a sequence into consecutive chunks of at most ``size`` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]


@contextmanager
def stopwatch() -> Iterator[Callable[[], float]]:
    """Yield a function returning the seconds elapsed since entering the block."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def apply_style() -> None:
    """Quiet chart style: thin marks, hairline grid, text in ink colors."""
    plt.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "axes.edgecolor": INK["axis"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK["secondary"],
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6.0,
            "font.family": "sans-serif",
            "font.size": 9.5,
            "text.color": INK["primary"],
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save ``figures/<name>.png`` at 160 dpi and close the figure."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
