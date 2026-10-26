"""E2: false positive rates of naive and selection-aware tests under the null.

Each simulation draws a universe of N zero-Sharpe strategies (five years of daily returns)
and asks every method whether the best strategy is significantly better than zero at
alpha = 0.05. With every null true, any rejection is a false positive. Rates come with
exact 95% binomial (Clopper-Pearson) intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from btaudit.sims import NullSpec, null_returns
from experiments.common import (
    COLORS,
    INK,
    apply_style,
    chunked,
    default_workers,
    parallel_map,
    rate_summary,
    save_figure,
    stopwatch,
    write_result,
)
from experiments.methods import LABELS, METHODS, TestSettings, apply_methods

SEED = 2002
N_OBS = 1260
PERIODS = 252
N_SIMS = 1000
CHUNK = 25
SETTINGS = TestSettings(alpha=0.05, n_boot=500, mean_block=10.0)
CORRECTED = ("dsr", "reality_check", "spa", "holm", "bh")
METHOD_COLORS = {
    "dsr": COLORS["blue"],
    "reality_check": COLORS["orange"],
    "spa": COLORS["aqua"],
    "holm": COLORS["yellow"],
    "bh": COLORS["magenta"],
    "naive_t": COLORS["red"],
    "naive_psr": COLORS["violet"],
}


@dataclass(frozen=True)
class Config:
    """A null universe: number of strategies and return process."""

    key: str
    label: str
    n_strategies: int
    spec: NullSpec


CONFIGS = (
    Config("normal_n1", "Normal iid", 1, NullSpec()),
    Config("normal_n10", "Normal iid", 10, NullSpec()),
    Config("normal_n100", "Normal iid", 100, NullSpec()),
    Config("normal_n1000", "Normal iid", 1000, NullSpec()),
    Config("t5_n100", "Student t(5) iid", 100, NullSpec(dist="t", df=5.0)),
    Config("garch_t5_n100", "GARCH(1,1) t(5)", 100, NullSpec(dist="t", df=5.0, garch=(0.05, 0.93))),
    Config("normal_rho05_n100", "Normal, correlation 0.5", 100, NullSpec(rho=0.5)),
)


def _simulate(task: tuple[Config, list[np.random.SeedSequence]]) -> np.ndarray:
    """Rejection indicators, one row per simulation and one column per method."""
    config, seeds = task
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x = null_returns(N_OBS, config.n_strategies, rng, config.spec)
        outcome = apply_methods(x, rng, SETTINGS)
        rows.append([outcome[m][0] for m in METHODS])
    return np.array(rows, dtype=bool)


def _dodge_plot(ax: plt.Axes, results: dict, keys: list[str], methods: tuple[str, ...]) -> None:
    """False positive rates with 95% intervals, methods side by side at each position."""
    width = 0.13
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        rates = [results[k]["methods"][method] for k in keys]
        x = np.arange(len(keys)) + offset
        y = np.array([100 * r["rate"] for r in rates])
        err = np.array(
            [
                [100 * (r["rate"] - r["ci_low"]) for r in rates],
                [100 * (r["ci_high"] - r["rate"]) for r in rates],
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt="o",
            color=METHOD_COLORS[method],
            markersize=5,
            elinewidth=1.2,
            capsize=0,
            markeredgecolor=INK["surface"],
            markeredgewidth=1,
            label=LABELS[method],
        )
    ax.axhline(100 * SETTINGS.alpha, color=INK["secondary"], linewidth=1)
    ax.set_xticks(np.arange(len(keys)))


def plot(results: dict) -> None:
    """Naive inflation, corrected tests versus N, and robustness to the return process."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    iid = ["normal_n1", "normal_n10", "normal_n100", "normal_n1000"]
    n_values = [results[k]["n_strategies"] for k in iid]
    ax = axes[0]
    grid = np.logspace(0, 3, 100)
    ax.plot(
        grid,
        100 * (1 - (1 - SETTINGS.alpha) ** grid),
        color=INK["axis"],
        linewidth=2,
        label="1 - 0.95^N (independent)",
    )
    naive_t = [100 * results[k]["methods"]["naive_t"]["rate"] for k in iid]
    naive_psr = [100 * results[k]["methods"]["naive_psr"]["rate"] for k in iid]
    ax.plot(
        n_values,
        naive_t,
        "o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor=METHOD_COLORS["naive_t"],
        markeredgewidth=1.8,
        label=LABELS["naive_t"],
    )
    ax.plot(
        n_values,
        naive_psr,
        "s",
        markersize=5,
        color=METHOD_COLORS["naive_psr"],
        label=LABELS["naive_psr"],
    )
    ax.set_xscale("log")
    ax.set_xticks(n_values, [f"{n:g}" for n in n_values])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of strategies N")
    ax.set_ylabel("False positive rate (%)")
    ax.set_title("A. Testing the selected best naively")
    ax.legend(loc="lower right")

    _dodge_plot(axes[1], results, iid, CORRECTED)
    axes[1].set_xticklabels([f"N = {n}" for n in n_values])
    axes[1].set_ylim(-0.4, 14)
    axes[1].set_title("B. Selection-aware tests, normal iid")
    axes[1].set_ylabel("False positive rate (%)")
    axes[1].legend(loc="upper right", ncol=2, fontsize=7.5)

    robust = ["normal_n100", "t5_n100", "garch_t5_n100", "normal_rho05_n100"]
    _dodge_plot(axes[2], results, robust, CORRECTED)
    axes[2].set_xticklabels(["Normal", "t(5)", "GARCH t(5)", "Corr. 0.5"])
    axes[2].set_ylim(-0.4, 14)
    axes[2].set_title("C. Return process, N = 100")
    fig.text(
        0.01,
        -0.03,
        f"{N_SIMS} null universes per setting, {N_OBS} daily returns each, "
        "alpha = 5% (horizontal line). Bars are exact 95% binomial intervals.",
        fontsize=8,
        color=INK["muted"],
    )
    save_figure(fig, "e2_calibration")


def run(workers: int = default_workers()) -> dict:
    """Run E2, write ``results/e2_calibration.json`` and its figure."""
    with stopwatch() as elapsed:
        roots = np.random.SeedSequence(SEED).spawn(len(CONFIGS))
        tasks = [
            (cfg, chunk)
            for cfg, root in zip(CONFIGS, roots, strict=True)
            for chunk in chunked(root.spawn(N_SIMS), CHUNK)
        ]
        outputs = parallel_map(_simulate, tasks, workers)
        per_config = N_SIMS // CHUNK
        results = {}
        for k, cfg in enumerate(CONFIGS):
            rejections = np.vstack(outputs[k * per_config : (k + 1) * per_config])
            results[cfg.key] = {
                "label": cfg.label,
                "n_strategies": cfg.n_strategies,
                "n_sims": int(rejections.shape[0]),
                "naive_rate_if_independent": 1 - (1 - SETTINGS.alpha) ** cfg.n_strategies,
                "methods": {
                    m: rate_summary(int(rejections[:, j].sum()), rejections.shape[0])
                    for j, m in enumerate(METHODS)
                },
            }
        plot(results)
        payload = {
            "seed": SEED,
            "n_obs": N_OBS,
            "periods_per_year": PERIODS,
            "alpha": SETTINGS.alpha,
            "n_boot": SETTINGS.n_boot,
            "mean_block": SETTINGS.mean_block,
            "configs": results,
        }
        write_result("e2_calibration", payload, elapsed())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=default_workers())
    run(parser.parse_args().workers)
