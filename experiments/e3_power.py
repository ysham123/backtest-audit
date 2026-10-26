"""E3: power to find one genuinely skilled strategy hidden among N - 1 zero-skill ones.

Exactly one of N strategies has a known true annualized Sharpe ratio. A method *detects*
it when it rejects the null and identifies that strategy: selection-based tests must pick
it as their best, and multiple-testing procedures must reject its own hypothesis. The
oracle benchmark is a one-sided t-test on the skilled strategy alone (no search).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from btaudit.sims import NullSpec, per_period_sharpe, planted_alpha_returns
from experiments.common import (
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
from experiments.e2_calibration import CORRECTED, METHOD_COLORS, N_OBS, PERIODS, SETTINGS
from experiments.methods import LABELS, METHODS, apply_methods

SEED = 3003
N_VALUES = (10, 100, 1000)
SHARPES = (0.5, 1.0, 1.5, 2.0)
N_SIMS = 500
CHUNK = 25


def oracle_power(annual_sharpe: float, n_obs: int = N_OBS, alpha: float = SETTINGS.alpha) -> float:
    """Power of a one-sided t-test on the skilled strategy alone (noncentral t)."""
    df = n_obs - 1
    noncentrality = per_period_sharpe(annual_sharpe, PERIODS) * np.sqrt(n_obs)
    return float(stats.nct.sf(stats.t.ppf(1 - alpha, df), df, noncentrality))


def _simulate(task: tuple[int, float, list[np.random.SeedSequence]]) -> np.ndarray:
    """Array ``(sims, methods, 2)`` of (rejects, detects) indicators."""
    n_strategies, sharpe, seeds = task
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x = planted_alpha_returns(N_OBS, n_strategies, sharpe, PERIODS, rng, NullSpec(), planted=0)
        outcome = apply_methods(x, rng, SETTINGS, planted=0)
        out.append([outcome[m] for m in METHODS])
    return np.array(out, dtype=bool)


def plot(results: dict) -> None:
    """Detection rate against the planted Sharpe ratio, one panel per N."""
    apply_style()
    fig, axes = plt.subplots(1, len(N_VALUES), figsize=(13, 4), sharey=True)
    fine = np.linspace(0.3, 2.2, 60)
    for ax, n in zip(axes, N_VALUES, strict=True):
        cells = [results[f"n{n}_sr{sr:g}"] for sr in SHARPES]
        ax.plot(
            fine,
            [100 * oracle_power(s) for s in fine],
            color=INK["axis"],
            linewidth=2,
            label="Oracle t-test, no search",
        )
        for method in CORRECTED:
            ax.plot(
                SHARPES,
                [100 * c["detect"][method]["rate"] for c in cells],
                "-o",
                color=METHOD_COLORS[method],
                markersize=5,
                markeredgecolor=INK["surface"],
                markeredgewidth=1,
                linewidth=1.8,
                label=LABELS[method],
            )
        ax.set_title(f"N = {n} strategies")
        ax.set_xlabel("True Sharpe of the skilled strategy (annualized)")
        ax.set_xticks(SHARPES)
        ax.set_ylim(-2, 102)
    axes[0].set_ylabel("Detection rate (%)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.text(
        0.01,
        -0.03,
        f"{N_SIMS} universes per point, {N_OBS} daily returns (5 years), "
        "alpha = 5%. Detection: the test rejects and identifies the skilled strategy.",
        fontsize=8,
        color=INK["muted"],
    )
    save_figure(fig, "e3_power")


def run(workers: int = default_workers()) -> dict:
    """Run E3, write ``results/e3_power.json`` and its figure."""
    with stopwatch() as elapsed:
        cells = [(n, sr) for n in N_VALUES for sr in SHARPES]
        roots = np.random.SeedSequence(SEED).spawn(len(cells))
        tasks = [
            (n, sr, chunk)
            for (n, sr), root in zip(cells, roots, strict=True)
            for chunk in chunked(root.spawn(N_SIMS), CHUNK)
        ]
        outputs = parallel_map(_simulate, tasks, workers)
        per_cell = N_SIMS // CHUNK
        results = {}
        for k, (n, sr) in enumerate(cells):
            arr = np.concatenate(outputs[k * per_cell : (k + 1) * per_cell])
            results[f"n{n}_sr{sr:g}"] = {
                "n_strategies": n,
                "annual_sharpe": sr,
                "n_sims": int(arr.shape[0]),
                "oracle_power": oracle_power(sr),
                "reject": {
                    m: rate_summary(int(arr[:, j, 0].sum()), arr.shape[0])
                    for j, m in enumerate(METHODS)
                },
                "detect": {
                    m: rate_summary(int(arr[:, j, 1].sum()), arr.shape[0])
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
            "cells": results,
        }
        write_result("e3_power", payload, elapsed())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=default_workers())
    run(parser.parse_args().workers)
