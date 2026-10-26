"""E1: selection inflation of the best in-sample Sharpe ratio under the null.

Every strategy has true Sharpe zero. For each replication we draw ``N_MAX`` independent
daily strategies, compute their annualized Sharpe ratios, and record the running maximum
at each ``N`` on the grid, so one draw serves all ``N``. The Monte Carlo mean of the
maximum is compared with

* the Bailey and Lopez de Prado (2014) approximation, using the measured cross-sectional
  variance of the trial Sharpe ratios, and
* the exact expectation for iid normal returns, where ``sqrt(T) * SR`` follows a Student
  t distribution with ``T - 1`` degrees of freedom.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import stats

from btaudit.sharpe import EULER_GAMMA, expected_max_iid, expected_max_sharpe
from btaudit.sims import NullSpec, null_returns
from experiments.common import (
    COLORS,
    INK,
    apply_style,
    default_workers,
    parallel_map,
    save_figure,
    stopwatch,
    write_result,
)

SEED = 1001
PERIODS = 252
N_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
N_MAX = N_GRID[-1]
N_REPS = 200
COLUMN_CHUNK = 500


@dataclass(frozen=True)
class Scenario:
    """A null market and sample length."""

    key: str
    label: str
    years: int
    spec: NullSpec


SCENARIOS = (
    Scenario("normal_1y", "Normal iid, 1 year", 1, NullSpec()),
    Scenario("normal_3y", "Normal iid, 3 years", 3, NullSpec()),
    Scenario("normal_10y", "Normal iid, 10 years", 10, NullSpec()),
    Scenario("t5_3y", "Student t(5) iid, 3 years", 3, NullSpec(dist="t", df=5.0)),
    Scenario(
        "garch_t5_3y", "GARCH(1,1) t(5), 3 years", 3, NullSpec(dist="t", df=5.0, garch=(0.05, 0.93))
    ),
)


def _replicate(task: tuple[Scenario, np.random.SeedSequence]) -> tuple[np.ndarray, float]:
    """Running maxima of annualized Sharpe at ``N_GRID`` and the per-period Sharpe variance."""
    scenario, seed = task
    rng = np.random.default_rng(seed)
    n_obs = scenario.years * PERIODS
    srs = []
    for _ in range(0, N_MAX, COLUMN_CHUNK):
        x = null_returns(n_obs, COLUMN_CHUNK, rng, scenario.spec)
        srs.append(x.mean(axis=0) / x.std(axis=0, ddof=1))
    sr = np.concatenate(srs)
    running = np.maximum.accumulate(sr) * np.sqrt(PERIODS)
    return running[np.array(N_GRID) - 1], float(sr.var(ddof=1))


def _bracket(n: np.ndarray) -> np.ndarray:
    """The Euler-Mascheroni bracket of the expected maximum formula (zero at N = 1)."""
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore"):
        z = (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(
            1 - 1 / (n * np.e)
        )
    return np.where(n == 1, 0.0, z)


def summarize(scenario: Scenario, maxima: np.ndarray, variances: np.ndarray) -> dict:
    """Compare Monte Carlo maxima with the approximation and the exact normal benchmark."""
    n_obs = scenario.years * PERIODS
    var_trials = float(variances.mean())
    t_dist = stats.t(df=n_obs - 1)
    rows = []
    for j, n in enumerate(N_GRID):
        mc = maxima[:, j]
        mc_mean, mc_se = float(mc.mean()), float(mc.std(ddof=1) / np.sqrt(mc.size))
        formula = expected_max_sharpe(n, var_trials) * np.sqrt(PERIODS)
        exact = expected_max_iid(n, t_dist) / np.sqrt(n_obs) * np.sqrt(PERIODS)
        rows.append(
            {
                "n_trials": n,
                "mc_mean": mc_mean,
                "mc_se": mc_se,
                "mc_q05": float(np.quantile(mc, 0.05)),
                "mc_q95": float(np.quantile(mc, 0.95)),
                "formula": formula,
                "exact_normal": exact,
                "formula_rel_error_vs_exact": formula / exact - 1 if n > 1 else None,
                "mc_rel_diff_vs_exact": mc_mean / exact - 1 if n > 1 else None,
                "mc_z_vs_exact": (mc_mean - exact) / mc_se,
            }
        )
    z = _bracket(np.array(N_GRID[1:]))
    means = np.array([r["mc_mean"] for r in rows[1:]])
    fitted_scale = float(z @ means / (z @ z))
    theory_scale = float(np.sqrt(var_trials * PERIODS))
    large_n = [r for r in rows if r["n_trials"] >= 10]
    return {
        "label": scenario.label,
        "years": scenario.years,
        "n_obs": n_obs,
        "n_reps": int(maxima.shape[0]),
        "var_trials_per_period": var_trials,
        "var_trials_null_theory": 1.0 / n_obs,
        "fitted_scale": fitted_scale,
        "theory_scale": theory_scale,
        "fitted_over_theory": fitted_scale / theory_scale,
        "max_abs_formula_error_vs_exact_n_ge_10": float(
            max(abs(r["formula_rel_error_vs_exact"]) for r in large_n)
        ),
        "max_abs_mc_z_vs_exact": float(max(abs(r["mc_z_vs_exact"]) for r in rows)),
        "rows": rows,
    }


def _count_label(value: float, _: int) -> str:
    """Tick labels 1, 10, 100, 1K, 10K."""
    return f"{value / 1000:g}K" if value >= 1000 else f"{value:g}"


def plot(results: dict) -> None:
    """Expected maximum Sharpe versus number of trials, and accuracy against the exact value."""
    apply_style()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    palette = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["aqua"],
        COLORS["yellow"],
        COLORS["magenta"],
    ]
    for color, scenario in zip(palette[:3], SCENARIOS[:3], strict=True):
        rows = results[scenario.key]["rows"]
        n = np.array([r["n_trials"] for r in rows])
        mc = np.array([r["mc_mean"] for r in rows])
        left.fill_between(
            n,
            [r["mc_q05"] for r in rows],
            [r["mc_q95"] for r in rows],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        left.plot(n, [r["formula"] for r in rows], color=color, linewidth=2)
        left.plot(
            n,
            mc,
            "o",
            color=color,
            markersize=5,
            markeredgecolor=INK["surface"],
            markeredgewidth=1.2,
            label=scenario.label,
        )
        left.annotate(
            f"{mc[-1]:.2f}",
            (n[-1], mc[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK["secondary"],
        )
    left.plot([], [], color=INK["secondary"], linewidth=2, label="Bailey-Lopez de Prado formula")
    left.set_xscale("log")
    left.set_xlim(0.8, 2.2e4)
    left.set_xlabel("Number of trials N")
    left.set_ylabel("Best in-sample Sharpe, annualized")
    left.set_title("Best of N zero-skill strategies")
    left.legend(loc="upper left")

    ref = results["normal_3y"]["rows"]
    keep = [j for j, r in enumerate(ref) if r["n_trials"] >= 10]
    n_ref = np.array([ref[j]["n_trials"] for j in keep])
    right.axhline(0, color=INK["axis"], linewidth=1)
    right.plot(
        n_ref,
        [100 * ref[j]["formula_rel_error_vs_exact"] for j in keep],
        color=INK["primary"],
        linewidth=2,
        label="Formula (3 years)",
    )
    offsets = np.linspace(-0.08, 0.08, len(SCENARIOS))
    for color, scenario, off in zip(palette, SCENARIOS, offsets, strict=True):
        rows = results[scenario.key]["rows"]
        diff = np.array([100 * rows[j]["mc_rel_diff_vs_exact"] for j in keep])
        half = np.array([196 * rows[j]["mc_se"] / rows[j]["exact_normal"] for j in keep])
        right.errorbar(
            n_ref * 10**off,
            diff,
            yerr=half,
            fmt="o",
            color=color,
            markersize=4.5,
            elinewidth=1,
            capsize=0,
            markeredgecolor=INK["surface"],
            markeredgewidth=1,
            label=f"Monte Carlo: {scenario.label}",
        )
    right.set_xscale("log")
    right.set_xlabel("Number of trials N")
    right.set_ylabel("Difference from exact normal value (%)")
    right.set_title("Accuracy against the exact expectation")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5, ncol=2)
    right.set_ylim(-8, 7)
    for ax in (left, right):
        ax.xaxis.set_major_formatter(FuncFormatter(_count_label))
    fig.text(
        0.01,
        -0.25,
        f"{results['normal_1y']['n_reps']} replications per scenario, daily "
        "returns, 252 per year. Left: markers are Monte Carlo means, bands the 5th to 95th "
        "percentile. Right: bars are 95% intervals of the Monte Carlo mean.",
        fontsize=8,
        color=INK["muted"],
    )
    save_figure(fig, "e1_selection_inflation")


def run(workers: int = default_workers()) -> dict:
    """Run E1, write ``results/e1_selection_inflation.json`` and its figure."""
    with stopwatch() as elapsed:
        seeds = np.random.SeedSequence(SEED).spawn(len(SCENARIOS))
        tasks = [
            (sc, s) for sc, root in zip(SCENARIOS, seeds, strict=True) for s in root.spawn(N_REPS)
        ]
        outputs = parallel_map(_replicate, tasks, workers)
        results = {}
        for k, scenario in enumerate(SCENARIOS):
            block = outputs[k * N_REPS : (k + 1) * N_REPS]
            maxima = np.array([b[0] for b in block])
            variances = np.array([b[1] for b in block])
            results[scenario.key] = summarize(scenario, maxima, variances)
        plot(results)
        payload = {
            "seed": SEED,
            "periods_per_year": PERIODS,
            "n_grid": list(N_GRID),
            "scenarios": results,
        }
        write_result("e1_selection_inflation", payload, elapsed())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=default_workers())
    run(parser.parse_args().workers)
