"""E4: does PBO behave as advertised on universes with a known answer?

* Pure noise universes (every true Sharpe zero) should give PBO near 0.5 or above.
* Universes with one strong planted signal should give PBO near zero.
* The 600-rule universe of the real-data experiment, applied to simulated markets with no
  predictability, shows what PBO and the SPA test report for that exact pipeline when
  there is nothing to find. A second set of null markets adds a constant drift close to the
  S&P mean return: still nothing is predictable, but long/flat rules now earn a premium
  that long/short rules do not.
"""

from __future__ import annotations

import argparse
from math import comb

import matplotlib.pyplot as plt
import numpy as np

from btaudit.bootstrap import bootstrap_means, spa_test
from btaudit.pbo import cscv_pbo
from btaudit.sharpe import sharpe_ratio
from btaudit.sims import NullSpec, null_returns, planted_alpha_returns
from btaudit.strategies import build_universe, default_grid
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
from experiments.e5_real_data import (
    MONTHLY_DRIFT,
    MONTHLY_SIGMA,
    N_BLOCKS,
    PRIMARY_CONFIG,
    SPA_BLOCK,
    SPA_BOOT,
)

SEED = 4004
N_OBS = 1260
PERIODS = 252
NOISE_N = (10, 100, 1000)
PLANTED_N = 100
PLANTED_SHARPES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
N_UNIVERSES = 200
N_MARKETS = 100
MARKET_LENGTH = 1829
# Normal GARCH innovations: fat-tailed t innovations at monthly volatility can produce
# simple returns below -100%, which no price series allows.
MARKET_SPEC = NullSpec(dist="normal", garch=(0.10, 0.85), sigma=MONTHLY_SIGMA)
CHUNK = 10


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _matrix_task(task: tuple[int, float, list[np.random.SeedSequence]]) -> list[dict]:
    """PBO statistics for synthetic return-matrix universes."""
    n_strategies, sharpe, seeds = task
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x = planted_alpha_returns(N_OBS, n_strategies, sharpe, PERIODS, rng, NullSpec())
        res = cscv_pbo(x, N_BLOCKS)
        out.append(
            {
                "pbo": res.pbo,
                "slope": res.degradation_slope,
                "prob_loss": res.prob_oos_loss,
                "planted_selected": float(np.mean(res.is_best == 0)),
            }
        )
    return out


def _market_task(task: tuple[float, list[np.random.SeedSequence]]) -> list[dict]:
    """PBO and best Sharpe of the rule universe on null markets; SPA when the drift is zero."""
    drift, seeds = task
    specs = default_grid()
    long_short = np.array([s.long_short for s in specs])
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        market = drift + null_returns(MARKET_LENGTH, 1, rng, MARKET_SPEC)[:, 0]
        universe = build_universe(market, specs, PRIMARY_CONFIG)
        r = universe.returns[universe.start :]
        row = {
            "pbo": cscv_pbo(r, N_BLOCKS).pbo,
            "pbo_long_flat": cscv_pbo(r[:, ~long_short], N_BLOCKS).pbo,
            "pbo_long_short": cscv_pbo(r[:, long_short], N_BLOCKS).pbo,
            "best_sharpe": float(np.nanmax(sharpe_ratio(r, 12))),
        }
        if drift == 0.0:
            diffs = r - market[universe.start :, None]
            row["spa_p"] = spa_test(bootstrap_means(diffs, SPA_BOOT, SPA_BLOCK, rng)).p_consistent
        out.append(row)
    return out


def _market_summary(rows: list[dict], drift: float) -> dict:
    """Distribution of PBO and best Sharpe over simulated markets."""
    pbo = np.array([r["pbo"] for r in rows])
    out = {
        "n_markets": len(rows),
        "n_rules": len(default_grid()),
        "market_length": MARKET_LENGTH,
        "monthly_drift": drift,
        "pbo": _summary(pbo),
        "share_pbo_above_half": float(np.mean(pbo > 0.5)),
        "pbo_long_flat": _summary(np.array([r["pbo_long_flat"] for r in rows])),
        "pbo_long_short": _summary(np.array([r["pbo_long_short"] for r in rows])),
        "best_full_sample_sharpe": _summary(np.array([r["best_sharpe"] for r in rows])),
    }
    if drift == 0.0:
        rejections = int(np.sum([r["spa_p"] < 0.05 for r in rows]))
        out["spa_vs_buy_and_hold_false_positive"] = rate_summary(rejections, len(rows))
    return out


def plot(results: dict) -> None:
    """PBO distributions when nothing is real, and PBO against planted signal strength."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    groups = [(f"Noise\nN = {n}", results["noise"][str(n)]["pbo"]) for n in NOISE_N]
    null_rules, drifted = results["rules_on_null_markets"], results["rules_on_drifted_null_markets"]
    groups.append((f"{null_rules['n_rules']} rules,\nnull market", null_rules["pbo"]))
    groups.append((f"{drifted['n_rules']} rules, null\nmarket + drift", drifted["pbo"]))
    for i, (_, s) in enumerate(groups):
        color = COLORS["blue"] if i < len(NOISE_N) else COLORS["orange"]
        ax.plot([i, i], [s["q05"], s["q95"]], color=color, linewidth=1.5)
        ax.plot([i, i], [s["q25"], s["q75"]], color=color, linewidth=6, solid_capstyle="butt")
        ax.plot(
            i,
            s["median"],
            "o",
            color=INK["surface"],
            markersize=5,
            markeredgecolor=color,
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"mean {s['mean']:.2f}",
            (i, s["q95"]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK["secondary"],
        )
    ax.axhline(0.5, color=INK["secondary"], linewidth=1)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PBO")
    ax.set_title("A. PBO when nothing is real")

    ax = axes[1]
    rows = [results["planted"][f"{s:g}"] for s in PLANTED_SHARPES]
    ax.fill_between(
        PLANTED_SHARPES,
        [r["pbo"]["q05"] for r in rows],
        [r["pbo"]["q95"] for r in rows],
        color=COLORS["blue"],
        alpha=0.10,
        linewidth=0,
    )
    ax.plot(
        PLANTED_SHARPES,
        [r["pbo"]["mean"] for r in rows],
        "-o",
        color=COLORS["blue"],
        markersize=5,
        markeredgecolor=INK["surface"],
        label="Mean PBO (band: 5th to 95th pct.)",
    )
    ax.plot(
        PLANTED_SHARPES,
        [r["planted_selected"] for r in rows],
        "-o",
        color=COLORS["aqua"],
        markersize=5,
        markeredgecolor=INK["surface"],
        label="Share of splits selecting the skilled strategy",
    )
    ax.set_xticks(PLANTED_SHARPES)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("True Sharpe of the one skilled strategy (annualized)")
    ax.set_title(f"B. One skilled strategy among {PLANTED_N}")
    ax.legend(loc="upper left", fontsize=7.5)

    fig.text(
        0.01,
        -0.03,
        f"CSCV with S = {N_BLOCKS} blocks ({results['n_splits']} splits). "
        f"Synthetic universes: {N_UNIVERSES} per setting, {N_OBS} daily returns. Null markets: "
        f"{N_MARKETS} GARCH(1,1) monthly series of {MARKET_LENGTH} months, drift 0 or "
        f"{MONTHLY_DRIFT} per month. Bars: 5th to 95th and 25th to 75th percentiles.",
        fontsize=8,
        color=INK["muted"],
    )
    save_figure(fig, "e4_pbo_sanity")


def run(workers: int = default_workers()) -> dict:
    """Run E4, write ``results/e4_pbo_sanity.json`` and its figure."""
    with stopwatch() as elapsed:
        root_noise, root_planted, root_market = np.random.SeedSequence(SEED).spawn(3)
        # The planted-Sharpe-zero cell is the noise cell with N = PLANTED_N, computed once.
        planted = [s for s in PLANTED_SHARPES if s > 0]
        matrix_cells = [(n, 0.0) for n in NOISE_N] + [(PLANTED_N, s) for s in planted]
        cell_roots = root_noise.spawn(len(NOISE_N)) + root_planted.spawn(len(planted))
        tasks = [
            (n, s, chunk)
            for (n, s), root in zip(matrix_cells, cell_roots, strict=True)
            for chunk in chunked(root.spawn(N_UNIVERSES), CHUNK)
        ]
        drifts = (0.0, MONTHLY_DRIFT)
        market_tasks = [
            (drift, chunk)
            for drift, root in zip(drifts, root_market.spawn(2), strict=True)
            for chunk in chunked(root.spawn(N_MARKETS), CHUNK)
        ]
        matrix_out = parallel_map(_matrix_task, tasks, workers)
        market_out = [
            row for chunk in parallel_map(_market_task, market_tasks, workers) for row in chunk
        ]
        per_cell = N_UNIVERSES // CHUNK
        cells = {}
        for k, key in enumerate(matrix_cells):
            rows = [row for chunk in matrix_out[k * per_cell : (k + 1) * per_cell] for row in chunk]
            pbo = np.array([r["pbo"] for r in rows])
            cells[key] = {
                "n_strategies": key[0],
                "annual_sharpe": key[1],
                "n_universes": len(rows),
                "pbo": _summary(pbo),
                "share_pbo_above_half": float(np.mean(pbo > 0.5)),
                "mean_degradation_slope": float(np.mean([r["slope"] for r in rows])),
                "mean_prob_oos_loss": float(np.mean([r["prob_loss"] for r in rows])),
                "planted_selected": float(np.mean([r["planted_selected"] for r in rows])),
            }
        results = {
            "n_blocks": N_BLOCKS,
            "n_splits": comb(N_BLOCKS, N_BLOCKS // 2),
            "noise": {str(n): cells[(n, 0.0)] for n in NOISE_N},
            "planted": {f"{s:g}": cells[(PLANTED_N, s)] for s in PLANTED_SHARPES},
            "rules_on_null_markets": _market_summary(market_out[:N_MARKETS], 0.0),
            "rules_on_drifted_null_markets": _market_summary(market_out[N_MARKETS:], MONTHLY_DRIFT),
        }
        plot(results)
        payload = {"seed": SEED, "n_obs": N_OBS, "periods_per_year": PERIODS, **results}
        write_result("e4_pbo_sanity", payload, elapsed())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=default_workers())
    run(parser.parse_args().workers)
