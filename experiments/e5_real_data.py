"""E5: a 600-rule timing sweep on S&P composite monthly total returns, audited.

Protocol (fixed before looking at results)
------------------------------------------
* Asset: S&P composite monthly total return, 1871-02 to 2023-06 (``btaudit.data``).
* Universe: :func:`btaudit.strategies.default_grid`, 600 variants, net of 10 bps per unit
  of turnover, 15% annualized volatility target where used, cash earns zero.
* Primary signal lag: two months. Shiller prices are monthly averages, so a one-month lag
  still overlaps the return it trades (see ``data/DATA.md``). The one-month lag is run as
  a second sweep to show the size of that artifact, and both lags are rerun on simulated
  averaged random walks where nothing is predictable.
* In-sample: evaluation start to 1989-12. Out-of-sample: 1990-01 to 2023-06.
* Selection: highest in-sample Sharpe. Tests: DSR with the number of trials actually run,
  PBO via CSCV on the full evaluation window with S = 16, White's Reality Check and
  Hansen's SPA against buy-and-hold on the full window (stationary bootstrap, mean block
  12 months, 2000 resamples).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btaudit.bootstrap import bootstrap_means, reality_check, spa_test
from btaudit.data import PERIODS_PER_YEAR, monthly_total_returns
from btaudit.pbo import cscv_pbo
from btaudit.sharpe import (
    effective_number_of_trials,
    probabilistic_sharpe_ratio,
    sample_moments,
    select_and_deflate,
    sharpe_ratio,
)
from btaudit.sims import NullSpec, averaged_random_walk_returns
from btaudit.strategies import StrategySpec, UniverseConfig, build_universe, default_grid
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

SEED = 5005
SPLIT_DATE = pd.Timestamp("1990-01-01")
COST_BPS = 10.0
N_BLOCKS = 16
SPA_BOOT = 2000
SPA_BLOCK = 12.0
BLOCK_SENSITIVITY = (3.0, 6.0, 24.0)
MONTHLY_SIGMA = 0.04  # close to the 0.041 sample volatility of the monthly series
MONTHLY_DRIFT = 0.008  # close to the 0.0082 sample mean of the monthly series
PRIMARY_CONFIG = UniverseConfig(lag=2, cost_bps=COST_BPS, periods_per_year=PERIODS_PER_YEAR)
LAG1_CONFIG = UniverseConfig(lag=1, cost_bps=COST_BPS, periods_per_year=PERIODS_PER_YEAR)
TRADING_DAYS_PER_MONTH = 21
DAILY_SPEC = NullSpec(
    dist="t", df=5.0, garch=(0.05, 0.93), sigma=MONTHLY_SIGMA / np.sqrt(TRADING_DAYS_PER_MONTH)
)
N_ARTIFACT_MARKETS = 100
ARTIFACT_CHUNK = 10
FAMILY_COLORS = {
    "ma_crossover": COLORS["blue"],
    "tsmom": COLORS["orange"],
    "mean_reversion": COLORS["aqua"],
}
FAMILY_LABELS = {
    "ma_crossover": "MA crossover",
    "tsmom": "Time-series momentum",
    "mean_reversion": "Mean reversion",
}


@dataclass(frozen=True)
class Sweep:
    """All rules on one asset, restricted to the common evaluation window."""

    specs: tuple[StrategySpec, ...]
    dates: pd.DatetimeIndex
    strat: np.ndarray
    bench: np.ndarray
    positions: np.ndarray
    is_mask: np.ndarray

    @property
    def sr_is(self) -> np.ndarray:
        return _annual_sr(self.strat[self.is_mask])

    @property
    def sr_oos(self) -> np.ndarray:
        return _annual_sr(self.strat[~self.is_mask])

    def family_columns(self, family: str) -> np.ndarray:
        return np.flatnonzero([s.family == family for s in self.specs])

    def side_columns(self) -> dict[str, np.ndarray]:
        flags = np.array([s.long_short for s in self.specs])
        return {"long_flat": np.flatnonzero(~flags), "long_short": np.flatnonzero(flags)}


def _annual_sr(x: np.ndarray) -> np.ndarray:
    return np.asarray(sharpe_ratio(x, PERIODS_PER_YEAR))


def build_sweep(returns: pd.Series, config: UniverseConfig) -> Sweep:
    """Apply the default grid and cut every series to the common evaluation window."""
    universe = build_universe(returns.to_numpy(), default_grid(), config)
    start = universe.start
    dates = returns.index[start:]
    return Sweep(
        universe.specs,
        dates,
        universe.returns[start:],
        returns.to_numpy()[start:],
        universe.positions[start:],
        np.asarray(dates < SPLIT_DATE),
    )


def selection_summary(sweep: Sweep) -> dict:
    """The in-sample winner, how it did out of sample, and buy-and-hold for reference."""
    sr_is, sr_oos = sweep.sr_is, sweep.sr_oos
    best = int(np.nanargmax(sr_is))
    spec = sweep.specs[best]
    skew, kurt = sample_moments(sweep.strat[sweep.is_mask, best])
    bh_oos = float(_annual_sr(sweep.bench[~sweep.is_mask]))
    return {
        "buy_and_hold": {
            "is_sharpe": float(_annual_sr(sweep.bench[sweep.is_mask])),
            "oos_sharpe": bh_oos,
            "full_sharpe": float(_annual_sr(sweep.bench)),
            "log10_growth": float(np.log10(1 + sweep.bench).sum()),
        },
        "best_in_sample": {
            "index": best,
            "rule": spec.name,
            "family": spec.family,
            "params": list(spec.params),
            "long_short": spec.long_short,
            "vol_target": spec.vol_target,
            "is_sharpe": float(sr_is[best]),
            "oos_sharpe": float(sr_oos[best]),
            "full_sharpe": float(_annual_sr(sweep.strat[:, best])),
            "log10_growth": float(np.log10(1 + sweep.strat[:, best]).sum()),
            "oos_percentile_among_rules": float(np.mean(sr_oos < sr_oos[best])),
            "annual_turnover": float(
                np.abs(np.diff(sweep.positions[:, best])).mean() * PERIODS_PER_YEAR
            ),
            "is_skew": float(skew),
            "is_kurtosis": float(kurt),
        },
        "median_is_sharpe": float(np.median(sr_is)),
        "median_oos_sharpe": float(np.median(sr_oos)),
        "share_rules_oos_above_buy_and_hold": float(np.mean(sr_oos > bh_oos)),
    }


def _deflate(returns: np.ndarray) -> dict:
    """DSR with the raw trial count and with the effective number of trials."""
    n_eff = effective_number_of_trials(returns)
    raw, eff = select_and_deflate(returns), select_and_deflate(returns, n_trials=n_eff)
    scale = np.sqrt(PERIODS_PER_YEAR)
    return {
        "best": raw.best,
        "sr_best_annual": raw.sr_best * scale,
        "n_trials": int(raw.n_trials),
        "sr0_annual": raw.sr0 * scale,
        "naive_psr_pvalue": raw.psr_pvalue,
        "dsr_pvalue": raw.dsr_pvalue,
        "var_trials_annual": float(np.var(_annual_sr(returns), ddof=1)),
        "effective_trials": n_eff,
        "sr0_annual_effective": eff.sr0 * scale,
        "dsr_pvalue_effective": eff.dsr_pvalue,
    }


def deflation_summary(sweep: Sweep) -> dict:
    """DSR of the in-sample winner (against zero) and of the best active return (against B&H)."""
    bench_is = sweep.bench[sweep.is_mask]
    skew, kurt = sample_moments(bench_is)
    bh_psr = probabilistic_sharpe_ratio(sharpe_ratio(bench_is), 0.0, bench_is.size, skew, kurt)
    active = _deflate(sweep.strat[sweep.is_mask] - bench_is[:, None])
    active["rule"] = sweep.specs[active["best"]].name
    return {
        "vs_zero": _deflate(sweep.strat[sweep.is_mask]),
        "active_vs_buy_and_hold": active,
        "buy_and_hold_psr_pvalue_vs_zero": 1.0 - float(bh_psr),
    }


def overfitting_summary(sweep: Sweep) -> tuple[dict, np.ndarray]:
    """CSCV PBO over the full window: all rules, each family, and long/flat versus long/short."""
    full = cscv_pbo(sweep.strat, N_BLOCKS)
    scale = np.sqrt(PERIODS_PER_YEAR)
    out = {
        "n_blocks": N_BLOCKS,
        "n_splits": full.n_combinations,
        "pbo": full.pbo,
        "degradation_slope": full.degradation_slope,
        "degradation_intercept_annual": full.degradation_intercept * scale,
        "degradation_r2": full.degradation_r2,
        "prob_oos_loss": full.prob_oos_loss,
        "by_family": {
            f: cscv_pbo(sweep.strat[:, sweep.family_columns(f)], N_BLOCKS).pbo
            for f in FAMILY_LABELS
        },
        "by_side": {
            side: cscv_pbo(sweep.strat[:, cols], N_BLOCKS).pbo
            for side, cols in sweep.side_columns().items()
        },
    }
    return out, full.logits


def spa_vs_benchmark(diffs: np.ndarray, mean_block: float, rng: np.random.Generator) -> dict:
    """Reality Check and SPA p-values for one matrix of performance differentials."""
    bm = bootstrap_means(diffs, SPA_BOOT, mean_block, rng)
    rc, spa = reality_check(bm), spa_test(bm)
    return {
        "mean_block": mean_block,
        "reality_check_p": rc.p_value,
        "spa_consistent_p": spa.p_consistent,
        "spa_lower_p": spa.p_lower,
        "spa_upper_p": spa.p_upper,
        "spa_statistic": spa.statistic,
        "spa_best": spa.best,
    }


def superiority_summary(sweep: Sweep, specs_names: list[str], rng: np.random.Generator) -> dict:
    """Does any rule beat buy-and-hold? Raw and volatility-matched differentials."""
    vol_ratio = sweep.bench.std(ddof=1) / sweep.strat.std(axis=0, ddof=1)
    raw = sweep.strat - sweep.bench[:, None]
    matched = sweep.strat * vol_ratio - sweep.bench[:, None]
    result = {
        "raw": spa_vs_benchmark(raw, SPA_BLOCK, rng),
        "vol_matched": spa_vs_benchmark(matched, SPA_BLOCK, rng),
        "raw_block_sensitivity": [spa_vs_benchmark(raw, b, rng) for b in BLOCK_SENSITIVITY],
    }
    for key in ("raw", "vol_matched"):
        result[key]["spa_best_rule"] = specs_names[result[key]["spa_best"]]
    return result


def family_summary(sweep: Sweep) -> dict:
    """Best in-sample rule of each family and the family's median OOS Sharpe."""
    sr_is, sr_oos = sweep.sr_is, sweep.sr_oos
    out = {}
    for family in FAMILY_LABELS:
        cols = sweep.family_columns(family)
        top = int(cols[np.argmax(sr_is[cols])])
        out[family] = {
            "n_rules": int(cols.size),
            "best_rule": sweep.specs[top].name,
            "is_sharpe": float(sr_is[top]),
            "oos_sharpe": float(sr_oos[top]),
            "median_oos_sharpe": float(np.median(sr_oos[cols])),
        }
    return out


def analyze(
    returns: pd.Series, config: UniverseConfig, rng: np.random.Generator
) -> tuple[dict, Sweep, np.ndarray]:
    """Full audit for one signal lag: summary, the sweep, and the CSCV logits."""
    sweep = build_sweep(returns, config)
    pbo, logits = overfitting_summary(sweep)
    summary = {
        "lag": config.lag,
        "n_rules": len(sweep.specs),
        "evaluation_start": str(sweep.dates[0].date()),
        "evaluation_end": str(sweep.dates[-1].date()),
        "n_months_is": int(sweep.is_mask.sum()),
        "n_months_oos": int((~sweep.is_mask).sum()),
        **selection_summary(sweep),
        "dsr": deflation_summary(sweep),
        "pbo": pbo,
        "spa_vs_buy_and_hold": superiority_summary(sweep, [s.name for s in sweep.specs], rng),
        "families": family_summary(sweep),
    }
    return summary, sweep, logits


def _artifact_task(task: tuple[int, list[np.random.SeedSequence]]) -> list[list[float]]:
    """SPA p-values (lag 1, lag 2) of the rule universe on averaged random walks."""
    n_months, seeds = task
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        market = averaged_random_walk_returns(n_months, TRADING_DAYS_PER_MONTH, rng, DAILY_SPEC)
        row = []
        for config in (LAG1_CONFIG, PRIMARY_CONFIG):
            universe = build_universe(market, default_grid(), config)
            s = universe.start
            diffs = universe.returns[s:] - market[s:, None]
            row.append(spa_vs_benchmark(diffs, SPA_BLOCK, rng)["spa_consistent_p"])
        out.append(row)
    return out


def artifact_check(n_months: int, seed: np.random.SeedSequence, workers: int) -> dict:
    """How often SPA 'finds' a rule beating buy-and-hold when prices are averaged noise."""
    tasks = [(n_months, chunk) for chunk in chunked(seed.spawn(N_ARTIFACT_MARKETS), ARTIFACT_CHUNK)]
    p_values = np.array(
        [row for chunk in parallel_map(_artifact_task, tasks, workers) for row in chunk]
    )
    reject = p_values < 0.05
    return {
        "n_markets": N_ARTIFACT_MARKETS,
        "n_months": n_months,
        "days_per_month": TRADING_DAYS_PER_MONTH,
        "spa_reject_lag1": rate_summary(int(reject[:, 0].sum()), reject.shape[0]),
        "spa_reject_lag2": rate_summary(int(reject[:, 1].sum()), reject.shape[0]),
        "median_spa_p_lag1": float(np.median(p_values[:, 0])),
        "median_spa_p_lag2": float(np.median(p_values[:, 1])),
    }


def _autocorrelation(x: np.ndarray, lag: int) -> float:
    c = x - x.mean()
    return float(c[lag:] @ c[:-lag] / (c @ c))


def plot(primary: dict, sweep: Sweep, logits: np.ndarray, lag1: dict, sweep1: Sweep) -> None:
    """Growth of the selected rules, IS versus OOS Sharpe, and the CSCV logits."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    b2, b1 = primary["best_in_sample"], lag1["best_in_sample"]
    lines = [
        (sweep.dates, sweep.bench, INK["secondary"], "Buy and hold"),
        (
            sweep.dates,
            sweep.strat[:, b2["index"]],
            COLORS["blue"],
            f"IS-best, 2-month lag: {b2['rule']}",
        ),
        (
            sweep1.dates,
            sweep1.strat[:, b1["index"]],
            COLORS["orange"],
            f"IS-best, 1-month lag: {b1['rule']}",
        ),
    ]
    for dates, r, color, label in lines:
        ax.plot(dates, np.cumsum(np.log10(1 + r)), color=color, linewidth=1.6, label=label)
    ax.axvspan(SPLIT_DATE, sweep.dates[-1], color=INK["grid"], alpha=0.5, linewidth=0)
    ax.annotate(
        "out of sample",
        (SPLIT_DATE, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=INK["secondary"],
    )
    ax.set_ylabel("log10 growth of 1, net of costs")
    ax.set_title("A. Selected rules and buy and hold")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    sr_is, sr_oos = sweep.sr_is, sweep.sr_oos
    for family, color in FAMILY_COLORS.items():
        cols = sweep.family_columns(family)
        ax.plot(
            sr_is[cols],
            sr_oos[cols],
            "o",
            color=color,
            markersize=3.5,
            alpha=0.6,
            markeredgewidth=0,
            label=FAMILY_LABELS[family],
        )
    ax.plot(
        sr_is[b2["index"]],
        sr_oos[b2["index"]],
        "o",
        color=COLORS["blue"],
        markersize=9,
        markeredgecolor=INK["primary"],
        markeredgewidth=1.2,
        label="IS-best rule",
    )
    bh = primary["buy_and_hold"]
    ax.axhline(bh["oos_sharpe"], color=INK["secondary"], linewidth=1)
    ax.axvline(bh["is_sharpe"], color=INK["secondary"], linewidth=1)
    ax.annotate(
        "buy and hold",
        (bh["is_sharpe"], bh["oos_sharpe"]),
        xytext=(-4, -12),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK["secondary"],
    )
    ax.set_xlabel("In-sample Sharpe, to 1989 (annualized)")
    ax.set_ylabel("Out-of-sample Sharpe, 1990 to 2023")
    ax.set_title(f"B. {primary['n_rules']} rules, 2-month lag")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[2]
    ax.hist(
        logits,
        bins=np.linspace(-7, 7, 43),
        color=COLORS["blue"],
        edgecolor=INK["surface"],
        linewidth=0.8,
    )
    ax.axvline(0, color=INK["secondary"], linewidth=1)
    ax.set_xlabel("Logit of the OOS rank of the IS-best rule")
    ax.set_ylabel("Number of CSCV splits")
    ax.set_title(f"C. CSCV, 2-month lag, PBO = {primary['pbo']['pbo']:.3f}")
    fig.text(
        0.01,
        -0.03,
        "S&P composite monthly total returns (Shiller data), 10 bps per unit "
        "turnover, cash earns zero. CSCV uses S = 16 blocks over the full evaluation window.",
        fontsize=8,
        color=INK["muted"],
    )
    save_figure(fig, "e5_real_data")


def run(workers: int = default_workers()) -> dict:
    """Run E5, write ``results/e5_real_data.json`` and its figure."""
    with stopwatch() as elapsed:
        returns = monthly_total_returns()
        seed_primary, seed_lag1, seed_artifact = np.random.SeedSequence(SEED).spawn(3)
        primary, sweep, logits = analyze(
            returns, PRIMARY_CONFIG, np.random.default_rng(seed_primary)
        )
        lag1, sweep1, _ = analyze(returns, LAG1_CONFIG, np.random.default_rng(seed_lag1))
        plot(primary, sweep, logits, lag1, sweep1)
        values = returns.to_numpy()
        payload = {
            "seed": SEED,
            "data": {
                "first_month": str(returns.index[0].date()),
                "last_month": str(returns.index[-1].date()),
                "n_months": len(returns),
                "monthly_volatility": float(values.std(ddof=1)),
                "autocorrelation_lag1": _autocorrelation(values, 1),
                "autocorrelation_lag2": _autocorrelation(values, 2),
            },
            "protocol": {
                "split_date": str(SPLIT_DATE.date()),
                "cost_bps": COST_BPS,
                "vol_target": PRIMARY_CONFIG.vol_target,
                "vol_window": PRIMARY_CONFIG.vol_window,
                "max_leverage": PRIMARY_CONFIG.max_leverage,
                "spa_boot": SPA_BOOT,
                "spa_mean_block": SPA_BLOCK,
            },
            "primary_lag2": primary,
            "artifact_lag1": lag1,
            "averaged_random_walk_check": artifact_check(len(returns), seed_artifact, workers),
        }
        write_result("e5_real_data", payload, elapsed())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=default_workers())
    run(parser.parse_args().workers)
