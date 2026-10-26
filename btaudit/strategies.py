"""Rule-based timing strategies on a single asset, built as a return matrix.

Timing convention
-----------------
A raw signal at index ``t`` uses only returns ``r_0 .. r_t`` (information available at the
end of period ``t``). The position held over period ``t`` is the signal from ``t - lag``
with ``lag >= 1``, so it is known before ``r_t`` is realized. The strategy return over
period ``t`` is ``w_t * r_t - c * |w_t - w_{t-1}|`` with ``c`` the cost per unit turnover.

Families
--------
* ``ma_crossover(fast, slow)``: long when the fast moving average of the price index is
  above the slow one.
* ``tsmom(lookback)``: long when the trailing ``lookback``-period return is positive
  (Moskowitz, Ooi and Pedersen 2012).
* ``mean_reversion(lookback, threshold)``: long when the log price is more than
  ``threshold`` rolling standard deviations below its rolling mean, short (or flat) when
  it is more than ``threshold`` above it, flat otherwise.

Each rule is either long/short (the bearish state is short) or long/flat (the bearish
state is cash, earning zero), and optionally volatility targeted: exposure is scaled by
``target / trailing realized volatility``, capped at ``max_leverage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

FAMILIES = ("ma_crossover", "tsmom", "mean_reversion")

MA_FAST = (1, 2, 3, 4, 5, 6, 8, 10, 12)
MA_SLOW = (6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 48, 60)
TSMOM_LOOKBACK = tuple(range(1, 25)) + (30, 36, 48, 60)
MR_LOOKBACK = (6, 9, 12, 18, 24, 36, 48, 60)
MR_THRESHOLD = (0.5, 1.0, 1.5)


@dataclass(frozen=True)
class StrategySpec:
    """One rule variant.

    Attributes:
        family: One of :data:`FAMILIES`.
        params: ``(fast, slow)``, ``(lookback,)`` or ``(lookback, threshold)``.
        long_short: Short in the bearish state if True, flat otherwise.
        vol_target: Scale exposure to a volatility target if True.
    """

    family: str
    params: tuple[float, ...]
    long_short: bool
    vol_target: bool

    def __post_init__(self) -> None:
        expected = {"ma_crossover": 2, "tsmom": 1, "mean_reversion": 2}
        if self.family not in expected:
            raise ValueError(f"family must be one of {FAMILIES}")
        if len(self.params) != expected[self.family]:
            raise ValueError(f"{self.family} takes {expected[self.family]} parameters")
        if self.family == "ma_crossover" and not 1 <= self.params[0] < self.params[1]:
            raise ValueError("ma_crossover needs 1 <= fast < slow")
        if self.family == "mean_reversion":
            lookback, threshold = self.params
            if lookback < 2:
                raise ValueError("mean_reversion needs a lookback of at least 2")
            # The z-score of the newest of L points is at most (L - 1) / sqrt(L) in absolute
            # value, so a higher threshold describes a rule that can never trade.
            if threshold >= (lookback - 1) / np.sqrt(lookback):
                raise ValueError("mean_reversion threshold is unreachable for this lookback")
        if self.family == "tsmom" and self.params[0] < 1:
            raise ValueError("tsmom needs a lookback of at least 1")

    @property
    def name(self) -> str:
        """Readable identifier such as ``tsmom(12)|LF|raw``."""
        params = ",".join(f"{p:g}" for p in self.params)
        side = "LS" if self.long_short else "LF"
        scaling = "VT" if self.vol_target else "raw"
        return f"{self.family}({params})|{side}|{scaling}"


@dataclass(frozen=True)
class UniverseConfig:
    """Execution settings shared by all rules.

    Attributes:
        lag: Periods between the signal and the return it is applied to (at least 1).
        cost_bps: Cost in basis points per unit of turnover ``|w_t - w_{t-1}|``.
        periods_per_year: Used to annualize realized volatility for targeting.
        vol_target: Annualized volatility target for volatility-targeted rules.
        vol_window: Trailing window (periods) of the realized volatility estimate.
        max_leverage: Cap on the volatility-targeting scale factor.
    """

    lag: int = 1
    cost_bps: float = 0.0
    periods_per_year: float = 12.0
    vol_target: float = 0.15
    vol_window: int = 12
    max_leverage: float = 2.0

    def __post_init__(self) -> None:
        if self.lag < 1:
            raise ValueError("lag must be at least 1: a signal cannot trade its own period")
        if self.cost_bps < 0 or self.vol_target <= 0 or self.max_leverage <= 0:
            raise ValueError("cost_bps must be non-negative, vol_target and max_leverage positive")
        if self.vol_window < 2:
            raise ValueError("vol_window must be at least 2")


@dataclass(frozen=True)
class Universe:
    """Strategy returns and positions; rows before ``start`` are ``nan``.

    Attributes:
        returns: ``T x K`` net strategy returns.
        positions: ``T x K`` positions ``w_t`` applied to ``r_t``.
        specs: The ``K`` rule specifications, in column order.
        start: First row where every rule has a position.
    """

    returns: NDArray[np.float64]
    positions: NDArray[np.float64]
    specs: tuple[StrategySpec, ...]
    start: int


def default_grid() -> list[StrategySpec]:
    """The 600-variant grid used in the real-data experiment.

    392 moving-average crossovers, 112 time-series momentum rules and 96 mean-reversion
    rules: each core rule in long/flat and long/short form, with and without volatility
    targeting.
    """
    cores: list[tuple[str, tuple[float, ...]]] = []
    cores += [("ma_crossover", (f, s)) for f, s in product(MA_FAST, MA_SLOW) if f < s]
    cores += [("tsmom", (lb,)) for lb in TSMOM_LOOKBACK]
    cores += [("mean_reversion", (lb, th)) for lb, th in product(MR_LOOKBACK, MR_THRESHOLD)]
    return [
        StrategySpec(family, params, long_short, vol_target)
        for family, params in cores
        for long_short in (False, True)
        for vol_target in (False, True)
    ]


def rolling_mean(x: NDArray, window: int) -> NDArray[np.float64]:
    """Trailing mean over ``x[t - window + 1 .. t]``; ``nan`` for the first ``window - 1`` rows."""
    out = np.full(x.shape[0], np.nan)
    out[window - 1 :] = sliding_window_view(x, window).mean(axis=-1)
    return out


def rolling_std(x: NDArray, window: int) -> NDArray[np.float64]:
    """Trailing sample standard deviation (ddof=1) over ``x[t - window + 1 .. t]``."""
    out = np.full(x.shape[0], np.nan)
    out[window - 1 :] = sliding_window_view(x, window).std(axis=-1, ddof=1)
    return out


def _state(bullish: NDArray, bearish: NDArray, valid: NDArray, long_short: bool) -> NDArray:
    """Map bullish/bearish indicators to +1 / (-1 or 0), ``nan`` where not yet valid."""
    signal = bullish.astype(float) - (bearish.astype(float) if long_short else 0.0)
    return np.where(valid, signal, np.nan)


def raw_signal(log_price: NDArray, spec: StrategySpec, cache: dict | None = None) -> NDArray:
    """Signal in ``{-1, 0, 1}`` at each index from log prices up to that index.

    Args:
        log_price: Log price index, ``log_price[t] = sum(log(1 + r_0..r_t))``.
        spec: Rule specification.
        cache: Optional dict reused across calls to share rolling statistics.
    """
    cache = {} if cache is None else cache

    def stat(kind: str, window: int) -> NDArray:
        key = (kind, window)
        if key not in cache:
            if kind == "ma":
                cache[key] = rolling_mean(np.exp(log_price), window)
            elif kind == "lp_mean":
                cache[key] = rolling_mean(log_price, window)
            else:
                cache[key] = rolling_std(log_price, window)
        return cache[key]

    if spec.family == "ma_crossover":
        fast, slow = (int(p) for p in spec.params)
        diff = stat("ma", fast) - stat("ma", slow)
        return _state(diff > 0, diff <= 0, ~np.isnan(diff), spec.long_short)
    if spec.family == "tsmom":
        lookback = int(spec.params[0])
        past = np.full(log_price.shape[0], np.nan)
        past[lookback:] = log_price[lookback:] - log_price[:-lookback]
        return _state(past > 0, past <= 0, ~np.isnan(past), spec.long_short)
    lookback, threshold = int(spec.params[0]), float(spec.params[1])
    std = stat("lp_std", lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (log_price - stat("lp_mean", lookback)) / std
    # A flat window (std == 0) gives z = nan, which maps to the neutral state.
    return _state(z < -threshold, z > threshold, ~np.isnan(std), spec.long_short)


def volatility_scale(returns: NDArray, config: UniverseConfig) -> NDArray[np.float64]:
    """Exposure multiplier ``min(target / realized vol, max_leverage)`` known at each index."""
    vol = rolling_std(returns, config.vol_window) * np.sqrt(config.periods_per_year)
    with np.errstate(divide="ignore"):
        scale = np.where(vol > 0, config.vol_target / vol, config.max_leverage)
    return np.where(np.isnan(vol), np.nan, np.minimum(scale, config.max_leverage))


def lag_series(x: NDArray, lag: int) -> NDArray[np.float64]:
    """Shift a series forward by ``lag`` periods, filling the first ``lag`` rows with ``nan``."""
    out = np.full(x.shape[0], np.nan)
    out[lag:] = x[:-lag]
    return out


def net_returns(asset_returns: NDArray, positions: NDArray, cost_bps: float) -> NDArray:
    """``w_t r_t - cost * |w_t - w_{t-1}|``; the position before the first one is zero."""
    held = np.nan_to_num(positions, nan=0.0)
    turnover = np.abs(np.diff(held, prepend=0.0, axis=0))
    net = held * asset_returns[:, None] - cost_bps * 1e-4 * turnover
    return np.where(np.isnan(positions), np.nan, net)


def build_universe(
    asset_returns: ArrayLike, specs: list[StrategySpec], config: UniverseConfig | None = None
) -> Universe:
    """Apply every rule in ``specs`` to one asset return series.

    Args:
        asset_returns: Simple returns ``r_t`` of the traded asset (all ``> -1``).
        specs: Rule specifications; one output column each.
        config: Execution settings.
    """
    config = config or UniverseConfig()
    r = np.asarray(asset_returns, dtype=float)
    if r.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r <= -1):
        raise ValueError("asset_returns must be a finite 1-D series with values above -1")
    log_price = np.cumsum(np.log1p(r))
    scale = volatility_scale(r, config)
    cache: dict = {}
    positions = np.empty((r.shape[0], len(specs)))
    for col, spec in enumerate(specs):
        exposure = raw_signal(log_price, spec, cache)
        if spec.vol_target:
            exposure = exposure * scale
        positions[:, col] = lag_series(exposure, config.lag)
    valid_rows = np.flatnonzero(~np.isnan(positions).any(axis=1))
    if valid_rows.size == 0:
        raise ValueError("series too short for the requested rules")
    start = int(valid_rows[0])
    positions[:start] = np.nan
    returns = net_returns(r, positions, config.cost_bps)
    return Universe(returns, positions, tuple(specs), start)
