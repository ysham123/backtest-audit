"""Synthetic strategy universes with known true Sharpe ratios, and synthetic markets.

Null universes give every strategy an expected return of exactly zero, so every true
Sharpe ratio is zero. Planted-alpha universes add a known mean to exactly one column.
Volatility is scaled so that the unconditional standard deviation equals ``sigma``.

:func:`averaged_random_walk_returns` reproduces how the Shiller price series is built
(monthly averages of daily closes) on a market with no predictability at all.

Reference: Working, H. (1960). Note on the correlation of first differences of averages
in a random chain. Econometrica, 28(4), 916-918.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_DISTRIBUTIONS = ("normal", "t")


@dataclass(frozen=True)
class NullSpec:
    """Configuration of a null return generator.

    Attributes:
        dist: Innovation distribution, ``"normal"`` or ``"t"`` (Student t scaled to unit
            variance).
        df: Degrees of freedom of the t distribution (must exceed 2).
        rho: Pairwise correlation between strategies, via one common factor.
        garch: Optional ``(alpha, beta)`` of a GARCH(1,1) variance process applied to each
            strategy, with ``alpha + beta < 1``.
        sigma: Unconditional standard deviation of per-period returns.
        burn_in: Number of discarded GARCH warm-up periods.
    """

    dist: str = "normal"
    df: float = 5.0
    rho: float = 0.0
    garch: tuple[float, float] | None = None
    sigma: float = 0.01
    burn_in: int = 500

    def __post_init__(self) -> None:
        if self.dist not in _DISTRIBUTIONS:
            raise ValueError(f"dist must be one of {_DISTRIBUTIONS}")
        if self.dist == "t" and self.df <= 2:
            raise ValueError("t innovations need df > 2 for a finite variance")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        if self.garch is not None and (min(self.garch) < 0 or sum(self.garch) >= 1):
            raise ValueError("GARCH parameters must be non-negative with alpha + beta < 1")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


def _innovations(shape: tuple[int, int], spec: NullSpec, rng: np.random.Generator) -> NDArray:
    """Unit-variance iid innovations of the configured distribution."""
    if spec.dist == "normal":
        return rng.standard_normal(shape)
    return rng.standard_t(spec.df, size=shape) / np.sqrt(spec.df / (spec.df - 2.0))


def _garch_filter(z: NDArray, alpha: float, beta: float, burn_in: int) -> NDArray:
    """Turn unit-variance innovations into GARCH(1,1) returns with unit unconditional variance."""
    omega = 1.0 - alpha - beta
    var = np.ones(z.shape[1])
    out = np.empty_like(z)
    for t in range(z.shape[0]):
        out[t] = np.sqrt(var) * z[t]
        var = omega + alpha * out[t] ** 2 + beta * var
    return out[burn_in:]


def null_returns(
    n_obs: int, n_strategies: int, rng: np.random.Generator, spec: NullSpec | None = None
) -> NDArray[np.float64]:
    """Simulate a ``n_obs x n_strategies`` matrix of zero-mean strategy returns.

    Correlation enters through a common factor, ``z = sqrt(rho) f + sqrt(1 - rho) e``,
    before the optional GARCH filter, so ``rho`` is the correlation of the innovations.
    """
    spec = spec or NullSpec()
    if n_obs < 1 or n_strategies < 1:
        raise ValueError("n_obs and n_strategies must be positive")
    extra = spec.burn_in if spec.garch is not None else 0
    z = _innovations((n_obs + extra, n_strategies), spec, rng)
    if spec.rho > 0:
        common = _innovations((n_obs + extra, 1), spec, rng)
        z = np.sqrt(spec.rho) * common + np.sqrt(1.0 - spec.rho) * z
    if spec.garch is not None:
        z = _garch_filter(z, spec.garch[0], spec.garch[1], extra)
    return spec.sigma * z


def per_period_sharpe(annual_sharpe: float, periods_per_year: float) -> float:
    """Convert an annualized Sharpe ratio to a per-period one."""
    return annual_sharpe / np.sqrt(periods_per_year)


def planted_alpha_returns(
    n_obs: int,
    n_strategies: int,
    annual_sharpe: float,
    periods_per_year: float,
    rng: np.random.Generator,
    spec: NullSpec | None = None,
    planted: int = 0,
) -> NDArray[np.float64]:
    """Null universe in which column ``planted`` has true annualized Sharpe ``annual_sharpe``.

    The planted column receives mean ``SR * sigma`` per period; all other columns keep a
    true mean of zero.
    """
    if not 0 <= planted < n_strategies:
        raise ValueError("planted index out of range")
    spec = spec or NullSpec()
    x = null_returns(n_obs, n_strategies, rng, spec)
    x[:, planted] += per_period_sharpe(annual_sharpe, periods_per_year) * spec.sigma
    return x


def averaged_random_walk_returns(
    n_months: int, days_per_month: int, rng: np.random.Generator, spec: NullSpec | None = None
) -> NDArray[np.float64]:
    """Monthly returns of monthly-average prices of a driftless daily price process.

    Daily prices follow ``P_d = P_{d-1} (1 + e_d)`` with zero-mean ``e_d`` from
    :func:`null_returns`, so the price is a martingale and nothing is predictable. Each
    month's price is the average of its ``days_per_month`` daily prices and the return is
    ``A_m / A_{m-1} - 1``. Averaging alone makes consecutive returns positively correlated,
    with first-order autocorrelation ``(m^2 - 1) / (2 (2 m^2 + 1))`` for ``m`` days per
    month (Working 1960), about 0.25.
    """
    if n_months < 1 or days_per_month < 1:
        raise ValueError("n_months and days_per_month must be positive")
    daily = null_returns((n_months + 1) * days_per_month, 1, rng, spec)[:, 0]
    if np.any(daily <= -1):
        raise ValueError("daily returns at or below -100%; lower sigma")
    averages = np.cumprod(1.0 + daily).reshape(n_months + 1, days_per_month).mean(axis=1)
    return averages[1:] / averages[:-1] - 1.0
