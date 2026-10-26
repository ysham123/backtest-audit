"""Sharpe ratio estimation, the Probabilistic Sharpe Ratio and the Deflated Sharpe Ratio.

Conventions
-----------
All inference functions work with *per-period* Sharpe ratios (for example monthly or
daily), exactly as in the source papers. Use :func:`sharpe_ratio` with
``periods_per_year`` only for reporting annualized values.

References
----------
Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier.
    Journal of Risk, 15(2), 3-44.
Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for
    selection bias, backtest overfitting and non-normality. Journal of Portfolio
    Management, 40(5), 94-107.
Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper,
    University of Basel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, stats

EULER_GAMMA: float = float(np.euler_gamma)


def sharpe_ratio(
    returns: ArrayLike, periods_per_year: float = 1.0, axis: int = 0
) -> float | NDArray[np.float64]:
    """Sample Sharpe ratio ``mean / std`` (ddof=1), scaled by ``sqrt(periods_per_year)``.

    Returns are taken as given (already in excess of any benchmark rate). Columns with
    zero sample variance yield ``nan``.

    Args:
        returns: Return series (1-D) or matrix of return series.
        periods_per_year: Annualization factor; 1 returns the per-period ratio.
        axis: Time axis.

    Returns:
        A float for 1-D input, otherwise an array with ``axis`` removed.
    """
    x = np.asarray(returns, dtype=float)
    if x.shape[axis] < 2:
        raise ValueError("need at least two observations")
    mean = x.mean(axis=axis)
    std = x.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, mean / std, np.nan) * np.sqrt(periods_per_year)
    return float(sr) if sr.ndim == 0 else sr


def sample_moments(returns: ArrayLike, axis: int = 0) -> tuple[NDArray, NDArray]:
    """Sample skewness and (non-excess) kurtosis using moment estimators.

    A normal distribution has skewness 0 and kurtosis 3.
    """
    x = np.asarray(returns, dtype=float)
    centered = x - x.mean(axis=axis, keepdims=True)
    m2 = np.mean(centered**2, axis=axis)
    m3 = np.mean(centered**3, axis=axis)
    m4 = np.mean(centered**4, axis=axis)
    return m3 / m2**1.5, m4 / m2**2


def sharpe_standard_error(
    sr: ArrayLike, n_obs: int, skew: ArrayLike = 0.0, kurtosis: ArrayLike = 3.0
) -> NDArray[np.float64]:
    """Standard error of a per-period Sharpe ratio estimate for non-normal iid returns.

    Uses ``sqrt((1 - skew*SR + (kurtosis - 1)/4 * SR^2) / (n - 1))`` (Mertens 2002), the
    expression inside the PSR of Bailey and Lopez de Prado (2012).
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    sr = np.asarray(sr, dtype=float)
    radicand = 1.0 - np.asarray(skew) * sr + (np.asarray(kurtosis) - 1.0) / 4.0 * sr**2
    if np.any(radicand <= 0):
        raise ValueError("moments imply a non-positive Sharpe ratio variance")
    return np.sqrt(radicand / (n_obs - 1))


def probabilistic_sharpe_ratio(
    sr: ArrayLike,
    sr_benchmark: ArrayLike,
    n_obs: int,
    skew: ArrayLike = 0.0,
    kurtosis: ArrayLike = 3.0,
) -> float | NDArray[np.float64]:
    """Probabilistic Sharpe Ratio, ``P(true SR > sr_benchmark)`` (Bailey and Lopez de Prado 2012).

    ``PSR = Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - skew*SR + (kurtosis - 1)/4 * SR^2))``.
    One minus the PSR is a one-sided p-value for ``H0: true SR <= sr_benchmark``.

    Args:
        sr: Observed per-period Sharpe ratio.
        sr_benchmark: Per-period benchmark Sharpe ratio ``SR*``.
        n_obs: Number of return observations.
        skew: Sample skewness of the returns.
        kurtosis: Sample (non-excess) kurtosis of the returns.
    """
    se = sharpe_standard_error(sr, n_obs, skew, kurtosis)
    psr = stats.norm.cdf((np.asarray(sr, dtype=float) - np.asarray(sr_benchmark)) / se)
    return float(psr) if np.ndim(psr) == 0 else psr


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Approximate expected maximum of ``n_trials`` Sharpe ratios whose true values are zero.

    Bailey and Lopez de Prado (2014), eq. for ``SR_0``::

        E[max SR] ~ sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))

    where ``g`` is the Euler-Mascheroni constant and ``V`` the variance of the trial
    Sharpe ratios. The approximation is undefined at ``N = 1``; a single trial has an
    expected value of zero under the null, which is returned.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if var_trials < 0:
        raise ValueError("var_trials must be non-negative")
    if n_trials == 1:
        return 0.0
    z = (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials) + EULER_GAMMA * stats.norm.ppf(
        1.0 - 1.0 / (n_trials * np.e)
    )
    return float(np.sqrt(var_trials) * z)


def expected_max_iid(n: int, dist: stats.rv_continuous = stats.norm) -> float:
    """Exact expected maximum of ``n`` iid draws from a continuous distribution.

    Computed by numerical integration of
    ``E[max] = int_0^inf (1 - F(x)^n) dx - int_-inf^0 F(x)^n dx``, which is accurate for
    large ``n`` because ``F(x)^n`` is evaluated in log space.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    def upper(x: float) -> float:
        return -np.expm1(n * dist.logcdf(x))

    def lower(x: float) -> float:
        return np.exp(n * dist.logcdf(x))

    hi = float(dist.isf(1e-12 / n))
    lo = float(dist.ppf(1e-12))
    pos, _ = integrate.quad(upper, 0.0, hi, limit=200)
    neg, _ = integrate.quad(lower, lo, 0.0, limit=200)
    return pos - neg


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    var_trials: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014).

    The PSR of the selected strategy against the benchmark ``SR_0 = E[max SR]`` implied by
    ``n_trials`` independent trials with Sharpe ratio variance ``var_trials``. All Sharpe
    ratios are per period.
    """
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return float(probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurtosis))


@dataclass(frozen=True)
class SelectionTest:
    """PSR and DSR of the in-sample best strategy in a set of trials.

    Attributes:
        best: Column index of the strategy with the highest sample Sharpe ratio.
        sr_best: Its per-period Sharpe ratio.
        sr0: Deflation benchmark ``E[max SR]`` (per period).
        n_trials: Number of trials used for deflation.
        psr: PSR of the best strategy against zero (no selection adjustment).
        dsr: Deflated Sharpe Ratio of the best strategy.
    """

    best: int
    sr_best: float
    sr0: float
    n_trials: float
    psr: float
    dsr: float

    @property
    def psr_pvalue(self) -> float:
        """One-sided p-value of the naive PSR test, ``1 - PSR``."""
        return 1.0 - self.psr

    @property
    def dsr_pvalue(self) -> float:
        """One-sided p-value of the DSR test, ``1 - DSR``."""
        return 1.0 - self.dsr


def select_and_deflate(returns: ArrayLike, n_trials: float | None = None) -> SelectionTest:
    """Pick the column with the highest Sharpe ratio and compute its PSR and DSR.

    Args:
        returns: ``T x N`` matrix of per-period strategy returns, one column per trial.
        n_trials: Number of trials to deflate by. Defaults to ``N``. A non-integer value
            (for example an effective number of independent trials) is rounded to the
            nearest integer of at least 1.

    The trial Sharpe ratio variance is the cross-sectional sample variance of the
    ``N`` per-period Sharpe ratios, as in Bailey and Lopez de Prado (2014).
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a T x N matrix")
    n_obs, n_cols = x.shape
    srs = np.asarray(sharpe_ratio(x))
    best = int(np.nanargmax(srs))
    skew, kurt = sample_moments(x[:, best])
    trials = n_cols if n_trials is None else max(1, int(round(n_trials)))
    var_trials = float(np.nanvar(srs, ddof=1)) if n_cols > 1 else 0.0
    sr0 = expected_max_sharpe(trials, var_trials)
    psr = float(probabilistic_sharpe_ratio(srs[best], 0.0, n_obs, skew, kurt))
    dsr = float(probabilistic_sharpe_ratio(srs[best], sr0, n_obs, skew, kurt))
    return SelectionTest(best, float(srs[best]), sr0, float(trials), psr, dsr)


def effective_number_of_trials(returns: ArrayLike) -> float:
    """Participation ratio of the correlation matrix eigenvalues, ``N^2 / sum(lambda^2)``.

    Equals ``N`` for uncorrelated trials and 1 for perfectly correlated ones. This is a
    heuristic for the number of independent trials behind a correlated parameter sweep,
    not part of the DSR papers, and is reported only as a sensitivity check.
    """
    x = np.asarray(returns, dtype=float)
    corr = np.corrcoef(x, rowvar=False)
    n = corr.shape[0]
    return float(n**2 / np.sum(corr**2))
