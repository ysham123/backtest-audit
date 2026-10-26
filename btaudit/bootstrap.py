"""Stationary bootstrap and data-snooping tests for the best of many strategies.

References
----------
Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. Journal of the
    American Statistical Association, 89(428), 1303-1313.
White, H. (2000). A reality check for data snooping. Econometrica, 68(5), 1097-1126.
Hansen, P. R. (2005). A test for superior predictive ability. Journal of Business and
    Economic Statistics, 23(4), 365-380.

Both tests take a ``T x K`` matrix of performance differentials ``d[t, k]`` (strategy
``k`` return minus benchmark return; use the raw returns for a zero benchmark) and test
``H0: max_k E[d_k] <= 0``, i.e. no strategy beats the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def stationary_bootstrap_indices(
    n_obs: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Resampling indices of the circular stationary bootstrap (Politis and Romano 1994).

    Each resample starts at a uniform random index. At every later step a new block
    starts at a uniform random index with probability ``q = 1 / mean_block``; otherwise
    the next index follows the previous one (wrapping around the end). Block lengths are
    therefore geometric with mean ``mean_block``.

    Returns:
        Integer array of shape ``(n_boot, n_obs)``.
    """
    if n_obs < 1 or n_boot < 1:
        raise ValueError("n_obs and n_boot must be positive")
    if mean_block < 1:
        raise ValueError("mean_block must be at least 1")
    starts = rng.integers(0, n_obs, size=(n_boot, n_obs))
    new_block = rng.random((n_boot, n_obs)) < 1.0 / mean_block
    new_block[:, 0] = True
    t = np.arange(n_obs)
    block_start = np.maximum.accumulate(np.where(new_block, t, 0), axis=1)
    origin = np.take_along_axis(starts, block_start, axis=1)
    return (origin + (t - block_start)) % n_obs


def resample_counts(indices: NDArray[np.intp], n_obs: int) -> NDArray[np.float64]:
    """Matrix ``C[b, t]``: how often observation ``t`` appears in resample ``b``.

    Resampled means of any data matrix ``X`` are then ``C @ X / n_obs``, which turns the
    bootstrap into one matrix product.
    """
    n_boot = indices.shape[0]
    flat = (indices + n_obs * np.arange(n_boot)[:, None]).ravel()
    return np.bincount(flat, minlength=n_boot * n_obs).reshape(n_boot, n_obs).astype(float)


def stationary_bootstrap_variance(x: ArrayLike, mean_block: float) -> NDArray[np.float64]:
    """Exact bootstrap variance of ``sqrt(T) * mean`` under the stationary bootstrap.

    Politis and Romano (1994, Lemma 1), also used by Hansen (2005)::

        omega^2 = g(0) + 2 * sum_{i=1}^{T-1} k(i) g(i),
        k(i) = (1 - i/T) (1 - q)^i + (i/T) (1 - q)^(T - i),

    with ``g(i)`` the sample autocovariances (divisor ``T``) and ``q = 1 / mean_block``.

    Args:
        x: ``T`` or ``T x K`` data.
        mean_block: Mean block length of the bootstrap.

    Returns:
        One variance per column.
    """
    data = np.asarray(x, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    n_obs = data.shape[0]
    # Autocovariances of all lags via the zero-padded FFT, one row per column (contiguous).
    centered = np.ascontiguousarray((data - data.mean(axis=0)).T)
    spectrum = np.fft.rfft(centered, n=2 * n_obs, axis=-1)
    power = spectrum.real**2 + spectrum.imag**2
    acov = np.fft.irfft(power, n=2 * n_obs, axis=-1)[:, :n_obs].T / n_obs
    q = 1.0 / mean_block
    lag = np.arange(1, n_obs)
    kernel = (1.0 - lag / n_obs) * (1.0 - q) ** lag + (lag / n_obs) * (1.0 - q) ** (n_obs - lag)
    return acov[0] + 2.0 * kernel @ acov[1:]


@dataclass(frozen=True)
class BootstrapMeans:
    """Sample means, stationary bootstrap means and HAC scales of performance differentials.

    Attributes:
        sample: Sample mean of each column, shape ``(K,)``.
        boot: Resampled means, shape ``(B, K)``.
        omega: Stationary bootstrap standard deviation of ``sqrt(T) * mean``, shape ``(K,)``.
        n_obs: Number of observations ``T``.
    """

    sample: NDArray[np.float64]
    boot: NDArray[np.float64]
    omega: NDArray[np.float64]
    n_obs: int


def bootstrap_means(
    diffs: ArrayLike, n_boot: int, mean_block: float, rng: np.random.Generator
) -> BootstrapMeans:
    """Resample the rows of ``diffs`` jointly with the stationary bootstrap.

    Rows (time points) are resampled together, which preserves the cross-sectional
    dependence between strategies that both tests rely on.
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n_obs = d.shape[0]
    counts = resample_counts(stationary_bootstrap_indices(n_obs, n_boot, mean_block, rng), n_obs)
    omega = np.sqrt(stationary_bootstrap_variance(d, mean_block))
    return BootstrapMeans(d.mean(axis=0), counts @ d / n_obs, omega, n_obs)


@dataclass(frozen=True)
class RealityCheckResult:
    """White's Reality Check: statistic ``max_k sqrt(T) mean_k``, p-value and best column."""

    statistic: float
    p_value: float
    best: int


def reality_check(bm: BootstrapMeans) -> RealityCheckResult:
    """White (2000) Reality Check p-value from bootstrap means.

    ``V = max_k sqrt(T) dbar_k`` and ``V*_b = max_k sqrt(T) (dbar*_bk - dbar_k)``; the
    p-value is the fraction of ``V*_b`` at least as large as ``V``.
    """
    root_n = np.sqrt(bm.n_obs)
    stat = float(root_n * bm.sample.max())
    boot_stat = root_n * (bm.boot - bm.sample).max(axis=1)
    return RealityCheckResult(stat, float(np.mean(boot_stat >= stat)), int(np.argmax(bm.sample)))


@dataclass(frozen=True)
class SPAResult:
    """Hansen's SPA test.

    Attributes:
        statistic: ``max(0, max_k sqrt(T) dbar_k / omega_k)``.
        p_consistent: p-value of the consistent test ``SPA_c`` (the recommended version).
        p_lower: p-value of the liberal ``SPA_l`` version.
        p_upper: p-value of the conservative ``SPA_u`` version (a studentized RC).
        best: Column with the largest studentized mean.
    """

    statistic: float
    p_consistent: float
    p_lower: float
    p_upper: float
    best: int


def spa_test(bm: BootstrapMeans) -> SPAResult:
    """Hansen (2005) test of superior predictive ability.

    The bootstrap statistic for each recentering ``g`` is
    ``T*_b = max(0, max_k sqrt(T) (dbar*_bk - g(dbar_k)) / omega_k)`` with

    * ``g_l(x) = max(x, 0)`` (lower),
    * ``g_c(x) = x * 1{x >= -omega_k * sqrt(2 log log T / T)}`` (consistent),
    * ``g_u(x) = x`` (upper).

    The p-value is the fraction of ``T*_b`` at least as large as ``T``. Both statistics
    have an atom at zero, so the weak inequality matters: when no strategy beats the
    benchmark in sample (``T = 0``) the p-value is 1, never a spurious rejection.

    Columns with ``omega_k = 0`` (constant differentials) carry no information and are
    excluded.
    """
    if bm.n_obs < 3:
        raise ValueError("the SPA threshold needs at least three observations")
    keep = bm.omega > 0
    if not np.any(keep):
        raise ValueError("all differential series are constant")
    sample, boot, omega = bm.sample[keep], bm.boot[:, keep], bm.omega[keep]
    root_n = np.sqrt(bm.n_obs)
    t_ratio = root_n * sample / omega
    stat = float(max(t_ratio.max(), 0.0))
    threshold = -omega * np.sqrt(2.0 * np.log(np.log(bm.n_obs)) / bm.n_obs)
    recentering = {
        "lower": np.maximum(sample, 0.0),
        "consistent": np.where(sample >= threshold, sample, 0.0),
        "upper": sample,
    }
    p_values = {}
    for name, g in recentering.items():
        boot_stat = np.maximum((root_n * (boot - g) / omega).max(axis=1), 0.0)
        p_values[name] = float(np.mean(boot_stat >= stat))
    best = int(np.flatnonzero(keep)[np.argmax(t_ratio)])
    return SPAResult(stat, p_values["consistent"], p_values["lower"], p_values["upper"], best)
