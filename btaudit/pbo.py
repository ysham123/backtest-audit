"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Reference
---------
Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability
    of backtest overfitting. Journal of Computational Finance, 20(4), 39-69.
    (Working paper version 2014.)

Procedure (CSCV)
----------------
1. Split the ``T x N`` matrix of strategy returns into ``S`` contiguous blocks of equal
   length (the earliest ``T mod S`` rows are dropped).
2. For every one of the ``C(S, S/2)`` ways to choose ``S/2`` blocks as the in-sample (IS)
   set, the remaining blocks form the out-of-sample (OOS) set.
3. Select the strategy with the best IS performance and find the relative rank
   ``w = rank / (N + 1)`` of its OOS performance among all ``N`` OOS performances
   (rank 1 is the worst, ties receive average ranks).
4. The logit is ``lambda = log(w / (1 - w))``. PBO is the fraction of combinations with
   ``lambda <= 0``, i.e. where the IS-best strategy ranks at or below the OOS median.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_METRICS = ("sharpe", "mean")


@dataclass(frozen=True)
class PBOResult:
    """Output of :func:`cscv_pbo`.

    Attributes:
        pbo: Probability of backtest overfitting.
        logits: Logit of the OOS relative rank of the IS-best strategy, per combination.
        is_best: Index of the IS-best strategy, per combination.
        is_performance: IS performance of the IS-best strategy, per combination.
        oos_performance: OOS performance of the same strategy, per combination.
        degradation_slope: OLS slope of OOS on IS performance of the selected strategy.
        degradation_intercept: OLS intercept of that regression.
        degradation_r2: R squared of that regression.
        prob_oos_loss: Fraction of combinations where the selected strategy has OOS
            performance below zero.
        n_blocks: Number of blocks ``S``.
        metric: Performance metric used for ranking.
    """

    pbo: float
    logits: NDArray[np.float64]
    is_best: NDArray[np.intp]
    is_performance: NDArray[np.float64]
    oos_performance: NDArray[np.float64]
    degradation_slope: float
    degradation_intercept: float
    degradation_r2: float
    prob_oos_loss: float
    n_blocks: int
    metric: str

    @property
    def n_combinations(self) -> int:
        """Number of IS/OOS splits evaluated, ``C(S, S/2)``."""
        return int(self.logits.size)


def split_membership(n_blocks: int) -> NDArray[np.float64]:
    """Rows are all ``C(S, S/2)`` IS block selections as 0/1 indicator vectors."""
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer of at least 2")
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    membership = np.zeros((len(combos), n_blocks))
    for row, combo in enumerate(combos):
        membership[row, list(combo)] = 1.0
    return membership


def relative_rank_logit(rank: ArrayLike, n_strategies: int) -> NDArray[np.float64]:
    """Logit of the relative rank ``w = rank / (N + 1)`` with ``rank`` in ``[1, N]``."""
    w = np.asarray(rank, dtype=float) / (n_strategies + 1.0)
    return np.log(w / (1.0 - w))


def _block_statistics(x: NDArray, n_blocks: int) -> tuple[NDArray, NDArray, NDArray, int]:
    """Column centers and per-block sums and sums of squares of centered returns."""
    n_obs = x.shape[0]
    if n_obs < 2 * n_blocks:
        raise ValueError("need at least two observations per block")
    block_len = n_obs // n_blocks
    trimmed = x[n_obs - block_len * n_blocks :]
    center = trimmed.mean(axis=0)
    blocks = (trimmed - center).reshape(n_blocks, block_len, x.shape[1])
    return center, blocks.sum(axis=1), (blocks**2).sum(axis=1), block_len


def _performance(
    sums: NDArray, sumsq: NDArray, n: float, center: NDArray, metric: str
) -> NDArray[np.float64]:
    """Mean or Sharpe ratio from sums of centered returns over ``n`` observations."""
    mean_c = sums / n
    mean = mean_c + center
    if metric == "mean":
        return mean
    var = np.maximum(sumsq - n * mean_c**2, 0.0) / (n - 1.0)
    # A strategy with no variation in a subsample (for example a rule that never trades
    # there) gets Sharpe 0 if its mean is 0, and +/- infinity otherwise.
    degenerate = np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0))
    return np.divide(mean, np.sqrt(var), out=degenerate, where=var > 0)


def _ols(x: NDArray, y: NDArray) -> tuple[float, float, float]:
    """Slope, intercept and R squared of a least-squares regression of y on x (finite pairs)."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    x_c = x - x.mean()
    y_c = y - y.mean()
    sxx = float(x_c @ x_c)
    if sxx == 0.0:
        return float("nan"), float(y.mean()), float("nan")
    slope = float(x_c @ y_c) / sxx
    intercept = float(y.mean() - slope * x.mean())
    syy = float(y_c @ y_c)
    r2 = float((x_c @ y_c) ** 2 / (sxx * syy)) if syy > 0 else float("nan")
    return slope, intercept, r2


def cscv_pbo(
    returns: ArrayLike, n_blocks: int = 16, metric: str = "sharpe", chunk_size: int = 2048
) -> PBOResult:
    """Estimate PBO with Combinatorially Symmetric Cross-Validation.

    Args:
        returns: ``T x N`` matrix of per-period strategy returns (``N >= 2``).
        n_blocks: Even number of contiguous blocks ``S``.
        metric: ``"sharpe"`` (per-period Sharpe ratio, the default) or ``"mean"``.
        chunk_size: Number of combinations processed at once (memory control only).

    Returns:
        A :class:`PBOResult`. Performance values are per period.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("returns must be a T x N matrix with N >= 2")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns must be finite")
    n_strat = x.shape[1]
    center, bsum, bsq, block_len = _block_statistics(x, n_blocks)
    membership = split_membership(n_blocks)
    total_sum, total_sq = bsum.sum(axis=0), bsq.sum(axis=0)
    half_n = block_len * n_blocks / 2.0

    best_parts, is_parts, oos_parts, rank_parts = [], [], [], []
    for start in range(0, membership.shape[0], chunk_size):
        m = membership[start : start + chunk_size]
        is_sum, is_sq = m @ bsum, m @ bsq
        perf_is = _performance(is_sum, is_sq, half_n, center, metric)
        perf_oos = _performance(total_sum - is_sum, total_sq - is_sq, half_n, center, metric)
        best = np.argmax(perf_is, axis=1)
        rows = np.arange(m.shape[0])
        oos_best = perf_oos[rows, best][:, None]
        rank = (perf_oos < oos_best).sum(axis=1) + ((perf_oos == oos_best).sum(axis=1) + 1) / 2.0
        best_parts.append(best)
        is_parts.append(perf_is[rows, best])
        oos_parts.append(oos_best[:, 0])
        rank_parts.append(rank)

    logits = relative_rank_logit(np.concatenate(rank_parts), n_strat)
    is_perf, oos_perf = np.concatenate(is_parts), np.concatenate(oos_parts)
    slope, intercept, r2 = _ols(is_perf, oos_perf)
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        is_best=np.concatenate(best_parts),
        is_performance=is_perf,
        oos_performance=oos_perf,
        degradation_slope=slope,
        degradation_intercept=intercept,
        degradation_r2=r2,
        prob_oos_loss=float(np.mean(oos_perf < 0.0)),
        n_blocks=n_blocks,
        metric=metric,
    )
