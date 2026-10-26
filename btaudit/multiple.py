"""Multiple-testing corrections for per-strategy p-values.

References
----------
Dunn, O. J. (1961). Multiple comparisons among means. Journal of the American Statistical
    Association, 56(293), 52-64. (Bonferroni correction.)
Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian
    Journal of Statistics, 6(2), 65-70.
Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical Society B,
    57(1), 289-300.

Bonferroni and Holm control the family-wise error rate under any dependence. BH controls
the false discovery rate under independence or positive regression dependence; under the
global null its FDR equals its family-wise error rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class MultipleTestResult:
    """Adjusted p-values and rejection decisions, in the original order of the input."""

    adjusted: NDArray[np.float64]
    reject: NDArray[np.bool_]

    @property
    def n_rejected(self) -> int:
        """Number of rejected hypotheses."""
        return int(self.reject.sum())


def _validate(p_values: ArrayLike, alpha: float) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    return p


def bonferroni(p_values: ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    """Bonferroni correction: adjusted ``p = min(1, m p)``."""
    p = _validate(p_values, alpha)
    adjusted = np.minimum(1.0, p * p.size)
    return MultipleTestResult(adjusted, adjusted <= alpha)


def holm(p_values: ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    """Holm step-down procedure.

    With sorted p-values ``p_(1) <= ... <= p_(m)``, the adjusted value of ``p_(i)`` is
    ``max_{j <= i} min(1, (m - j + 1) p_(j))``.
    """
    p = _validate(p_values, alpha)
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, stepped)
    return MultipleTestResult(adjusted, adjusted <= alpha)


def benjamini_hochberg(p_values: ArrayLike, alpha: float = 0.05) -> MultipleTestResult:
    """Benjamini-Hochberg step-up procedure at false discovery rate ``alpha``.

    The adjusted value of ``p_(i)`` is ``min_{j >= i} min(1, m p_(j) / j)``.
    """
    p = _validate(p_values, alpha)
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    stepped = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, stepped)
    return MultipleTestResult(adjusted, adjusted <= alpha)


def one_sided_t_pvalues(returns: ArrayLike) -> NDArray[np.float64]:
    """One-sided p-values of ``H0: mean <= 0`` per column from a Student t-test.

    Exact under iid normal returns, asymptotically valid for martingale differences.
    """
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least two observations")
    t_stat = x.mean(axis=0) / (x.std(axis=0, ddof=1) / np.sqrt(n))
    return stats.t.sf(t_stat, df=n - 1)
