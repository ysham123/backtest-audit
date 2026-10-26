"""Apply every selection-aware and naive test to one simulated universe (used by E2 and E3)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from btaudit.bootstrap import bootstrap_means, reality_check, spa_test
from btaudit.multiple import benjamini_hochberg, bonferroni, holm, one_sided_t_pvalues
from btaudit.sharpe import select_and_deflate

METHODS = (
    "naive_t",
    "naive_psr",
    "dsr",
    "reality_check",
    "spa",
    "spa_iid",
    "bonferroni",
    "holm",
    "bh",
)
LABELS = {
    "naive_t": "Naive t-test on best",
    "naive_psr": "Naive PSR on best",
    "dsr": "Deflated Sharpe",
    "reality_check": "Reality Check",
    "spa": "SPA (consistent)",
    "spa_iid": "SPA, iid scale (diagnostic)",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "bh": "Benjamini-Hochberg",
}


@dataclass(frozen=True)
class TestSettings:
    """Settings shared by all tests in a simulation study."""

    alpha: float = 0.05
    n_boot: int = 500
    mean_block: float = 10.0


def apply_methods(
    x: np.ndarray, rng: np.random.Generator, settings: TestSettings, planted: int | None = None
) -> dict[str, tuple[bool, bool]]:
    """Run all tests on a ``T x N`` return matrix.

    Returns:
        For each method, ``(rejects the global null, rejects and identifies the planted
        strategy)``. Selection-based tests identify their own argmax; FWER and FDR
        procedures identify the planted strategy if they reject its hypothesis.

    ``spa_iid`` is a diagnostic: the same bootstrap draws, but studentized by the iid
    sample standard deviation instead of the stationary bootstrap long-run deviation.
    """
    alpha = settings.alpha
    p_values = one_sided_t_pvalues(x)
    selection = select_and_deflate(x)
    boot = bootstrap_means(x, settings.n_boot, settings.mean_block, rng)
    rc, spa = reality_check(boot), spa_test(boot)
    spa_iid = spa_test(replace(boot, omega=x.std(axis=0)))

    def hit(index: int) -> bool:
        return planted is not None and index == planted

    out = {
        "naive_t": (p_values[selection.best] < alpha, hit(selection.best)),
        "naive_psr": (selection.psr_pvalue < alpha, hit(selection.best)),
        "dsr": (selection.dsr_pvalue < alpha, hit(selection.best)),
        "reality_check": (rc.p_value < alpha, hit(rc.best)),
        "spa": (spa.p_consistent < alpha, hit(spa.best)),
        "spa_iid": (spa_iid.p_consistent < alpha, hit(spa_iid.best)),
    }
    for name, procedure in (("bonferroni", bonferroni), ("holm", holm), ("bh", benjamini_hochberg)):
        result = procedure(p_values, alpha)
        out[name] = (result.n_rejected > 0, planted is not None and bool(result.reject[planted]))
    return {name: (bool(rej), bool(rej and ident)) for name, (rej, ident) in out.items()}
