"""Synthetic markets have the moments and true Sharpe ratios they claim."""

import numpy as np
import pytest

from btaudit.sharpe import sample_moments, sharpe_ratio
from btaudit.sims import (
    NullSpec,
    averaged_random_walk_returns,
    null_returns,
    per_period_sharpe,
    planted_alpha_returns,
)


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float(x[lag:] @ x[:-lag] / (x @ x))


@pytest.mark.parametrize(
    "spec",
    [NullSpec(), NullSpec(dist="t", df=5.0), NullSpec(garch=(0.05, 0.93), dist="t")],
    ids=["normal", "t5", "garch_t5"],
)
def test_null_returns_have_zero_mean_and_target_volatility(spec: NullSpec) -> None:
    x = null_returns(20_000, 20, np.random.default_rng(0), spec)
    assert x.shape == (20_000, 20)
    # Mean zero: pooled t-statistic within sampling error.
    pooled_t = x.mean() / (x.std() / np.sqrt(x.size))
    assert abs(pooled_t) < 4
    assert x.std() == pytest.approx(spec.sigma, rel=0.05)


def test_t_innovations_are_fat_tailed() -> None:
    x = null_returns(200_000, 1, np.random.default_rng(1), NullSpec(dist="t", df=8.0))
    _, kurt = sample_moments(x[:, 0])
    assert kurt == pytest.approx(3 + 6 / (8 - 4), rel=0.1)


def test_garch_has_volatility_clustering_but_no_return_autocorrelation() -> None:
    x = null_returns(50_000, 1, np.random.default_rng(2), NullSpec(garch=(0.08, 0.90)))[:, 0]
    assert abs(_autocorr(x, 1)) < 0.02
    assert _autocorr(x**2, 1) > 0.1


def test_common_factor_sets_pairwise_correlation() -> None:
    x = null_returns(50_000, 4, np.random.default_rng(3), NullSpec(rho=0.6))
    corr = np.corrcoef(x, rowvar=False)
    off_diagonal = corr[~np.eye(4, dtype=bool)]
    assert np.allclose(off_diagonal, 0.6, atol=0.02)


def test_planted_alpha_has_known_sharpe_and_others_zero() -> None:
    x = planted_alpha_returns(
        400_000, 3, 2.0, 252, np.random.default_rng(4), NullSpec(sigma=0.02), planted=1
    )
    srs = sharpe_ratio(x)
    assert srs[1] == pytest.approx(per_period_sharpe(2.0, 252), abs=0.005)
    assert np.all(np.abs(srs[[0, 2]]) < 0.006)


def test_seeding_is_deterministic() -> None:
    spec = NullSpec(dist="t", garch=(0.05, 0.9), rho=0.3)
    a = null_returns(100, 5, np.random.default_rng(7), spec)
    b = null_returns(100, 5, np.random.default_rng(7), spec)
    c = null_returns(100, 5, np.random.default_rng(8), spec)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize(
    "kwargs", [{"dist": "cauchy"}, {"dist": "t", "df": 2.0}, {"rho": 1.0}, {"garch": (0.5, 0.6)}]
)
def test_invalid_specs_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        NullSpec(**kwargs)


def test_averaging_a_random_walk_creates_working_autocorrelation() -> None:
    m = 21
    r = averaged_random_walk_returns(40_000, m, np.random.default_rng(9), NullSpec(sigma=0.005))
    assert _autocorr(r, 1) == pytest.approx((m**2 - 1) / (2 * (2 * m**2 + 1)), abs=0.015)
    assert abs(_autocorr(r, 2)) < 0.015
    assert abs(r.mean()) < 4 * r.std() / np.sqrt(r.size)
