"""Stationary bootstrap properties and Reality Check / SPA behavior."""

import numpy as np
import pytest
from scipy import stats

from btaudit import bootstrap as bs


def _block_lengths(indices: np.ndarray) -> np.ndarray:
    """Lengths of complete blocks (a block ends where the next index is not +1 mod T)."""
    n_obs = indices.shape[1]
    lengths = []
    for row in indices:
        breaks = np.flatnonzero(np.diff(row) % n_obs != 1)
        lengths.append(np.diff(breaks))  # blocks strictly between the first and last break
    return np.concatenate(lengths)


def test_indices_shape_and_range() -> None:
    idx = bs.stationary_bootstrap_indices(50, 30, 5.0, np.random.default_rng(0))
    assert idx.shape == (30, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_block_lengths_are_geometric_with_requested_mean() -> None:
    mean_block = 8.0
    idx = bs.stationary_bootstrap_indices(5_000, 200, mean_block, np.random.default_rng(1))
    lengths = _block_lengths(idx)
    q = 1.0 / mean_block
    # A new block can restart exactly at the next index with probability q / T: negligible.
    assert lengths.mean() == pytest.approx(mean_block, rel=0.02)
    assert lengths.var() == pytest.approx((1 - q) / q**2, rel=0.05)
    observed = np.array([(lengths == k).sum() for k in range(1, 16)])
    expected = lengths.size * q * (1 - q) ** np.arange(15)
    assert np.max(np.abs(observed / expected - 1)) < 0.05


def test_resampled_indices_are_uniform_at_every_position() -> None:
    n_obs = 40
    idx = bs.stationary_bootstrap_indices(n_obs, 40_000, 6.0, np.random.default_rng(2))
    for position in (0, n_obs // 2, n_obs - 1):
        counts = np.bincount(idx[:, position], minlength=n_obs)
        assert stats.chisquare(counts).pvalue > 0.001
    # Stationarity: the joint law of consecutive pairs does not depend on the position.
    continue_early = np.mean((idx[:, 1] - idx[:, 0]) % n_obs == 1)
    continue_late = np.mean((idx[:, -1] - idx[:, -2]) % n_obs == 1)
    expected = (1 - 1 / 6.0) + (1 / 6.0) / n_obs
    assert continue_early == pytest.approx(expected, abs=0.01)
    assert continue_late == pytest.approx(expected, abs=0.01)


def test_mean_block_one_is_the_iid_bootstrap() -> None:
    idx = bs.stationary_bootstrap_indices(100, 20_000, 1.0, np.random.default_rng(3))
    continues = np.mean((idx[:, 1:] - idx[:, :-1]) % 100 == 1)
    assert continues == pytest.approx(0.01, abs=0.002)


def test_counts_reproduce_resampled_means() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((60, 3))
    idx = bs.stationary_bootstrap_indices(60, 25, 4.0, rng)
    counts = bs.resample_counts(idx, 60)
    assert np.all(counts.sum(axis=1) == 60)
    assert np.allclose(counts @ x / 60, x[idx].mean(axis=1))


def test_kernel_variance_matches_bootstrap_variance() -> None:
    rng = np.random.default_rng(5)
    n_obs, mean_block = 200, 10.0
    e = rng.standard_normal(n_obs + 1)
    x = e[1:] + 0.6 * e[:-1]  # MA(1) data, so the kernel weights matter
    idx = bs.stationary_bootstrap_indices(n_obs, 40_000, mean_block, rng)
    boot = np.sqrt(n_obs) * (x[idx].mean(axis=1) - x.mean())
    kernel = bs.stationary_bootstrap_variance(x, mean_block)[0]
    assert boot.var() == pytest.approx(kernel, rel=0.03)


def test_kernel_variance_with_unit_blocks_is_sample_variance() -> None:
    x = np.random.default_rng(6).standard_normal(500)
    assert bs.stationary_bootstrap_variance(x, 1.0)[0] == pytest.approx(x.var())


def test_reality_check_detects_strong_alpha_and_accepts_losers() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((500, 20))
    strong = x.copy()
    strong[:, 0] += 0.4
    weak = x - 0.3
    assert bs.reality_check(bs.bootstrap_means(strong, 500, 5.0, rng)).p_value < 0.01
    assert bs.reality_check(bs.bootstrap_means(weak, 500, 5.0, rng)).p_value > 0.5


def test_spa_versions_are_ordered() -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal((400, 30)) + np.linspace(-0.4, 0.08, 30)
    bm = bs.bootstrap_means(x, 1_000, 5.0, rng)
    result = bs.spa_test(bm)
    assert result.p_lower <= result.p_consistent <= result.p_upper
    assert result.best == int(np.argmax(bm.sample / bm.omega))


def test_spa_consistent_ignores_very_poor_strategies() -> None:
    # Adding hopeless strategies inflates the RC p-value but leaves SPA_c nearly unchanged.
    rng = np.random.default_rng(9)
    good = rng.standard_normal((600, 1)) + 0.09
    poor = rng.standard_normal((600, 200)) - 1.0
    with_poor = np.hstack([good, poor])
    seed = 123
    alone = bs.spa_test(bs.bootstrap_means(good, 2_000, 5.0, np.random.default_rng(seed)))
    mixed = bs.spa_test(bs.bootstrap_means(with_poor, 2_000, 5.0, np.random.default_rng(seed)))
    assert mixed.p_consistent == pytest.approx(alone.p_consistent, abs=0.01)


def test_spa_p_value_is_one_when_nothing_beats_the_benchmark() -> None:
    # All strategies far below zero: T = 0 and most bootstrap statistics are exactly 0 too.
    rng = np.random.default_rng(13)
    x = rng.standard_normal((500, 3)) - 0.5
    result = bs.spa_test(bs.bootstrap_means(x, 500, 5.0, rng))
    assert result.statistic == 0.0
    assert result.p_consistent == 1.0 and result.p_lower == 1.0 and result.p_upper == 1.0


def test_spa_with_one_strategy_matches_reality_check_when_mean_is_positive() -> None:
    # With K = 1 studentization is a constant rescaling, so both tests agree when dbar > 0.
    rng = np.random.default_rng(14)
    x = rng.standard_normal((300, 1)) + 0.05
    bm = bs.bootstrap_means(x, 1_000, 4.0, rng)
    assert bm.sample[0] > 0
    assert bs.spa_test(bm).p_consistent == pytest.approx(bs.reality_check(bm).p_value)


def test_spa_size_under_null_is_controlled() -> None:
    rng = np.random.default_rng(10)
    rejections = [
        bs.spa_test(bs.bootstrap_means(rng.standard_normal((250, 5)), 300, 3.0, rng)).p_consistent
        < 0.05
        for _ in range(300)
    ]
    ci_low = stats.binomtest(int(np.sum(rejections)), 300).proportion_ci().low
    assert ci_low <= 0.05  # the 95% CI includes or lies below the nominal level


def test_tests_are_invariant_to_a_common_shift_of_strategies_and_benchmark() -> None:
    rng = np.random.default_rng(11)
    strategies = rng.standard_normal((300, 8)) * 0.02
    benchmark = rng.standard_normal(300) * 0.02
    shift = rng.standard_normal(300) * 0.05
    diffs = strategies - benchmark[:, None]
    shifted = (strategies + shift[:, None]) - (benchmark + shift)[:, None]
    a = bs.spa_test(bs.bootstrap_means(diffs, 400, 4.0, np.random.default_rng(1)))
    b = bs.spa_test(bs.bootstrap_means(shifted, 400, 4.0, np.random.default_rng(1)))
    assert a.p_consistent == pytest.approx(b.p_consistent)


def test_seeding_is_deterministic() -> None:
    x = np.random.default_rng(12).standard_normal((200, 10))
    a = bs.spa_test(bs.bootstrap_means(x, 300, 5.0, np.random.default_rng(42)))
    b = bs.spa_test(bs.bootstrap_means(x, 300, 5.0, np.random.default_rng(42)))
    assert a == b
    c = bs.bootstrap_means(x, 300, 5.0, np.random.default_rng(43))
    assert not np.allclose(c.boot, bs.bootstrap_means(x, 300, 5.0, np.random.default_rng(42)).boot)


def test_spa_rejects_constant_differentials() -> None:
    with pytest.raises(ValueError):
        bs.spa_test(bs.bootstrap_means(np.zeros((50, 3)), 10, 2.0, np.random.default_rng(0)))
