"""PBO via CSCV on constructed universes with known answers."""

from math import comb

import numpy as np
import pytest

from btaudit.pbo import cscv_pbo, relative_rank_logit, split_membership
from btaudit.sharpe import sharpe_ratio


def test_split_membership_enumerates_all_half_splits() -> None:
    m = split_membership(8)
    assert m.shape == (comb(8, 4), 8)
    assert np.all(m.sum(axis=1) == 4)
    rows = {tuple(r) for r in m.astype(int)}
    assert len(rows) == m.shape[0]
    assert all(tuple(1 - np.array(r)) in rows for r in rows)  # complements are present


@pytest.mark.parametrize("bad", [0, 3, 7])
def test_split_membership_rejects_odd_or_small(bad: int) -> None:
    with pytest.raises(ValueError):
        split_membership(bad)


def test_relative_rank_logit_values() -> None:
    logits = relative_rank_logit(np.array([1, 5, 9]), 9)
    assert logits[0] == pytest.approx(np.log(0.1 / 0.9))
    assert logits[1] == pytest.approx(0.0)
    assert logits[2] == pytest.approx(np.log(0.9 / 0.1))


def test_mirror_pair_gives_pbo_one() -> None:
    # B = -A and A has zero full-sample mean: whichever wins IS loses OOS.
    rng = np.random.default_rng(0)
    a = rng.standard_normal(320)
    a -= a.mean()
    result = cscv_pbo(np.column_stack([a, -a]), n_blocks=8)
    assert result.pbo == 1.0
    assert result.prob_oos_loss == 1.0
    assert np.all(result.logits == pytest.approx(np.log(0.5)))


def test_zero_sum_universe_gives_pbo_one_under_mean_metric() -> None:
    # Every strategy has the same full-sample total, so OOS mean = -IS mean for all of them
    # and the IS-best is always the OOS-worst.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 12))
    x -= x.mean(axis=0)
    result = cscv_pbo(x, n_blocks=10, metric="mean")
    assert result.pbo == 1.0
    assert np.all(result.logits == pytest.approx(np.log(1 / 12)))
    assert result.degradation_slope == pytest.approx(-1.0)


def test_dominant_strategy_gives_pbo_zero() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((480, 20))
    x[:, 3] = 1.0 + 0.01 * rng.standard_normal(480)
    result = cscv_pbo(x, n_blocks=16)
    assert result.pbo == 0.0
    assert result.prob_oos_loss == 0.0
    assert np.all(result.is_best == 3)
    assert np.all(result.logits == pytest.approx(np.log(20)))
    assert result.n_combinations == comb(16, 8)


def test_pbo_invariant_to_column_order_and_positive_scaling() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((300, 15)) + 0.05 * rng.standard_normal(15)
    base = cscv_pbo(x, n_blocks=10)
    perm = rng.permutation(15)
    assert cscv_pbo(x[:, perm], n_blocks=10).pbo == pytest.approx(base.pbo)
    assert cscv_pbo(3.0 * x, n_blocks=10).pbo == pytest.approx(base.pbo)


def test_block_sharpe_matches_direct_computation() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((203, 6)) * 0.02 + 0.001  # 203 rows: 3 earliest are dropped
    result = cscv_pbo(x, n_blocks=4)
    trimmed = x[3:].reshape(4, 50, 6)
    first_split = split_membership(4)[0].astype(bool)  # IS blocks (0, 1)
    is_rows = trimmed[first_split].reshape(-1, 6)
    oos_rows = trimmed[~first_split].reshape(-1, 6)
    is_sr = sharpe_ratio(is_rows)
    best = int(np.argmax(is_sr))
    assert result.is_best[0] == best
    assert result.is_performance[0] == pytest.approx(is_sr[best], rel=1e-10)
    assert result.oos_performance[0] == pytest.approx(sharpe_ratio(oos_rows)[best], rel=1e-10)


def test_degradation_regression_matches_polyfit() -> None:
    rng = np.random.default_rng(5)
    result = cscv_pbo(rng.standard_normal((320, 30)), n_blocks=8)
    slope, intercept = np.polyfit(result.is_performance, result.oos_performance, 1)
    assert result.degradation_slope == pytest.approx(slope)
    assert result.degradation_intercept == pytest.approx(intercept)


def test_noise_universes_have_pbo_near_one_half() -> None:
    rng = np.random.default_rng(6)
    pbos = [cscv_pbo(rng.standard_normal((400, 50)), n_blocks=10).pbo for _ in range(60)]
    assert np.mean(pbos) == pytest.approx(0.5, abs=0.1)


def test_cscv_input_validation() -> None:
    with pytest.raises(ValueError):
        cscv_pbo(np.ones((100, 1)))
    with pytest.raises(ValueError):
        cscv_pbo(np.ones((20, 4)), n_blocks=16)
    with pytest.raises(ValueError):
        cscv_pbo(np.ones((100, 4)), metric="sortino")
    bad = np.ones((100, 4))
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        cscv_pbo(bad)


def test_flat_strategy_ranks_as_zero_sharpe() -> None:
    # A strategy that never trades has Sharpe 0: above every loser, below every winner.
    rng = np.random.default_rng(7)
    x = np.column_stack(
        [np.zeros(320), rng.standard_normal(320) - 0.5, rng.standard_normal(320) + 0.5]
    )
    result = cscv_pbo(x, n_blocks=8)
    assert np.all(result.is_best == 2)
    assert np.isfinite(result.degradation_slope)
    assert result.pbo == 0.0
