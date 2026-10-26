"""Formula checks for Sharpe ratio, PSR, expected maximum Sharpe and DSR."""

import numpy as np
import pytest
from scipy import stats

from btaudit import sharpe as sh


def test_sharpe_ratio_hand_computed() -> None:
    returns = np.array([0.01, 0.03, -0.01, 0.01])
    # mean 0.01, sample std sqrt(0.0008 / 3)
    expected = 0.01 / np.sqrt(0.0008 / 3)
    assert sh.sharpe_ratio(returns) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("periods", [12, 52, 252])
def test_sharpe_annualization_scales_by_sqrt_periods(periods: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=500)
    assert sh.sharpe_ratio(x, periods) == pytest.approx(np.sqrt(periods) * sh.sharpe_ratio(x))


def test_sharpe_matrix_matches_columns_and_flags_zero_variance() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 3))
    x[:, 2] = 0.5
    srs = sh.sharpe_ratio(x)
    assert srs[0] == pytest.approx(sh.sharpe_ratio(x[:, 0]))
    assert srs[1] == pytest.approx(sh.sharpe_ratio(x[:, 1]))
    assert np.isnan(srs[2])


def test_sample_moments_of_normal_and_exponential() -> None:
    rng = np.random.default_rng(2)
    skew, kurt = sh.sample_moments(rng.standard_normal(400_000))
    assert skew == pytest.approx(0.0, abs=0.02)
    assert kurt == pytest.approx(3.0, abs=0.05)
    skew, kurt = sh.sample_moments(rng.exponential(size=400_000))
    assert skew == pytest.approx(2.0, abs=0.05)
    assert kurt == pytest.approx(9.0, abs=0.4)


def test_psr_is_one_half_at_the_benchmark() -> None:
    assert sh.probabilistic_sharpe_ratio(0.1, 0.1, 120, -1.0, 6.0) == pytest.approx(0.5)


def test_psr_hand_computed_normal_case() -> None:
    sr, n = 0.2, 60
    z = sr * np.sqrt(n - 1) / np.sqrt(1 + 0.5 * sr**2)
    assert sh.probabilistic_sharpe_ratio(sr, 0.0, n) == pytest.approx(stats.norm.cdf(z))


def test_psr_penalizes_negative_skew_and_fat_tails() -> None:
    base = sh.probabilistic_sharpe_ratio(0.15, 0.0, 100, 0.0, 3.0)
    neg_skew = sh.probabilistic_sharpe_ratio(0.15, 0.0, 100, -2.0, 3.0)
    fat = sh.probabilistic_sharpe_ratio(0.15, 0.0, 100, 0.0, 12.0)
    assert neg_skew < base and fat < base


def test_sharpe_standard_error_matches_monte_carlo() -> None:
    rng = np.random.default_rng(3)
    n, true_sr = 250, 0.1
    draws = rng.normal(true_sr, 1.0, size=(20_000, n))
    srs = sh.sharpe_ratio(draws, axis=1)
    assert np.std(srs) == pytest.approx(float(sh.sharpe_standard_error(true_sr, n)), rel=0.03)


def test_deflated_sharpe_matches_published_example() -> None:
    # Bailey and Lopez de Prado (2014), numerical example: N = 100 trials, annualized
    # V[SR] = 1/2, T = 1250 daily returns, annualized SR = 2.5, skew = -3, kurtosis = 10.
    var_trials = 0.5 / 250
    sr0 = sh.expected_max_sharpe(100, var_trials)
    assert sr0 == pytest.approx(0.1132, abs=1e-4)
    dsr = sh.deflated_sharpe_ratio(2.5 / np.sqrt(250), 1250, 100, var_trials, -3.0, 10.0)
    assert dsr == pytest.approx(0.9004, abs=1e-4)


def test_expected_max_sharpe_single_trial_is_zero() -> None:
    assert sh.expected_max_sharpe(1, 0.3) == 0.0
    with pytest.raises(ValueError):
        sh.expected_max_sharpe(0, 0.3)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_expected_max_approximation_close_to_exact_normal(n: int) -> None:
    exact = sh.expected_max_iid(n)
    approx = sh.expected_max_sharpe(n, 1.0)
    assert approx == pytest.approx(exact, rel=0.03)
    assert approx > exact  # the approximation is slightly conservative in this range


@pytest.mark.parametrize("n", [2, 5, 20])
def test_expected_max_exact_matches_monte_carlo(n: int) -> None:
    rng = np.random.default_rng(n)
    mc = rng.standard_normal((200_000, n)).max(axis=1).mean()
    assert sh.expected_max_iid(n) == pytest.approx(mc, abs=0.01)


def test_expected_max_exact_known_values() -> None:
    assert sh.expected_max_iid(1) == pytest.approx(0.0, abs=1e-9)
    assert sh.expected_max_iid(2) == pytest.approx(1 / np.sqrt(np.pi), abs=1e-9)
    assert sh.expected_max_iid(3) == pytest.approx(1.5 / np.sqrt(np.pi), abs=1e-9)


def test_expected_max_of_small_sharpe_sets_against_monte_carlo() -> None:
    rng = np.random.default_rng(4)
    n_obs, n_trials = 120, 10
    srs = sh.sharpe_ratio(rng.standard_normal((5_000, n_obs, n_trials)), axis=1)
    mc_max = srs.max(axis=1).mean()
    predicted = sh.expected_max_sharpe(n_trials, 1.0 / n_obs)
    assert predicted == pytest.approx(mc_max, rel=0.04)


def test_select_and_deflate_single_trial_equals_psr() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0.05, 1.0, size=(300, 1))
    result = sh.select_and_deflate(x)
    assert result.dsr == pytest.approx(result.psr)
    assert result.sr0 == 0.0


def test_select_and_deflate_picks_best_and_deflates() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal((500, 50))
    x[:, 7] += 0.3
    result = sh.select_and_deflate(x)
    assert result.best == 7
    assert result.dsr < result.psr
    assert result.dsr_pvalue == pytest.approx(1 - result.dsr)


def test_effective_number_of_trials_limits() -> None:
    rng = np.random.default_rng(7)
    independent = rng.standard_normal((20_000, 10))
    assert sh.effective_number_of_trials(independent) == pytest.approx(10, rel=0.02)
    common = rng.standard_normal((1_000, 1))
    duplicated = common + 1e-9 * rng.standard_normal((1_000, 10))
    assert sh.effective_number_of_trials(duplicated) == pytest.approx(1.0, rel=1e-6)
