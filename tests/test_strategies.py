"""Strategy construction: timing, costs, and the no-look-ahead guarantee."""

import numpy as np
import pytest

from btaudit.strategies import (
    StrategySpec,
    UniverseConfig,
    build_universe,
    default_grid,
    net_returns,
    raw_signal,
    volatility_scale,
)

SMALL_GRID = [
    StrategySpec("ma_crossover", (1, 10), False, False),
    StrategySpec("ma_crossover", (3, 12), True, True),
    StrategySpec("tsmom", (1,), True, False),
    StrategySpec("tsmom", (12,), False, True),
    StrategySpec("mean_reversion", (6, 1.0), True, False),
    StrategySpec("mean_reversion", (24, 0.5), False, True),
]


def _market(n: int = 400, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.005, 0.04, size=n)


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_no_look_ahead_future_perturbation_leaves_past_unchanged(lag: int) -> None:
    returns = _market()
    config = UniverseConfig(lag=lag, cost_bps=10.0)
    base = build_universe(returns, SMALL_GRID, config)
    for t0 in (150, 250, 398):
        perturbed = returns.copy()
        rng = np.random.default_rng(t0)
        perturbed[t0 + 1 :] = rng.normal(-0.02, 0.1, size=perturbed.size - t0 - 1)
        other = build_universe(perturbed, SMALL_GRID, config)
        # Positions up to t0 + lag depend only on returns up to t0.
        assert np.array_equal(
            base.positions[: t0 + lag + 1], other.positions[: t0 + lag + 1], equal_nan=True
        )
        assert np.array_equal(base.returns[: t0 + 1], other.returns[: t0 + 1], equal_nan=True)
        if t0 + lag + 1 < returns.size:  # the check is not vacuous: later positions react
            assert not np.array_equal(
                base.positions[t0 + lag + 1 :], other.positions[t0 + lag + 1 :]
            )


def test_full_grid_has_no_look_ahead() -> None:
    returns = _market(300, seed=1)
    grid = default_grid()
    config = UniverseConfig(lag=1)
    base = build_universe(returns, grid, config)
    perturbed = returns.copy()
    perturbed[200:] *= -3.0
    other = build_universe(perturbed, grid, config)
    assert np.array_equal(base.positions[:201], other.positions[:201], equal_nan=True)


def test_lag_must_be_positive() -> None:
    with pytest.raises(ValueError):
        UniverseConfig(lag=0)


def test_positions_are_lagged_signals() -> None:
    returns = _market()
    spec = StrategySpec("tsmom", (3,), True, False)
    log_price = np.cumsum(np.log1p(returns))
    signal = raw_signal(log_price, spec)
    universe = build_universe(returns, [spec], UniverseConfig(lag=2))
    assert np.array_equal(universe.positions[universe.start :, 0], signal[universe.start - 2 : -2])


def test_transaction_costs_hand_computed() -> None:
    asset = np.array([0.01, 0.02, -0.01, 0.03])
    positions = np.array([[np.nan], [1.0], [-1.0], [-1.0]])
    net = net_returns(asset, positions, cost_bps=10.0)[:, 0]
    assert np.isnan(net[0])
    # entry |1 - 0| = 1 unit, then a flip of 2 units, then no trade
    assert net[1:] == pytest.approx([0.02 - 0.001, 0.01 - 0.002, -0.03])


def test_zero_cost_gives_gross_returns() -> None:
    returns = _market()
    gross = build_universe(returns, SMALL_GRID, UniverseConfig(cost_bps=0.0))
    s = gross.start
    assert np.allclose(gross.returns[s:], gross.positions[s:] * returns[s:, None])
    costly = build_universe(returns, SMALL_GRID, UniverseConfig(cost_bps=25.0))
    assert np.all(costly.returns[s:] <= gross.returns[s:] + 1e-15)


def test_tsmom_and_ma_on_a_rising_market_are_long() -> None:
    returns = np.full(100, 0.01)
    specs = [
        StrategySpec("tsmom", (6,), True, False),
        StrategySpec("ma_crossover", (2, 8), True, False),
    ]
    universe = build_universe(returns, specs)
    assert np.all(universe.positions[universe.start :] == 1.0)


def test_mean_reversion_goes_long_after_a_crash() -> None:
    returns = np.concatenate([np.full(30, 0.005), [-0.3], np.full(5, 0.0)])
    spec = StrategySpec("mean_reversion", (12, 1.5), True, False)
    signal = raw_signal(np.cumsum(np.log1p(returns)), spec)
    assert signal[30] == 1.0
    assert signal[25] == -1.0  # steady rise puts the price above its band


def test_volatility_targeting_respects_target_and_cap() -> None:
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.001, 50)
    wild = rng.normal(0, 0.2, 50)
    config = UniverseConfig(vol_target=0.15, vol_window=12, max_leverage=2.0)
    scale = volatility_scale(np.concatenate([calm, wild]), config)
    assert np.all(scale[11:50] == 2.0)  # tiny volatility hits the cap
    realized = np.std(wild[-12:], ddof=1) * np.sqrt(12)
    assert scale[-1] == pytest.approx(0.15 / realized)


def test_mean_reversion_threshold_bound_is_tight() -> None:
    # (L - 1) / sqrt(L) is attained when the first L - 1 points are equal.
    log_price = np.concatenate([np.zeros(5), [1.0]])
    spec = StrategySpec("mean_reversion", (6, 2.0), True, False)
    assert (6 - 1) / np.sqrt(6) > 2.0
    assert raw_signal(log_price, spec)[-1] == -1.0


def test_default_grid_trades_on_real_like_data() -> None:
    universe = build_universe(_market(1_000, seed=4), default_grid())
    held = universe.positions[universe.start :]
    assert np.all(np.abs(held).max(axis=0) > 0)  # every rule takes a position at some point
    assert np.all(np.isfinite(universe.returns[universe.start :]))


def test_default_grid_size_and_unique_names() -> None:
    grid = default_grid()
    assert len(grid) == 600
    assert len({s.name for s in grid}) == 600


def test_zero_drift_market_gives_zero_mean_strategies() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.04, size=200_000)
    universe = build_universe(returns, SMALL_GRID)
    x = universe.returns[universe.start :]
    t_stats = x.mean(axis=0) / (x.std(axis=0, ddof=1) / np.sqrt(x.shape[0]))
    assert np.all(np.abs(t_stats) < 4)


@pytest.mark.parametrize(
    "args",
    [
        ("ma_crossover", (10, 5)),
        ("tsmom", (0,)),
        ("mean_reversion", (1, 1.0)),
        ("mean_reversion", (3, 1.2)),  # |z| of the newest of 3 points never exceeds 1.155
        ("breakout", (5,)),
    ],
)
def test_invalid_specs_raise(args: tuple) -> None:
    with pytest.raises(ValueError):
        StrategySpec(args[0], args[1], False, False)
