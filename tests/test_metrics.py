import math

import numpy as np
import pandas as pd
import pytest

from portfolio import metrics as m

DATES = pd.bdate_range("2024-01-01", periods=504)


def series(values, name=None, index=None):
    index = DATES[: len(values)] if index is None else index
    return pd.Series(values, index=index, name=name, dtype=float)


def test_daily_returns_drops_first_row():
    prices = series([100, 110, 99])
    r = m.daily_returns(prices)
    assert list(r.round(10)) == [0.1, -0.1]
    assert r.index[0] == DATES[1]


def test_rebalanced_portfolio_is_weighted_sum():
    returns = pd.DataFrame({"A": [0.10, -0.05], "B": [0.00, 0.05]}, index=DATES[:2])
    r = m.portfolio_returns(returns, {"A": 0.6, "B": 0.4}, rebalance=True)
    assert r.tolist() == pytest.approx([0.06, -0.01])


def test_buy_and_hold_lets_weights_drift():
    # A doubles on day 1 then is flat; B never moves. 50/50 start.
    returns = pd.DataFrame({"A": [1.0, 0.0, 0.10], "B": [0.0, 0.0, 0.0]}, index=DATES[:3])
    r = m.portfolio_returns(returns, {"A": 0.5, "B": 0.5})
    # Day 1: 0.5*2 + 0.5 = 1.5 -> +50%. Day 3: A is now 2/3 of the portfolio, so +10% * 2/3.
    assert r.tolist() == pytest.approx([0.5, 0.0, 0.1 * 2 / 3])
    rebalanced = m.portfolio_returns(returns, {"A": 0.5, "B": 0.5}, rebalance=True)
    assert rebalanced.iloc[2] == pytest.approx(0.05)


def test_weights_must_sum_to_one():
    returns = pd.DataFrame({"A": [0.01]}, index=DATES[:1])
    with pytest.raises(ValueError, match="sum to 1"):
        m.portfolio_returns(returns, {"A": 0.9})


def test_weights_for_unknown_ticker_rejected():
    returns = pd.DataFrame({"A": [0.01]}, index=DATES[:1])
    with pytest.raises(ValueError, match="No returns for: B"):
        m.portfolio_returns(returns, {"A": 0.5, "B": 0.5})


def test_normalise_weights():
    assert m.normalise_weights({"A": 30, "B": 90}).tolist() == pytest.approx([0.25, 0.75])


def test_wealth_index_prepends_start():
    r = series([0.1, 0.1])
    w = m.wealth_index(r, start=pd.Timestamp("2023-12-29"))
    assert w.tolist() == pytest.approx([1.0, 1.1, 1.21])
    assert m.cumulative_returns(r).tolist() == pytest.approx([0.1, 0.21])


@pytest.mark.parametrize("n_days", [252, 504])
def test_annualised_return_of_constant_daily_return(n_days):
    r = series([0.001] * n_days)
    assert m.annualised_return(r) == pytest.approx(1.001**252 - 1)


def test_annualised_return_half_year():
    # +10% over half a year compounds to 21% a year.
    r = series([0.0] * 125 + [0.10])
    assert m.annualised_return(r) == pytest.approx(0.21)


def test_volatility_of_constant_return_is_zero():
    assert m.annualised_volatility(series([0.001] * 100)) == pytest.approx(0.0)


def test_volatility_scales_daily_std():
    values = [0.01, -0.01] * 50
    expected = np.std(values, ddof=1) * math.sqrt(252)
    assert m.annualised_volatility(series(values)) == pytest.approx(expected)


def test_sharpe_matches_hand_calculation():
    values = np.array([0.02, -0.01, 0.015, 0.0, -0.005])
    rf = 0.03
    excess = values - ((1 + rf) ** (1 / 252) - 1)
    expected = excess.mean() * 252 / (excess.std(ddof=1) * math.sqrt(252))
    assert m.sharpe_ratio(series(values), rf) == pytest.approx(expected)


def test_sharpe_is_nan_without_volatility():
    assert math.isnan(m.sharpe_ratio(series([0.001] * 10)))


def test_m_squared_of_levered_benchmark_equals_benchmark():
    rng = np.random.default_rng(0)
    rf = 0.04
    rf_d = m.daily_risk_free(rf)
    bench = series(rng.normal(0.0004, 0.01, 504))
    levered = rf_d + 2 * (bench - rf_d)  # same Sharpe, twice the risk
    assert m.sharpe_ratio(levered, rf) == pytest.approx(m.sharpe_ratio(bench, rf))
    assert m.m_squared_excess(levered, bench, rf) == pytest.approx(0.0, abs=1e-12)
    expected = m.sharpe_ratio(bench, rf) * m.annualised_volatility(bench) + rf
    assert m.m_squared(levered, bench, rf) == pytest.approx(expected)


def test_m_squared_rewards_better_sharpe():
    rng = np.random.default_rng(1)
    bench = series(rng.normal(0.0003, 0.01, 504))
    better = bench + 0.0002  # same risk, more return
    assert m.m_squared_excess(better, bench, 0.02) > 0


def test_beta():
    rng = np.random.default_rng(2)
    bench = series(rng.normal(0, 0.01, 300))
    assert m.beta(bench, bench) == pytest.approx(1.0)
    assert m.beta(2 * bench + 0.001, bench) == pytest.approx(2.0)
    assert m.beta(-0.5 * bench, bench) == pytest.approx(-0.5)


def test_beta_uses_shared_dates_only():
    bench = series([0.01, -0.02, 0.03, 0.01])
    port = (2 * bench).iloc[1:]
    assert m.beta(port, bench) == pytest.approx(2.0)


def test_max_drawdown_with_recovery():
    wealth = series([1.0, 1.2, 0.9, 1.0, 1.3])
    dd = m.max_drawdown(wealth)
    assert dd.max_drawdown == pytest.approx(-0.25)
    assert (dd.peak, dd.trough, dd.recovery) == (DATES[1], DATES[2], DATES[4])


def test_max_drawdown_without_recovery():
    dd = m.max_drawdown(series([1.0, 0.8, 0.6, 0.7]))
    assert dd.max_drawdown == pytest.approx(-0.4)
    assert dd.peak == DATES[0]
    assert dd.recovery is None


def test_max_drawdown_of_rising_series_is_zero():
    dd = m.max_drawdown(series([1.0, 1.1, 1.2]))
    assert dd.max_drawdown == 0.0
    assert dd.peak is None


def test_correlation_matrix():
    a = series([0.01, -0.02, 0.03, 0.0])
    corr = m.correlation_matrix(pd.DataFrame({"A": a, "B": -a, "C": 3 * a}))
    assert corr.loc["A", "B"] == pytest.approx(-1.0)
    assert corr.loc["A", "C"] == pytest.approx(1.0)


def test_summarise_benchmark_against_itself():
    rng = np.random.default_rng(3)
    bench = series(rng.normal(0.0004, 0.01, 252))
    s = m.summarise(bench, bench, 0.03)
    assert s.beta == pytest.approx(1.0)
    assert s.m_squared_excess == pytest.approx(0.0, abs=1e-12)
    assert s.drawdown.max_drawdown <= 0


def test_observed_periods_per_year_single_exchange():
    # Weekdays over exactly two calendar years: ~261 a year (252 + ~9 holidays we didn't remove).
    index = pd.bdate_range("2024-01-02", "2026-01-01")
    ppy = m.observed_periods_per_year(index, start=pd.Timestamp("2024-01-01"))
    assert ppy == pytest.approx(len(index) / (731 / 365.25))
    assert 255 < ppy < 265


def test_observed_periods_per_year_falls_back_without_span():
    assert m.observed_periods_per_year(pd.DatetimeIndex(["2024-01-02"])) == 252


def test_annualised_return_uses_calendar_time_on_mixed_calendars():
    # 21% growth over exactly two calendar years is 10% a year, however many rows the
    # combined calendar has (here 520, i.e. 260 a year rather than 252).
    start = pd.Timestamp("2024-01-01")
    index = pd.date_range(start, "2026-01-01", periods=521)[1:]
    r = pd.Series((1.21 ** (1 / 520)) - 1, index=index)
    ppy = m.observed_periods_per_year(r.index, start)
    assert m.annualised_return(r, ppy) == pytest.approx(0.10, abs=1e-3)
    assert m.annualised_return(r) < 0.098  # the 252 convention understates it


def test_volatility_and_sharpe_scale_with_periods_per_year():
    values = np.array([0.02, -0.01, 0.015, 0.0, -0.005])
    rf, ppy = 0.03, 260
    excess = values - ((1 + rf) ** (1 / ppy) - 1)
    assert m.annualised_volatility(series(values), ppy) == pytest.approx(np.std(values, ddof=1) * math.sqrt(ppy))
    assert m.sharpe_ratio(series(values), rf, ppy) == pytest.approx(
        excess.mean() * ppy / (excess.std(ddof=1) * math.sqrt(ppy))
    )


def test_to_weekly_compounds_series_and_dates_by_last_trading_day():
    dates = pd.bdate_range("2024-01-01", periods=7)  # Mon..Fri, then Mon, Tue
    weekly = m.to_weekly(pd.Series(0.01, index=dates))
    assert list(weekly.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-09")]
    assert weekly.tolist() == pytest.approx([1.01**5 - 1, 1.01**2 - 1])


def test_weekly_beta_recovers_exposure_hidden_by_a_one_day_lag():
    # A portfolio that reacts to the benchmark one day late, like US stocks measured
    # against a fund that closes hours earlier: daily beta sees almost nothing.
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2023-01-02", periods=780)
    bench = pd.Series(rng.normal(0.0003, 0.01, 780), index=index)
    port = bench.shift(1).fillna(0.0)
    daily = m.summarise(port, bench, beta_frequency="daily")
    weekly = m.summarise(port, bench, beta_frequency="weekly")
    assert abs(daily.beta) < 0.15
    assert weekly.beta == pytest.approx(0.8, abs=0.15)  # 4 of each week's 5 moves land in the same week
