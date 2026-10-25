import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio import factors, values as v

COMPETITION_CSV = Path(__file__).resolve().parent.parent / "data" / "competition_values.csv"


def csv(text):
    return io.StringIO(text)


def snapshots(dates, numbers):
    return pd.Series(numbers, index=pd.DatetimeIndex(dates), dtype=float)


# ---------------------------------------------------------------- reading the CSV


def test_read_values_csv_sorts_and_accepts_any_column_case():
    series = v.read_values_csv(csv(" Date ,VALUE\n2026-03-01,110\n2026-02-01,100\n2026-04-01,121\n"))
    assert list(series.index) == list(pd.to_datetime(["2026-02-01", "2026-03-01", "2026-04-01"]))
    assert series.tolist() == [100, 110, 121]


@pytest.mark.parametrize(
    "text, message",
    [
        ("day,amount\n2026-01-01,1\n", "'date' column"),
        ("date,value\n2026-01-01,1\n03/04/2026,2\n2026-01-09,3\n", "Row(s) 3"),
        ("date,value\n2026-01-01,1\n2026-01-02,abc\n2026-01-09,3\n", "Row(s) 3"),
        ("date,value\n2026-01-01,1\n2026-01-01,2\n2026-01-09,3\n", "more than once: 2026-01-01"),
        ("date,value\n2026-01-01,1\n2026-01-02,0\n2026-01-09,3\n", "above zero"),
        ("date,value\n2026-01-01,1\n2026-01-02,2\n", "three snapshots"),
    ],
)
def test_read_values_csv_errors(text, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        v.read_values_csv(csv(text))


def test_competition_file_loads():
    series = v.read_values_csv(COMPETITION_CSV)
    assert len(series) == 13
    assert series.index[0] == pd.Timestamp("2026-02-17")
    assert series.iloc[0] == pytest.approx(297.4)
    assert len(v.interval_returns(series)) == 12


# ---------------------------------------------------------------- intervals


def test_interval_returns():
    iv = v.interval_returns(snapshots(["2026-01-01", "2026-01-08", "2026-01-22"], [100, 110, 99]))
    assert iv["days"].tolist() == [7, 14]
    assert iv["return"].tolist() == pytest.approx([0.10, -0.10])
    assert list(iv.index) == list(pd.to_datetime(["2026-01-08", "2026-01-22"]))


def test_weekly_equivalent():
    assert v.weekly_equivalent(0.05, 7) == pytest.approx(0.05)
    assert v.weekly_equivalent(0.21, 14) == pytest.approx(0.10)
    assert v.weekly_equivalent(0.01, 1) == pytest.approx(1.01**7 - 1)
    frame = pd.DataFrame({"a": [0.21, 0.05]})
    assert v.weekly_equivalent(frame, [14, 7])["a"].tolist() == pytest.approx([0.10, 0.05])


def test_values_at_uses_last_close_on_or_before():
    prices = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-04-08", "2026-04-09", "2026-04-10"]))
    sunday = pd.DatetimeIndex(["2026-04-09", "2026-04-12"])
    assert v.values_at(prices, sunday).tolist() == [2.0, 3.0]  # Sunday -> Friday's close
    with pytest.raises(ValueError, match="No data on or before 2026-04-01"):
        v.values_at(prices, pd.DatetimeIndex(["2026-04-01"]))


def test_compound_over_intervals():
    days = pd.bdate_range("2026-01-05", periods=10)  # Mon 5 Jan .. Fri 16 Jan
    daily = pd.DataFrame({"x": 0.01}, index=days)
    # From the Sunday before (nothing earlier) to Friday, then to the Sunday after the second Friday.
    dates = pd.DatetimeIndex(["2026-01-04", "2026-01-09", "2026-01-18"])
    result = v.compound_over_intervals(daily, dates)
    assert result["x"].tolist() == pytest.approx([1.01**5 - 1, 1.01**5 - 1])


# ---------------------------------------------------------------- performance metrics


def test_annualised_return_uses_calendar_time():
    series = snapshots(["2024-01-01", "2024-09-10", "2026-01-01"], [100, 107, 121])
    assert v.annualised_return(series) == pytest.approx(1.21 ** (365.25 / 731) - 1)


def test_volatility_with_exactly_weekly_snapshots_is_standard():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-03", periods=60, freq="7D")
    series = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60))), index=dates)
    log_returns = np.diff(np.log(series.to_numpy()))
    expected = np.std(log_returns, ddof=1) * math.sqrt(365.25 / 7)
    assert v.annualised_volatility(series) == pytest.approx(expected)


def random_walk_snapshots(sigma_annual=0.20, n_intervals=3000, seed=1, max_gap=28):
    """Snapshots of a random walk with known volatility, at irregular gaps of 1 to max_gap days."""
    rng = np.random.default_rng(seed)
    gaps = rng.integers(1, max_gap + 1, n_intervals)
    sigma_daily = sigma_annual / math.sqrt(365.25)
    log_steps = rng.normal(0, 1, n_intervals) * sigma_daily * np.sqrt(gaps)
    dates = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.concatenate([[0], np.cumsum(gaps)]), unit="D")
    return pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(log_steps)])), index=dates)


@pytest.mark.parametrize("max_gap", [14, 28])
def test_time_scaled_volatility_recovers_true_volatility_from_irregular_snapshots(max_gap):
    series = random_walk_snapshots(sigma_annual=0.20, max_gap=max_gap)
    assert v.annualised_volatility(series) == pytest.approx(0.20, rel=0.05)


def test_naive_weekly_equivalent_volatility_is_biased_by_harmonic_mean_gap():
    # Scaling a d-day return to a week multiplies its variance by 7/d, so the naive
    # std(weekly-equivalent) * sqrt(52) is off by sqrt(7 / harmonic mean of the gaps):
    # unbiased only when that harmonic mean happens to be 7 days. Gaps of 1-14 days
    # have a harmonic mean of ~4.3, so the naive estimate is ~27% too high.
    series = random_walk_snapshots(sigma_annual=0.20, max_gap=14)
    iv = v.interval_returns(series)
    harmonic_mean_gap = len(iv) / (1 / iv["days"]).sum()
    naive = v.weekly_equivalent(iv["return"], iv["days"]).std() * math.sqrt(365.25 / 7)
    assert naive == pytest.approx(0.20 * math.sqrt(7 / harmonic_mean_gap), rel=0.06)
    assert naive > 0.24


def test_beta_and_correlation_on_irregular_snapshots():
    bench = random_walk_snapshots(n_intervals=200, seed=2)
    portfolio = np.exp(1.5 * np.log(bench / 100)) * 50  # log returns exactly 1.5x the benchmark's
    assert v.beta(portfolio, bench) == pytest.approx(1.5)
    assert v.correlation(portfolio, bench) == pytest.approx(1.0)


def test_sharpe_uses_expected_return_convention():
    series = random_walk_snapshots(n_intervals=200, seed=3)
    drift, _, _ = v._scaled_log_returns(series)
    vol = v.annualised_volatility(series)
    expected = (drift * 365.25 + vol**2 / 2 - math.log1p(0.02)) / vol
    assert v.sharpe_ratio(series, 0.02) == pytest.approx(expected)


def test_summarise_benchmark_against_itself():
    bench = random_walk_snapshots(n_intervals=40, seed=4)
    s = v.summarise(bench, bench, 0.02)
    assert s.beta == pytest.approx(1.0)
    assert s.m_squared_excess == pytest.approx(0.0, abs=1e-12)
    assert s.m_squared == pytest.approx(s.sharpe * s.annualised_volatility + 0.02)
    assert s.drawdown.max_drawdown <= 0


def test_summarise_requires_same_dates():
    a = snapshots(["2026-01-01", "2026-01-08", "2026-01-15"], [1, 2, 3])
    with pytest.raises(ValueError, match="same dates"):
        v.summarise(a, a.iloc[:2], 0.0)


# ---------------------------------------------------------------- factor regression


def synthetic_factor_snapshots(n_days=2000, seed=5):
    """Daily factors, plus portfolio values built from known loadings, sampled at irregular dates."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=n_days)
    factor_data = pd.DataFrame(rng.normal(0, 0.008, (n_days, 6)), index=days, columns=factors.FACTORS)
    factor_data["RF"] = 0.0001
    loadings = pd.Series({"Mkt-RF": 1.1, "SMB": 0.4, "HML": -0.3, "RMW": 0.0, "CMA": 0.2, "Mom": 0.3})
    daily = factor_data["RF"] + factor_data[factors.FACTORS] @ loadings + rng.normal(0, 0.001, n_days)
    wealth = (1 + daily).cumprod()

    # Irregular snapshot dates, including weekends: gaps of 2 to 20 calendar days.
    gaps = rng.integers(2, 21, 400)
    dates = days[0] + pd.to_timedelta(np.cumsum(gaps), unit="D")
    dates = dates[dates <= days[-1]]
    values = v.values_at(wealth, pd.DatetimeIndex(dates))
    return values, factor_data, loadings


def test_factor_regression_recovers_loadings_from_irregular_snapshots():
    values, factor_data, loadings = synthetic_factor_snapshots()
    aligned = v.align_with_factors(values, factor_data)
    assert aligned.n == len(values) - 1
    result = factors.run_factor_regression(aligned, "weekly")
    for factor, beta in loadings.items():
        assert result.table.loc[factor, "coefficient"] == pytest.approx(beta, abs=0.05)
    assert abs(result.alpha_annual) < 0.03  # true alpha is zero


def test_align_counts_snapshots_after_factor_data():
    values, factor_data, _ = synthetic_factor_snapshots()
    # End the factor data on a weekday snapshot near the end (a weekend snapshot would
    # itself fall after the last factor date).
    k = next(i for i in range(len(values) - 6, 0, -1) if values.index[i].day_of_week < 5)
    aligned = v.align_with_factors(values, factor_data.loc[: values.index[k]])
    assert aligned.days_after_factor_data == len(values) - 1 - k
    assert aligned.n == k


def test_short_snapshot_series_runs_with_lower_minimum():
    values, factor_data, _ = synthetic_factor_snapshots()
    short = v.align_with_factors(values.iloc[:13], factor_data)  # like the competition file
    assert short.n == 12 < v.MIN_OBSERVATIONS
    with pytest.raises(ValueError):
        factors.run_factor_regression(short, "weekly")  # default minimum is 26
    result = factors.run_factor_regression(short, "weekly", min_observations=v.MIN_REGRESSION_OBSERVATIONS)
    assert result.n == 12
    too_short = v.align_with_factors(values.iloc[:8], factor_data)
    with pytest.raises(ValueError):
        factors.run_factor_regression(too_short, "weekly", min_observations=v.MIN_REGRESSION_OBSERVATIONS)


def test_align_with_no_overlap_is_empty():
    values, factor_data, _ = synthetic_factor_snapshots()
    aligned = v.align_with_factors(values, factor_data.loc[: values.index[0] - pd.Timedelta(days=1)])
    assert aligned.n == 0
