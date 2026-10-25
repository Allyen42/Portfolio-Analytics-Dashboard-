"""Performance and factor analysis from irregular portfolio value snapshots.

For when you only know what the portfolio was worth on certain dates (e.g. a
competition statement), not its holdings. Pure functions over pandas objects:
no Streamlit, no network access.

Snapshots are taken as end-of-day values. The benchmark and factors are
measured over exactly the same intervals, using the last close on or before
each snapshot date (so a weekend snapshot uses Friday's close).

Returns between snapshots are converted to weekly-equivalent returns,
(1 + R) ** (7 / days) - 1, for the return series and the factor regression.
Risk (volatility, Sharpe, M2, beta) is measured on log returns scaled by
sqrt(7 / days) instead, because risk grows with the square root of time.
Raising a d-day return to the power 7/d multiplies its variance by 7/d, so
the standard deviation of weekly-equivalent returns is off by a factor of
sqrt(7 / harmonic mean of the interval lengths): right only when that
harmonic mean happens to be 7 days, and noisy because short intervals are
amplified the most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio import metrics
from portfolio.factors import FACTORS, AlignedFactors

DAYS_PER_WEEK = 7
MIN_OBSERVATIONS = 26  # fewer intervals than this get a warning
MIN_REGRESSION_OBSERVATIONS = len(FACTORS) + 2  # below this the regression can't run at all


def read_values_csv(source) -> pd.Series:
    """Read a CSV with `date` and `value` columns into a sorted Series of positive values.

    Dates must be YYYY-MM-DD so that 03-04 can't be misread as March or April.
    `source` is a path or a file-like object (e.g. a Streamlit upload).
    """
    table = pd.read_csv(source)
    table.columns = [str(c).strip().lower() for c in table.columns]
    if not {"date", "value"} <= set(table.columns):
        raise ValueError("The CSV needs a 'date' column and a 'value' column.")

    dates = pd.to_datetime(table["date"].astype(str).str.strip(), format="ISO8601", errors="coerce")
    values = pd.to_numeric(table["value"], errors="coerce")
    bad_rows = [i + 2 for i in table.index[dates.isna() | values.isna()]]  # +2: header row, 1-based
    if bad_rows:
        raise ValueError(
            f"Row(s) {', '.join(map(str, bad_rows))} have a date or value that can't be read. "
            "Use dates like 2026-02-17 and plain numbers like 297.4."
        )

    series = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name="Portfolio").sort_index()
    duplicates = series.index[series.index.duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"Dates listed more than once: {', '.join(d.strftime('%Y-%m-%d') for d in duplicates)}.")
    if (series <= 0).any():
        raise ValueError("Portfolio values must all be above zero.")
    if len(series) < 3:
        raise ValueError("At least three snapshots (two intervals) are needed.")
    return series


def interval_returns(values: pd.Series) -> pd.DataFrame:
    """Return and length in calendar days of each interval between consecutive snapshots.

    Indexed by the interval's end date; columns: start, days, return.
    """
    ends = values.index[1:]
    starts = values.index[:-1]
    return pd.DataFrame(
        {
            "start": starts,
            "days": (ends - starts).days.to_numpy(),
            "return": values.to_numpy()[1:] / values.to_numpy()[:-1] - 1,
        },
        index=ends,
    )


def weekly_equivalent(returns, days):
    """Weekly-equivalent of a return earned over `days` calendar days: (1 + R) ** (7 / days) - 1.

    Works element-wise on numbers, Series, or a DataFrame with `days` aligned to its rows.
    """
    exponent = DAYS_PER_WEEK / np.asarray(days, dtype=float)
    if isinstance(returns, pd.DataFrame):
        return (1 + returns).pow(exponent, axis=0) - 1
    return (1 + returns) ** exponent - 1


def values_at(series: pd.Series | pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series | pd.DataFrame:
    """The last value on or before each date (e.g. Friday's close for a Sunday snapshot)."""
    if dates.min() < series.index.min():
        raise ValueError(
            f"No data on or before {dates.min():%Y-%m-%d}; it starts on {series.index.min():%Y-%m-%d}."
        )
    combined = series.reindex(series.index.union(dates)).ffill()
    return combined.loc[dates]


def compound_over_intervals(daily_returns: pd.DataFrame, snapshot_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compound daily returns over each interval (start, end] between snapshot dates.

    A snapshot before the first daily return counts as the starting point (growth 1.0).
    Indexed by the interval's end date.
    """
    wealth = (1 + daily_returns).cumprod()
    origin = pd.DataFrame(1.0, index=[min(daily_returns.index[0], snapshot_dates[0]) - pd.Timedelta(days=1)],
                          columns=daily_returns.columns)
    wealth = pd.concat([origin, wealth])
    at = values_at(wealth, snapshot_dates)
    return pd.DataFrame(at.to_numpy()[1:] / at.to_numpy()[:-1] - 1, index=snapshot_dates[1:], columns=wealth.columns)


# ---------------------------------------------------------------- performance metrics


def _scaled_log_returns(values: pd.Series) -> tuple[float, pd.Series, pd.Series]:
    """Daily log drift, and each interval's deviation from it scaled to one day.

    Model: log return over d days = drift * d + sigma_daily * sqrt(d) * noise.
    Returns (drift per day, scaled residuals, interval lengths).
    """
    intervals = interval_returns(values)
    log_returns = np.log1p(intervals["return"])
    days = intervals["days"].astype(float)
    drift = log_returns.sum() / days.sum()
    residuals = (log_returns - drift * days) / np.sqrt(days)
    return float(drift), residuals, days


def annualised_return(values: pd.Series) -> float:
    """Growth from the first to the last snapshot over calendar time: (V_last / V_first) ** (365.25 / days) - 1."""
    days = (values.index[-1] - values.index[0]).days
    return float((values.iloc[-1] / values.iloc[0]) ** (metrics.DAYS_PER_YEAR / days) - 1)


def annualised_volatility(values: pd.Series) -> float:
    """Volatility from irregular snapshots: the spread of time-scaled log returns, scaled to a year.

    sigma_daily^2 = sum(residual^2) / (n - 1); annual volatility = sigma_daily * sqrt(365.25).
    With exactly weekly snapshots this is the usual std of weekly log returns * sqrt(52.18).
    """
    _, residuals, _ = _scaled_log_returns(values)
    if len(residuals) < 2:
        return float("nan")
    return float(np.sqrt((residuals**2).sum() / (len(residuals) - 1) * metrics.DAYS_PER_YEAR))


def sharpe_ratio(values: pd.Series, rf_annual: float = 0.0) -> float:
    """Annual expected excess return over annual volatility.

    The expected (arithmetic) annual return is the log drift plus sigma^2 / 2, the same
    convention as the daily Sharpe ratio, which averages simple returns.
    """
    drift, _, _ = _scaled_log_returns(values)
    vol = annualised_volatility(values)
    if not np.isfinite(vol) or vol < metrics.ZERO_VOLATILITY:
        return float("nan")
    expected = drift * metrics.DAYS_PER_YEAR + vol**2 / 2
    return float((expected - np.log1p(rf_annual)) / vol)


def beta(values: pd.Series, benchmark_values: pd.Series) -> float:
    """Beta on time-scaled log returns, so long and short intervals count equally (as for volatility)."""
    _, portfolio, _ = _scaled_log_returns(values)
    _, bench, _ = _scaled_log_returns(benchmark_values)
    denominator = (bench**2).sum()
    if denominator < metrics.ZERO_VOLATILITY**2:
        return float("nan")
    return float((portfolio * bench).sum() / denominator)


def correlation(values: pd.Series, benchmark_values: pd.Series) -> float:
    """Correlation of time-scaled log returns."""
    _, portfolio, _ = _scaled_log_returns(values)
    _, bench, _ = _scaled_log_returns(benchmark_values)
    return float((portfolio * bench).sum() / np.sqrt((portfolio**2).sum() * (bench**2).sum()))


def summarise(values: pd.Series, benchmark_values: pd.Series, rf_annual: float = 0.0) -> metrics.Summary:
    """Headline metrics from snapshots. `benchmark_values` must be on the same dates as `values`."""
    if not values.index.equals(benchmark_values.index):
        raise ValueError("Portfolio and benchmark values must be on the same dates.")
    sharpe = sharpe_ratio(values, rf_annual)
    bench_sharpe = sharpe_ratio(benchmark_values, rf_annual)
    bench_vol = annualised_volatility(benchmark_values)
    return metrics.Summary(
        annualised_return=annualised_return(values),
        annualised_volatility=annualised_volatility(values),
        sharpe=sharpe,
        m_squared=sharpe * bench_vol + rf_annual,
        m_squared_excess=(sharpe - bench_sharpe) * bench_vol,
        beta=beta(values, benchmark_values),
        drawdown=metrics.max_drawdown(values / values.iloc[0]),
    )


# ---------------------------------------------------------------- factor regression


def align_with_factors(values: pd.Series, factor_data: pd.DataFrame) -> AlignedFactors:
    """Weekly-equivalent excess returns and factor returns over the same snapshot intervals.

    The factors (and the market's total return, so that Mkt-RF is rebuilt correctly) are
    compounded over each interval, then everything is converted to weekly equivalents.
    Snapshots outside the factor data's date range are dropped; those after the last
    factor date are counted in `days_after_factor_data` (as snapshots, not days).
    """
    factor_data = factor_data[[*FACTORS, "RF"]].dropna()
    first, last = factor_data.index.min(), factor_data.index.max()
    after = int((values.index > last).sum())
    kept = values[(values.index >= first) & (values.index <= last)]
    if len(kept) < 2:
        empty = pd.DataFrame(columns=FACTORS, index=pd.DatetimeIndex([]), dtype=float)
        return AlignedFactors(pd.Series(dtype=float, name="Excess return"), empty, last, after)

    intervals = interval_returns(kept)
    total = factor_data.copy()
    total["Mkt-RF"] = total["Mkt-RF"] + total["RF"]
    compounded = weekly_equivalent(compound_over_intervals(total, kept.index), intervals["days"])
    compounded["Mkt-RF"] = compounded["Mkt-RF"] - compounded["RF"]
    portfolio = weekly_equivalent(intervals["return"], intervals["days"])

    return AlignedFactors(
        excess=(portfolio - compounded["RF"]).rename("Excess return"),
        factors=compounded[FACTORS],
        last_factor_date=last,
        days_after_factor_data=after,
    )
