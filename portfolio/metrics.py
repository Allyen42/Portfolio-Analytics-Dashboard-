"""Portfolio performance metrics.

Pure functions over pandas objects: no Streamlit, no network access. Returns
are simple (not log) daily returns as decimals (0.01 == 1%). Rates such as the
risk-free rate are annual decimals.

Annualising functions take `periods_per_year`. It defaults to 252, but when
holdings trade on exchanges with different holidays the combined calendar has
more rows per year; pass `observed_periods_per_year(...)` so that annual
figures are measured against calendar time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25
WEIGHT_TOLERANCE = 1e-6
ZERO_VOLATILITY = 1e-12  # below this, volatility is floating-point noise

Frequency = Literal["daily", "weekly"]


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple daily returns from a price series/frame. The first row is dropped."""
    return (prices / prices.shift(1) - 1).iloc[1:]


def normalise_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    """Scale weights so they sum to 1."""
    w = pd.Series(weights, dtype=float)
    total = w.sum()
    if w.empty or total <= 0:
        raise ValueError("Weights must have a positive sum.")
    return w / total


def validate_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    """Return weights as a Series, raising if they are empty or don't sum to 1."""
    w = pd.Series(weights, dtype=float)
    if w.empty:
        raise ValueError("No weights given.")
    if abs(w.sum() - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"Weights must sum to 1 (got {w.sum():.6f}).")
    return w


def portfolio_returns(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    rebalance: bool = False,
) -> pd.Series:
    """Daily returns of a weighted portfolio.

    rebalance=False (buy and hold): weights are set on day 0 and then drift
    with prices. rebalance=True: weights are reset to target every day, so the
    portfolio return is simply the weighted sum of asset returns.
    """
    w = validate_weights(weights)
    missing = set(w.index) - set(returns.columns)
    if missing:
        raise ValueError(f"No returns for: {', '.join(sorted(missing))}")
    asset_returns = returns[w.index]

    if rebalance:
        result = asset_returns @ w
    else:
        value = (1 + asset_returns).cumprod() @ w
        result = value / value.shift(1, fill_value=1.0) - 1
    return result.rename("Portfolio")


def wealth_index(returns: pd.Series, start: pd.Timestamp | None = None) -> pd.Series:
    """Growth of 1 unit invested. If `start` is given, the series begins at 1.0 on
    that date (the day before the first return), so charts start from 0%."""
    wealth = (1 + returns).cumprod()
    if start is not None:
        wealth = pd.concat([pd.Series([1.0], index=[pd.Timestamp(start)]), wealth])
        wealth.name = returns.name
    return wealth


def cumulative_returns(returns: pd.Series, start: pd.Timestamp | None = None) -> pd.Series:
    """Cumulative return to date: prod(1 + r) - 1."""
    return wealth_index(returns, start) - 1


def observed_periods_per_year(index: pd.DatetimeIndex, start: pd.Timestamp | None = None) -> float:
    """Number of return observations per calendar year actually in the data.

    About 252 for a single exchange; more when exchanges with different holidays
    are combined (each holiday on one exchange adds a row for the other). `start`
    is the price date before the first return; without it the span starts at
    the first return. Falls back to 252 if the span is empty.
    """
    if len(index) == 0:
        raise ValueError("No dates given.")
    first = pd.Timestamp(start) if start is not None else index[0]
    days = (index[-1] - first).days
    if days <= 0:
        return float(TRADING_DAYS)
    return len(index) / (days / DAYS_PER_YEAR)


def annualised_return(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """Compound annual growth rate: (1 + total return) ** (periods_per_year / n) - 1.

    With observed periods per year this is (1 + total return) ** (1 / years).
    """
    if len(returns) == 0:
        raise ValueError("No returns given.")
    growth = float((1 + returns).prod())
    return growth ** (periods_per_year / len(returns)) - 1


def annualised_volatility(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """Standard deviation of returns scaled by sqrt(periods per year)."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def daily_risk_free(rf_annual: float, periods_per_year: float = TRADING_DAYS) -> float:
    """Convert an annual risk-free rate to the equivalent rate per period."""
    return (1 + rf_annual) ** (1 / periods_per_year) - 1


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: float = TRADING_DAYS) -> float:
    """Annualised mean excess return divided by annualised volatility of excess return."""
    excess = returns - daily_risk_free(rf_annual, periods_per_year)
    vol = excess.std(ddof=1) * np.sqrt(periods_per_year)
    if not np.isfinite(vol) or vol < ZERO_VOLATILITY:
        return float("nan")
    return float(excess.mean() * periods_per_year / vol)


def m_squared(
    returns: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0, periods_per_year: float = TRADING_DAYS
) -> float:
    """Modigliani-Modigliani measure: the portfolio's return if it were levered or
    de-levered to the benchmark's volatility. M2 = Sharpe_p * sigma_b + rf."""
    return (
        sharpe_ratio(returns, rf_annual, periods_per_year) * annualised_volatility(benchmark, periods_per_year)
        + rf_annual
    )


def m_squared_excess(
    returns: pd.Series, benchmark: pd.Series, rf_annual: float = 0.0, periods_per_year: float = TRADING_DAYS
) -> float:
    """M2 minus the benchmark's own (Sharpe-consistent) return: (Sharpe_p - Sharpe_b) * sigma_b.
    Positive means the portfolio delivered more return per unit of risk than the benchmark."""
    sharpe_gap = sharpe_ratio(returns, rf_annual, periods_per_year) - sharpe_ratio(
        benchmark, rf_annual, periods_per_year
    )
    return sharpe_gap * annualised_volatility(benchmark, periods_per_year)


def to_weekly(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compound daily returns into Monday-to-Friday weeks.

    Each week is dated by its last trading day, so a partial week at the end of
    the data isn't labelled with a Friday that hasn't happened yet.
    """
    week = pd.Grouper(freq="W-FRI")
    grouped = (1 + returns).groupby(week)
    has_data = grouped.size() > 0
    weekly = (grouped.prod() - 1)[has_data]
    weekly.index = pd.Series(returns.index, index=returns.index).groupby(week).max()[has_data].to_numpy()
    return weekly


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Cov(r_p, r_b) / Var(r_b) on the dates both series share."""
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    p, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    var_b = b.var(ddof=1)
    if var_b < ZERO_VOLATILITY**2:
        return float("nan")
    return float(p.cov(b, ddof=1) / var_b)


def drawdown_series(wealth: pd.Series) -> pd.Series:
    """Percentage below the running peak at each date (0 at a new high, negative otherwise)."""
    return wealth / wealth.cummax() - 1


@dataclass(frozen=True)
class Drawdown:
    max_drawdown: float  # negative decimal, e.g. -0.25 for a 25% fall
    peak: pd.Timestamp | None
    trough: pd.Timestamp | None
    recovery: pd.Timestamp | None  # None if never back to the peak value


def max_drawdown(wealth: pd.Series) -> Drawdown:
    """Largest peak-to-trough fall of a wealth index, with its dates."""
    dd = drawdown_series(wealth)
    worst = float(dd.min())
    if worst >= 0:
        return Drawdown(0.0, None, None, None)
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered = after[after >= wealth.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return Drawdown(worst, peak, trough, recovery)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily returns."""
    return returns.corr()


@dataclass(frozen=True)
class Summary:
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    m_squared: float
    m_squared_excess: float
    beta: float
    drawdown: Drawdown


def summarise(
    returns: pd.Series,
    benchmark: pd.Series,
    rf_annual: float = 0.0,
    start: pd.Timestamp | None = None,
    periods_per_year: float = TRADING_DAYS,
    beta_frequency: Frequency = "daily",
) -> Summary:
    """All headline metrics for `returns` measured against `benchmark`.

    beta_frequency="weekly" measures beta on weekly returns, which avoids the
    downward bias when the two series are priced at different times of day
    (e.g. US stocks against a Europe-listed fund).
    """
    if beta_frequency == "weekly":
        weekly = to_weekly(pd.concat([returns, benchmark], axis=1, join="inner"))
        portfolio_beta = beta(weekly.iloc[:, 0], weekly.iloc[:, 1])
    else:
        portfolio_beta = beta(returns, benchmark)
    return Summary(
        annualised_return=annualised_return(returns, periods_per_year),
        annualised_volatility=annualised_volatility(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, rf_annual, periods_per_year),
        m_squared=m_squared(returns, benchmark, rf_annual, periods_per_year),
        m_squared_excess=m_squared_excess(returns, benchmark, rf_annual, periods_per_year),
        beta=portfolio_beta,
        drawdown=max_drawdown(wealth_index(returns, start)),
    )
