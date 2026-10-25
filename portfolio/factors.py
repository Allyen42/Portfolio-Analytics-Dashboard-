"""Fama-French five-factor plus momentum regression.

Pure functions over pandas objects: no Streamlit, no network access. Factor
data is expected as decimals with columns Mkt-RF, SMB, HML, RMW, CMA, Mom and
RF, as returned by `portfolio.data.download_french_factors`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from portfolio import metrics

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
FACTOR_NAMES = {
    "Alpha": "Alpha (annualised)",
    "Mkt-RF": "Market (Mkt-RF)",
    "SMB": "Size (SMB)",
    "HML": "Value (HML)",
    "RMW": "Profitability (RMW)",
    "CMA": "Investment (CMA)",
    "Mom": "Momentum (Mom)",
}

Frequency = Literal["daily", "weekly"]
CovType = Literal["newey-west", "ols"]

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}
ROLLING_WINDOW = {"daily": 252, "weekly": 52}  # 12 months
MIN_OBSERVATIONS = {"daily": 60, "weekly": 26}
NEWEY_WEST_LAGS = {"daily": 5, "weekly": 4}  # about a week / a month of autocorrelation
SIGNIFICANCE = 0.05

# Thresholds for the plain-English summary. A style loading is only described
# if it is statistically significant AND at least this large in absolute value.
STYLE_THRESHOLD = 0.10
MARKET_BAND = 0.10  # market beta within 1 +/- this counts as "about the market"
LOW_R_SQUARED = 0.50


@dataclass
class AlignedFactors:
    excess: pd.Series  # portfolio return minus RF
    factors: pd.DataFrame  # FACTORS columns
    last_factor_date: pd.Timestamp
    days_after_factor_data: int  # portfolio days dropped because factor data hasn't caught up
    # Portfolio days inside the factor data's date range that have no factor row (e.g. a US
    # holiday when European exchanges were open). Dropped in daily mode; always 0 in weekly
    # mode, where those days are compounded into their week.
    days_dropped_mid_sample: int = 0

    @property
    def n(self) -> int:
        return len(self.excess)


def align_with_factors(
    portfolio_returns: pd.Series, factor_data: pd.DataFrame, frequency: Frequency = "daily"
) -> AlignedFactors:
    """Match portfolio returns to factor returns and subtract RF.

    The French library is updated monthly with a lag, so recent portfolio days
    usually have no factor data; they are dropped and counted.

    Daily: an inner join on date, so portfolio days with no factor row are
    dropped (and counted). Weekly: the portfolio and the factors are each
    compounded to weeks on their own calendars and then joined by week, so no
    portfolio day is lost.
    """
    factor_data = factor_data[[*FACTORS, "RF"]].dropna()
    first_factor_date, last_factor_date = factor_data.index.min(), factor_data.index.max()
    days_after = int((portfolio_returns.index > last_factor_date).sum())
    in_range = portfolio_returns.loc[first_factor_date:last_factor_date].rename("Portfolio")

    if in_range.empty:  # e.g. the whole date range is newer than the factor data
        joined = pd.DataFrame(columns=["Portfolio", *FACTORS, "RF"], index=pd.DatetimeIndex([]), dtype=float)
        dropped_mid_sample = 0
    elif frequency == "weekly":
        portfolio_weekly = metrics.to_weekly(in_range)
        factors_weekly = to_weekly(factor_data.loc[in_range.index.min() : in_range.index.max()])
        portfolio_weeks = portfolio_weekly.index.to_period("W-FRI")
        factor_weeks = factors_weekly.index.to_period("W-FRI")
        joined = pd.concat(
            [portfolio_weekly.set_axis(portfolio_weeks), factors_weekly.set_axis(factor_weeks)],
            axis=1,
            join="inner",
        )
        # Date each week by the last day either calendar has data for it.
        last_days = pd.concat(
            [
                pd.Series(portfolio_weekly.index, index=portfolio_weeks),
                pd.Series(factors_weekly.index, index=factor_weeks),
            ],
            axis=1,
        ).max(axis=1)
        joined.index = pd.DatetimeIndex(last_days.loc[joined.index].to_numpy())
        dropped_mid_sample = 0
    else:
        joined = pd.concat([in_range, factor_data], axis=1, join="inner")
        dropped_mid_sample = len(in_range) - len(joined)

    return AlignedFactors(
        excess=(joined["Portfolio"] - joined["RF"]).rename("Excess return"),
        factors=joined[FACTORS],
        last_factor_date=last_factor_date,
        days_after_factor_data=days_after,
        days_dropped_mid_sample=dropped_mid_sample,
    )


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns into Monday-to-Friday weeks.

    Each week is dated by its last trading day, so a partial week at the end of
    the data isn't labelled with a Friday that hasn't happened yet. The market
    excess return is rebuilt from the market's total return so that weekly
    Mkt-RF = compounded market return - compounded RF. Long-short factors and
    the portfolio are compounded directly.
    """
    total = daily.copy()
    if "Mkt-RF" in total and "RF" in total:
        total["Mkt-RF"] = total["Mkt-RF"] + total["RF"]
    weekly = metrics.to_weekly(total)
    if "Mkt-RF" in weekly and "RF" in weekly:
        weekly["Mkt-RF"] = weekly["Mkt-RF"] - weekly["RF"]
    return weekly


@dataclass
class FactorRegression:
    table: pd.DataFrame  # rows: Alpha + FACTORS; columns below
    r_squared: float
    adj_r_squared: float
    n: int
    frequency: Frequency
    cov_type: CovType

    @property
    def alpha_annual(self) -> float:
        return float(self.table.loc["Alpha", "coefficient"])

    @property
    def loadings(self) -> pd.DataFrame:
        return self.table.drop(index="Alpha")


def run_factor_regression(
    aligned: AlignedFactors,
    frequency: Frequency = "daily",
    cov_type: CovType = "newey-west",
    min_observations: int | None = None,
) -> FactorRegression:
    """OLS of portfolio excess return on the six factors.

    Coefficients are identical for both `cov_type`s; only standard errors,
    t-statistics and p-values differ. Alpha (and its confidence interval) is
    annualised by multiplying by the number of periods per year.

    Table columns: coefficient, std_error, t_stat, p_value, ci_low, ci_high, significant.
    `min_observations` overrides MIN_OBSERVATIONS[frequency], e.g. for short snapshot series.
    """
    minimum = MIN_OBSERVATIONS[frequency] if min_observations is None else min_observations
    if aligned.n < minimum:
        raise ValueError(f"Need at least {minimum} {frequency} observations, got {aligned.n}.")
    X = sm.add_constant(aligned.factors[FACTORS]).rename(columns={"const": "Alpha"})
    model = sm.OLS(aligned.excess, X)
    if cov_type == "newey-west":
        # use_t: p-values from the t-distribution rather than the normal, which matters in short samples
        fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": NEWEY_WEST_LAGS[frequency]}, use_t=True)
    else:
        fit = model.fit()

    ci = fit.conf_int(alpha=SIGNIFICANCE)
    table = pd.DataFrame(
        {
            "coefficient": fit.params,
            "std_error": fit.bse,
            "t_stat": fit.tvalues,
            "p_value": fit.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    table["significant"] = table["p_value"] < SIGNIFICANCE
    scale = PERIODS_PER_YEAR[frequency]
    table.loc["Alpha", ["coefficient", "std_error", "ci_low", "ci_high"]] *= scale

    return FactorRegression(
        table=table,
        r_squared=float(fit.rsquared),
        adj_r_squared=float(fit.rsquared_adj),
        n=int(fit.nobs),
        frequency=frequency,
        cov_type=cov_type,
    )


def rolling_loadings(aligned: AlignedFactors, frequency: Frequency = "daily") -> pd.DataFrame:
    """Factor loadings from the full six-factor regression over a rolling 12-month window.

    Each row is dated at the end of its window. Empty if there isn't a full window of data.
    """
    window = ROLLING_WINDOW[frequency]
    if aligned.n < window:
        return pd.DataFrame(columns=FACTORS, dtype=float)
    X = sm.add_constant(aligned.factors[FACTORS])
    fit = RollingOLS(aligned.excess, X, window=window).fit(params_only=True)
    return fit.params[FACTORS].dropna()


def _tilt(result: FactorRegression, factor: str) -> int:
    """+1 / -1 for a significant loading beyond STYLE_THRESHOLD, else 0."""
    row = result.table.loc[factor]
    if not row["significant"] or abs(row["coefficient"]) < STYLE_THRESHOLD:
        return 0
    return 1 if row["coefficient"] > 0 else -1


def _join_and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def describe_exposures(result: FactorRegression) -> list[str]:
    """Two or three plain-English sentences interpreting the regression."""
    size = {1: "small-cap", -1: "large-cap", 0: ""}[_tilt(result, "SMB")]
    style = {1: "value", -1: "growth", 0: ""}[_tilt(result, "HML")]
    extras = [
        {1: "high momentum exposure", -1: "a contrarian (negative momentum) tilt", 0: ""}[_tilt(result, "Mom")],
        {1: "a tilt towards highly profitable companies", -1: "a tilt towards less profitable companies", 0: ""}[
            _tilt(result, "RMW")
        ],
        {1: "a tilt towards conservatively investing companies", -1: "a tilt towards fast-growing, heavily investing companies", 0: ""}[
            _tilt(result, "CMA")
        ],
    ]
    extras = [e for e in extras if e]
    adjectives = " ".join(a for a in (size, style) if a)

    if adjectives and extras:
        style_text = f"Your portfolio behaves like a {adjectives} tilt with {_join_and(extras)}"
    elif adjectives:
        style_text = f"Your portfolio behaves like a {adjectives} tilt"
    elif extras:
        style_text = f"Your portfolio shows {_join_and(extras)}"
    else:
        style_text = "Your portfolio shows no clear size, value, profitability, investment or momentum tilt"

    beta = float(result.table.loc["Mkt-RF", "coefficient"])
    if beta > 1 + MARKET_BAND:
        market_text = f"its market beta of {beta:.2f} means it tends to move more than the market"
    elif beta < 1 - MARKET_BAND:
        market_text = f"its market beta of {beta:.2f} means it tends to move less than the market"
    else:
        market_text = f"its market beta of {beta:.2f} means it moves roughly in line with the market"

    alpha = result.alpha_annual
    if result.table.loc["Alpha", "significant"]:
        direction = "positive" if alpha > 0 else "negative"
        alpha_text = (
            f"Alpha is {alpha:+.1%} a year and statistically significant: a {direction} return "
            "that these factors don't explain."
        )
    else:
        alpha_text = (
            f"Alpha ({alpha:+.1%} a year) is not statistically significant, so it can't be "
            "distinguished from luck."
        )

    r2_text = f"The factors explain {result.r_squared:.0%} of the variation in your excess returns"
    if result.r_squared < LOW_R_SQUARED:
        r2_text += ", so much of your risk is stock-specific rather than driven by these factors."
    else:
        r2_text += "."

    return [f"{style_text}; {market_text}.", alpha_text, r2_text]
