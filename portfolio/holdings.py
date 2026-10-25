"""Turn the user's ticker/weight table into validated weights."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from portfolio.metrics import WEIGHT_TOLERANCE

PERCENT_TOLERANCE = 0.01  # weights summing to 99.99-100.01% count as 100%


@dataclass
class ParsedHoldings:
    weights: pd.Series  # ticker -> weight as a decimal
    errors: list[str] = field(default_factory=list)

    @property
    def total_percent(self) -> float:
        return float(self.weights.sum() * 100)

    @property
    def sums_to_one(self) -> bool:
        return abs(self.weights.sum() - 1) <= max(WEIGHT_TOLERANCE, PERCENT_TOLERANCE / 100)


def parse_holdings(table: pd.DataFrame) -> ParsedHoldings:
    """Read a table with 'Ticker' and 'Weight (%)' columns.

    Blank rows are ignored and tickers are upper-cased. Duplicates, missing or
    negative weights are reported as errors (short positions aren't supported).
    """
    errors: list[str] = []
    rows = table.copy()
    rows["Ticker"] = rows["Ticker"].fillna("").astype(str).str.strip().str.upper()
    rows = rows[rows["Ticker"] != ""]

    if rows.empty:
        return ParsedHoldings(pd.Series(dtype=float), ["Add at least one ticker."])

    weights = pd.to_numeric(rows["Weight (%)"], errors="coerce")
    for ticker in rows.loc[weights.isna(), "Ticker"]:
        errors.append(f"{ticker} has no weight.")
    for ticker in rows.loc[weights < 0, "Ticker"]:
        errors.append(f"{ticker} has a negative weight; short positions aren't supported.")
    dupes = sorted(rows.loc[rows["Ticker"].duplicated(), "Ticker"].unique())
    if dupes:
        errors.append(f"Listed more than once: {', '.join(dupes)}.")

    series = pd.Series(weights.to_numpy() / 100, index=rows["Ticker"].to_numpy(), dtype=float)
    if not errors and series.sum() <= 0:
        errors.append("Weights must add up to more than 0%.")
    return ParsedHoldings(series, errors)
