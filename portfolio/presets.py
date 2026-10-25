"""Saved portfolios that can be loaded from the sidebar."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

DEFAULT_BENCHMARK = "^SP500TR"
# iShares MSCI World EUR Hedged UCITS ETF (Acc), ISIN IE00B441G979, on Euronext Amsterdam.
# Same fund elsewhere: IBCH.DE (Xetra), IWDE.L (London), IWDE.MI (Milan); all quoted in EUR.
COMPETITION_BENCHMARK = "IWDE.AS"


@dataclass(frozen=True)
class Preset:
    weights: dict[str, float]  # ticker -> weight in percent
    start: dt.date | None = None
    end: dt.date | None = None
    benchmark: str = DEFAULT_BENCHMARK
    notes: str = ""

    @property
    def is_filled(self) -> bool:
        return bool(self.weights)


PRESETS: dict[str, Preset] = {
    # TODO: fill in your competition holdings. Weights are percentages and
    # should add to 100, e.g. {"AAPL": 25, "MSFT": 25, "ASML.AS": 50}.
    # Use Yahoo Finance tickers (non-US listings need a suffix: .L, .AS, .PA, .DE ...).
    "Student competition": Preset(
        weights={},
        start=None,  # e.g. dt.date(2026, 1, 15)
        end=None,  # None means today
        benchmark=COMPETITION_BENCHMARK,
        notes="Fill in portfolio/presets.py with your competition holdings.",
    ),
    "Example: 60/40 stocks/bonds": Preset(
        weights={"VTI": 60, "BND": 40},
        notes="A simple reference portfolio to try the app with.",
    ),
}
