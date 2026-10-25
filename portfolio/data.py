"""Price data: download from Yahoo Finance via yfinance and clean it for analysis."""

from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import requests
import yfinance as yf

RISK_FREE_TICKER = "^IRX"  # 13-week US T-bill yield, quoted in percent

FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {  # region -> (five-factor zip, momentum zip), daily data
    "US": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"),
    "Developed Europe": ("Europe_5_Factors_Daily_CSV.zip", "Europe_Mom_Factor_Daily_CSV.zip"),
}
FRENCH_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
FRENCH_MISSING = -99.99  # the library marks missing values as -99.99 or -999
_DATE_ROW = re.compile(r"^\s*\d{8}\s*,")


def download_prices(tickers: Sequence[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Adjusted daily closes (dividends and splits included), one column per ticker.

    `end` is inclusive here; yfinance treats it as exclusive, so one day is added.
    """
    raw = yf.download(
        list(tickers),
        start=start,
        end=end + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
        multi_level_index=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(tickers), dtype=float)
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes


@dataclass
class CleanPrices:
    prices: pd.DataFrame
    missing: list[str] = field(default_factory=list)  # tickers with no data at all
    late_starters: dict[str, pd.Timestamp] = field(default_factory=dict)  # ticker -> first date
    early_enders: dict[str, pd.Timestamp] = field(default_factory=dict)  # ticker -> last date


def clean_prices(raw: pd.DataFrame, tickers: Sequence[str]) -> CleanPrices:
    """Keep the dates on which every ticker has a price.

    - Tickers with no data are reported in `missing` and dropped.
    - The window is trimmed to start when the last ticker starts trading and
      end when the first one stops, so no asset is assumed flat outside its history.
    - Gaps inside the window (e.g. a holiday on one exchange only) are forward-filled.
    """
    missing = [t for t in tickers if t not in raw.columns or raw[t].dropna().empty]
    present = [t for t in tickers if t not in missing]
    if not present:
        return CleanPrices(pd.DataFrame(), missing)

    prices = raw[present].dropna(how="all")
    first = {t: prices[t].first_valid_index() for t in present}
    last = {t: prices[t].last_valid_index() for t in present}
    start, end = max(first.values()), min(last.values())
    late = {t: d for t, d in first.items() if d > prices.index[0]}
    early = {t: d for t, d in last.items() if d < prices.index[-1]}

    prices = prices.loc[start:end].ffill().dropna()
    return CleanPrices(prices, missing, late, early)


def risk_free_from_irx(irx: pd.Series) -> float | None:
    """Average annual risk-free rate (decimal) from ^IRX closes, which are quoted in percent."""
    irx = irx.dropna()
    if irx.empty:
        return None
    return float(irx.mean()) / 100


def fetch_risk_free_rate(start: dt.date, end: dt.date) -> float | None:
    """Average 13-week T-bill yield over the period, or None if it can't be downloaded."""
    try:
        prices = download_prices([RISK_FREE_TICKER], start, end)
    except Exception:
        return None
    if RISK_FREE_TICKER not in prices.columns:
        return None
    return risk_free_from_irx(prices[RISK_FREE_TICKER])


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse one table from a Ken French data library CSV.

    The files start with free-text notes, then a header row beginning with a
    comma, then rows of `YYYYMMDD, value, ...` in percent, then a copyright
    footer. Returns decimals (percent / 100) indexed by date; missing-value
    markers become NaN.
    """
    lines = text.splitlines()
    header_at = next((i for i, line in enumerate(lines) if line.startswith(",")), None)
    if header_at is None:
        raise ValueError("No header row found in the factor file.")
    columns = [c.strip() for c in lines[header_at].split(",")[1:]]

    rows = []
    for line in lines[header_at + 1 :]:
        if not _DATE_ROW.match(line):
            break
        rows.append([cell.strip() for cell in line.split(",")])
    if not rows:
        raise ValueError("No data rows found in the factor file.")

    table = pd.DataFrame([r[1:] for r in rows], columns=columns, dtype=float)
    table.index = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    table = table.mask(table <= FRENCH_MISSING, np.nan)
    return table / 100


def read_french_zip(content: bytes) -> pd.DataFrame:
    """Parse the CSV inside a Ken French zip file."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(name).decode("latin-1")
    return parse_french_csv(text)


def download_french_factors(region: str, timeout: float = 30) -> pd.DataFrame:
    """Daily Fama-French five factors plus momentum and the risk-free rate, as decimals.

    Columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF. Only dates present in both
    files and with no missing values are kept.
    """
    if region not in FRENCH_FILES:
        raise ValueError(f"Unknown factor region: {region}")
    tables = []
    for filename in FRENCH_FILES[region]:
        response = requests.get(FRENCH_BASE_URL + filename, timeout=timeout)
        response.raise_for_status()
        tables.append(read_french_zip(response.content))
    five, momentum = tables
    momentum.columns = ["Mom"]  # "Mom" in the US file, "WML" in the regional ones
    return five.join(momentum, how="inner")[FRENCH_COLUMNS].dropna()
