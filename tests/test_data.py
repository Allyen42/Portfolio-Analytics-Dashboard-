import datetime as dt

import numpy as np
import pandas as pd
import pytest

from portfolio import data

DATES = pd.bdate_range("2024-01-01", periods=6)


def test_clean_prices_reports_missing_and_trims_to_common_window():
    raw = pd.DataFrame(
        {
            "A": [10, 11, 12, 13, 14, 15],
            "B": [np.nan, np.nan, 20, np.nan, 22, 23],  # starts late, one gap
            "C": [5, 5, 5, 5, 5, np.nan],  # stops early
            "D": [np.nan] * 6,  # no data
        },
        index=DATES,
        dtype=float,
    )
    result = data.clean_prices(raw, ["A", "B", "C", "D", "E"])
    assert result.missing == ["D", "E"]
    assert list(result.prices.columns) == ["A", "B", "C"]
    assert result.prices.index[0] == DATES[2]
    assert result.prices.index[-1] == DATES[4]
    assert result.prices.loc[DATES[3], "B"] == 20  # gap forward-filled
    assert result.late_starters == {"B": DATES[2]}
    assert result.early_enders == {"C": DATES[4]}


def test_clean_prices_with_nothing_usable():
    result = data.clean_prices(pd.DataFrame(index=DATES), ["X"])
    assert result.missing == ["X"]
    assert result.prices.empty


def test_download_prices_requests_inclusive_end(monkeypatch):
    calls = {}
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]])
    frame = pd.DataFrame(np.ones((3, 4)), index=DATES[:3], columns=columns)

    def fake_download(tickers, **kwargs):
        calls.update(kwargs, tickers=tickers)
        return frame

    monkeypatch.setattr(data.yf, "download", fake_download)
    prices = data.download_prices(["AAA", "BBB"], dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert calls["end"] == dt.date(2024, 1, 4)
    assert calls["auto_adjust"] is True
    assert list(prices.columns) == ["AAA", "BBB"]


def test_download_prices_empty_result(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: pd.DataFrame())
    prices = data.download_prices(["AAA"], dt.date(2024, 1, 1), dt.date(2024, 1, 3))
    assert prices.empty
    assert list(prices.columns) == ["AAA"]


def test_risk_free_from_irx_converts_percent():
    assert data.risk_free_from_irx(pd.Series([4.0, 5.0, np.nan])) == pytest.approx(0.045)
    assert data.risk_free_from_irx(pd.Series([], dtype=float)) is None


def test_fetch_risk_free_rate_survives_download_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(data.yf, "download", boom)
    assert data.fetch_risk_free_rate(dt.date(2024, 1, 1), dt.date(2024, 2, 1)) is None


FRENCH_CSV = """This file was created using the 202608 database.
Missing data are indicated by -99.99 or -999.

,Mkt-RF,SMB,Mom
20260828   ,-0.34    ,1.00   ,-99.99
20260831,   0.50,   -0.25,    -999

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_csv_converts_percent_and_missing():
    table = data.parse_french_csv(FRENCH_CSV)
    assert list(table.columns) == ["Mkt-RF", "SMB", "Mom"]
    assert list(table.index) == [pd.Timestamp("2026-08-28"), pd.Timestamp("2026-08-31")]
    assert table.loc["2026-08-28", "Mkt-RF"] == pytest.approx(-0.0034)
    assert table.loc["2026-08-31", "SMB"] == pytest.approx(-0.0025)
    assert table["Mom"].isna().all()


def test_parse_french_csv_rejects_unexpected_format():
    with pytest.raises(ValueError, match="header"):
        data.parse_french_csv("not a factor file")


def _zip(name, text):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def test_download_french_factors_joins_files_and_renames_momentum(monkeypatch):
    five = ",Mkt-RF,SMB,HML,RMW,CMA,RF\n20260827,1,2,3,4,5,0.01\n20260828,1,2,3,4,5,0.01\n"
    momentum = ",WML\n20260828,0.5\n20260831,0.7\n"
    files = {
        "Europe_5_Factors_Daily_CSV.zip": _zip("Europe_5_Factors_Daily.csv", five),
        "Europe_Mom_Factor_Daily_CSV.zip": _zip("Europe_MOM_Factor_Daily.csv", momentum),
    }

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        return Response(files[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(data.requests, "get", fake_get)
    factors = data.download_french_factors("Developed Europe")
    assert list(factors.columns) == data.FRENCH_COLUMNS
    assert list(factors.index) == [pd.Timestamp("2026-08-28")]  # only the shared date
    assert factors.loc["2026-08-28", "Mom"] == pytest.approx(0.005)


def test_download_french_factors_unknown_region():
    with pytest.raises(ValueError, match="Unknown factor region"):
        data.download_french_factors("Mars")
