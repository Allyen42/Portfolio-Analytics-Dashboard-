import pandas as pd
import pytest

from portfolio.holdings import parse_holdings


def table(rows):
    return pd.DataFrame(rows, columns=["Ticker", "Weight (%)"])


def test_parses_and_cleans_tickers():
    parsed = parse_holdings(table([[" aapl ", 60], ["MSFT", 40], ["", None], [None, None]]))
    assert parsed.errors == []
    assert parsed.weights.to_dict() == pytest.approx({"AAPL": 0.6, "MSFT": 0.4})
    assert parsed.sums_to_one


def test_reports_weights_not_summing_to_100():
    parsed = parse_holdings(table([["AAPL", 50], ["MSFT", 30]]))
    assert parsed.errors == []
    assert not parsed.sums_to_one
    assert parsed.total_percent == pytest.approx(80)


@pytest.mark.parametrize(
    "rows, message",
    [
        ([], "at least one ticker"),
        ([["AAPL", None]], "AAPL has no weight"),
        ([["AAPL", -10], ["MSFT", 110]], "negative weight"),
        ([["AAPL", 50], ["aapl", 50]], "more than once: AAPL"),
        ([["AAPL", 0]], "more than 0%"),
    ],
)
def test_errors(rows, message):
    parsed = parse_holdings(table(rows))
    assert any(message in e for e in parsed.errors)
