"""Streamlit UI for portfolio analytics. All finance logic lives in the `portfolio` package."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio import data, factors, metrics, values
from portfolio.holdings import parse_holdings
from portfolio.presets import COMPETITION_BENCHMARK, DEFAULT_BENCHMARK, PRESETS

BENCHMARKS = {
    "^SP500TR": "S&P 500 Total Return (includes dividends)",
    "^GSPC": "S&P 500 price index (excludes dividends)",
    "SPY": "SPDR S&P 500 ETF (includes dividends, net of fees)",
    COMPETITION_BENCHMARK: "iShares MSCI World EUR Hedged (Acc), competition benchmark",
}
CUSTOM = "Custom"
VALUES_MODE = "Portfolio values (CSV)"
COMPETITION_VALUES_CSV = Path(__file__).parent / "data" / "competition_values.csv"
OTHER_BENCHMARK = "Other ticker…"
PORTFOLIO_COLOR = "#2563eb"
BENCHMARK_COLOR = "#9ca3af"
MOMENTUM_COLOR = "#f59e0b"
SHORT_FACTOR_NAMES = {
    "Mkt-RF": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "Mom": "Momentum",
}

st.set_page_config(page_title="Portfolio Analytics", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading prices…")
def load_prices(tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    return data.download_prices(tickers, start, end)


@st.cache_data(ttl=3600, show_spinner="Downloading T-bill yields…")
def load_risk_free(start: dt.date, end: dt.date) -> float | None:
    return data.fetch_risk_free_rate(start, end)


def pct(x: float, digits: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{digits}%}"


def num(x: float, digits: int = 2) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


def date_str(d: pd.Timestamp | None) -> str:
    return "—" if d is None else d.strftime("%d %b %Y")


def is_us_listing(ticker: str) -> bool:
    """Yahoo gives non-US listings an exchange suffix (ASML.AS, IWDE.L); US tickers and indices have none."""
    return "." not in ticker


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading Fama-French factors…")
def load_factors(region: str) -> pd.DataFrame:
    return data.download_french_factors(region)


def render_factor_tab(port_r: pd.Series | None = None, snapshots: pd.Series | None = None) -> None:
    """Fama-French five-factor plus momentum regression of the portfolio's excess returns.

    Pass daily portfolio returns (`port_r`), or portfolio value snapshots (`snapshots`), which are
    always analysed as weekly-equivalent returns over the snapshot intervals.
    """
    c1, c2, c3 = st.columns(3)
    region = c1.selectbox(
        "Factor region",
        list(data.FRENCH_FILES),
        help="Use Developed Europe factors if most of your holdings are European stocks.",
    )
    if snapshots is not None:
        frequency = "weekly"
        c2.markdown("**Return frequency**  \nWeekly-equivalent, over your snapshot intervals")
    else:
        frequency = c2.radio(
            "Return frequency",
            ["Daily", "Weekly"],
            horizontal=True,
            help="Weekly returns reduce the distortion from markets closing at different times "
            "(e.g. European stocks against US factors). See the README.",
        ).lower()
    # Newey-West estimates autocorrelation from the residuals; with only a handful of observations
    # it can't, and its standard errors come out far too small. Default to plain OLS then.
    few_snapshots = snapshots is not None and len(snapshots) - 1 < values.MIN_OBSERVATIONS
    cov_type = (
        "newey-west"
        if c3.radio(
            "Standard errors",
            ["Newey-West", "Plain OLS"],
            index=1 if few_snapshots else 0,
            horizontal=True,
            key=f"cov-type-{few_snapshots}",
            help="Newey-West corrects t-statistics for volatility clustering and autocorrelation. "
            "The coefficients are the same either way. With fewer than "
            f"{values.MIN_OBSERVATIONS} observations Newey-West is unreliable (it overstates significance), "
            "so plain OLS is the default then.",
        )
        == "Newey-West"
        else "ols"
    )

    if region == "Developed Europe":
        st.warning(
            "Kenneth French's Developed Europe factors are measured in **US dollars**, but Yahoo Finance prices "
            "European stocks in their local currency (EUR, GBP, CHF…). Currency moves will show up as "
            "unexplained noise, and may distort the market beta.",
            icon="💱",
        )

    try:
        factor_data = load_factors(region)
    except Exception as exc:  # network or format problems shouldn't break the rest of the app
        st.error(f"Couldn't download the {region} factors from the Kenneth French data library: {exc}")
        return

    if snapshots is not None:
        aligned = values.align_with_factors(snapshots, factor_data)
        minimum = values.MIN_REGRESSION_OBSERVATIONS
        unit = "snapshots"
    else:
        aligned = factors.align_with_factors(port_r, factor_data, frequency)
        minimum = factors.MIN_OBSERVATIONS[frequency]
        unit = "trading days"
    left_out = []
    if aligned.days_after_factor_data:
        left_out.append(
            f"Factor data currently ends on **{date_str(aligned.last_factor_date)}** (the library is updated "
            f"monthly with a lag), so the last {aligned.days_after_factor_data} {unit} of your date range "
            "are left out."
        )
    if aligned.days_dropped_mid_sample:
        left_out.append(
            f"**{aligned.days_dropped_mid_sample}** days of your portfolio's returns fall on dates with no "
            f"{region} factor data (usually holidays in that region while another exchange in your analysis "
            "was open) "
            "and are left out of the daily regression. Weekly returns keep them."
        )
    if left_out:
        st.warning(" ".join(left_out))
    if snapshots is not None and minimum <= aligned.n < values.MIN_OBSERVATIONS:
        st.warning(
            f"Only **{aligned.n}** intervals overlap with the factor data, fewer than {values.MIN_OBSERVATIONS}. "
            "Seven numbers (alpha and six loadings) are being estimated from very few observations, so "
            "expect wide confidence intervals and treat the results as rough."
        )
    if aligned.n < minimum:
        st.info(
            f"Only {aligned.n} {frequency} observations overlap with the factor data; at least {minimum} are "
            "needed for a meaningful regression. Choose an earlier start date."
        )
        return

    result = factors.run_factor_regression(aligned, frequency, cov_type, min_observations=minimum)

    cards = st.columns(4)
    alpha_sig = result.table.loc["Alpha", "significant"]
    cards[0].metric(
        "Alpha (annualised)",
        pct(result.alpha_annual),
        "significant at 5%" if alpha_sig else "not significant",
        delta_color="normal" if alpha_sig else "off",
        delta_arrow="auto" if alpha_sig else "off",
    )
    cards[1].metric("R²", num(result.r_squared))
    cards[2].metric("Adjusted R²", num(result.adj_r_squared))
    cards[3].metric("Observations", f"{result.n:,} {frequency}")

    table = result.table
    shown = pd.DataFrame(
        {
            "Factor": [factors.FACTOR_NAMES[t] for t in table.index],
            "Coefficient": [
                pct(v) if t == "Alpha" else num(v, 3) for t, v in table["coefficient"].items()
            ],
            "t-statistic": table["t_stat"].map(lambda v: num(v)),
            "p-value": table["p_value"].map(lambda v: f"{v:.3f}"),
            "95% interval": [
                f"{pct(lo)} to {pct(hi)}" if t == "Alpha" else f"{lo:.2f} to {hi:.2f}"
                for t, lo, hi in zip(table.index, table["ci_low"], table["ci_high"])
            ],
            "Significant at 5%": table["significant"].map({True: "✓", False: "✗"}),
        }
    )
    st.dataframe(shown, hide_index=True, width="stretch")
    st.markdown(" ".join(factors.describe_exposures(result)))
    st.caption(
        f"OLS of {'weekly-equivalent' if snapshots is not None else frequency} portfolio return minus the risk-free rate (French's RF) on the factors, "
        f"{date_str(aligned.excess.index[0])} – {date_str(aligned.excess.index[-1])}. "
        + (
            f"Newey-West standard errors with {factors.NEWEY_WEST_LAGS[frequency]} lags."
            if cov_type == "newey-west"
            else "Plain OLS standard errors."
        )
    )

    left, right = st.columns(2)
    with left:
        loadings = result.loadings
        fig = go.Figure(
            go.Bar(
                x=[SHORT_FACTOR_NAMES[t] for t in loadings.index],
                y=loadings["coefficient"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=loadings["ci_high"] - loadings["coefficient"],
                    arrayminus=loadings["coefficient"] - loadings["ci_low"],
                ),
                marker_color=[PORTFOLIO_COLOR if s else BENCHMARK_COLOR for s in loadings["significant"]],
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            )
        )
        fig.update_layout(title="Factor loadings (95% intervals)")
        fig.add_hline(y=0, line_width=1, line_color=BENCHMARK_COLOR)
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue bars are significant at 5%; grey bars are not.")

    with right:
        if snapshots is not None:
            st.info("Rolling 12-month exposure isn't available for irregular snapshots.")
            return
        rolling = factors.rolling_loadings(aligned, frequency)
        window = factors.ROLLING_WINDOW[frequency]
        if rolling.empty:
            st.info(
                f"The rolling chart needs at least 12 months ({window} {frequency} observations) of overlapping "
                f"data; there are {aligned.n}."
            )
        else:
            fig = go.Figure()
            fig.add_scatter(x=rolling.index, y=rolling["Mkt-RF"], name="Market beta", line_color=PORTFOLIO_COLOR)
            fig.add_scatter(x=rolling.index, y=rolling["Mom"], name="Momentum loading", line_color=MOMENTUM_COLOR)
            fig.add_hline(y=0, line_width=1, line_color=BENCHMARK_COLOR)
            fig.update_layout(
                title="Rolling 12-month exposure",
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.15, x=0),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("Each point is the full six-factor regression over the 12 months ending on that date.")


def render_summary(
    port: metrics.Summary,
    bench: metrics.Summary,
    benchmark: str,
    beta_label: str,
    extra_cards: dict[str, str],
) -> None:
    """Metric cards (portfolio, with differences from the benchmark) and the comparison table."""
    cards = [*st.columns(4), *st.columns(4)]
    cards[0].metric(
        "Annualised return", pct(port.annualised_return), f"{port.annualised_return - bench.annualised_return:+.2%} vs bm"
    )
    cards[1].metric(
        "Volatility",
        pct(port.annualised_volatility),
        f"{port.annualised_volatility - bench.annualised_volatility:+.2%} vs bm",
        delta_color="inverse",
    )
    cards[2].metric("Sharpe ratio", num(port.sharpe), f"{port.sharpe - bench.sharpe:+.2f} vs bm")
    cards[3].metric("M²", pct(port.m_squared), f"{port.m_squared_excess:+.2%} vs bm")
    cards[4].metric(beta_label, num(port.beta))
    cards[5].metric(
        "Max drawdown",
        pct(port.drawdown.max_drawdown),
        f"{port.drawdown.max_drawdown - bench.drawdown.max_drawdown:+.2%} vs bm",
    )
    for card, (label, value) in zip(cards[6:], extra_cards.items()):
        card.metric(label, value)

    with st.expander("Portfolio vs benchmark table"):

        def row(s: metrics.Summary) -> dict[str, str]:
            return {
                "Annualised return": pct(s.annualised_return),
                "Annualised volatility": pct(s.annualised_volatility),
                "Sharpe ratio": num(s.sharpe),
                "M²": pct(s.m_squared),
                "M² vs benchmark": pct(s.m_squared_excess),
                "Beta": num(s.beta),
                "Max drawdown": pct(s.drawdown.max_drawdown),
                "Drawdown peak": date_str(s.drawdown.peak),
                "Drawdown low": date_str(s.drawdown.trough),
                "Recovered": date_str(s.drawdown.recovery) if s.drawdown.peak else "—",
            }

        st.dataframe(
            pd.DataFrame({"Portfolio": row(port), f"Benchmark ({benchmark})": row(bench)}), width="stretch"
        )


def render_values_page(upload, benchmark: str, use_tbill: bool, manual_rf: float) -> None:
    """Performance and factor analysis from portfolio value snapshots (uploaded, or the competition file)."""
    source = upload if upload is not None else COMPETITION_VALUES_CSV
    try:
        snapshots = values.read_values_csv(source)
    except ValueError as exc:
        st.error(f"Couldn't read the portfolio values: {exc}")
        return
    if not benchmark:
        st.info("Enter a benchmark ticker in the sidebar.")
        return

    first, last = snapshots.index[0], snapshots.index[-1]
    # Start a little early so a snapshot on a weekend or holiday can use the previous close.
    raw = load_prices((benchmark,), (first - pd.Timedelta(days=10)).date(), last.date())
    cleaned = data.clean_prices(raw, [benchmark])
    if cleaned.missing:
        st.error(f"No price data for the benchmark {benchmark}.")
        return
    bench_prices = cleaned.prices[benchmark]
    try:
        bench_at = values.values_at(bench_prices, snapshots.index).rename(benchmark)
    except ValueError as exc:
        st.error(f"The benchmark doesn't cover your first snapshot. {exc}")
        return

    if use_tbill:
        rf = load_risk_free(first.date(), last.date())
        if rf is None:
            st.warning("Couldn't download T-bill yields; using the manual risk-free rate instead.")
            rf = manual_rf / 100
    else:
        rf = manual_rf / 100

    intervals = values.interval_returns(snapshots)
    n = len(intervals)
    source_name = "uploaded file" if upload is not None else "competition values"
    st.caption(
        f"{date_str(first)} – {date_str(last)} · {len(snapshots)} snapshots, {n} intervals of "
        f"{intervals['days'].min()}–{intervals['days'].max()} days · benchmark {benchmark} at the same dates · "
        f"risk-free rate {rf:.2%} · source: {source_name}"
    )
    if n < values.MIN_OBSERVATIONS:
        st.warning(
            f"Only **{n}** intervals between snapshots, fewer than {values.MIN_OBSERVATIONS}. Volatility, Sharpe, "
            "M², beta and the factor regression are estimated from very few observations, so treat them as rough."
        )

    perf_tab, factor_tab = st.tabs(["Performance", "Factor exposure"])

    with perf_tab:
        port = values.summarise(snapshots, bench_at, rf)
        bench = values.summarise(bench_at, bench_at, rf)
        render_summary(
            port,
            bench,
            benchmark,
            "Beta",
            {
                "Total return": pct(snapshots.iloc[-1] / snapshots.iloc[0] - 1),
                "Correlation with bm": num(values.correlation(snapshots, bench_at)),
            },
        )

        bench_daily = bench_prices.loc[first:last]
        fig = go.Figure()
        fig.add_scatter(
            x=bench_daily.index,
            y=bench_daily / bench_at.iloc[0] - 1,
            name=f"{benchmark} (daily)",
            line_color=BENCHMARK_COLOR,
        )
        fig.add_scatter(
            x=snapshots.index,
            y=snapshots / snapshots.iloc[0] - 1,
            name="Portfolio (snapshots)",
            mode="lines+markers",
            line_color=PORTFOLIO_COLOR,
        )
        fig.update_layout(title="Cumulative return", hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_scatter(
                x=bench_at.index,
                y=metrics.drawdown_series(bench_at),
                name=benchmark,
                mode="lines+markers",
                line_color=BENCHMARK_COLOR,
            )
            fig.add_scatter(
                x=snapshots.index,
                y=metrics.drawdown_series(snapshots),
                name="Portfolio",
                mode="lines+markers",
                line_color=PORTFOLIO_COLOR,
                fill="tozeroy",
            )
            fig.update_layout(
                title="Drawdown at snapshot dates", hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0)
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("Both are measured only at your snapshot dates, so falls in between aren't seen.")
        with right:
            bench_intervals = values.interval_returns(bench_at)
            table = pd.DataFrame(
                {
                    "From": intervals["start"].dt.strftime("%d %b"),
                    "To": intervals.index.strftime("%d %b"),
                    "Days": intervals["days"],
                    "Return": intervals["return"].map(pct),
                    "Weekly-equivalent": values.weekly_equivalent(intervals["return"], intervals["days"]).map(pct),
                    f"{benchmark} return": bench_intervals["return"].map(pct),
                }
            )
            st.markdown("**Returns between snapshots**")
            st.dataframe(table, hide_index=True, width="stretch")

        st.caption(
            "Returns are annualised over calendar time; risk is measured on time-scaled log returns between "
            "snapshots. See README: Portfolio values mode."
        )

    with factor_tab:
        render_factor_tab(snapshots=snapshots)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Portfolio")
    preset_name = st.selectbox(
        "Portfolio input",
        [CUSTOM, *PRESETS, VALUES_MODE],
        help=f"Enter tickers and weights, load a preset, or choose '{VALUES_MODE}' to analyse a series of "
        "portfolio values on any dates.",
    )
    preset = PRESETS.get(preset_name)
    values_mode = preset_name == VALUES_MODE
    values_upload = None

    if values_mode:
        values_upload = st.file_uploader(
            "Portfolio values CSV",
            type="csv",
            help="Columns: date (YYYY-MM-DD) and value. Dates can be irregular.",
        )
        if values_upload is None:
            st.caption("No file uploaded: using the competition values in data/competition_values.csv.")
    else:
        if preset is not None and not preset.is_filled:
            st.info(preset.notes or "This preset is empty.")
            initial = pd.DataFrame({"Ticker": [""], "Weight (%)": [None]})
        elif preset is not None:
            initial = pd.DataFrame({"Ticker": list(preset.weights), "Weight (%)": list(preset.weights.values())})
        else:
            initial = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "JNJ"], "Weight (%)": [40.0, 40.0, 20.0]})

        holdings_table = st.data_editor(
            initial,
            key=f"holdings-{preset_name}",  # new key per preset, so switching reloads the table
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Ticker": st.column_config.TextColumn(help="Yahoo Finance ticker, e.g. AAPL or ASML.AS"),
                "Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.2f"),
            },
        )

        today = dt.date.today()
        default_start = (preset.start if preset and preset.start else None) or today.replace(year=today.year - 3)
        default_end = (preset.end if preset and preset.end else None) or today
        start = st.date_input("Start date", default_start, max_value=today, key=f"start-{preset_name}")
        end = st.date_input("End date", default_end, max_value=today, key=f"end-{preset_name}")

    st.header("Settings")
    preset_benchmark = COMPETITION_BENCHMARK if values_mode else preset.benchmark if preset else DEFAULT_BENCHMARK
    benchmark_options = list(dict.fromkeys([preset_benchmark, *BENCHMARKS, OTHER_BENCHMARK]))
    benchmark = st.selectbox(
        "Benchmark",
        benchmark_options,
        format_func=lambda t: BENCHMARKS.get(t, t),
        help=f"Pick one, or choose '{OTHER_BENCHMARK}' to enter any Yahoo Finance ticker.",
        key=f"benchmark-{preset_name}",
    )
    if benchmark == OTHER_BENCHMARK:
        benchmark = st.text_input("Benchmark ticker", placeholder="e.g. URTH or IWDA.AS").strip().upper()

    rebalance = not values_mode and (  # not used in values mode
        st.radio(
            "Rebalancing",
            ["Buy and hold", "Rebalance daily"],
            help="Buy and hold sets the weights on the start date and lets them drift. "
            "Rebalance daily resets them to target every day.",
        )
        == "Rebalance daily"
    )

    use_tbill = st.checkbox("Risk-free rate from 13-week T-bill", value=True)
    manual_rf = st.number_input(
        "Risk-free rate (% a year)", value=4.0, step=0.25, disabled=use_tbill, format="%.2f"
    )
    include_bm = not values_mode and st.checkbox("Include benchmark in correlations", value=True)

# ---------------------------------------------------------------- validation

st.title("Portfolio Analytics")

if values_mode:
    render_values_page(values_upload, benchmark, use_tbill, manual_rf)
    st.stop()

if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

holdings = parse_holdings(holdings_table)
if holdings.errors:
    for message in holdings.errors:
        st.error(message)
    st.stop()

weights = holdings.weights
if not holdings.sums_to_one:
    st.warning(f"Your weights add up to {holdings.total_percent:.2f}%, not 100%.")
    if not st.toggle("Scale weights so they add to 100%", value=False):
        st.stop()
    weights = metrics.normalise_weights(weights)

tickers = list(weights.index)
if not benchmark:
    st.info("Enter a benchmark ticker in the sidebar.")
    st.stop()
if benchmark in tickers:
    st.error(f"{benchmark} is both a holding and the benchmark; pick a different benchmark.")
    st.stop()

raw = load_prices(tuple([*tickers, benchmark]), start, end)
cleaned = data.clean_prices(raw, [*tickers, benchmark])

if cleaned.missing:
    st.error(
        f"No price data for: {', '.join(cleaned.missing)}. "
        "Check the ticker on finance.yahoo.com; non-US listings need a suffix such as .L or .AS."
    )
    st.stop()
if len(cleaned.prices) < 3:
    st.error("Not enough overlapping price history in this date range.")
    st.stop()
for ticker, first in cleaned.late_starters.items():
    st.info(f"{ticker} has no prices before {date_str(first)}, so the analysis starts then.")
for ticker, last in cleaned.early_enders.items():
    st.info(f"{ticker} has no prices after {date_str(last)}, so the analysis ends then.")

prices = cleaned.prices
returns = metrics.daily_returns(prices)
port_r = metrics.portfolio_returns(returns[tickers], weights, rebalance=rebalance)
bench_r = returns[benchmark].rename(benchmark)
first_day = prices.index[0]
periods_per_year = metrics.observed_periods_per_year(returns.index, first_day)

if use_tbill:
    rf = load_risk_free(prices.index[0].date(), prices.index[-1].date())
    if rf is None:
        st.warning("Couldn't download T-bill yields; using the manual risk-free rate instead.")
        rf = manual_rf / 100
else:
    rf = manual_rf / 100

st.caption(
    f"{date_str(prices.index[0])} – {date_str(prices.index[-1])} · {len(returns)} days of returns "
    f"(≈{periods_per_year:.0f} a year) · "
    f"benchmark {benchmark} · risk-free rate {rf:.2%} · {'rebalanced daily' if rebalance else 'buy and hold'}"
)

perf_tab, factor_tab = st.tabs(["Performance", "Factor exposure"])

with perf_tab:
    # ---------------------------------------------------------------- metrics

    mixed_listings = len({is_us_listing(t) for t in [*tickers, benchmark]}) > 1
    beta_frequency = (
        st.radio(
            "Beta and correlations from",
            ["Daily returns", "Weekly returns"],
            index=1 if mixed_listings else 0,
            horizontal=True,
            key=f"beta-frequency-{mixed_listings}",  # re-default when the mix of listings changes
            help="Markets close at different times, so daily returns understate beta and correlations "
            "between, say, US stocks and a Europe-listed fund. Weekly returns fix this at the cost of "
            "fewer observations. Defaults to weekly when US and non-US listings are mixed.",
        )
        .split()[0]
        .lower()
    )

    port = metrics.summarise(port_r, bench_r, rf, first_day, periods_per_year, beta_frequency)
    bench = metrics.summarise(bench_r, bench_r, rf, first_day, periods_per_year, beta_frequency)

    render_summary(
        port,
        bench,
        benchmark,
        f"Beta ({beta_frequency})",
        {"Total return": pct(metrics.cumulative_returns(port_r).iloc[-1])},
    )

    # ---------------------------------------------------------------- charts

    cumulative = pd.DataFrame(
        {
            "Portfolio": metrics.cumulative_returns(port_r, first_day),
            benchmark: metrics.cumulative_returns(bench_r, first_day),
        }
    )
    fig = px.line(
        cumulative,
        labels={"index": "", "value": "Cumulative return", "variable": ""},
        color_discrete_sequence=[PORTFOLIO_COLOR, BENCHMARK_COLOR],
        title="Cumulative return",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)

    with left:
        drawdowns = pd.DataFrame(
            {
                "Portfolio": metrics.drawdown_series(metrics.wealth_index(port_r, first_day)),
                benchmark: metrics.drawdown_series(metrics.wealth_index(bench_r, first_day)),
            }
        )
        fig = go.Figure()
        fig.add_scatter(
            x=drawdowns.index, y=drawdowns[benchmark], name=benchmark, line_color=BENCHMARK_COLOR
        )
        fig.add_scatter(
            x=drawdowns.index,
            y=drawdowns["Portfolio"],
            name="Portfolio",
            line_color=PORTFOLIO_COLOR,
            fill="tozeroy",
        )
        fig.update_layout(
            title="Drawdown from previous peak",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    with right:
        corr_input = returns[[*tickers, benchmark]] if include_bm else returns[tickers]
        if beta_frequency == "weekly":
            corr_input = metrics.to_weekly(corr_input)
        corr = metrics.correlation_matrix(corr_input)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title=f"Correlation of {beta_frequency} returns",
        )
        st.plotly_chart(fig, width="stretch")

    st.caption(
        "Prices: Yahoo Finance via yfinance (adjusted for splits and dividends). "
        "Past performance does not predict future results. See README for how each metric is calculated."
    )

with factor_tab:
    render_factor_tab(port_r)
