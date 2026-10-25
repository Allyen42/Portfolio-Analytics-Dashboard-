# Portfolio Analytics Dashboard

A Streamlit app that analyses a stock portfolio against a benchmark. Enter tickers, weights and a date
range; the app downloads prices from Yahoo Finance and shows:

- cumulative return of the portfolio vs the benchmark (default: S&P 500 Total Return)
- annualised return and volatility
- Sharpe ratio and M² (Modigliani risk-adjusted return)
- beta
- maximum drawdown (with a drawdown chart)
- a correlation heatmap of the holdings
- **factor exposure**: a Fama-French five-factor plus momentum regression, with alpha, loadings,
  rolling betas and a plain-English summary
- **portfolio values mode**: all of the above from a CSV of portfolio values on irregular dates,
  when you don't have the holdings

## Getting started

Requires Python 3.11 or newer.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

(On macOS/Linux: `source .venv/bin/activate`.)

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Run the tests (they don't need an internet connection):

```bash
pytest
```

## Project layout

```
app.py                  Streamlit UI only: inputs, charts, metric cards
portfolio/
  metrics.py            the finance calculations (pure functions, no UI, no network)
  factors.py            the factor regression, rolling loadings and summary text (also pure)
  values.py             metrics and factor alignment for irregular value snapshots (also pure)
  data.py               downloading prices (Yahoo Finance) and factors (Kenneth French library)
  holdings.py           validating the ticker/weight table
  presets.py            saved portfolios, including the student competition preset
data/
  competition_values.csv  the Midas competition portfolio values
tests/                  pytest tests for metrics, factors, values, data parsing and holdings
```

The finance logic in `portfolio/metrics.py` takes pandas Series/DataFrames and returns numbers, so you can
reuse it in a notebook or a report without Streamlit.

## Filling in the student competition preset

Open `portfolio/presets.py` and fill in the `"Student competition"` entry:

```python
"Student competition": Preset(
    weights={"AAPL": 25, "MSFT": 25, "ASML.AS": 50},  # percentages, adding to 100
    start=dt.date(2026, 1, 15),                       # competition start
    end=None,                                         # None = today
    benchmark=COMPETITION_BENCHMARK,                  # already set, see below
),
```

Use Yahoo Finance tickers. Non-US listings need an exchange suffix: `.L` (London), `.AS` (Amsterdam),
`.PA` (Paris), `.DE` (Xetra), `.TO` (Toronto) and so on. Search the ticker on finance.yahoo.com if unsure.

## Settings

**Benchmark.** The default is `^SP500TR`, the S&P 500 *Total Return* index, which includes reinvested
dividends. The app uses dividend-adjusted prices for your holdings, so comparing them with `^GSPC` (the
S&P 500 *price* index, no dividends) would flatter your portfolio by roughly the index's dividend yield,
about 1.5% a year. `SPY` (the ETF) also includes dividends, minus a small fee. You can type any other
Yahoo ticker too.

**Competition benchmark.** The student competition preset uses the competition's benchmark, the
*iShares MSCI World EUR Hedged UCITS ETF (Acc)* (ISIN IE00B441G979), on Euronext Amsterdam as `IWDE.AS`.
The same fund also trades as `IBCH.DE` (Xetra), `IWDE.L` (London) and `IWDE.MI` (Milan), all in EUR, with
nearly identical returns. It's in the benchmark list for custom portfolios too. Things to bear in mind:

- *It reinvests dividends* (Acc share class), so it's a fair comparison with the dividend-adjusted prices
  of your holdings.
- *It's EUR-hedged.* The fund owns stocks worldwide (about 70% US) but hedges the currency back to euros,
  so its return is roughly the stocks' local-currency return plus the euro–dollar interest-rate gap. Your
  holdings' returns are in their own trading currencies (see Limitations), so a USD stock and the
  benchmark aren't measured in quite the same terms.
- *It trades in European hours.* The ETF's daily close is at 17:30 Amsterdam time, while US stocks close
  4½ hours later. Daily beta and correlations between US holdings and this benchmark are therefore
  understated. With a sample of US stocks, daily beta came out at 0.39 and weekly beta at 0.89. Use
  weekly returns for beta and correlations (the app does this by default when US and non-US listings are
  mixed); see *Beta* below.
- *Risk-free rate.* For a euro benchmark, a euro rate (e.g. the ECB deposit rate or €STR) is a better
  risk-free rate than US T-bills. Untick the T-bill box and enter it manually.

**Risk-free rate.** By default this is the average yield on 13-week US Treasury bills (`^IRX`) over your
chosen dates. Untick the box to enter a fixed rate, e.g. one set by your competition rules.

**Rebalancing.**
- *Buy and hold* (default): you buy the target weights on the start date and never trade again, so
  winners grow into a larger share of the portfolio. This is how most competition portfolios behave.
- *Rebalance daily*: weights are reset to target every day. Useful as a theoretical comparison.

## The metrics

Notation: `r_t` is the return on day `t` (0.01 = 1%), `n` is the number of days of returns, `rf` is the
annual risk-free rate, and `N` is the number of days of returns per year. Subscript `p` means portfolio,
`b` means benchmark.

**How `N` is set.** Textbooks use N = 252, the trading days in a US year. That's only right when every
holding trades on the same calendar. The app lines up all prices on the dates *any* of the markets was
open. If you mix exchanges with different holidays (say US stocks and a Dutch ETF), each holiday on one
exchange adds an extra row in which the closed market shows a 0% return. A three-year sample then has about
773 rows instead of 751. So the app measures `N` from the data instead:

```
years = calendar days from the first price to the last price / 365.25
N = n / years
```

For a US-only portfolio this comes out close to 252. The page header shows the value used. Annualised
return then equals growth over actual calendar time, and volatility isn't understated by the extra 0% days.

### Daily and portfolio returns

The percentage change in adjusted closing price from one trading day to the next. Adjusted prices treat
dividends as reinvested and remove the effect of stock splits.

```
r_t = P_t / P_(t-1) − 1
```

With **buy and hold**, the portfolio's value is each holding's growth weighted by its *starting* weight:
`V_t = Σ w_i × Π(1 + r_i)`. Its daily return is `V_t / V_(t-1) − 1`. With **daily rebalancing**, the
portfolio's return is simply `Σ w_i × r_i,t`.

### Cumulative return

How much £1 (or $1) invested on the first day has grown or shrunk by each date.

```
Cumulative return_t = (1 + r_1) × (1 + r_2) × … × (1 + r_t) − 1
```

*Reading it:* 0.35 means the investment is up 35% since the start.

### Annualised return (CAGR)

The constant yearly growth rate that would have taken you from the starting value to the ending value.
It lets you compare periods of different lengths.

```
Annualised return = (1 + total return) ^ (N / n) − 1 = (1 + total return) ^ (1 / years) − 1
```

*Reading it:* +20% over half a year is shown as about +44% a year, because the gain is assumed to keep
compounding at the same pace. Be careful with short periods: a few good weeks annualise into enormous
numbers.

### Annualised volatility

How much the daily returns bounce around their average: the standard deviation of daily returns, scaled
up to a year. It's the most common measure of risk.

```
Volatility = std(r_t) × √N
```

*Reading it:* 20% volatility means that in a typical year, returns land within about ±20 percentage points
of the average roughly two-thirds of the time. The S&P 500 has historically run at about 15–20%.

### Sharpe ratio

How much extra return you earned per unit of risk taken, compared with holding risk-free T-bills.

```
Sharpe = (mean(r_t − rf_daily) × N) / (std(r_t − rf_daily) × √N)
where rf_daily = (1 + rf)^(1/N) − 1
```

*Reading it:* higher is better. Above 1 is generally considered good, and above 2 is excellent (and rare
over long periods). A negative Sharpe means you'd have done better in T-bills. The numerator is the
*arithmetic* average return, which is the textbook convention. That's why it doesn't exactly match the
annualised return (CAGR) card.

### M² (Modigliani risk-adjusted return)

A Sharpe ratio is hard to explain ("0.8 of what?"). M² turns it back into a percentage return: *what would
the portfolio have returned if we'd dialled its risk up or down to match the benchmark's?* (You dial risk
by mixing the portfolio with cash, or borrowing.)

```
M² = Sharpe_p × σ_b + rf
M² vs benchmark = M² − (Sharpe_b × σ_b + rf) = (Sharpe_p − Sharpe_b) × σ_b
```

*Reading it:* if M² vs benchmark is +2%, then at the same level of risk as the S&P 500, your portfolio
would have earned 2 percentage points a year more. Unlike comparing raw returns, a risky portfolio can't
look good just by taking more risk. Because M² ranks portfolios exactly as Sharpe does, the comparison uses
the benchmark's Sharpe-consistent return rather than its CAGR.

### Beta

How strongly the portfolio moves with the benchmark.

```
Beta = Cov(r_p, r_b) / Var(r_b)
```

*Reading it:* a beta of 1.2 means that, on average, when the market moved 1% the portfolio moved 1.2% in the
same direction. Below 1 is more defensive, and near 0 means the portfolio barely tracks the market. Beta
only measures the part of the risk that comes from the market. A single stock can have a low beta and still
be very volatile.

*Daily or weekly.* The Performance tab can measure beta (and the correlation heatmap) on daily or weekly
returns. Weekly returns are compounded Monday–Friday. Use weekly when your holdings and the benchmark trade
in different time zones: news that arrives after one market closes shows up in the other market's price on
the *same* day but in the first market's price the *next* day. Daily returns then look only loosely related,
and beta is biased towards zero. Over a whole week those few hours hardly matter. The app defaults to weekly
when US and non-US listings (tickers with a suffix such as `.AS` or `.L`) are mixed, and to daily
otherwise, since daily gives about five times as many observations. Bloomberg's standard beta also uses
weekly returns.

### Maximum drawdown

The worst fall from a high point to a later low point, the most you'd have lost if you'd bought at the
worst possible time and sold at the worst possible time.

```
Drawdown_t = V_t / max(V_1 … V_t) − 1
Max drawdown = min over t of Drawdown_t
```

*Reading it:* −25% means that at some point the portfolio was a quarter below its previous peak. The table
also shows when the peak and low happened and when (if ever) the portfolio got back to that peak. The
drawdown chart shows how far below its peak the portfolio was on each day.

### Correlation heatmap

The Pearson correlation between each pair of holdings' returns: daily or weekly, following the same
setting as beta.

```
Corr(a, b) = Cov(r_a, r_b) / (σ_a × σ_b)
```

*Reading it:* +1 means two stocks move in lockstep, 0 means no relationship, and −1 means they move in
opposite directions. Lots of high correlations (dark red) mean your holdings tend to fall together, so you
get less diversification than the number of stocks suggests.

## Factor exposure

The **Factor exposure** tab asks: *what kinds of stocks does this portfolio behave like?* Instead of
comparing with one benchmark, it compares the portfolio with six well-known "factors", each a long-short
portfolio that captures one style of investing. Fama and French, and later others, found that these
styles explain most of the differences in returns between diversified portfolios.

### Why it matters

- **It tells you where your returns came from.** A portfolio that beat the S&P 500 may have done it by
  holding small companies during a good year for small caps. That's a factor bet, not stock-picking skill.
- **It reveals hidden bets.** Ten stocks that look different may all be expensive, fast-growing tech
  companies: one growth bet ten times over.
- **Alpha is measured after the factors.** Alpha here is the return that's left after accounting for
  every factor exposure. It's a much stricter test of skill than "beat the index".

### The factors

All factors come from the [Kenneth R. French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
as daily returns. A loading (coefficient) of 0.3 on a factor means that, holding the other factors fixed,
the portfolio moved about 0.3% for every 1% the factor moved.

| Factor | Name | Long minus short | Positive loading means | Negative loading means |
|---|---|---|---|---|
| **Mkt-RF** | Market | The whole stock market minus T-bills | Market beta. 1 = moves with the market | (Rare) moves against the market |
| **SMB** | Size: Small Minus Big | Small companies − large companies | Small-cap tilt | Large-cap tilt |
| **HML** | Value: High Minus Low | Cheap stocks (high book-to-market) − expensive ones | Value tilt | Growth tilt |
| **RMW** | Profitability: Robust Minus Weak | Highly profitable companies − unprofitable ones | Quality/profitable tilt | Tilt towards weak profitability |
| **CMA** | Investment: Conservative Minus Aggressive | Companies that invest/expand slowly − those growing assets fast | Conservative, mature companies | Fast-expanding companies |
| **Mom** | Momentum (also called WML: Winners Minus Losers) | Stocks that rose most over the past year − those that fell most | Holding recent winners | Holding recent losers (contrarian) |

The tab can use **US** or **Developed Europe** factors. Pick whichever region most of your holdings
come from.

### The regression

```
r_p,t − RF_t = α + β_mkt·(Mkt−RF)_t + β_smb·SMB_t + β_hml·HML_t + β_rmw·RMW_t + β_cma·CMA_t + β_mom·Mom_t + ε_t
```

- **Left-hand side:** the portfolio's return minus the risk-free rate (`RF` from the French data, the
  1-month T-bill), estimated by ordinary least squares (statsmodels).
- **Alpha (α):** the average return the factors don't explain, annualised as daily α × 252 (or weekly
  α × 52). It's significant at 5% if its p-value is below 0.05, roughly a t-statistic beyond ±2.
- **t-statistic:** coefficient ÷ standard error. It measures how many standard errors the estimate is
  from zero. Large t means the loading is unlikely to be zero by chance.
- **R²:** the share of the ups and downs in your excess return that the factors explain. A diversified
  fund is often above 0.9. A concentrated portfolio of a few stocks is much lower, because most of its
  risk is specific to those companies.
- **Rolling chart:** the same regression, re-run on each 12-month window (252 days or 52 weeks), plotting
  the market and momentum loadings so you can see whether the portfolio's exposure drifted.

### Settings

**Standard errors: Newey-West (default) or plain OLS.** The coefficients are identical either way. Only
the standard errors, t-statistics and significance change. Plain OLS assumes each day's surprise is
independent and equally variable. Real returns have calm and turbulent spells, and some knock-on effects
from one day to the next, which makes plain OLS overconfident. Newey-West standard errors allow for both,
with 5 lags for daily data and 4 for weekly. p-values use the t-distribution, which is more cautious
in short samples. Newey-West needs a reasonable number of observations to work; see *Few observations*
under Portfolio values mode.

**Frequency: daily (default) or weekly.** Weekly returns are compounded Monday–Friday and dated by each
week's last trading day.

*Why weekly helps with non-US stocks.* Markets close at different times. A London stock's daily return is
measured at 16:30 London time, but the US factors are measured at the New York close, 4½ hours later. Any
news in that gap shows up in the US factors *today* but in the London stock *tomorrow*. Daily regressions
therefore see the two as only partly related, which biases market beta towards zero and pushes some of the
relationship into the next day. Over a week, the few hours of mismatch are a small part of the window, so
weekly returns line up much better. The cost is about one-fifth as many observations, so the
estimates are noisier. If your holdings and the factor region are both US, daily is fine.

**Minimum data:** the regression needs 60 daily or 26 weekly observations, and the rolling chart needs
12 months.

### Plain-English summary

The summary under the table is generated by fixed rules, so the same numbers always give the same text:

- **Style tilts.** A size, value, profitability, investment or momentum tilt is only mentioned if the
  loading is significant at 5% *and* at least 0.10 in size.
- **Market risk.** Market beta above 1.10 is described as moving more than the market, below 0.90 as
  moving less, and anything in between as roughly in line.
- **Alpha.** The summary says whether alpha is statistically significant, and if not, that it can't be
  told apart from luck.
- **R².** Below 0.5, the summary notes that much of the risk is stock-specific.

### Caveats

- **Data lag.** The French library is updated monthly, usually one to two months behind. The tab shows
  the last available date and leaves out later days rather than failing.
- **Currency.** The Developed Europe factors are in **US dollars**. Yahoo Finance prices European stocks
  in local currency (EUR, GBP, CHF…), so exchange-rate moves end up in the unexplained part of the
  regression. The tab shows a warning when you pick Europe.
- **Short samples are noisy.** With a year or two of data, confidence intervals are wide. A loading that
  isn't significant may still be real; the data just can't tell yet.

## Portfolio values mode

Sometimes you don't have the holdings, only what the portfolio was worth on certain dates: a competition
statement, a broker screenshot, a spreadsheet you update now and then. Choose **Portfolio values (CSV)**
under *Portfolio input* and upload a file like this:

```
date,value
2026-02-17,297.4
2026-02-25,301.7
2026-02-27,298.3
```

- **Format:** dates in `YYYY-MM-DD` format, so 03-04 can't be misread. Values are in any currency or
  units, since only their ratios matter. Snapshots can be any number of days apart, including weekends.
- **Default file:** if you don't upload one, the app uses the Midas competition series in
  `data/competition_values.csv`, with the competition benchmark preselected.

### How it works

**Returns between snapshots.** Each pair of consecutive snapshots gives one interval return,
`R = V_end / V_start − 1`, over `d` calendar days. The *Returns between snapshots* table shows each one.

**Weekly-equivalent returns.** To put intervals of different lengths on the same footing, each return is
converted to the weekly return that would compound to it:

```
weekly-equivalent = (1 + R) ^ (7 / d) − 1
```

A 21% gain over 14 days becomes 10% a week; a 1% gain over one day becomes about 7.2% a week.

**Benchmark and factors over the same intervals.** The benchmark is measured at the same dates, using
the last close on or before each snapshot date (a Sunday snapshot uses Friday's close). The factors are
compounded over exactly the same intervals and converted to weekly equivalents the same way.

**Annualised return** uses calendar time: `(V_last / V_first) ^ (365.25 / days) − 1`.

**Risk (volatility, Sharpe, M², beta, correlation) is measured differently from returns.** Risk grows
with the *square root* of time, not in proportion to it. Raising a d-day return to the power 7/d therefore
distorts its riskiness, multiplying its variance by about 7/d:
- A 2-day move is amplified 3.5×.
- A 4-week move is shrunk to a quarter.

The standard deviation of weekly-equivalent returns ends up off by a factor of `√(7 / harmonic mean of the
gap lengths)`. It's right only when that harmonic mean happens to be 7 days, and noisy even then. For
the competition file (gaps of 2–31 days, harmonic mean ≈ 5 days) it would overstate volatility by about
18%. The app instead scales each interval's log return to a common length:

```
log return over d days = drift × d + σ_daily × √d × noise
drift      = Σ log(1 + R) / Σ d
residual_i = (log(1 + R_i) − drift × d_i) / √d_i
σ_annual   = √( Σ residual² / (n − 1) × 365.25 )
```

With exactly weekly snapshots this is the usual weekly volatility × √52. A test with a simulated random
walk sampled at irregular dates checks that it recovers the true volatility.
- **Sharpe** is `(drift × 365.25 + σ²/2 − ln(1 + rf)) / σ`. The `σ²/2` turns the log drift into the
  average simple return, the same convention as the daily Sharpe ratio.
- **Beta and correlation** use the same scaled residuals for the portfolio and the benchmark.

**Max drawdown** and the drawdown chart only see the snapshot dates, for both the portfolio and the
benchmark. A fall that recovered between two snapshots is invisible.

### Factor regression

The Factor exposure tab regresses the weekly-equivalent excess return (portfolio minus RF, each converted
over the interval) on the weekly-equivalent factors. Because the portfolio and the factors go through the
same conversion, the loadings keep their usual meaning, and alpha is a weekly alpha × 52. The rolling
chart isn't available, since irregular snapshots can't fill fixed 12-month windows.

### Few observations

Each interval is one observation. The app shows a warning with **fewer than 26 intervals** (about six
months of weekly data), and won't run the regression below 8: it estimates 7 numbers, so it needs
at least 8 observations.

The competition file has 13 snapshots, so 12 intervals. That's enough to run everything, but not to be
confident about any of it:
- **Standard errors default to plain OLS** in this case. Newey-West estimates how the noise is related
  from one period to the next using the residuals themselves. With 12 observations and 7 parameters
  there are only 5 residual degrees of freedom, which isn't enough, and its standard errors collapse. On
  the competition data with US factors, Newey-West called alpha "significant" (p < 0.001) at −203% a
  year, while plain OLS gives p = 0.13 and a 95% interval of −486% to +81%. The OLS figures are the
  honest ones.
- **p-values come from the t-distribution** with n − 7 degrees of freedom for both kinds of standard
  error. That makes a big difference with 12 observations and almost none with hundreds.

## How the data is handled

- Prices are dividend- and split-adjusted closes from Yahoo Finance (`yfinance`, `auto_adjust=True`).
- The analysis only covers dates on which **every** holding and the benchmark have prices. If a stock
  listed after your start date, the app starts when it began trading and tells you so.
- Short gaps (for example, a holiday on one exchange but not another) are filled with the previous day's
  price, which counts as a 0% return for that day.
- Price downloads are cached for an hour and factor downloads for a day, so changing weights or settings
  doesn't download them again.

## Limitations

- **Past performance doesn't predict future results.** All of these metrics describe the chosen period
  only, and can change a lot with the start and end dates.
- **Mixed trading calendars.** On a day when one exchange is closed, its holdings count as a 0% return,
  and the move is caught up the next day they trade. Annualising by calendar time (see *How `N` is set*)
  keeps annual return and volatility right overall. Daily beta and correlations are still distorted, so use
  weekly returns for those when calendars or time zones differ.
- **Currencies aren't converted.** Each stock's returns are in its own trading currency. If you mix, say,
  London-listed and US stocks, the portfolio return ignores exchange-rate moves that a USD- or GBP-based
  investor would experience.
- **No trading costs or taxes** are modelled.
- **Yahoo Finance data** is free and generally good, but occasionally has missing days or bad prices,
  especially for smaller non-US listings. If a result looks odd, check the stock's chart on Yahoo Finance.
- Short positions (negative weights) aren't supported.
