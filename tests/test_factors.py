import numpy as np
import pandas as pd
import pytest

from portfolio import factors as f

TRUE_LOADINGS = {"Mkt-RF": 1.2, "SMB": 0.5, "HML": -0.3, "RMW": 0.0, "CMA": 0.2, "Mom": 0.4}
TRUE_ALPHA_DAILY = 0.0002
RF_DAILY = 0.0001


def synthetic(n_days=1500, noise=0.001, seed=42):
    """Factor data plus a portfolio built from TRUE_LOADINGS and TRUE_ALPHA_DAILY."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    factor_data = pd.DataFrame(rng.normal(0, 0.01, (n_days, 6)), index=dates, columns=f.FACTORS)
    factor_data["RF"] = RF_DAILY
    excess = TRUE_ALPHA_DAILY + factor_data[f.FACTORS] @ pd.Series(TRUE_LOADINGS)
    portfolio = (RF_DAILY + excess + rng.normal(0, noise, n_days)).rename("Portfolio")
    return portfolio, factor_data


@pytest.mark.parametrize("cov_type", ["newey-west", "ols"])
def test_recovers_known_loadings_and_alpha(cov_type):
    portfolio, factor_data = synthetic()
    aligned = f.align_with_factors(portfolio, factor_data)
    result = f.run_factor_regression(aligned, cov_type=cov_type)

    for factor, beta in TRUE_LOADINGS.items():
        assert result.table.loc[factor, "coefficient"] == pytest.approx(beta, abs=0.01)
    # Alpha is estimated less precisely: its annualised standard error is
    # noise / sqrt(n) * 252 ~= 0.65 percentage points, so allow 3 standard errors.
    true_alpha = TRUE_ALPHA_DAILY * 252
    assert result.alpha_annual == pytest.approx(true_alpha, abs=3 * 0.001 / np.sqrt(1500) * 252)
    assert result.table.loc["Alpha", "ci_low"] < true_alpha < result.table.loc["Alpha", "ci_high"]
    assert result.r_squared > 0.95
    assert result.n == 1500


def test_significance_flags():
    portfolio, factor_data = synthetic()
    result = f.run_factor_regression(f.align_with_factors(portfolio, factor_data))
    assert not result.table.loc["RMW", "significant"]  # true loading is zero
    for factor in ["Mkt-RF", "SMB", "HML", "CMA", "Mom"]:
        assert result.table.loc[factor, "significant"]
    row = result.table.loc["Mkt-RF"]
    assert row["ci_low"] < 1.2 < row["ci_high"]


def test_newey_west_changes_errors_not_coefficients():
    portfolio, factor_data = synthetic()
    aligned = f.align_with_factors(portfolio, factor_data)
    nw = f.run_factor_regression(aligned, cov_type="newey-west")
    ols = f.run_factor_regression(aligned, cov_type="ols")
    pd.testing.assert_series_equal(nw.table["coefficient"], ols.table["coefficient"])
    assert not np.allclose(nw.table["std_error"], ols.table["std_error"])


def test_alpha_is_annualised_with_its_interval():
    portfolio, factor_data = synthetic()
    result = f.run_factor_regression(f.align_with_factors(portfolio, factor_data))
    alpha = result.table.loc["Alpha"]
    assert alpha["ci_low"] < alpha["coefficient"] < alpha["ci_high"]
    # t-stat is unit-free, so annualising coefficient and error together leaves it unchanged
    assert alpha["t_stat"] == pytest.approx(alpha["coefficient"] / alpha["std_error"])


def test_weekly_regression_recovers_loadings_and_annualises_by_52():
    portfolio, factor_data = synthetic(n_days=2500)
    aligned = f.align_with_factors(portfolio, factor_data, frequency="weekly")
    assert aligned.n == pytest.approx(2500 / 5, abs=2)
    result = f.run_factor_regression(aligned, frequency="weekly")
    for factor, beta in TRUE_LOADINGS.items():
        assert result.table.loc[factor, "coefficient"] == pytest.approx(beta, abs=0.05)
    # a weekly alpha is ~5x the daily alpha, times 52 is still ~ daily alpha x 252
    assert result.alpha_annual == pytest.approx(TRUE_ALPHA_DAILY * 252, abs=0.03)


def test_alignment_reports_lagging_factor_data():
    portfolio, factor_data = synthetic(n_days=300)
    lagging = factor_data.iloc[:-40]
    aligned = f.align_with_factors(portfolio, lagging)
    assert aligned.last_factor_date == factor_data.index[-41]
    assert aligned.days_after_factor_data == 40
    assert aligned.n == 260
    assert aligned.excess.index.max() == aligned.last_factor_date


def test_alignment_subtracts_risk_free_rate():
    portfolio, factor_data = synthetic(n_days=100)
    aligned = f.align_with_factors(portfolio, factor_data)
    pd.testing.assert_series_equal(
        aligned.excess, (portfolio - RF_DAILY).rename("Excess return"), check_freq=False
    )


def test_to_weekly_compounds_and_rebuilds_market_excess():
    dates = pd.bdate_range("2024-01-01", periods=7)  # Mon..Fri, then Mon, Tue
    daily = pd.DataFrame(
        {"Portfolio": [0.01] * 7, "Mkt-RF": [0.02] * 7, "SMB": [0.0] * 7, "RF": [0.001] * 7},
        index=dates,
    )
    weekly = f.to_weekly(daily)
    # dated by the last trading day in each week, so the partial week ends on Tuesday
    assert list(weekly.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-09")]
    assert weekly["Portfolio"].iloc[0] == pytest.approx(1.01**5 - 1)
    assert weekly["Mkt-RF"].iloc[0] == pytest.approx(1.021**5 - 1.001**5)
    assert weekly["Portfolio"].iloc[1] == pytest.approx(1.01**2 - 1)


def test_too_few_observations_raises():
    portfolio, factor_data = synthetic(n_days=40)
    with pytest.raises(ValueError, match="at least 60"):
        f.run_factor_regression(f.align_with_factors(portfolio, factor_data))


def test_rolling_loadings_are_flat_for_constant_exposure():
    portfolio, factor_data = synthetic(n_days=400, noise=0.0)
    rolling = f.rolling_loadings(f.align_with_factors(portfolio, factor_data))
    assert len(rolling) == 400 - 252 + 1
    for factor, beta in TRUE_LOADINGS.items():
        assert np.allclose(rolling[factor], beta)


def test_rolling_loadings_empty_without_full_window():
    portfolio, factor_data = synthetic(n_days=100)
    assert f.rolling_loadings(f.align_with_factors(portfolio, factor_data)).empty


def make_result(coefficients, significant, alpha=0.0, alpha_significant=False, r_squared=0.9):
    table = pd.DataFrame(
        {
            "coefficient": {"Alpha": alpha, **coefficients},
            "significant": {"Alpha": alpha_significant, **significant},
        }
    )
    return f.FactorRegression(table, r_squared, r_squared, 500, "daily", "newey-west")


def test_describe_small_cap_growth_momentum():
    result = make_result(
        {"Mkt-RF": 1.0, "SMB": 0.4, "HML": -0.3, "RMW": 0.05, "CMA": 0.0, "Mom": 0.3},
        {"Mkt-RF": True, "SMB": True, "HML": True, "RMW": True, "CMA": False, "Mom": True},
        alpha=0.021,
    )
    sentences = f.describe_exposures(result)
    assert len(sentences) == 3
    assert sentences[0].startswith(
        "Your portfolio behaves like a small-cap growth tilt with high momentum exposure;"
    )
    assert "roughly in line with the market" in sentences[0]
    assert "profitable" not in sentences[0]  # 0.05 is below the threshold
    assert "not statistically significant" in sentences[1]
    assert "+2.1%" in sentences[1]


def test_describe_ignores_insignificant_tilts_and_flags_low_r_squared():
    result = make_result(
        {"Mkt-RF": 0.7, "SMB": 0.6, "HML": 0.5, "RMW": 0.0, "CMA": 0.0, "Mom": 0.0},
        {"Mkt-RF": True, "SMB": False, "HML": False, "RMW": False, "CMA": False, "Mom": False},
        alpha=-0.04,
        alpha_significant=True,
        r_squared=0.3,
    )
    first, alpha, r2 = f.describe_exposures(result)
    assert "no clear" in first
    assert "move less than the market" in first
    assert "statistically significant: a negative return" in alpha
    assert "stock-specific" in r2


def test_describe_profitability_and_investment_without_size_or_value():
    result = make_result(
        {"Mkt-RF": 1.3, "SMB": 0.0, "HML": 0.0, "RMW": 0.3, "CMA": -0.2, "Mom": 0.0},
        {"Mkt-RF": True, "SMB": False, "HML": False, "RMW": True, "CMA": True, "Mom": False},
    )
    first = f.describe_exposures(result)[0]
    assert first.startswith(
        "Your portfolio shows a tilt towards highly profitable companies and "
        "a tilt towards fast-growing, heavily investing companies;"
    )
    assert "more than the market" in first


def constant_factor_data(index, value=0.0):
    data = pd.DataFrame(value, index=index, columns=f.FACTORS)
    data["RF"] = 0.0
    return data


def test_weekly_keeps_portfolio_days_missing_from_factor_data():
    # Regression: the inner join used to run before weekly compounding, so a portfolio
    # day with no factor row (e.g. a US holiday when Europe trades) was silently lost.
    days = pd.bdate_range("2024-01-08", periods=15)  # three Mon-Fri weeks
    portfolio = pd.Series(0.01, index=days)
    factor_data = constant_factor_data(days.delete(7))  # Wednesday of week 2 missing
    aligned = f.align_with_factors(portfolio, factor_data, frequency="weekly")
    assert aligned.n == 3
    assert aligned.excess.tolist() == pytest.approx([1.01**5 - 1] * 3)
    assert aligned.days_dropped_mid_sample == 0


def test_daily_counts_days_dropped_mid_sample():
    days = pd.bdate_range("2024-01-08", periods=20)
    portfolio = pd.Series(0.01, index=days)
    factor_data = constant_factor_data(days[:-3].delete([5, 9]))  # two gaps, ends 3 days early
    aligned = f.align_with_factors(portfolio, factor_data)
    assert aligned.days_dropped_mid_sample == 2
    assert aligned.days_after_factor_data == 3
    assert aligned.n == 15


def test_weekly_stops_portfolio_at_last_factor_date():
    # Factor data ends on a Wednesday: the final week must compare Mon-Wed with Mon-Wed,
    # not five portfolio days with three factor days.
    days = pd.bdate_range("2024-01-08", periods=10)
    portfolio = pd.Series(0.01, index=days)
    factor_data = constant_factor_data(days[:8], value=0.002)
    aligned = f.align_with_factors(portfolio, factor_data, frequency="weekly")
    assert aligned.days_after_factor_data == 2
    assert list(aligned.excess.index) == [days[4], days[7]]
    assert aligned.excess.iloc[1] == pytest.approx(1.01**3 - 1)
    assert aligned.factors["SMB"].iloc[1] == pytest.approx(1.002**3 - 1)


@pytest.mark.parametrize("frequency", ["daily", "weekly"])
def test_no_overlap_gives_zero_observations(frequency):
    days = pd.bdate_range("2024-01-08", periods=10)
    factor_data = constant_factor_data(pd.bdate_range("2023-01-02", periods=10))
    aligned = f.align_with_factors(pd.Series(0.01, index=days), factor_data, frequency)
    assert aligned.n == 0
    assert aligned.days_after_factor_data == 10


def test_newey_west_p_values_use_t_distribution():
    # With few observations the normal distribution overstates significance; p-values
    # must come from the t-distribution with n - 7 degrees of freedom.
    from scipy import stats

    portfolio, factor_data = synthetic(n_days=70)
    result = f.run_factor_regression(f.align_with_factors(portfolio, factor_data), cov_type="newey-west")
    t_stats = result.table["t_stat"]
    expected = 2 * stats.t.sf(t_stats.abs(), df=70 - 7)
    assert result.table["p_value"].to_numpy() == pytest.approx(expected)
