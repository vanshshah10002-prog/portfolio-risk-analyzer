"""Offline tests: returns are synthetic or supplied directly, and Yahoo Finance is faked."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import portfolio_risk
from portfolio_risk import RiskAnalyzer

WEIGHTS = {"AAA": 0.6, "BBB": 0.4}


def frame(rows: list[list[float]], columns=("AAA", "BBB")) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns), index=pd.bdate_range("2025-01-01", periods=len(rows)))


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = rng.multivariate_normal([0.0005, 0.0003], [[0.0004, 0.0001], [0.0001, 0.0002]], size=750)
    return frame(data.tolist())


@pytest.fixture
def analyzer(returns) -> RiskAnalyzer:
    return RiskAnalyzer(WEIGHTS, returns=returns)


# ---------- portfolio returns ----------

def test_portfolio_returns_weight_each_asset_by_ticker_not_column_order(returns):
    reordered = RiskAnalyzer(WEIGHTS, returns=returns[["BBB", "AAA"]])
    expected = returns["AAA"] * 0.6 + returns["BBB"] * 0.4
    pd.testing.assert_series_equal(reordered.portfolio_returns(), expected, check_names=False)


def test_extra_return_columns_are_ignored(returns):
    wider = returns.assign(CCC=0.01)
    assert RiskAnalyzer(WEIGHTS, returns=wider).portfolio_returns().equals(
        RiskAnalyzer(WEIGHTS, returns=returns).portfolio_returns())


# ---------- VaR and CVaR: losses are positive, gains negative ----------

def test_historical_var_is_the_loss_at_the_lower_quantile(analyzer):
    port = analyzer.portfolio_returns()
    assert analyzer.calculate_var(0.95, "historical") == pytest.approx(-np.percentile(port, 5))


def test_parametric_var_matches_the_normal_formula(analyzer):
    port = analyzer.portfolio_returns()
    expected = -(port.mean() + stats.norm.ppf(0.05) * port.std())
    assert analyzer.calculate_var(0.95, "parametric") == pytest.approx(expected)


def test_monte_carlo_var_is_reproducible_and_close_to_parametric(analyzer):
    first = analyzer.monte_carlo_var(0.95, simulations=200_000, seed=11)
    assert first == analyzer.monte_carlo_var(0.95, simulations=200_000, seed=11)
    assert first == pytest.approx(analyzer.calculate_var(0.95, "parametric"), abs=5e-4)


@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_var_is_negative_when_even_the_worst_days_are_gains(method):
    steady_gains = frame([[0.020, 0.018], [0.021, 0.020], [0.019, 0.021], [0.022, 0.019]])
    assert RiskAnalyzer(WEIGHTS, returns=steady_gains).calculate_var(0.95, method, seed=5) < 0


@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_var_is_positive_when_the_worst_days_are_losses(method):
    steady_losses = frame([[-0.020, -0.018], [-0.021, -0.020], [-0.019, -0.021], [-0.022, -0.019]])
    assert RiskAnalyzer(WEIGHTS, returns=steady_losses).calculate_var(0.95, method, seed=5) > 0


def test_cvar_is_negative_when_every_day_is_a_gain():
    gains_only = RiskAnalyzer(WEIGHTS, returns=frame([[0.01, 0.02], [0.03, 0.01], [0.02, 0.02], [0.04, 0.03]]))
    assert gains_only.calculate_cvar(0.95) < 0


def test_cvar_averages_the_returns_at_or_beyond_var():
    single = RiskAnalyzer({"AAA": 1.0}, returns=frame([[r] for r in (-0.10, -0.06, -0.02, 0.01, 0.03)], ["AAA"]))
    var = single.calculate_var(0.75, "historical")  # 25th percentile = -0.06, so VaR = 0.06
    assert var == pytest.approx(0.06)
    assert single.calculate_cvar(0.75) == pytest.approx(0.08)  # mean of -0.10 and -0.06


def test_cvar_is_at_least_var(analyzer):
    assert analyzer.calculate_cvar(0.99) >= analyzer.calculate_var(0.99, "historical")


def test_calculate_var_passes_the_seed_to_monte_carlo(analyzer):
    assert analyzer.calculate_var(0.95, "monte_carlo", seed=9) == analyzer.monte_carlo_var(0.95, seed=9)


# ---------- stress test ----------

def test_stress_test_reports_a_loss_as_positive_and_a_gain_as_negative(analyzer):
    assert analyzer.stress_test({"AAA": -0.05, "BBB": -0.05}) == pytest.approx(0.05)
    assert analyzer.stress_test({"AAA": 0.05, "BBB": 0.05}) == pytest.approx(-0.05)


def test_stress_test_treats_unlisted_holdings_as_flat_and_ignores_other_tickers(analyzer):
    assert analyzer.stress_test({"AAA": -0.10, "ZZZ": -0.90}) == pytest.approx(0.06)


# ---------- drawdown and summary ----------

def test_max_drawdown_compounds_returns_from_a_starting_value_of_one():
    falling = RiskAnalyzer({"AAA": 1.0}, returns=frame([[-0.5], [-0.5]], ["AAA"]))
    assert falling.max_drawdown() == pytest.approx(0.75)  # 1 -> 0.5 -> 0.25 is a 75% loss


def test_max_drawdown_is_measured_from_the_running_peak():
    path = RiskAnalyzer({"AAA": 1.0}, returns=frame([[0.10], [-0.50], [0.20]], ["AAA"]))
    assert path.max_drawdown() == pytest.approx(0.5)  # 1.10 -> 0.55


def test_max_drawdown_is_zero_when_value_only_rises():
    rising = RiskAnalyzer({"AAA": 1.0}, returns=frame([[0.01], [0.02]], ["AAA"]))
    assert str(rising.max_drawdown()) == "0.0"  # positive zero, so it never prints as -0.00%


def test_summary_reports_every_metric(analyzer):
    report = analyzer.summary(0.95, seed=3)
    assert report["historical_var"] == analyzer.calculate_var(0.95, "historical")
    assert report["monte_carlo_var"] == analyzer.monte_carlo_var(0.95, seed=3)
    assert report["max_drawdown"] == analyzer.max_drawdown()
    assert report["portfolio_annualised_vol"] == pytest.approx(report["portfolio_std_daily"] * np.sqrt(252))
    assert report["confidence_level"] == 0.95


# ---------- input validation ----------

@pytest.mark.parametrize("weights", [{}, {"AAA": 0.5, "BBB": 0.4}, {"AAA": float("nan"), "BBB": 1.0},
                                     {"AAA": float("inf"), "BBB": 1.0}])
def test_weights_must_be_present_finite_and_sum_to_one(returns, weights):
    with pytest.raises(ValueError, match="weight"):
        RiskAnalyzer(weights, returns=returns)


def test_short_positions_are_allowed(returns):
    long_short = RiskAnalyzer({"AAA": 1.5, "BBB": -0.5}, returns=returns)
    assert long_short.stress_test({"AAA": -0.10, "BBB": -0.10}) == pytest.approx(0.10)


def test_weights_are_copied_not_shared(returns):
    weights = dict(WEIGHTS)
    analyzer = RiskAnalyzer(weights, returns=returns)
    weights["AAA"] = 0.0
    assert analyzer.weights["AAA"] == 0.6


def test_missing_return_history_is_reported_by_ticker(returns):
    with pytest.raises(ValueError, match="CCC"):
        RiskAnalyzer({"AAA": 0.5, "CCC": 0.5}, returns=returns)


@pytest.mark.parametrize("rows", [[[np.nan, 0.01]], [[0.01, 0.02]]])
def test_too_little_return_history_is_rejected(rows):
    with pytest.raises(ValueError, match="at least 2 days"):
        RiskAnalyzer(WEIGHTS, returns=frame(rows))


@pytest.mark.parametrize("confidence", [0, 1, 1.5, -0.1])
def test_confidence_must_be_strictly_between_zero_and_one(analyzer, confidence):
    with pytest.raises(ValueError, match="confidence"):
        analyzer.calculate_var(confidence)
    with pytest.raises(ValueError, match="confidence"):
        analyzer.monte_carlo_var(confidence)


def test_unknown_var_method_is_rejected(analyzer):
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.calculate_var(0.95, "bootstrap")


# ---------- Yahoo Finance download (faked) ----------

def fake_download(prices: pd.DataFrame):
    def download(tickers, period, progress, auto_adjust):
        assert period == "2y" and progress is False and auto_adjust is True
        return pd.concat({"Close": prices, "Open": prices}, axis=1)  # yfinance returns (field, ticker) columns
    return download


def test_prices_are_downloaded_and_converted_to_daily_returns(monkeypatch):
    prices = frame([[100.0, 50.0], [110.0, 50.0], [99.0, 55.0]])
    monkeypatch.setattr(portfolio_risk.yf, "download", fake_download(prices))
    analyzer = RiskAnalyzer(WEIGHTS)
    assert analyzer.returns["AAA"].tolist() == pytest.approx([0.10, -0.10])
    assert analyzer.returns["BBB"].tolist() == pytest.approx([0.0, 0.10])


def test_days_when_any_ticker_did_not_trade_are_skipped(monkeypatch):
    prices = frame([[100.0, 50.0], [110.0, np.nan], [121.0, 55.0], [121.0, 55.0]])
    monkeypatch.setattr(portfolio_risk.yf, "download", fake_download(prices))
    analyzer = RiskAnalyzer(WEIGHTS)
    assert analyzer.returns["AAA"].tolist() == pytest.approx([0.21, 0.0])  # no fake 0% day for BBB
    assert analyzer.returns["BBB"].tolist() == pytest.approx([0.10, 0.0])


def test_flat_single_ticker_download_from_older_yfinance_is_supported(monkeypatch):
    flat = pd.DataFrame({"Open": [10.0, 11.0, 12.1], "Close": [10.0, 11.0, 12.1]},
                        index=pd.bdate_range("2025-01-01", periods=3))
    monkeypatch.setattr(portfolio_risk.yf, "download", lambda tickers, **kwargs: flat)
    assert RiskAnalyzer({"AAA": 1.0}).returns["AAA"].tolist() == pytest.approx([0.10, 0.10])


def test_a_ticker_with_no_prices_is_named(monkeypatch):
    prices = frame([[100.0, np.nan], [110.0, np.nan], [99.0, np.nan]])
    monkeypatch.setattr(portfolio_risk.yf, "download", fake_download(prices))
    with pytest.raises(ValueError, match="no prices for: BBB"):
        RiskAnalyzer(WEIGHTS)


def test_an_empty_download_is_reported_clearly(monkeypatch):
    monkeypatch.setattr(portfolio_risk.yf, "download", lambda tickers, **kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="no prices for: AAA, BBB"):
        RiskAnalyzer(WEIGHTS)


def test_command_line_report(monkeypatch, capsys):
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (60, 4)), axis=0),
                          columns=["AAPL", "MSFT", "GOOGL", "AMZN"],
                          index=pd.bdate_range("2025-01-01", periods=60))
    monkeypatch.setattr(portfolio_risk.yf, "download", fake_download(prices))
    portfolio_risk.main()
    out = capsys.readouterr().out
    assert "Historical VaR (95%)" in out and "Stress test (-5% all):   5.0000%" in out
