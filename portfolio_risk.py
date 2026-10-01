"""
Portfolio Risk Analyzer
========================
One-day downside risk for an equity portfolio: VaR (historical, parametric,
Monte Carlo), CVaR / Expected Shortfall, shock scenarios and drawdown.

Sign convention: every risk figure (VaR, CVaR, stress loss, max drawdown) is a
loss expressed as a positive fraction of portfolio value. A negative VaR, CVaR or
stress figure means the scenario is a gain.

Author: Vansh Shah
Course: MSc Financial Technology, Warwick Business School
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

TRADING_DAYS = 252
WEIGHT_TOLERANCE = 1e-6
MIN_OBSERVATIONS = 2  # volatility and covariance need at least two daily returns


def fetch_returns(tickers: list[str], lookback_years: int) -> pd.DataFrame:
    """Daily simple returns of split- and dividend-adjusted closing prices from Yahoo Finance."""
    data = yf.download(tickers, period=f"{lookback_years}y", progress=False, auto_adjust=True)
    try:
        prices = data["Close"]
    except KeyError:
        raise ValueError(f"Yahoo Finance returned no prices for: {', '.join(tickers)}") from None
    if isinstance(prices, pd.Series):  # older yfinance returns a flat frame for a single ticker
        prices = prices.to_frame(tickers[0])
    failed = [str(t) for t in prices.columns[prices.isna().all()]]
    if failed:
        raise ValueError(f"Yahoo Finance returned no prices for: {', '.join(failed)}")
    return prices.dropna().pct_change().dropna()  # keep only days when every ticker traded


def _validated_weights(weights: dict[str, float]) -> dict[str, float]:
    """Weights may be negative (short positions) but must be finite and sum to 1."""
    if not weights:
        raise ValueError("portfolio_weights must contain at least one ticker")
    if not all(np.isfinite(w) for w in weights.values()):
        raise ValueError("portfolio weights must be finite numbers")
    total = sum(weights.values())
    if not abs(total - 1) <= WEIGHT_TOLERANCE:
        raise ValueError(f"portfolio weights must sum to 1, got {total:.6f}")
    return dict(weights)


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")


class RiskAnalyzer:
    """
    Compute portfolio risk metrics from daily returns.

    Parameters:
        portfolio_weights: {ticker: weight}; weights must sum to 1.
        lookback_years: history to download when `returns` is not given.
        returns: optional daily returns with one column per ticker
            (skips the download, e.g. for your own data or for tests).
    """

    def __init__(
        self,
        portfolio_weights: dict[str, float],
        lookback_years: int = 2,
        returns: pd.DataFrame | None = None,
    ):
        self.weights = _validated_weights(portfolio_weights)
        self.lookback_years = lookback_years
        if returns is None:
            returns = fetch_returns(list(self.weights), lookback_years)
        missing = [t for t in self.weights if t not in returns.columns]
        if missing:
            raise ValueError(f"no return data for: {', '.join(missing)}")
        self.returns = returns[list(self.weights)].dropna()
        if len(self.returns) < MIN_OBSERVATIONS:
            raise ValueError(f"need at least {MIN_OBSERVATIONS} days of overlapping return history, "
                             f"got {len(self.returns)}")
        self._weight_vector = np.array(list(self.weights.values()))

    def portfolio_returns(self) -> pd.Series:
        """Daily portfolio returns, weighting each asset by its ticker."""
        return self.returns @ self._weight_vector

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def calculate_var(self, confidence: float = 0.95, method: str = "historical", seed: int | None = None) -> float:
        """
        One-day Value at Risk.

        Parameters:
            confidence: confidence level, e.g. 0.95 for 95%.
            method: 'historical' | 'parametric' | 'monte_carlo'.
            seed: random seed for the Monte Carlo method (ignored by the others).

        Returns:
            Loss at the (1 - confidence) quantile, as a positive fraction.
        """
        _check_confidence(confidence)
        if method == "monte_carlo":
            return self.monte_carlo_var(confidence, seed=seed)
        port = self.portfolio_returns()
        if method == "historical":
            return -float(np.percentile(port, (1 - confidence) * 100))
        if method == "parametric":
            return -float(stats.norm.ppf(1 - confidence, port.mean(), port.std()))
        raise ValueError(f"Unknown method: {method}")

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """Expected Shortfall: average loss on the days at or beyond the historical VaR."""
        port = self.portfolio_returns()
        threshold = -self.calculate_var(confidence, "historical")
        return -float(port[port <= threshold].mean())  # never empty: the quantile is >= the minimum

    def monte_carlo_var(
        self,
        confidence: float = 0.95,
        simulations: int = 10_000,
        seed: int | None = None,
    ) -> float:
        """VaR from one-day returns drawn from a multivariate normal fitted to the assets (seed for repeatability)."""
        _check_confidence(confidence)
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(self.returns.mean().to_numpy(), self.returns.cov().to_numpy(), simulations)
        return -float(np.percentile(draws @ self._weight_vector, (1 - confidence) * 100))

    # ------------------------------------------------------------------
    # Shock scenarios and drawdown
    # ------------------------------------------------------------------

    def stress_test(self, scenario_returns: dict[str, float]) -> float:
        """
        Portfolio loss for a one-day shock.

        Parameters:
            scenario_returns: {ticker: return}; holdings not listed are assumed flat,
                tickers outside the portfolio are ignored.

        Returns:
            Loss as a positive fraction (a gain comes back negative).
        """
        return -float(sum(weight * scenario_returns.get(t, 0.0) for t, weight in self.weights.items()))

    def max_drawdown(self) -> float:
        """Largest fall in compounded portfolio value from its running peak (starting at 1), as a positive fraction."""
        wealth = (1 + self.portfolio_returns()).cumprod()
        peak = wealth.cummax().clip(lower=1.0)
        return float(1 - (wealth / peak).min())

    # ------------------------------------------------------------------
    # Summary Report
    # ------------------------------------------------------------------

    def summary(self, confidence: float = 0.95, seed: int | None = None) -> dict:
        """All metrics at one confidence level (seed makes the Monte Carlo figure repeatable)."""
        port = self.portfolio_returns()
        return {
            "portfolio_mean_daily": float(port.mean()),
            "portfolio_std_daily": float(port.std()),
            "portfolio_annualised_vol": float(port.std() * np.sqrt(TRADING_DAYS)),
            "historical_var": self.calculate_var(confidence, "historical"),
            "parametric_var": self.calculate_var(confidence, "parametric"),
            "monte_carlo_var": self.monte_carlo_var(confidence, seed=seed),
            "cvar": self.calculate_cvar(confidence),
            "max_drawdown": self.max_drawdown(),
            "confidence_level": confidence,
        }


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    weights = {"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.20}  # example: US tech portfolio
    print("=" * 60)
    print("  Portfolio Risk Analyzer")
    print("=" * 60)
    print(f"\nPortfolio: {weights}")
    print("Fetching market data...\n")

    analyzer = RiskAnalyzer(weights)
    report = analyzer.summary(confidence=0.95)

    print(f"  Mean daily return:       {report['portfolio_mean_daily']:+.4%}")
    print(f"  Daily volatility:        {report['portfolio_std_daily']:.4%}")
    print(f"  Annualised volatility:   {report['portfolio_annualised_vol']:.2%}")
    print()
    print(f"  Historical VaR (95%):    {report['historical_var']:.4%}")
    print(f"  Parametric VaR (95%):    {report['parametric_var']:.4%}")
    print(f"  Monte Carlo VaR (95%):   {report['monte_carlo_var']:.4%}")
    print(f"  CVaR / ES (95%):         {report['cvar']:.4%}")
    print(f"  Max drawdown:            {report['max_drawdown']:.4%}")
    print()
    print(f"  Stress test (-5% all):   {analyzer.stress_test({t: -0.05 for t in weights}):.4%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
