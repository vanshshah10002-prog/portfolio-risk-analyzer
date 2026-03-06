"""
Portfolio Risk Analyzer
========================
Quantitative portfolio risk analysis tool calculating VaR, CVaR,
and stress testing scenarios using multiple methodologies.

Author: Vansh Shah
Course: MSc Financial Technology, Warwick Business School
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Optional


class RiskAnalyzer:
    """
    Compute portfolio risk metrics including VaR, CVaR, and
    Monte Carlo stress testing.

    Parameters:
        portfolio_weights (dict): {ticker: weight} mapping.
        lookback_years (int): Historical data window in years.
    """

    def __init__(self, portfolio_weights: dict, lookback_years: int = 2):
        self.weights = portfolio_weights
        self.lookback_years = lookback_years
        self.returns: Optional[pd.DataFrame] = None
        self._fetch_returns()

    def _fetch_returns(self) -> None:
        """Fetch historical returns for portfolio assets."""
        tickers = list(self.weights.keys())
        data = yf.download(
            tickers,
            period=f"{self.lookback_years}y",
            progress=False,
        )["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        self.returns = data.pct_change().dropna()

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def portfolio_returns(self) -> pd.Series:
        """Compute weighted portfolio return series."""
        w = np.array([self.weights[t] for t in self.returns.columns])
        return (self.returns * w).sum(axis=1)

    def calculate_var(
        self,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk.

        Parameters:
            confidence: Confidence level (e.g. 0.95 for 95%).
            method: 'historical' | 'parametric' | 'monte_carlo'.

        Returns:
            VaR as a positive float (loss magnitude).
        """
        port = self.portfolio_returns()

        if method == "historical":
            var = np.percentile(port, (1 - confidence) * 100)
        elif method == "parametric":
            mu, sigma = port.mean(), port.std()
            var = stats.norm.ppf(1 - confidence, mu, sigma)
        elif method == "monte_carlo":
            return self.monte_carlo_var(confidence)
        else:
            raise ValueError(f"Unknown method: {method}")

        return abs(float(var))

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Parameters:
            confidence: Confidence level.

        Returns:
            CVaR as a positive float.
        """
        port = self.portfolio_returns()
        var = self.calculate_var(confidence)
        tail = port[port <= -var]
        if tail.empty:
            return var  # fallback: no tail observations
        return abs(float(tail.mean()))

    def monte_carlo_var(
        self,
        confidence: float = 0.95,
        simulations: int = 10_000,
    ) -> float:
        """
        Monte Carlo VaR using multivariate normal simulation.

        Parameters:
            confidence: Confidence level.
            simulations: Number of Monte Carlo paths.

        Returns:
            VaR as a positive float.
        """
        mu = self.returns.mean().values
        cov = self.returns.cov().values
        w = np.array([self.weights[t] for t in self.returns.columns])

        sim = np.random.multivariate_normal(mu, cov, simulations)
        port_sim = sim @ w
        var = np.percentile(port_sim, (1 - confidence) * 100)
        return abs(float(var))

    # ------------------------------------------------------------------
    # Stress Testing
    # ------------------------------------------------------------------

    def stress_test(self, scenario_returns: dict) -> float:
        """
        Apply a stress scenario to the portfolio.

        Parameters:
            scenario_returns: {ticker: daily_return} for the stress day.

        Returns:
            Portfolio loss as a positive float.
        """
        loss = sum(
            self.weights.get(t, 0) * r
            for t, r in scenario_returns.items()
        )
        return abs(loss)

    # ------------------------------------------------------------------
    # Summary Report
    # ------------------------------------------------------------------

    def summary(self, confidence: float = 0.95) -> dict:
        """
        Generate a full risk summary.

        Returns:
            dict with VaR, CVaR, Monte Carlo VaR, portfolio stats.
        """
        port = self.portfolio_returns()
        return {
            "portfolio_mean_daily": float(port.mean()),
            "portfolio_std_daily": float(port.std()),
            "portfolio_annualised_vol": float(port.std() * np.sqrt(252)),
            "historical_var": self.calculate_var(confidence, "historical"),
            "parametric_var": self.calculate_var(confidence, "parametric"),
            "monte_carlo_var": self.monte_carlo_var(confidence),
            "cvar": self.calculate_cvar(confidence),
            "max_drawdown": float((port.cumsum() - port.cumsum().cummax()).min()),
            "confidence_level": confidence,
        }


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Portfolio Risk Analyzer")
    print("=" * 60)

    # Example: US tech portfolio
    weights = {"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.20}

    print(f"\nPortfolio: {weights}")
    print("Fetching market data...\n")

    analyzer = RiskAnalyzer(weights)
    report = analyzer.summary(confidence=0.95)

    print(f"  Mean Daily Return:      {report['portfolio_mean_daily']:+.4%}")
    print(f"  Daily Volatility:       {report['portfolio_std_daily']:.4%}")
    print(f"  Annualised Volatility:  {report['portfolio_annualised_vol']:.2%}")
    print()
    print(f"  Historical VaR (95%):   {report['historical_var']:.4%}")
    print(f"  Parametric VaR (95%):   {report['parametric_var']:.4%}")
    print(f"  Monte Carlo VaR (95%):  {report['monte_carlo_var']:.4%}")
    print(f"  CVaR / ES (95%):        {report['cvar']:.4%}")
    print(f"  Max Drawdown:           {report['max_drawdown']:.4%}")
    print()

    # Stress test: simulate a -5% day for all assets
    stress = {t: -0.05 for t in weights}
    print(f"  Stress Test (-5% all):  {analyzer.stress_test(stress):.4%}")
    print("=" * 60)
