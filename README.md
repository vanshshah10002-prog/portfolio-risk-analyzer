# Portfolio Risk Analyzer

[![tests](https://github.com/vanshshah10002-prog/portfolio-risk-analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/vanshshah10002-prog/portfolio-risk-analyzer/actions/workflows/tests.yml)

A compact Python module that measures one-day downside risk for an equity portfolio. It uses daily returns from Yahoo Finance, or returns you supply yourself, and calculates Value at Risk (VaR) using three methods plus Conditional VaR (Expected Shortfall).

**Sign convention:** every risk figure (VaR, CVaR, shock loss and maximum drawdown) is a loss, expressed as a positive fraction of portfolio value. A negative VaR, CVaR or shock figure means the scenario is a gain.

## What it calculates

| Metric | Method |
|---|---|
| **Historical VaR** | Loss at the empirical lower percentile of the portfolio's daily returns |
| **Parametric VaR** | Loss at the lower percentile of a normal distribution fitted to the portfolio's daily mean and standard deviation |
| **Monte Carlo VaR** | Loss at the lower percentile of 10,000 one-day draws from a multivariate normal fitted to the assets' means and covariances. Pass a seed to make it repeatable |
| **CVaR / Expected Shortfall** | Average loss on the days at or beyond the historical VaR |
| **Shock scenario** | Portfolio loss for a user-supplied one-day return per ticker, such as −5% on every holding. Holdings left out are treated as flat |
| **Maximum drawdown** | Largest fall in compounded portfolio value from its running peak, over the whole lookback window |
| **Summary statistics** | Mean daily return, and daily and annualised volatility |

## Usage

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
python portfolio_risk.py
```

```python
from portfolio_risk import RiskAnalyzer

analyzer = RiskAnalyzer({"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.2}, lookback_years=2)
analyzer.calculate_var(confidence=0.95, method="historical")   # or "parametric" / "monte_carlo" (seed=...)
analyzer.calculate_cvar(confidence=0.95)
analyzer.monte_carlo_var(confidence=0.99, seed=42)
analyzer.stress_test({"AAPL": -0.05, "MSFT": -0.05, "GOOGL": -0.05, "AMZN": -0.05})  # 0.05 = a 5% loss
analyzer.max_drawdown()
analyzer.summary(seed=42)

# Use your own daily returns (one column per ticker) instead of downloading prices
analyzer = RiskAnalyzer({"FUND_A": 0.7, "FUND_B": 0.3}, returns=my_returns_dataframe)
```

Inputs are validated:

- Weights must be finite and sum to 1. Negative weights (short positions) are allowed.
- The tickers need at least two days of overlapping return history. A ticker that Yahoo Finance can't find is named in the error.
- The confidence level must be strictly between 0 and 1.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run offline, using synthetic returns and a faked Yahoo Finance download. They cover:

- each VaR method against its formula
- the sign convention for gains and losses, for every method
- CVaR on a hand-worked example
- compounded drawdown
- input validation and download failures
- the command-line report

Coverage is enforced at ≥ 80%. GitHub Actions runs the tests on pushes to `main` and on pull requests.

## Limitations

- VaR, CVaR and shocks are one-day figures based on simple daily returns. Prices are split- and dividend-adjusted, and only days when every holding traded are used.
- Fixed weights imply the portfolio is rebalanced daily.
- Parametric and Monte Carlo VaR assume normally distributed returns, so they understate fat-tailed losses.
- The shock scenario is user-specified. No historical crisis scenarios are built in.

## Licence

[MIT](LICENSE)

---

*A practical implementation of risk management concepts from the MSc Financial Technology at Warwick Business School.*
