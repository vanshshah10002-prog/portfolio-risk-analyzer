# Portfolio Risk Analyzer

A compact Python module that measures one-day downside risk for an equity portfolio. It uses daily returns from Yahoo Finance and calculates Value at Risk (VaR) and Conditional VaR (Expected Shortfall) using three methods.

## What it calculates

| Metric | Method |
|---|---|
| **Historical VaR** | Empirical percentile of the portfolio's daily returns |
| **Parametric VaR** | Normal distribution fitted to the portfolio's daily mean and standard deviation |
| **Monte Carlo VaR** | 10,000 draws of one-day returns from a multivariate normal fitted to the assets' mean and covariance |
| **CVaR / Expected Shortfall** | Average of historical returns at or beyond the historical VaR |
| **Shock scenario** | Size of the portfolio move for a user-supplied one-day return per ticker, such as −5% on every holding (returned as a positive number) |
| **Summary statistics** | Mean daily return, daily and annualised volatility, maximum drawdown (approximated from cumulative simple returns) |

## Usage

```bash
pip install numpy pandas scipy yfinance
python portfolio_risk.py
```

```python
from portfolio_risk import RiskAnalyzer

analyzer = RiskAnalyzer({"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.2}, lookback_years=2)
analyzer.calculate_var(confidence=0.95, method="historical")   # or "parametric" / "monte_carlo"
analyzer.calculate_cvar(confidence=0.95)
analyzer.stress_test({"AAPL": -0.05, "MSFT": -0.05, "GOOGL": -0.05, "AMZN": -0.05})
analyzer.summary()
```

## Limitations

- All measures are one-day horizon figures on simple returns. Weights must sum to 1.
- Parametric and Monte Carlo VaR assume normally distributed returns, so they understate fat-tailed losses.
- The shock scenario is user-specified. No historical crisis scenarios are built in.

## Licence

[MIT](LICENSE)

---

*A practical implementation of risk management concepts from the MSc Financial Technology at Warwick Business School.*
