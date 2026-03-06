# Portfolio Risk Analyzer

A quantitative risk management tool for analyzing portfolio downside risk using Value at Risk (VaR), Conditional Value at Risk (CVaR), and Monte Carlo simulation.

## Features

- **Value at Risk (VaR) Calculation:** Historical, parametric, and Monte Carlo methods
- **Expected Shortfall (CVaR):** Tail risk quantification
- **Stress Testing:** Historical scenario analysis (2008 crisis, COVID-19 crash)
- **Correlation Analysis:** Portfolio diversification metrics
- **Risk Decomposition:** Contribution of individual assets to portfolio risk

## Installation

```bash
pip install numpy pandas yfinance matplotlib scipy
```

## Usage

```python
from portfolio_risk import RiskAnalyzer

# Define portfolio
weights = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.2, 'AMZN': 0.2}

# Calculate risk metrics
analyzer = RiskAnalyzer(weights)
var_95 = analyzer.calculate_var(confidence=0.95)
cvar_95 = analyzer.calculate_cvar(confidence=0.95)
```

## Methodologies

### 1. Historical VaR
Uses empirical return distribution from historical data

### 2. Parametric VaR
Assumes normal distribution of returns

### 3. Monte Carlo VaR
Simulates 10,000+ future price paths using multivariate GBM

### 4. Stress Testing
Applies historical crisis scenarios to current portfolio

## Tech Stack

- **NumPy** — numerical computations
- **pandas** — data handling
- **SciPy** — statistical distributions
- **yfinance** — market data
- **matplotlib** — visualization

## Key Concepts

| Concept | Description |
|---|---|
| **Value at Risk (VaR)** | Maximum expected loss at given confidence level |
| **Conditional VaR (CVaR)** | Expected loss beyond VaR threshold |
| **Monte Carlo Simulation** | Stochastic modeling of asset returns |
| **Stress Testing** | Scenario-based risk assessment |

---

*Practical implementation of risk management concepts from MSc Financial Technology curriculum at Warwick Business School.*
