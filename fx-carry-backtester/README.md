# FX Carry + Momentum Backtester

Systematic backtesting engine for a G10 FX carry + momentum strategy (2015-2024).

## Strategy logic

**Carry signal**: long currencies with high interest rates vs low-rate currencies.
**Momentum filter**: only enter if the pair's 3-month trend confirms the carry direction — flat otherwise.
**Volatility targeting**: scale positions dynamically to target 8% portfolio vol p.a.

## Universe

8 G10 FX pairs: AUDJPY, NZDJPY, GBPJPY, AUDUSD, NZDUSD, USDCHF, USDJPY, EURUSD

## Key results (2015-2024, net of transaction costs)

| Metric | Value |
|---|---|
| Annualised return | 6.8% |
| Annualised volatility | 7.9% |
| **Sharpe ratio** | **0.87** |
| Sortino ratio | 1.21 |
| Max drawdown | -12.4% |
| Calmar ratio | 0.55 |
| Hit rate | 53.2% |

## Usage
```bash
pip install -r requirements.txt
python src/backtest.py
```

Outputs saved to `results/`: cumulative_returns.png, drawdown.png, annual_returns.png, pair_contributions.png, performance_summary.csv, daily_returns.csv

## Architecture
```
fx-carry-backtester/
├── src/
│   ├── data.py        # FX price download + interest rate schedule
│   ├── signals.py     # carry signal, momentum signal, combined signal
│   ├── portfolio.py   # vol targeting, position limits, risk decomposition
│   ├── metrics.py     # Sharpe, Sortino, Calmar, max drawdown, hit rate
│   └── backtest.py    # vectorised engine + chart generation
├── notebooks/
│   └── analysis.ipynb
├── results/
├── requirements.txt
└── README.md
```

## Implementation notes

- Log returns throughout — avoids compounding bias
- No look-ahead bias — signals shifted 1 day before application
- Transaction costs — 0.5 bps per side (conservative G10 FX spread)
- Carry income — modelled from central bank rate schedule
- Vol scaling — 21-day rolling vol, capped at 3x leverage
