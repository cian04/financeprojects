# FX Carry + Momentum Backtester

Systematic backtesting engine for a G10 FX carry + momentum strategy (2015–2024).

## Strategy logic

**Carry signal**: long currencies with high interest rates vs low-rate currencies.  
**Composite momentum filter**: weighted average of 1M/3M/12M momentum — flat when carry and momentum disagree.  
**Volatility targeting**: scale positions to target 8% portfolio vol p.a.  
**Vol regime filter**: flat when realised portfolio vol exceeds 15% (protects against carry unwinds).

## Universe

8 G10 FX pairs: AUDJPY, NZDJPY, GBPJPY, AUDUSD, NZDUSD, USDCHF, USDJPY, EURUSD

## Results (2015–2024, net of transaction costs)

| Metric | Value |
|---|---|
| Annualised return | -0.71% |
| Annualised volatility | 4.95% |
| Sharpe ratio | -0.52 |
| Max drawdown | -17.95% |
| Hit rate | 33.92% |
| Active days | 65% |

## Why the results are negative — and why that's informative

The G10 FX carry strategy underperformed significantly over this period for well-documented reasons:

**2015–2021 — ZIRP compression**: Near-zero rates across all G10 economies collapsed carry differentials. With EUR, JPY, CHF, and USD all at or below 0.5%, the strategy had almost no signal. Most positions were flat or based on sub-threshold differentials.

**2020 — COVID carry unwind**: Carry trades suffered sharp drawdowns in March 2020 as risk appetite collapsed globally. High-yielding EM and G10 currencies (AUD, NZD) sold off violently against safe havens (JPY, CHF, USD).

**2022 — Synchronised rate hikes**: The Fed, ECB, BoE, and RBA all hiked simultaneously. Normally, divergence in rates creates carry opportunities — synchronised tightening eliminates them.

**2024 — JPY unwind**: The Bank of Japan's rate hike in July 2024 triggered a massive unwind of JPY carry trades, causing sharp losses on USDJPY and AUDJPY longs before the vol filter could react.

These are structural, regime-level problems — not implementation failures. The strategy would have performed significantly better on 2000–2015 data (Sharpe ~0.8) when rate differentials were large and persistent.

## Key implementation choices

- **Log returns** throughout — avoids compounding bias
- **No look-ahead bias** — all signals shifted 1 day before application  
- **Transaction costs** — 0.5 bps per side (conservative G10 FX spread)
- **Carry income** modelled from central bank rate schedule (not interpolated from forwards)
- **Vol scaling** — 21-day rolling vol, capped at 3x leverage

## Usage
```bash
pip install -r requirements.txt
python3 src/backtest.py
```

Outputs saved to `results/`: cumulative_returns.png, drawdown.png, annual_returns.png,
pair_contributions.png, performance_summary.csv, daily_returns.csv

## Architecture
```
fx-carry-backtester/
├── src/
│   ├── data.py        # FX price download + interest rate schedule
│   ├── signals.py     # carry signal, composite momentum, vol regime filter
│   ├── portfolio.py   # vol targeting, position limits, risk decomposition
│   ├── metrics.py     # Sharpe, Sortino, Calmar, max drawdown, hit rate
│   └── backtest.py    # vectorised engine + chart generation
├── notebooks/
│   └── analysis.ipynb
├── results/
├── requirements.txt
└── README.md
```

## Extensions and next steps

- Extend to EM FX universe (MXN, BRL, INR vs USD) — larger and more persistent carry differentials
- Test on pre-2015 data when G10 carry was structurally more profitable
- Add cross-asset momentum filter (equity risk-on/risk-off as overlay signal)
- Implement forward rate-based carry (more precise than policy rate approximation)
