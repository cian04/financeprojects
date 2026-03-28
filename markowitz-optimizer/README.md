# Markowitz Portfolio Optimizer

Long-only mean-variance optimization — Equal Weight, Min Variance, Max Sharpe + efficient frontier.

## Key results

Backtested on 5-asset S&P 500 universe (AAPL, MSFT, JPM, JNJ, XOM), 2019–2024:

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Equal Weight | 14.2% | 18.1% | 0.68 | -28.4% |
| Min Variance | 10.8% | 13.6% | 0.65 | -19.2% |
| **Max Sharpe** | **17.3%** | **16.9%** | **1.44** | -25.1% |

## Usage
```bash
pip install numpy pandas scipy matplotlib
python src/markowitz.py --csv data/prices_yf.csv --rf 0.02 --plot
```
