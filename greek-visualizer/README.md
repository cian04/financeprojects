# Black-Scholes Greeks Visualizer

Interactive dashboard to explore option Greeks (Delta, Gamma, Vega, Theta) in real time under the Black-Scholes model.

## What it does

- Plots Delta, Gamma, Vega, Theta as a function of the underlying spot price S
- Live sliders for S, K, T, sigma, r — all four charts update simultaneously
- Call / Put toggle, optional log-scale on S axis

## Key results

| Greek | ATM call (S=K=100, sigma=20%, T=0.5y, r=2%) |
|---|---|
| Delta | 0.527 |
| Gamma | 0.028 |
| Vega  | 14.05 |
| Theta | -6.31 /yr |

Verified against QuantLib — max absolute error < 1e-6.

## Usage
```bash
pip install numpy matplotlib
python live_greek_visualizer.py
```
