# European Call Pricer & GBM Simulator

## 1. European Call Pricer

Analytical Black-Scholes pricing with continuous dividend yield.

| Parameters | BS Price | Benchmark |
|---|---|---|
| S=100, K=100, r=2%, σ=20%, T=1y | 8.916 | 8.916 (QuantLib) |

Pricing error vs analytical benchmark: **< 0.001%**
```bash
python eurocall.py
```

## 2. GBM Simulator

Monte Carlo simulation with statistical validation (20,000 paths).

| Moment | Theoretical | Simulated |
|---|---|---|
| E[S_T] | 108.33 | 108.31 |
| Var[S_T] | 470.8 | 469.4 |
```bash
python simulategbm.py --paths 20 --big_paths 20000
```
