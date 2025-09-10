# Live Greek Visualizer 

An interactive dashboard to explore **option Greeks** (Delta, Gamma, Vega, Theta) under the Black–Scholes model.  
The tool allows real-time visualization with sliders for **spot (S)**, **strike (K)**, **maturity (T)**, **volatility (σ)**, and **risk-free rate (r)**.

---

# The Greeks and their variations

- Δ = ∂V/∂S
  1) Sensitivity of option price to changes in the underlying price.
  2) Interpretable as the probability a call/put finishes ITM (heuristic).

- Quick rules (signs only, no arrows):
  - If K is much smaller than S (K ≪ S): deep ITM call; Δ ≈ 1.
  - If K is much larger than S (K ≫ S): deep OTM call; Δ ≈ 0.
  - Higher volatility (σ): Δ tends toward 0.5.
  - For an OTM call, longer maturity (T): Δ increases.
