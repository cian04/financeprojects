# Live Greek Visualizer 

An interactive dashboard to explore **option Greeks** (Delta, Gamma, Vega, Theta) under the Black–Scholes model.  
The tool allows real-time visualization with sliders for **spot (S)**, **strike (K)**, **maturity (T)**, **volatility (σ)**, and **risk-free rate (r)**.

---

# The Greeks and theyre variation

- Delta (∂V/∂S): 
 1) Sensitivity of option price to changes in the underlying price.
 2) Probability of call or put to finish ITM
- If \( S \) is much greater than \( K \):  
  → the call is deeply **ITM** → \(\Delta \approx 1\).  

- If \( S \) is much smaller than \( K \):  
  → the call is deeply **OTM** → \(\Delta \approx 0\).  

- Delta tends toward **0.5** when volatility \( \sigma \) ↑ (more uncertainty → the call looks like a 50/50 bet).  

- For an OTM call: \(\Delta\) ↑ when maturity \( T \) ↑ (more time = higher chance to end up ITM).  