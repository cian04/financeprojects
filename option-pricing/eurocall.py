"""
Black–Scholes: prix d'un CALL européen
Notation:
- S0    : spot (prix actuel du sous-jacent)
- X     : strike (prix d'exercice)
- r     : taux sans risque (continu, annuel)
- q     : taux de dividende (continu, annuel) – 0 si aucun
- sigma : volatilité annualisée
- T     : maturité en années

Formule:
C = S0 * e^{-qT} * N(d1) - X * e^{-rT} * N(d2)
d1 = [ln(S0/X) + (r - q + 0.5*sigma^2)T] / (sigma * sqrt(T))
d2 = d1 - sigma*sqrt(T)
"""

import math
from typing import Tuple

#stats tools
def _norm_cdf(x: float) -> float:
    """Fonction de répartition N(x) de la N(0,1)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

#input cheks & d1/d2
def _check_inputs(S0: float, X: float, sigma: float, T: float):
    if S0 <= 0 or X <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S0, X, sigma, T doivent être > 0.")

def d1_d2(S0: float, X: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    _check_inputs(S0, X, sigma, T)
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S0 / X) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2

#prix du call
def bs_call_price(S0: float, X: float, r: float, q: float, sigma: float, T: float) -> float:
    """Prix Black–Scholes du CALL européen."""
    d1, d2 = d1_d2(S0, X, r, q, sigma, T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    return S0 * disc_q * _norm_cdf(d1) - X * disc_r * _norm_cdf(d2)

#démo
if __name__ == "__main__":
    # Exemple: ATM, r=2%, q=0, sigma=20%, T=1 an
    S0, X, r, q, sigma, T = 100.0, 100.0, 0.02, 0.00, 0.20, 1.0
    price = bs_call_price(S0, X, r, q, sigma, T)
    print(f"Call EU (BS) = {price:.6f}")  # ~ 8.916