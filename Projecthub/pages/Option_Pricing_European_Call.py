import streamlit as st
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

st.set_page_config(page_title="European Call", page_icon="📈", layout="wide")
st.title("📈 Option Pricing — European Call (Black–Scholes)")

# Inputs
col = st.columns(3)
with col[0]:
    S0 = st.number_input("Spot S0", 1.0, 1e6, 100.0, step=1.0)
    K  = st.number_input("Strike K", 1.0, 1e6, 100.0, step=1.0)
with col[1]:
    r  = st.number_input("Taux sans risque r", -0.05, 0.2, 0.02, step=0.001, format="%.3f")
    q  = st.number_input("Dividende q", -0.05, 0.2, 0.00, step=0.001, format="%.3f")
with col[2]:
    sigma = st.number_input("Volatilité σ", 0.0001, 5.0, 0.2, step=0.01, format="%.2f")
    T     = st.number_input("Maturité T (années)", 0.0001, 50.0, 1.0, step=0.1, format="%.2f")

def bs_call_price(S0, K, r, q, sigma, T):
    d1 = (log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S0*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2), d1, d2

def bs_call_greeks(S0, K, r, q, sigma, T, d1, d2):
    Nd1 = norm.cdf(d1); nd1 = norm.pdf(d1)
    delta = exp(-q*T) * Nd1
    gamma = exp(-q*T) * nd1 / (S0*sigma*sqrt(T))
    vega  = S0*exp(-q*T) * nd1 * sqrt(T)
    theta = (-S0*exp(-q*T)*nd1*sigma/(2*sqrt(T))
             + q*S0*exp(-q*T)*Nd1
             - r*K*exp(-r*T)*norm.cdf(d2))
    rho   = K*T*exp(-r*T)*norm.cdf(d2)
    return delta, gamma, vega, theta, rho

price, d1, d2 = bs_call_price(S0,K,r,q,sigma,T)
Δ, Γ, 𝑉, Θ, ρ = bs_call_greeks(S0,K,r,q,sigma,T,d1,d2)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.metric("Prix Call (BS)", f"{price:,.4f}")
with c2:
    st.write("**d1** =", round(d1,4), " • **d2** =", round(d2,4))
with c3:
    st.write("")

st.subheader("Greeks")
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta (Δ)", f"{Δ:.4f}")
g2.metric("Gamma (Γ)", f"{Γ:.6f}")
g3.metric("Vega (𝑉)", f"{𝑉:.4f}")
g4.metric("Theta (Θ / an)", f"{Θ:.4f}")
g5.metric("Rho (ρ)", f"{ρ:.4f}")

st.caption("Formule : C = S₀ e^{-qT} N(d₁) − K e^{-rT} N(d₂).")