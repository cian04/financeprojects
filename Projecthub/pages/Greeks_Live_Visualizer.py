import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

st.set_page_config(page_title="Greeks Live Visualizer", page_icon="📈", layout="wide")
st.title("📈 Greeks Live Visualizer — Call Européen Black–Scholes")

# --------- Sidebar : paramètres ----------
with st.sidebar:
    st.header("Paramètres")
    S0 = st.number_input("Spot S0", value=100.0, step=1.0, format="%.2f")
    K = st.number_input("Strike K", value=100.0, step=1.0, format="%.2f")
    T = st.number_input("Maturité T (années)", value=0.5, step=0.25, min_value=0.01, format="%.2f")
    r = st.number_input("Taux sans risque r", value=0.02, step=0.01, format="%.4f")
    q = st.number_input("Dividend yield q", value=0.00, step=0.01, format="%.4f")
    sigma = st.number_input("Volatilité σ", value=0.25, step=0.01, min_value=0.0001, format="%.4f")
    s_min = st.number_input("S min (pour le graphe)", value=20.0, step=5.0, format="%.2f")
    s_max = st.number_input("S max (pour le graphe)", value=180.0, step=5.0, format="%.2f")
    n_pts = st.slider("Points sur la grille S", min_value=101, max_value=2001, value=601, step=100)

# --------- Fonctions BS ----------
def d1_d2(S, K, T, r, q, sigma):
    S = np.asarray(S, dtype=float)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

def greeks_call(S, K, T, r, q, sigma):
    """Retourne Delta, Gamma, Vega, Theta pour un call européen."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    Nd1 = norm.cdf(d1)
    nd1 = norm.pdf(d1)
    Delta = np.exp(-q * T) * Nd1
    Gamma = np.exp(-q * T) * nd1 / (S * sigma * np.sqrt(T))
    Vega  = S * np.exp(-q * T) * nd1 * np.sqrt(T)
    Theta = (-S * np.exp(-q * T) * nd1 * sigma / (2 * np.sqrt(T))
             + q * S * np.exp(-q * T) * Nd1
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    return Delta, Gamma, Vega, Theta

# --------- Calculs sur grille ----------
S_grid = np.linspace(min(s_min, S0*0.2), max(s_max, S0*1.8), n_pts)
Delta, Gamma, Vega, Theta = greeks_call(S_grid, K, T, r, q, sigma)

Delta_spot, Gamma_spot, Vega_spot, Theta_spot = greeks_call(S0, K, T, r, q, sigma)

# --------- Fonction pour tracer un Greek ----------
def plot_greek(y, name, ytitle, spot_value):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_grid, y=y,
        mode="lines",
        name=name,
        hovertemplate="S=%{x:.2f}<br>"+name+"=%{y:.6f}<extra></extra>"
    ))
    # Ligne verticale au Spot
    fig.add_vline(
        x=S0, line_width=2, line_dash="dash", line_color="black",
        annotation_text=f"S0 = {S0:.2f}", annotation_position="top"
    )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Prix sous-jacent S",
        yaxis_title=ytitle,
        template="plotly_white"
    )
    return fig

# --------- Affichage 4 colonnes métriques ---------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Δ (Delta)", f"{Delta_spot:.4f}")
col2.metric("Γ (Gamma)", f"{Gamma_spot:.6f}")
col3.metric("Vega", f"{Vega_spot:.4f}")
col4.metric("Θ (Theta)", f"{Theta_spot:.4f}")

st.divider()

# --------- Graphiques ---------
st.subheader("Courbes des Greeks vs S")

st.plotly_chart(plot_greek(Delta, "Δ", "Delta", Delta_spot), use_container_width=True)
st.plotly_chart(plot_greek(Gamma, "Γ", "Gamma", Gamma_spot), use_container_width=True)
st.plotly_chart(plot_greek(Vega, "Vega", "Vega", Vega_spot), use_container_width=True)
st.plotly_chart(plot_greek(Theta, "Θ", "Theta", Theta_spot), use_container_width=True)

st.divider()
st.latex(r"""
\textbf{Formules Black–Scholes (call européen)} \\
\Delta = e^{-qT}N(d_1),\quad
\Gamma = \frac{e^{-qT}n(d_1)}{S\sigma\sqrt{T}},\quad
Vega = S e^{-qT} n(d_1) \sqrt{T},\quad
\Theta = -\frac{S e^{-qT}n(d_1)\sigma}{2\sqrt{T}}
+ q S e^{-qT}N(d_1)
- rK e^{-rT}N(d_2)
""")