# Projecthub/pages/GBM_Simulator.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="GBM Simulator", page_icon="📈", layout="wide")

st.title("📈 Simulation du Mouvement Brownien Géométrique (GBM)")

# --- Sidebar : paramètres ---
with st.sidebar:
    st.header("Paramètres")
    S0 = st.number_input("Prix initial S₀", min_value=0.0, value=100.0, step=1.0)
    mu = st.number_input("Drift μ (ann.)", value=0.08, step=0.01, format="%.4f")
    sigma = st.number_input("Volatilité σ (ann.)", min_value=0.0, value=0.20, step=0.01, format="%.4f")
    T = st.number_input("Horizon T (années)", min_value=0.0, value=1.0, step=0.25, format="%.4f")
    n_steps = st.slider("Nombre de pas de temps", min_value=10, max_value=2000, value=252)
    n_paths = st.slider("Nombre de trajectoires", min_value=1, max_value=20000, value=2000, step=100)
    seed = st.number_input("Seed aléatoire (optionnel)", min_value=0, value=0, step=1)
    show_paths = st.slider("Chemins à tracer (pour ne pas surcharger le graphe)", 1, 200, 50)
    st.caption("Astuce : augmente les trajectoires pour un histogramme plus lisse, "
               "mais limite les chemins tracés pour garder un affichage fluide.")

col_left, col_right = st.columns([2, 1])

# --- Simulation GBM vectorisée ---
dt = T / n_steps
rng = np.random.default_rng(seed if seed != 0 else None)

# Incréments brownien ~ N(0, dt)
Z = rng.standard_normal(size=(n_steps, n_paths))
dW = np.sqrt(dt) * Z

# Solution exacte du GBM : S_t = S0 * exp((μ-0.5σ²)t + σ W_t)
# On cumule W_t par colonne (trajectoire), puis on applique la formule fermée.
W = dW.cumsum(axis=0)
t_grid = np.linspace(dt, T, n_steps).reshape(-1, 1)

drift = (mu - 0.5 * sigma**2) * t_grid
diffusion = sigma * W
S = S0 * np.exp(drift + diffusion)  # shape: (n_steps, n_paths)

# On ajoute S0 au début pour avoir n_steps+1 points (t=0 … T)
S = np.vstack([np.full((1, n_paths), S0), S])
t_grid_full = np.linspace(0.0, T, n_steps + 1)

# DataFrame pour export
df = pd.DataFrame(S, index=t_grid_full)
df.index.name = "Time"
df.columns = [f"path_{i}" for i in range(n_paths)]

# --- Stats empiriques vs théoriques ---
S_T = S[-1, :]
emp_mean = float(np.mean(S_T))
emp_std = float(np.std(S_T))
emp_median = float(np.median(S_T))

# Théorie :  E[S_T] = S0 * exp(μ T)
theo_mean = float(S0 * np.exp(mu * T))
# Var[S_T] = S0² * exp(2μT) * (exp(σ²T) - 1)
theo_var = (S0**2) * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1.0)
theo_std = float(np.sqrt(theo_var))

with col_right:
    st.subheader("Statistiques (prix terminal S_T)")
    st.metric("Moyenne empirique", f"{emp_mean:,.4f}")
    st.metric("Moyenne théorique", f"{theo_mean:,.4f}")
    st.metric("Écart-type empirique", f"{emp_std:,.4f}")
    st.metric("Écart-type théorique", f"{theo_std:,.4f}")
    st.metric("Médiane empirique", f"{emp_median:,.4f}")

    st.download_button(
        "💾 Télécharger les trajectoires (CSV)",
        data=df.to_csv().encode("utf-8"),
        file_name="gbm_paths.csv",
        mime="text/csv",
    )

# --- Graphique des chemins (Plotly) ---
with col_left:
    st.subheader("Trajectoires simulées")
    fig_paths = go.Figure()
    # On trace seulement 'show_paths' chemins pour rester lisible
    for i in range(min(show_paths, n_paths)):
        fig_paths.add_trace(go.Scatter(
            x=t_grid_full, y=S[:, i], mode="lines", name=f"Chemin {i+1}",
            line=dict(width=1)
        ))
    fig_paths.update_layout(
        xaxis_title="Temps (années)",
        yaxis_title="Prix",
        showlegend=False,
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_paths, use_container_width=True)

# --- Histogramme du prix terminal ---
st.subheader("Distribution de S_T (histogramme)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=S_T, nbinsx=60, name="S_T"))
fig_hist.add_vline(x=emp_mean, line_width=2, line_dash="dash", annotation_text="Mean (emp.)")
fig_hist.add_vline(x=theo_mean, line_width=2, line_dash="dot", annotation_text="Mean (theo.)")
fig_hist.update_layout(
    xaxis_title="S_T",
    yaxis_title="Fréquence",
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_hist, use_container_width=True)

# --- Rappel formule ---
st.caption("GBM : dS/S = μ dt + σ dW  →  S(t) = S₀ · exp((μ−½σ²)t + σW(t))")