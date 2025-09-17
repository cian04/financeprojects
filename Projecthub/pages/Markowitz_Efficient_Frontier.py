import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Markowitz Efficient Frontier", page_icon="📊", layout="wide")
st.title("📊 Markowitz — Efficient Frontier")

st.markdown("""
Cette page te permet de :
- Charger des prix historiques ou utiliser des données simulées,
- Calculer les rendements/volatilités annualisés,
- Générer une **frontière efficiente** par simulation Monte Carlo,
- Mettre en avant 3 portefeuilles clés : **Equal Weight**, **Min Variance**, **Max Sharpe**.
""")

# ---- Chargement données ----
uploaded = st.file_uploader("Charge un fichier CSV (Date + colonnes de prix)", type=["csv"])

if uploaded is not None:
    prices = pd.read_csv(uploaded, parse_dates=[0]).set_index(prices := None or 0)
    prices = pd.read_csv(uploaded, parse_dates=[0]).set_index(prices.columns[0])
else:
    st.info("Aucun fichier chargé — génération de prix synthétiques (random walk).")
    np.random.seed(42)
    T, n = 600, 4
    rets = np.random.normal(0.0005, 0.02, size=(T, n))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    prices = pd.DataFrame(prices, columns=["AAPL", "MSFT", "NVDA", "TSLA"])
    prices.index.name = "t"

# ---- Calculs ----
returns = prices.pct_change().dropna()
mu = returns.mean() * 252  # annualisé
cov = returns.cov() * 252  # annualisé
rf = st.number_input("Taux sans risque (annualisé)", 0.0, 0.1, 0.01, step=0.001)

def port_stats(w):
    mu_p = np.dot(w, mu)
    vol_p = np.sqrt(np.dot(w, np.dot(cov, w)))
    sharpe = (mu_p - rf) / (vol_p + 1e-12)
    return mu_p, vol_p, sharpe

# ---- Monte Carlo ----
n_sim = st.slider("Nombre de portefeuilles simulés", 500, 10000, 3000, step=500)

results = []
weights = []
for _ in range(n_sim):
    w = np.random.random(len(mu))
    w /= w.sum()
    mu_p, vol_p, sharpe = port_stats(w)
    results.append((mu_p, vol_p, sharpe))
    weights.append(w)

results = np.array(results)
mu_grid, vol_grid, sh_grid = results[:, 0], results[:, 1], results[:, 2]

# ---- Portefeuilles spéciaux ----
# Equal weight
n = len(mu)
w_ew = np.repeat(1/n, n)
mu_ew, vol_ew, sh_ew = port_stats(w_ew)

# Max Sharpe
idx_maxsh = np.argmax(sh_grid)
w_maxs = weights[idx_maxsh]
mu_maxs, vol_maxs, sh_maxs = results[idx_maxsh]

# Min Variance
idx_minv = np.argmin(vol_grid)
w_minv = weights[idx_minv]
mu_minv, vol_minv, sh_minv = results[idx_minv]

# ---- Graphique ----
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=vol_grid, y=mu_grid,
    mode="markers",
    marker=dict(size=5, color=sh_grid, colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
    name="Portefeuilles aléatoires",
    opacity=0.6
))

fig.add_trace(go.Scatter(
    x=[vol_ew], y=[mu_ew],
    mode="markers+text",
    text=["EW"],
    textposition="top center",
    marker=dict(size=12, color="gray", symbol="star"),
    name="Equal Weight"
))

fig.add_trace(go.Scatter(
    x=[vol_maxs], y=[mu_maxs],
    mode="markers+text",
    text=["Max Sharpe"],
    textposition="top center",
    marker=dict(size=12, color="green", symbol="star"),
    name="Max Sharpe"
))

fig.add_trace(go.Scatter(
    x=[vol_minv], y=[mu_minv],
    mode="markers+text",
    text=["Min Var"],
    textposition="top center",
    marker=dict(size=12, color="red", symbol="star"),
    name="Min Variance"
))

fig.update_layout(
    title="Frontière efficiente (approximation Monte Carlo)",
    xaxis_title="Volatilité annualisée",
    yaxis_title="Rendement annualisé",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ---- Tableau des poids ----
st.subheader("Poids optimisés")
df_w = pd.DataFrame(
    [w_ew, w_maxs, w_minv],
    index=["Equal Weight", "Max Sharpe", "Min Variance"],
    columns=mu.index
)
st.dataframe(df_w.round(4))