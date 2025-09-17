import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Quant Portfolio • Project Hub", page_icon="🧠", layout="wide")

# ---------- Header ----------
st.title("🧠 Quant/Structuring – Project Hub")
st.caption("Bienvenue ! Voici un sommaire de tes projets (pricing, greeks, Markowitz, simulations).")
st.markdown("---")

# ---------- Helper: Card component ----------
def project_card(title, subtitle, badges=None, go_to=None, repo_rel_path=None, image_path=None):
    cols = st.columns([1, 4, 2])
    with cols[0]:
        if image_path and Path(image_path).exists():
            st.image(image_path, use_container_width=True)
        else:
            st.markdown("### 📦")
    with cols[1]:
        st.markdown(f"### {title}")
        st.write(subtitle)
        if badges:
            st.write(" ".join([f"`{b}`" for b in badges]))
    with cols[2]:
        if go_to:
            st.page_link(go_to, label="Ouvrir la page →", icon="➡️")
        if repo_rel_path:
            st.caption(f"**Repo path**: `{repo_rel_path}`")
    st.divider()

# ---------- Cards ----------
project_card(
    title="Option Pricing — European Call (BS + GBM)",
    subtitle="Pricing Black–Scholes, calcul des Greeks, simulation GBM et visuels.",
    badges=["Black–Scholes", "Greeks", "Simulation"],
    go_to="pages/Option_Pricing_European_Call.py",
    repo_rel_path="OptionPricing/EuropeenCall/",
    image_path="OptionPricing/EuropeenCall/results/gbm_paths.png",
)

project_card(
    title="Markowitz — Efficient Frontier",
    subtitle="Frontière efficiente, Min Var, Max Sharpe et Equal-Weight, avec visualisation interactive.",
    badges=["Portfolio", "Optimization", "Viz"],
    go_to="pages/Markowitz_Efficient_Frontier.py",
    repo_rel_path="Markowitz/",
    image_path="results/frontier_efficient.png",
)

project_card(
    title="Greeks — Live Visualizer",
    subtitle="Visualisation interactive des Greeks (Δ, Γ, Vega, Theta, Rho) et sensibilités en fonction de S.",
    badges=["Greeks", "Visualisation", "Education"],
    go_to="pages/Greeks_Live_Visualizer.py",
    repo_rel_path="LiveGreekVisuliser/",
    image_path=None,
)

st.info(
    "💡 Déploie ce dossier sur GitHub puis publie-le sur **Streamlit Community Cloud**. "
    "Clique sur les cartes pour accéder à chaque projet."
)