#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Markowitz (long-only) avec:
- Max Sharpe / Min Variance / Equal Weight
- Frontière efficiente (option --plot)
- Export CSV des résultats + PNG du graphique

Entrée: CSV de prix avec colonnes: Date,AAPL,MSFT,...
Sorties:
- results/results_markowitz.csv
- results/efficient_frontier.png (si --plot)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# Utilitaires rendement/risque
# -----------------------------

def load_prices(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    # garde uniquement colonnes numériques
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        raise ValueError("Le CSV doit contenir au moins 2 colonnes d'actifs numériques.")
    return df

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """log-returns journaliers"""
    r = np.log(prices / prices.shift(1)).dropna()
    return r

def annualize_from_logs(r_log: pd.DataFrame):
    """
    À partir de log-returns journaliers:
    - mu_log_ann: moyenne log annualisée
    - cov_log_ann: covariance log annualisée
    - mu_arith_ann: espérance arithmétique annualisée ≈ exp(mu + 0.5 * var) - 1
    """
    mu_log_ann = r_log.mean() * 252.0
    cov_log_ann = r_log.cov() * 252.0
    var_diag = np.diag(cov_log_ann.values)
    mu_arith_ann = np.exp(mu_log_ann.values + 0.5 * var_diag) - 1.0
    mu_arith_ann = pd.Series(mu_arith_ann, index=r_log.columns)
    return mu_log_ann, cov_log_ann, mu_arith_ann

def portfolio_path_mdd(r_log: pd.DataFrame, w: np.ndarray) -> float:
    """
    Max drawdown (MDD) à partir de log-returns journaliers du portefeuille.
    Retourne une valeur <= 0 (ex. -0.35 = -35%).
    """
    port_log = r_log.values @ w  # série log-return port
    cum = np.exp(np.cumsum(port_log))  # valeur cumulée (base 1)
    rolling_max = np.maximum.accumulate(cum)
    drawdown = cum / rolling_max - 1.0
    return float(np.min(drawdown))

def portfolio_perf(
    w: np.ndarray,
    mu_arith_ann: pd.Series,
    cov_log_ann: pd.DataFrame,
    r_log_hist: pd.DataFrame,
    rf: float,
):
    """
    Renvoie (exp_return, exp_vol, sharpe, max_drawdown) annualisés.
    - exp_return: espérance arithmétique annualisée
    - exp_vol: volatilité annualisée (à partir de covariance log)
    - sharpe: (exp_return - rf) / exp_vol
    - max_drawdown: sur historique (journaliers)
    """
    w = np.asarray(w)
    exp_return = float(np.dot(w, mu_arith_ann.values))
    vol = float(np.sqrt(w @ cov_log_ann.values @ w))
    sharpe = (exp_return - rf) / vol if vol > 0 else np.nan
    mdd = portfolio_path_mdd(r_log_hist, w)
    return exp_return, vol, sharpe, mdd

# -----------------------------
# Optimisations
# -----------------------------

def solve_min_variance(
    mu_arith_ann: pd.Series,
    cov_log_ann: pd.DataFrame,
    max_weight: float,
):
    n = len(mu_arith_ann)
    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    obj = lambda w: w @ cov_log_ann.values @ w
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Echec optimisation min-variance: {res.message}")
    return res.x

def solve_max_sharpe(
    mu_arith_ann: pd.Series,
    cov_log_ann: pd.DataFrame,
    rf: float,
    max_weight: float,
):
    n = len(mu_arith_ann)
    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        ret = np.dot(w, mu_arith_ann.values)
        vol = np.sqrt(w @ cov_log_ann.values @ w)
        return - (ret - rf) / vol if vol > 0 else 1e6

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Echec optimisation max-sharpe: {res.message}")
    return res.x

def efficient_frontier(
    mu_arith_ann: pd.Series,
    cov_log_ann: pd.DataFrame,
    max_weight: float,
    grid_size: int = 40,
):
    """
    Construit la frontière en minimisant la variance pour différents
    rendements cibles (contrainte dot(w, mu) = target).
    """
    n = len(mu_arith_ann)
    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n

    rets, vols = [], []
    # grille entre ~80% du min et 110% du max pour plus de couverture
    t_min, t_max = mu_arith_ann.min(), mu_arith_ann.max()
    targets = np.linspace(t_min * 0.8, t_max * 1.1, grid_size)

    for tr in targets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=tr: np.dot(w, mu_arith_ann.values) - tr},
        )
        obj = lambda w: w @ cov_log_ann.values @ w
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            w = res.x
            vol = float(np.sqrt(w @ cov_log_ann.values @ w))
            rets.append(float(tr))
            vols.append(vol)
        # si échec, on saute le point
    return pd.DataFrame({"exp_vol": vols, "exp_return": rets}).dropna()

# -----------------------------
# I/O résultats
# -----------------------------

def append_portfolio_rows(rows, name, metrics, tickers, weights):
    exp_return, exp_vol, sharpe, mdd = metrics
    rows.append((name, "exp_return", exp_return))
    rows.append((name, "exp_vol", exp_vol))
    rows.append((name, "sharpe", sharpe))
    rows.append((name, "max_drawdown", mdd))
    for t, w in zip(tickers, weights):
        rows.append((name, f"weight_{t}", float(w)))

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Markowitz (long-only) avec benchmark Equal Weight.")
    parser.add_argument("--csv", type=str, default="data/prices_yf.csv", help="Chemin du CSV de prix.")
    parser.add_argument("--rf", type=float, default=0.02, help="Taux sans risque annualisé (ex: 0.02 = 2%).")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Poids max par actif (ex: 0.4).")
    parser.add_argument("--plot", action="store_true", help="Tracer la frontière efficiente + points MS/MV/EW.")
    args = parser.parse_args()

    # 1) Données
    prices = load_prices(args.csv)
    tickers = list(prices.columns)
    n = len(tickers)

    # 2) Log-returns & annualisation
    r_log = compute_log_returns(prices)
    mu_log_ann, cov_log_ann, mu_arith_ann = annualize_from_logs(r_log)

    # 3) Portefeuilles
    w_ew = np.ones(n) / n
    w_mv = solve_min_variance(mu_arith_ann, cov_log_ann, args.max_weight)
    w_ms = solve_max_sharpe(mu_arith_ann, cov_log_ann, args.rf, args.max_weight)

    # 4) Métriques
    rows = []
    append_portfolio_rows(rows, "equal_weight",
                          portfolio_perf(w_ew, mu_arith_ann, cov_log_ann, r_log, args.rf),
                          tickers, w_ew)
    append_portfolio_rows(rows, "min_variance",
                          portfolio_perf(w_mv, mu_arith_ann, cov_log_ann, r_log, args.rf),
                          tickers, w_mv)
    append_portfolio_rows(rows, "max_sharpe",
                          portfolio_perf(w_ms, mu_arith_ann, cov_log_ann, r_log, args.rf),
                          tickers, w_ms)

    summary_df = pd.DataFrame(rows, columns=["portfolio", "metric", "value"])

    # 5) Sauvegarde CSV
    Path("results").mkdir(exist_ok=True)
    out_csv = Path("results/results_markowitz.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"✅ Résultats sauvegardés -> {out_csv}")

    # 6) Frontière efficiente (optionnel)
    if args.plot:
        ef = efficient_frontier(mu_arith_ann, cov_log_ann, args.max_weight, grid_size=60)

        plt.figure(figsize=(7, 5))
        # nuage de la frontière
        plt.scatter(ef["exp_vol"], ef["exp_return"], s=12, label="Efficient Frontier")

        # scatter des 3 portefeuilles
        for name, w in [("Equal Weight", w_ew), ("Max Sharpe", w_ms), ("Min Variance", w_mv)]:
            r_, v_, s_, _ = portfolio_perf(w, mu_arith_ann, cov_log_ann, r_log, args.rf)
            plt.scatter([v_], [r_], s=70, marker="x", label=name)

        plt.xlabel("Volatilité annualisée")
        plt.ylabel("Rendement annualisé")
        plt.title("Frontière efficiente (long-only) + EW / Max Sharpe / Min Var")
        plt.legend()
        plt.tight_layout()

        out_png = Path("results/efficient_frontier.png")
        plt.savefig(out_png, dpi=150)
        plt.show()
        print(f"🖼️ Graphique sauvegardé -> {out_png}")

if __name__ == "__main__":
    main()