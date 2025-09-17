"""
Simulateur de Mouvement Brownien Géométrique (GBM)
SDE: dS_t = mu * S_t dt + sigma * S_t dW_t
Solution exacte discrétisée:
S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),  Z~N(0,1)

Sorties:
- results/gbm_paths.png : N trajectoires (par défaut 10)
- results/gbm_terminal_hist.png : histogramme de S_T vs densité log-normale théorique
- results/gbm_one_path.csv : une trajectoire (pour inspection)
- Impression des moments empiriques vs théoriques
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, log, pi

def simulate_gbm_paths(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: int | None = 42):
    """
    Simule n_paths trajectoires GBM avec solution exacte.

    Paramètres
    ----------
    S0 : float
        Prix initial S(0)
    mu : float
        Drift (rendement espéré) annuel
    sigma : float
        Volatilité annuelle (écart-type)
    T : float
        Horizon (années)
    n_steps : int
        Nombre de pas de temps
    n_paths : int
        Nombre de trajectoires simulées
    seed : int | None
        Graine pour reproductibilité

    Retour
    ------
    t : np.ndarray shape (n_steps+1,)
        Grille de temps
    paths : np.ndarray shape (n_paths, n_steps+1)
        Trajectoires simulées
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    # bruit gaussien
    Z = np.random.normal(size=(n_paths, n_steps))

    drift = (mu - 0.5 * sigma**2) * dt
    diff = sigma * sqrt(dt)

    # construction multiplicative
    for k in range(n_steps):
        increment = drift + diff * Z[:, k]
        paths[:, k + 1] = paths[:, k] * np.exp(increment)

    return t, paths

def lognormal_pdf(x, m, v):
    """
    Densité log-normale pour X ~ LogNormal(m, v) avec ln(X) ~ N(m, v).
    Ici, sous GBM: ln(S_T) ~ N( ln(S0) + (mu - 0.5*sigma^2)T , sigma^2 T )
    """
    # éviter division par zéro
    out = np.zeros_like(x, dtype=float)
    positive = x > 0
    out[positive] = (1.0 / (x[positive] * np.sqrt(2 * pi * v))) * np.exp(-(np.log(x[positive]) - m)**2 / (2 * v))
    return out

def main():
    parser = argparse.ArgumentParser(description="Simulation de GBM et comparaisons analytiques.")
    parser.add_argument("--S0", type=float, default=100.0, help="Spot initial")
    parser.add_argument("--mu", type=float, default=0.08, help="Drift annuel")
    parser.add_argument("--sigma", type=float, default=0.20, help="Volatilité annuelle")
    parser.add_argument("--T", type=float, default=1.0, help="Horizon en années")
    parser.add_argument("--steps", type=int, default=252, help="Pas de temps (ex: 252 jours)")
    parser.add_argument("--paths", type=int, default=10, help="Nombre de trajectoires à tracer")
    parser.add_argument("--big_paths", type=int, default=20000, help="Nb de trajectoires pour histogramme de S_T")
    parser.add_argument("--seed", type=int, default=42, help="Graine RNG")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # --- 1) Trajectoires à afficher
    t, paths = simulate_gbm_paths(args.S0, args.mu, args.sigma, args.T, args.steps, args.paths, seed=args.seed)

    # Plot des trajectoires
    plt.figure(figsize=(10, 6))
    for i in range(args.paths):
        plt.plot(t, paths[i, :], alpha=0.9, linewidth=1.2)
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix S(t)")
    plt.title(f"GBM paths — S0={args.S0}, mu={args.mu}, sigma={args.sigma}, T={args.T}, steps={args.steps}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/gbm_paths.png", dpi=200)
    plt.close()

    # Sauvegarde d'une trajectoire pour inspection
    np.savetxt("results/gbm_one_path.csv", np.column_stack([t, paths[0]]), delimiter=",", header="t,S", comments="")

    # --- 2) Histogramme de S_T + densité théorique
    # Pour une bonne comparaison statistique, on simule beaucoup de trajectoires juste pour le terminal
    _, big_paths = simulate_gbm_paths(args.S0, args.mu, args.sigma, args.T, args.steps, args.big_paths, seed=args.seed)
    ST = big_paths[:, -1]

    # Moments théoriques sous dynamique P (mu, sigma constants)
    # E[S_T] = S0 * exp(mu * T)
    mean_th = args.S0 * exp(args.mu * args.T)
    # Var[S_T] = S0^2 * exp(2 mu T) * (exp(sigma^2 T) - 1)
    var_th = (args.S0**2) * exp(2 * args.mu * args.T) * (exp((args.sigma**2) * args.T) - 1)

    # Moments empiriques
    mean_emp = float(np.mean(ST))
    var_emp = float(np.var(ST, ddof=0))

    print("=== Moments S_T ===")
    print(f"E[S_T]   théorique = {mean_th:.6f}   | empirique = {mean_emp:.6f}")
    print(f"Var[S_T] théorique = {var_th:.6f}    | empirique = {var_emp:.6f}")

    # Paramètres de la loi log-normale pour S_T
    m = log(args.S0) + (args.mu - 0.5 * args.sigma**2) * args.T  # moyenne de ln(S_T)
    v = (args.sigma**2) * args.T                                 # variance de ln(S_T)

    # Histogramme + densité
    plt.figure(figsize=(10, 6))
    # histogramme normalisé (densité)
    counts, bins, _ = plt.hist(ST, bins=80, density=True, alpha=0.5, label="Simulé S_T")

    # Courbe de densité log-normale
    x = np.linspace(max(1e-8, np.min(ST)), np.max(ST), 400)
    pdf = lognormal_pdf(x, m, v)
    plt.plot(x, pdf, linewidth=2.0, label="Théorie (log-normale)")

    plt.xlabel("S_T")
    plt.ylabel("Densité")
    plt.title(f"Distribution de S_T — comparaison simulation vs théorie\nS0={args.S0}, mu={args.mu}, sigma={args.sigma}, T={args.T}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/gbm_terminal_hist.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()