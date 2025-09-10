# live_greek_visualizer.py
# Interactive Black-Scholes Greeks visualizer (call/put) with sliders and real-time values.

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

# -----------------------------
# Utils: normal pdf / cdf
# -----------------------------
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / SQRT2PI

def _norm_cdf_scalar(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))

# -----------------------------
# Black-Scholes d1, d2
# -----------------------------
def d1(S, K, r, sigma, T):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * math.sqrt(T)

# -----------------------------
# Greeks
# -----------------------------
def greeks(S, K, r, sigma, T, option_type="call"):
    D1 = d1(S, K, r, sigma, T)
    D2 = D1 - sigma * math.sqrt(T)
    if np.isnan(D1):
        return np.nan, np.nan, np.nan, np.nan

    pdf_d1 = norm_pdf(D1)
    cdf_d1 = _norm_cdf_scalar(D1)
    cdf_m_d1 = _norm_cdf_scalar(-D1)
    cdf_d2 = _norm_cdf_scalar(D2)
    cdf_m_d2 = _norm_cdf_scalar(-D2)

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T)

    if option_type == "call":
        delta = cdf_d1
        theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * cdf_d2)
    else:
        delta = -cdf_m_d1
        theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * cdf_m_d2)

    return float(delta), float(gamma), float(vega), float(theta)

# -----------------------------
# Plot helper
# -----------------------------
def compute_curves(S0, K, r, sigma, T, opt):
    s_min = max(0.1, 0.2 * K)
    s_max = 2.0 * K
    S_grid = np.linspace(s_min, s_max, 400)

    Delta, Gamma, Vega, Theta = [], [], [], []
    for s in S_grid:
        dlt, gma, vga, tht = greeks(s, K, r, sigma, T, opt)
        Delta.append(dlt)
        Gamma.append(gma)
        Vega.append(vga)
        Theta.append(tht)

    return S_grid, np.array(Delta), np.array(Gamma), np.array(Vega), np.array(Theta)

def main():
    S0, K0, r0, sigma0, T0, opt0 = 100.0, 100.0, 0.02, 0.20, 0.5, "call"

    plt.close("all")

    # === 1. Delta ===
    fig_delta, ax_delta = plt.subplots()
    fig_delta.canvas.manager.set_window_title("Delta")
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # === 2. Gamma ===
    fig_gamma, ax_gamma = plt.subplots()
    fig_gamma.canvas.manager.set_window_title("Gamma")
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # === 3. Vega ===
    fig_vega, ax_vega = plt.subplots()
    fig_vega.canvas.manager.set_window_title("Vega")
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # === 4. Theta ===
    fig_theta, ax_theta = plt.subplots()
    fig_theta.canvas.manager.set_window_title("Theta")
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # Courbes initiales
    Sgrid, D, G, V, Th = compute_curves(S0, K0, r0, sigma0, T0, opt0)

    # Delta
    ln_delta, = ax_delta.plot(Sgrid, D, lw=2)
    vline_delta = ax_delta.axvline(S0, ls="--")
    point_delta, = ax_delta.plot([S0], [np.interp(S0, Sgrid, D)], "ro")
    text_delta = ax_delta.text(0.05, 0.95, f"Delta={D[200]:.3f}",
                               transform=ax_delta.transAxes, va="top")
    ax_delta.set_xlabel("Prix du sous-jacent S")
    ax_delta.set_ylabel("Delta (∂V/∂S)\n→ Sensibilité au spot")
    ax_delta.grid(True)

    # Gamma
    ln_gamma, = ax_gamma.plot(Sgrid, G, lw=2)
    vline_gamma = ax_gamma.axvline(S0, ls="--")
    point_gamma, = ax_gamma.plot([S0], [np.interp(S0, Sgrid, G)], "ro")
    text_gamma = ax_gamma.text(0.05, 0.95, f"Gamma={G[200]:.3f}",
                               transform=ax_gamma.transAxes, va="top")
    ax_gamma.set_xlabel("Prix du sous-jacent S")
    ax_gamma.set_ylabel("Gamma (∂²V/∂S²)\n→ Variation du Delta")
    ax_gamma.grid(True)

    # Vega
    ln_vega, = ax_vega.plot(Sgrid, V, lw=2)
    vline_vega = ax_vega.axvline(S0, ls="--")
    point_vega, = ax_vega.plot([S0], [np.interp(S0, Sgrid, V)], "ro")
    text_vega = ax_vega.text(0.05, 0.95, f"Vega={V[200]:.3f}",
                             transform=ax_vega.transAxes, va="top")
    ax_vega.set_xlabel("Prix du sous-jacent S")
    ax_vega.set_ylabel("Vega (∂V/∂σ)\n→ Sensibilité à la volatilité")
    ax_vega.grid(True)

    # Theta
    ln_theta, = ax_theta.plot(Sgrid, Th, lw=2)
    vline_theta = ax_theta.axvline(S0, ls="--")
    point_theta, = ax_theta.plot([S0], [np.interp(S0, Sgrid, Th)], "ro")
    text_theta = ax_theta.text(0.05, 0.95, f"Theta={Th[200]:.3f}",
                               transform=ax_theta.transAxes, va="top")
    ax_theta.set_xlabel("Prix du sous-jacent S")
    ax_theta.set_ylabel("Theta (∂V/∂t)\n→ Perte de valeur temps")
    ax_theta.grid(True)

    # === Sliders sur la figure Delta ===
    axcolor = "lightgoldenrodyellow"
    ax_S = fig_delta.add_axes([0.25, 0.24, 0.65, 0.03], facecolor=axcolor)
    ax_K = fig_delta.add_axes([0.25, 0.19, 0.65, 0.03], facecolor=axcolor)
    ax_T = fig_delta.add_axes([0.25, 0.14, 0.65, 0.03], facecolor=axcolor)
    ax_r = fig_delta.add_axes([0.25, 0.09, 0.65, 0.03], facecolor=axcolor)
    ax_sig = fig_delta.add_axes([0.25, 0.04, 0.65, 0.03], facecolor=axcolor)

    s_S = Slider(ax_S, "S (spot)", 10.0, 300.0, valinit=S0, valstep=0.5)
    s_K = Slider(ax_K, "K (strike)", 10.0, 300.0, valinit=K0, valstep=0.5)
    s_T = Slider(ax_T, "T (années)", 1/365, 3.0, valinit=T0)
    s_r = Slider(ax_r, "r (sans risque)", -0.02, 0.10, valinit=r0)
    s_sig = Slider(ax_sig, "σ (vol)", 0.01, 1.00, valinit=sigma0)

    # Radio Call/Put
    ax_radio = fig_delta.add_axes([0.05, 0.55, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(ax_radio, ("call", "put"), active=0)

    # Checkbox log-scale
    ax_chk = fig_delta.add_axes([0.05, 0.45, 0.15, 0.05], facecolor=axcolor)
    chk = CheckButtons(ax_chk, ["log(S)"], [False])

    # === Update callback ===
    def update(_=None):
        S, K, T, r, sig, opt = s_S.val, s_K.val, max(s_T.val, 1e-6), s_r.val, max(s_sig.val, 1e-6), radio.value_selected
        Sgrid, D, G, V, Th = compute_curves(S, K, r, sig, T, opt)

        for ax, ln, arr, vline, point, text, name in [
            (ax_delta, ln_delta, D, vline_delta, point_delta, text_delta, "Delta"),
            (ax_gamma, ln_gamma, G, vline_gamma, point_gamma, text_gamma, "Gamma"),
            (ax_vega, ln_vega, V, vline_vega, point_vega, text_vega, "Vega"),
            (ax_theta, ln_theta, Th, vline_theta, point_theta, text_theta, "Theta"),
        ]:
            ln.set_xdata(Sgrid)
            ln.set_ydata(arr)
            vline.set_xdata([S, S])
            point.set_xdata([S])
            point.set_ydata([np.interp(S, Sgrid, arr)])
            text.set_text(f"{name}={np.interp(S, Sgrid, arr):.3f}")
            ax.relim()
            ax.autoscale_view()

        for ax in [ax_delta, ax_gamma, ax_vega, ax_theta]:
            ax.set_xscale("log" if chk.get_status()[0] else "linear")

        fig_delta.canvas.draw_idle()
        fig_gamma.canvas.draw_idle()
        fig_vega.canvas.draw_idle()
        fig_theta.canvas.draw_idle()

    for s in [s_S, s_K, s_T, s_r, s_sig]:
        s.on_changed(update)
    radio.on_clicked(update)
    chk.on_clicked(update)

    update()
    plt.show()

if __name__ == "__main__":
    main()