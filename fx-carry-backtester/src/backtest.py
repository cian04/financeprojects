"""
backtest.py — Vectorised FX carry + momentum backtest engine v2.

Improvements:
- Composite momentum signal (1M + 3M + 12M)
- Carry threshold raised to 1% (filter ZIRP periods)
- Volatility regime filter (flat when portfolio vol > 15%)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from data import download_fx_prices, build_carry_differential
from signals import combined_signal, rank_signal, vol_regime_filter
from portfolio import apply_vol_targeting, apply_position_limits, portfolio_summary
from metrics import summary

START            = "2015-01-01"
END              = "2024-12-31"
TARGET_VOL       = 0.08
VOL_WINDOW       = 21
VOL_CAP          = 0.15
TRANSACTION_COST = 0.00005
RF_ANNUAL        = 0.02
RESULTS_DIR      = Path(__file__).parent.parent / "results"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})
BLUE, RED, GRAY, ORANGE = "#1B5FA8", "#C0392B", "#7F8C8D", "#E67E22"


def run_backtest() -> dict:
    print("Downloading FX prices...")
    prices  = download_fx_prices(START, END)
    pairs   = list(prices.columns)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    index   = log_ret.index
    carry_diff = build_carry_differential(index)[pairs]

    print(f"  Pairs  : {pairs}")
    print(f"  Period : {index[0].date()} -> {index[-1].date()} ({len(index)} days)")

    # Signals
    raw_signal    = combined_signal(carry_diff, log_ret, carry_threshold=0.01)[pairs]
    weights_rank  = rank_signal(raw_signal)
    weights_vt    = apply_vol_targeting(weights_rank, log_ret, TARGET_VOL, VOL_WINDOW)
    weights_final = apply_position_limits(weights_vt)

    # Vol regime filter — zero out all weights in high-vol periods
    vol_filter = vol_regime_filter(log_ret, weights_rank, vol_cap=VOL_CAP)
    weights_final = weights_final.multiply(vol_filter, axis=0)

    port = portfolio_summary(weights_final, log_ret, carry_diff, TRANSACTION_COST)

    # Fraction of days active
    active_days = (weights_final.abs().sum(axis=1) > 0).mean()
    print(f"  Active days: {active_days:.1%}")

    return {**port,
            "pairs": pairs,
            "pair_returns": weights_final * log_ret,
            "gross_returns": port["gross_returns"]}


def plot_cumulative(net: pd.Series, gross: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot((1 + net).cumprod(),   color=BLUE, lw=2,   label="Net (FX + carry - costs)")
    ax.plot((1 + gross).cumprod(), color=GRAY, lw=1.2, ls="--", label="Gross FX only")
    ax.axhline(1.0, color="black", lw=0.5, ls=":")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"),
               alpha=0.08, color=RED,    label="COVID crash")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               alpha=0.08, color=ORANGE, label="2022 rate shock")
    ax.set_title("FX Carry + Momentum v2 — Cumulative returns (2015–2024)", fontweight="bold")
    ax.set_ylabel("Portfolio value (base = 1)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cumulative_returns.png", dpi=180)
    plt.close(fig)
    print("  Saved: cumulative_returns.png")


def plot_drawdown(net: pd.Series) -> None:
    cum = (1 + net).cumprod()
    dd  = (cum / cum.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.5)
    ax.set_title("Drawdown (%)", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "drawdown.png", dpi=180)
    plt.close(fig)
    print("  Saved: drawdown.png")


def plot_annual(net: pd.Series) -> None:
    annual = net.resample("YE").apply(lambda r: (1 + r).prod() - 1) * 100
    colors = [BLUE if v >= 0 else RED for v in annual]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(annual.index.year, annual.values, color=colors, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Annual returns (%)", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xticks(annual.index.year)
    for yr, val in zip(annual.index.year, annual.values):
        ax.text(yr, val + (0.3 if val >= 0 else -0.8), f"{val:.1f}%",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "annual_returns.png", dpi=180)
    plt.close(fig)
    print("  Saved: annual_returns.png")


def plot_contributions(pair_returns: pd.DataFrame) -> None:
    total  = pair_returns.sum().sort_values()
    colors = [BLUE if v >= 0 else RED for v in total]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(total.index, total.values * 100, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Total return contribution by pair (2015–2024, %)", fontweight="bold")
    ax.set_xlabel("Contribution (%)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pair_contributions.png", dpi=180)
    plt.close(fig)
    print("  Saved: pair_contributions.png")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run_backtest()
    net, gross = results["net_returns"], results["gross_returns"]

    print("\nGenerating charts...")
    plot_cumulative(net, gross)
    plot_drawdown(net)
    plot_annual(net)
    plot_contributions(results["pair_returns"])

    perf = summary(net, "FX Carry + Momentum v2", RF_ANNUAL)
    print("\n" + "="*45)
    print("  PERFORMANCE SUMMARY (Net, 2015-2024)")
    print("="*45)
    for k, v in perf.items():
        print(f"  {k:<22} {v}")
    print("="*45)

    perf.to_frame().to_csv(RESULTS_DIR / "performance_summary.csv")
    net.to_csv(RESULTS_DIR / "daily_returns.csv", header=["net_return"])
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
