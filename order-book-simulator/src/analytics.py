"""
analytics.py — Metrics and visualisation for order book simulation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})
BLUE, RED, GREEN, GRAY, ORANGE = "#1B5FA8", "#C0392B", "#27AE60", "#7F8C8D", "#E67E22"


def _to_series(history: list[tuple[float, float]], name: str) -> pd.Series:
    if not history:
        return pd.Series(name=name, dtype=float)
    t, v = zip(*history)
    # Reset index to avoid duplicate timestamp issues
    return pd.Series(list(v), index=list(t), name=name)


def plot_mid_vs_fundamental(results: dict, out_dir: Path) -> None:
    mid  = _to_series(results["mid_price_history"], "Mid price")
    fund = _to_series(results["fundamental_history"], "Fundamental value")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(mid.index,  mid.values,  color=BLUE, lw=1.2, label="Mid price (LOB)")
    ax.plot(fund.index, fund.values, color=RED,  lw=1.0, ls="--", alpha=0.8,
            label="Fundamental value V(t)")
    ax.set_title("Mid price vs fundamental value", fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mid_price_vs_fundamental.png", dpi=180)
    plt.close(fig)
    print("  Saved: mid_price_vs_fundamental.png")


def plot_spread_dynamics(results: dict, out_dir: Path) -> None:
    spread = _to_series(results["spread_history"], "Spread")
    mm     = results["market_maker"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(spread.index, spread.values * 100, color=BLUE, lw=1, alpha=0.7)
    axes[0].set_title("Realised bid-ask spread (bps)", fontweight="bold")
    axes[0].set_ylabel("Spread (× 100)")

    if mm.state.quote_history:
        t_q, bid_q, ask_q, _ = zip(*mm.state.quote_history)
        quoted_spread = [a - b for a, b in zip(ask_q, bid_q)]
        axes[1].plot(list(t_q), quoted_spread, color=ORANGE, lw=1.2)
        axes[1].set_title("Market maker quoted spread", fontweight="bold")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Spread")

    fig.tight_layout()
    fig.savefig(out_dir / "spread_dynamics.png", dpi=180)
    plt.close(fig)
    print("  Saved: spread_dynamics.png")


def plot_market_maker_pnl(results: dict, out_dir: Path) -> None:
    mm = results["market_maker"]
    if not mm.state.pnl_history:
        return

    t_pnl, pnl = zip(*mm.state.pnl_history)
    t_inv, inv  = zip(*mm.state.inventory_history)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(t_pnl, pnl, color=GREEN, lw=1.5, label="Cumulative P&L")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].fill_between(t_pnl, pnl, 0,
                          where=[p >= 0 for p in pnl], alpha=0.2, color=GREEN)
    axes[0].fill_between(t_pnl, pnl, 0,
                          where=[p < 0 for p in pnl], alpha=0.2, color=RED)
    axes[0].set_title("Market maker cumulative P&L", fontweight="bold")
    axes[0].set_ylabel("P&L")
    axes[0].legend()

    axes[1].plot(t_inv, inv, color=ORANGE, lw=1.2)
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].fill_between(t_inv, inv, 0, alpha=0.2, color=ORANGE)
    axes[1].set_title("Market maker net inventory", fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Inventory (units)")

    fig.tight_layout()
    fig.savefig(out_dir / "market_maker_pnl.png", dpi=180)
    plt.close(fig)
    print("  Saved: market_maker_pnl.png")


def plot_depth_profile(results: dict, out_dir: Path) -> None:
    ob   = results["orderbook"]
    snap = ob.snapshot(levels=10)
    if not snap["bids"] or not snap["asks"]:
        return

    bid_prices = [p for p, _ in snap["bids"]]
    bid_qtys   = [q for _, q in snap["bids"]]
    ask_prices = [p for p, _ in snap["asks"]]
    ask_qtys   = [q for _, q in snap["asks"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(bid_prices,  bid_qtys,              height=0.008, color=GREEN, alpha=0.8, label="Bids")
    ax.barh(ask_prices, [-q for q in ask_qtys], height=0.008, color=RED,   alpha=0.8, label="Asks")
    if snap["mid"]:
        ax.axhline(snap["mid"], color="black", lw=1, ls="--",
                   label=f"Mid {snap['mid']:.4f}")
    ax.set_title("Order book depth snapshot (final state)", fontweight="bold")
    ax.set_xlabel("Quantity (+ bids, - asks)")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "depth_profile.png", dpi=180)
    plt.close(fig)
    print("  Saved: depth_profile.png")


def plot_trade_distribution(results: dict, out_dir: Path) -> None:
    trades = results["trades"]
    if not trades:
        return

    sizes      = [t.quantity for t in trades]
    informed   = [t.quantity for t in trades if t.is_informed]
    t_times    = [t.timestamp for t in trades]
    prices     = [t.price for t in trades]
    colors     = [RED if t.is_informed else BLUE for t in trades]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(sizes, bins=20, color=BLUE, alpha=0.7, label="All trades")
    if informed:
        axes[0].hist(informed, bins=20, color=RED, alpha=0.6, label="Informed")
    axes[0].set_title("Trade size distribution", fontweight="bold")
    axes[0].set_xlabel("Trade size (units)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].scatter(t_times, prices, c=colors, s=8, alpha=0.5)
    axes[1].set_title("Trade prices over time\n(red=informed, blue=uninformed)",
                       fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Trade price")

    fig.tight_layout()
    fig.savefig(out_dir / "trade_distribution.png", dpi=180)
    plt.close(fig)
    print("  Saved: trade_distribution.png")


def compute_metrics(results: dict) -> pd.Series:
    ob     = results["orderbook"]
    mm     = results["market_maker"]
    trades = results["trades"]
    p      = results["params"]

    mid_series    = _to_series(results["mid_price_history"], "mid")
    spread_series = _to_series(results["spread_history"], "spread")
    fund_series   = _to_series(results["fundamental_history"], "fund")

    price_range = float(mid_series.max() - mid_series.min()) if len(mid_series) > 0 else 0

    # Tracking error: interpolate both series onto common numeric index
    if len(mid_series) > 10 and len(fund_series) > 10:
        # Use numpy interpolation to avoid duplicate index issues
        fund_interp = np.interp(
            mid_series.index,
            fund_series.index,
            fund_series.values
        )
        tracking_error = float(np.abs(mid_series.values - fund_interp).mean())
    else:
        tracking_error = float("nan")

    n_informed = sum(1 for t in trades if t.is_informed)

    return pd.Series({
        "Simulation horizon (s)":   p.T,
        "Total trades":             len(trades),
        "Informed trades":          n_informed,
        "Informed fraction":        f"{n_informed/len(trades)*100:.1f}%" if trades else "N/A",
        "Avg trade size":           f"{np.mean([t.quantity for t in trades]):.1f}" if trades else "N/A",
        "Avg spread":               f"{spread_series.mean():.4f}" if len(spread_series) > 0 else "N/A",
        "Min spread":               f"{spread_series.min():.4f}" if len(spread_series) > 0 else "N/A",
        "Max spread":               f"{spread_series.max():.4f}" if len(spread_series) > 0 else "N/A",
        "Price range":              f"{price_range:.4f}",
        "Mid-fundamental tracking": f"{tracking_error:.4f}",
        "MM final P&L":             f"{mm.state.pnl:.4f}",
        "MM final inventory":       mm.state.inventory,
        "MM spread earned":         f"{mm.state.total_spread_earned:.4f}",
        "MM informed flow EWM":     f"{mm._informed_flow_ewm:.3f}",
    }, name="Simulation results")


def run_analytics(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    print("\nGenerating charts...")

    plot_mid_vs_fundamental(results, out_dir)
    plot_spread_dynamics(results, out_dir)
    plot_market_maker_pnl(results, out_dir)
    plot_depth_profile(results, out_dir)
    plot_trade_distribution(results, out_dir)

    metrics = compute_metrics(results)
    print("\n" + "="*50)
    print("  SIMULATION SUMMARY")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<35} {v}")
    print("="*50)

    metrics.to_csv(out_dir / "performance_summary.csv")
    print(f"\nResults saved to {out_dir}/")
