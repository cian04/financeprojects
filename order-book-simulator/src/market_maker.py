"""
market_maker.py — Market maker agent with adverse selection model.

The market maker continuously quotes bid and ask prices around the mid-price.
It adjusts its spread based on:
  1. Inventory risk    — wider spread when holding too much inventory
  2. Adverse selection — wider spread when detecting informed flow
  3. Volatility        — wider spread in high-vol environments

Based on:
  - Glosten-Milgrom (1985): adverse selection in bid-ask spreads
  - Avellaneda-Stoikov (2008): inventory-based market making
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from order import Order


@dataclass
class MarketMakerState:
    """Tracks the market maker's current state."""
    inventory:          int   = 0       # net position (+ long, - short)
    pnl:                float = 0.0     # cumulative P&L
    n_trades:           int   = 0       # total trades executed
    n_informed_trades:  int   = 0       # trades against informed flow
    total_spread_earned: float = 0.0   # total spread revenue
    quote_history:      list  = field(default_factory=list)
    pnl_history:        list  = field(default_factory=list)
    inventory_history:  list  = field(default_factory=list)


class MarketMaker:
    """
    Adaptive market maker agent.

    Quotes bid and ask around mid-price, adjusting for:
    - Base half-spread (sigma * sqrt(dt) / 2 — Avellaneda-Stoikov)
    - Inventory penalty (risk aversion * inventory)
    - Adverse selection premium (learned from informed trade frequency)
    - Volatility regime (wider in high vol)

    Parameters
    ----------
    base_spread     : Minimum half-spread in price units
    risk_aversion   : Inventory penalty coefficient (gamma in A-S model)
    max_inventory   : Hard limit — refuses to quote one side if exceeded
    informed_memory : EWM decay for informed flow detection (0 < alpha < 1)
    tick_size       : Minimum price increment
    """

    def __init__(
        self,
        base_spread:     float = 0.02,
        risk_aversion:   float = 0.1,
        max_inventory:   int   = 50,
        informed_memory: float = 0.05,
        tick_size:       float = 0.01,
    ):
        self.base_spread     = base_spread
        self.risk_aversion   = risk_aversion
        self.max_inventory   = max_inventory
        self.informed_memory = informed_memory
        self.tick_size       = tick_size

        self.state               = MarketMakerState()
        self._informed_flow_ewm  = 0.0   # EWM estimate of informed flow fraction
        self._vol_ewm            = 0.0   # EWM estimate of mid-price volatility
        self._last_mid           = None

    # ── Quoting ───────────────────────────────────────────────────────────────

    def compute_spread(self, mid: float, timestamp: float) -> float:
        """
        Compute current quoted half-spread.

        Half-spread = base + inventory_penalty + adverse_selection_premium + vol_premium
        """
        # 1. Inventory penalty: grow spread linearly with |inventory|
        inventory_penalty = (
            self.risk_aversion * abs(self.state.inventory) * self.tick_size
        )

        # 2. Adverse selection premium: grow with estimated informed flow fraction
        adverse_premium = self._informed_flow_ewm * self.base_spread * 3.0

        # 3. Volatility premium: grow with realised vol estimate
        vol_premium = self._vol_ewm * 0.5

        half_spread = (
            self.base_spread
            + inventory_penalty
            + adverse_premium
            + vol_premium
        )

        # Minimum 1 tick
        return max(half_spread, self.tick_size)

    def get_quotes(
        self, mid: float, timestamp: float, quantity: int = 10
    ) -> tuple[Optional[Order], Optional[Order]]:
        """
        Generate bid and ask limit orders around mid.

        Returns (bid_order, ask_order).
        Returns None for a side if inventory limit exceeded.
        """
        half_spread = self.compute_spread(mid, timestamp)

        # Update vol estimate
        if self._last_mid is not None:
            ret = abs(mid - self._last_mid) / self._last_mid
            self._vol_ewm = (
                0.05 * ret + 0.95 * self._vol_ewm
            )
        self._last_mid = mid

        bid_price = round(mid - half_spread, 4)
        ask_price = round(mid + half_spread, 4)

        # Record quote
        self.state.quote_history.append((timestamp, bid_price, ask_price, mid))

        # Inventory limits: refuse to quote side that worsens inventory beyond limit
        bid_order = None
        ask_order = None

        if self.state.inventory > -self.max_inventory:
            bid_order = Order(
                side="bid", order_type="limit",
                price=bid_price, quantity=quantity,
                timestamp=timestamp, is_informed=False,
            )

        if self.state.inventory < self.max_inventory:
            ask_order = Order(
                side="ask", order_type="limit",
                price=ask_price, quantity=quantity,
                timestamp=timestamp, is_informed=False,
            )

        return bid_order, ask_order

    # ── Trade processing ──────────────────────────────────────────────────────

    def process_trade(
        self,
        trade_price: float,
        trade_qty: int,
        aggressor_side: str,
        is_informed: bool,
        mid: float,
        timestamp: float,
    ) -> None:
        """
        Update state after a trade against the market maker's quotes.

        aggressor_side: 'bid' (buyer hit our ask) or 'ask' (seller hit our bid)
        """
        self.state.n_trades += 1

        if aggressor_side == "bid":
            # Buyer lifted our ask: we sold, inventory decreases
            self.state.inventory -= trade_qty
            half_spread = trade_price - mid
            self.state.total_spread_earned += half_spread * trade_qty
            self.state.pnl += half_spread * trade_qty
        else:
            # Seller hit our bid: we bought, inventory increases
            self.state.inventory += trade_qty
            half_spread = mid - trade_price
            self.state.total_spread_earned += half_spread * trade_qty
            self.state.pnl += half_spread * trade_qty

        # Adverse selection: informed trades move price against us
        if is_informed:
            self.state.n_informed_trades += 1
            # Informed trades cause inventory losses
            adverse_loss = abs(mid - trade_price) * trade_qty * 0.5
            self.state.pnl -= adverse_loss

        # Update informed flow EWM
        informed_flag = 1.0 if is_informed else 0.0
        self._informed_flow_ewm = (
            self.informed_memory * informed_flag
            + (1 - self.informed_memory) * self._informed_flow_ewm
        )

        self.state.pnl_history.append((timestamp, self.state.pnl))
        self.state.inventory_history.append((timestamp, self.state.inventory))

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        informed_pct = (
            self.state.n_informed_trades / self.state.n_trades * 100
            if self.state.n_trades > 0 else 0
        )
        return {
            "Total trades":        self.state.n_trades,
            "Informed trades":     f"{self.state.n_informed_trades} ({informed_pct:.1f}%)",
            "Final inventory":     self.state.inventory,
            "Total spread earned": f"{self.state.total_spread_earned:.4f}",
            "Final P&L":           f"{self.state.pnl:.4f}",
            "Informed flow EWM":   f"{self._informed_flow_ewm:.3f}",
        }
