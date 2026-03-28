"""
simulator.py — Poisson arrival process simulation engine.

Simulates order flow with:
- Uninformed traders: random limit/market orders (noise flow)
- Informed traders  : directional orders based on private signal
- Market maker      : continuous two-sided quoting with adverse selection
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from order import Order
from orderbook import OrderBook
from market_maker import MarketMaker


@dataclass
class SimulationParams:
    """All parameters for one simulation run."""
    # Time
    T:                  float = 3600.0   # simulation horizon (seconds)
    # Uninformed flow
    lambda_uninformed:  float = 2.0      # uninformed arrival rate (orders/sec)
    prob_market_order:  float = 0.3      # fraction of uninformed that are market orders
    prob_bid:           float = 0.5      # fraction of orders that are buys
    # Informed flow
    lambda_informed:    float = 0.2      # informed arrival rate (orders/sec)
    informed_fraction:  float = 0.1      # prob any given order is informed
    # Price process
    S0:                 float = 100.0    # initial mid-price
    sigma:              float = 0.01     # fundamental vol per second (GBM)
    # Order sizing
    min_qty:            int   = 1
    max_qty:            int   = 20
    informed_qty:       int   = 10       # informed traders trade larger
    # Market maker
    mm_base_spread:     float = 0.02
    mm_risk_aversion:   float = 0.05
    mm_max_inventory:   int   = 100
    mm_quote_qty:       int   = 15
    # Book initialisation
    n_levels:           int   = 10       # initial depth levels each side
    tick_size:          float = 0.01
    # Seed
    seed:               Optional[int] = 42


class Simulator:
    """
    Event-driven simulation of a limit order book.

    Flow model:
        - Fundamental value V(t) follows GBM
        - Uninformed traders submit random orders around current mid
        - Informed traders know V(t) and trade when mid deviates from V(t)
        - Market maker quotes continuously, updating after each event
    """

    def __init__(self, params: SimulationParams):
        self.p   = params
        self.rng = np.random.default_rng(params.seed)
        self.ob  = OrderBook(tick_size=params.tick_size)
        self.mm  = MarketMaker(
            base_spread=params.mm_base_spread,
            risk_aversion=params.mm_risk_aversion,
            max_inventory=params.mm_max_inventory,
            tick_size=params.tick_size,
        )
        self.fundamental_value_history: list[tuple[float, float]] = []
        self._fundamental_value = params.S0
        self._mm_bid_id: Optional[int] = None
        self._mm_ask_id: Optional[int] = None

    # ── Fundamental value ─────────────────────────────────────────────────────

    def _update_fundamental(self, dt: float) -> float:
        """GBM step for fundamental value."""
        shock = self.rng.normal(0, self.p.sigma * np.sqrt(dt))
        self._fundamental_value *= np.exp(shock)
        return self._fundamental_value

    # ── Order generation ──────────────────────────────────────────────────────

    def _uninformed_order(self, timestamp: float, mid: float) -> Order:
        """Random order from uninformed (noise) trader."""
        side     = "bid" if self.rng.random() < self.p.prob_bid else "ask"
        is_mkt   = self.rng.random() < self.p.prob_market_order
        qty      = int(self.rng.integers(self.p.min_qty, self.p.max_qty + 1))

        if is_mkt:
            return Order(side=side, order_type="market",
                         quantity=qty, timestamp=timestamp, is_informed=False)
        else:
            # Limit price: offset from mid by random number of ticks
            n_ticks = int(self.rng.integers(1, 6))
            offset  = n_ticks * self.p.tick_size
            price   = round(mid - offset if side == "bid" else mid + offset, 4)
            return Order(side=side, order_type="limit",
                         price=price, quantity=qty, timestamp=timestamp,
                         is_informed=False)

    def _informed_order(self, timestamp: float, mid: float, V: float) -> Optional[Order]:
        """
        Informed trader submits order when mid deviates from fundamental V.
        If V > mid: buy (price will rise). If V < mid: sell (price will fall).
        Only trades if deviation > 1 tick (otherwise not worth it).
        """
        deviation = V - mid
        if abs(deviation) < self.p.tick_size:
            return None  # deviation too small — skip

        side    = "bid" if deviation > 0 else "ask"
        is_mkt  = abs(deviation) > 5 * self.p.tick_size  # urgent if large deviation
        qty     = self.p.informed_qty

        if is_mkt:
            return Order(side=side, order_type="market",
                         quantity=qty, timestamp=timestamp, is_informed=True)
        else:
            # Limit at fundamental value
            price = round(V, 4)
            return Order(side=side, order_type="limit",
                         price=price, quantity=qty, timestamp=timestamp,
                         is_informed=True)

    # ── Market maker quoting ──────────────────────────────────────────────────

    def _refresh_mm_quotes(self, timestamp: float) -> None:
        """Cancel old MM quotes and submit fresh ones."""
        if self._mm_bid_id is not None:
            self.ob.cancel(self._mm_bid_id)
        if self._mm_ask_id is not None:
            self.ob.cancel(self._mm_ask_id)

        mid = self.ob.mid_price or self.p.S0
        bid, ask = self.mm.get_quotes(mid, timestamp, self.p.mm_quote_qty)

        if bid is not None:
            self.ob.submit(bid)
            self._mm_bid_id = bid.order_id
        if ask is not None:
            self.ob.submit(ask)
            self._mm_ask_id = ask.order_id

    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run full simulation. Returns results dict with all history arrays.

        Event loop:
          1. Sample next arrival time from Poisson process
          2. Update fundamental value
          3. Refresh MM quotes
          4. Generate and submit trader order
          5. Process any resulting trades
          6. Record state
        """
        p   = self.p
        T   = p.T
        t   = 0.0
        lam = p.lambda_uninformed + p.lambda_informed  # total arrival rate

        # Initialise book with MM quotes
        self._fundamental_value = p.S0
        # Seed the book with some initial depth
        for i in range(p.n_levels):
            price_bid = round(p.S0 - (i + 1) * p.tick_size, 4)
            price_ask = round(p.S0 + (i + 1) * p.tick_size, 4)
            qty = int(self.rng.integers(5, 30))
            self.ob.submit(Order("bid", "limit", qty, 0.0, price_bid))
            self.ob.submit(Order("ask", "limit", qty, 0.0, price_ask))

        n_events = 0
        print(f"Running simulation: T={T}s, λ={lam:.1f} orders/s, "
              f"informed fraction={p.informed_fraction:.0%}")

        while t < T:
            # Inter-arrival time (exponential)
            dt = self.rng.exponential(1.0 / lam)
            t  = min(t + dt, T)

            # Update fundamental
            V = self._update_fundamental(dt)
            self.fundamental_value_history.append((t, V))

            # Refresh MM quotes
            self._refresh_mm_quotes(t)

            # Decide: informed or uninformed?
            mid = self.ob.mid_price or p.S0
            is_informed = self.rng.random() < p.informed_fraction

            if is_informed:
                order = self._informed_order(t, mid, V)
                if order is None:
                    continue
            else:
                order = self._uninformed_order(t, mid)

            # Submit order and process trades
            trades = self.ob.submit(order)

            # Notify MM of any trades against its quotes
            for trade in trades:
                is_mm_bid = (trade.bid_order_id == self._mm_bid_id)
                is_mm_ask = (trade.ask_order_id == self._mm_ask_id)
                if is_mm_bid or is_mm_ask:
                    self.mm.process_trade(
                        trade_price=trade.price,
                        trade_qty=trade.quantity,
                        aggressor_side=trade.aggressor,
                        is_informed=trade.is_informed,
                        mid=mid,
                        timestamp=t,
                    )

            n_events += 1

        print(f"Simulation complete: {n_events} events, {len(self.ob.trades)} trades")

        return {
            "orderbook":             self.ob,
            "market_maker":          self.mm,
            "fundamental_history":   self.fundamental_value_history,
            "mid_price_history":     self.ob.mid_price_history,
            "spread_history":        self.ob.spread_history,
            "depth_history":         self.ob.depth_history,
            "trades":                self.ob.trades,
            "params":                p,
        }
