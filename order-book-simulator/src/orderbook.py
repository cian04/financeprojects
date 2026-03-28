"""
orderbook.py — Limit Order Book matching engine.

Maintains two sorted queues:
- Bids: sorted descending by price (highest bid first)
- Asks: sorted ascending by price (lowest ask first)

Matching rule: price-time priority (FIFO at same price level).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import heapq

from order import Order, Trade


class PriceLevel:
    """All orders at a single price level, FIFO queue."""

    def __init__(self, price: float):
        self.price    = price
        self.orders:  deque[Order] = deque()
        self.total_qty: int = 0

    def add(self, order: Order) -> None:
        self.orders.append(order)
        self.total_qty += order.quantity

    def is_empty(self) -> bool:
        return len(self.orders) == 0

    def __repr__(self):
        return f"PriceLevel({self.price:.4f} qty={self.total_qty} n={len(self.orders)})"


class OrderBook:
    """
    Limit Order Book with price-time priority matching.

    State:
        bids: dict[price -> PriceLevel], sorted descending
        asks: dict[price -> PriceLevel], sorted ascending
        trades: list of all executed trades
        mid_price_history: list of (timestamp, mid_price)
        spread_history: list of (timestamp, spread)
    """

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids:  dict[float, PriceLevel] = {}
        self.asks:  dict[float, PriceLevel] = {}
        self.trades: list[Trade] = []
        self.mid_price_history:  list[tuple[float, float]] = []
        self.spread_history:     list[tuple[float, float]] = []
        self.depth_history:      list[tuple[float, int, int]] = []
        self._cancelled_ids:     set[int] = set()

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def bid_depth(self) -> int:
        return sum(pl.total_qty for pl in self.bids.values())

    @property
    def ask_depth(self) -> int:
        return sum(pl.total_qty for pl in self.asks.values())

    def snapshot(self, levels: int = 5) -> dict:
        """Top-N price levels on each side."""
        bid_levels = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_levels = sorted(self.asks.keys())[:levels]
        return {
            "bids": [(p, self.bids[p].total_qty) for p in bid_levels],
            "asks": [(p, self.asks[p].total_qty) for p in ask_levels],
            "mid":  self.mid_price,
            "spread": self.spread,
        }

    # ── Order submission ──────────────────────────────────────────────────────

    def submit(self, order: Order) -> list[Trade]:
        """Submit an order. Returns list of trades generated."""
        if order.order_type == "market":
            return self._match_market(order)
        else:
            return self._match_limit(order)

    def _match_market(self, order: Order) -> list[Trade]:
        """Market order: match against best available prices."""
        trades = []
        remaining = order.quantity

        if order.side == "bid":
            # Buy market: match against asks (ascending)
            while remaining > 0 and self.asks:
                best_ask = min(self.asks.keys())
                level    = self.asks[best_ask]
                trades += self._fill_from_level(
                    order, level, remaining, best_ask, "bid"
                )
                remaining -= sum(t.quantity for t in trades) - sum(
                    t.quantity for t in self.trades[:-len(trades)]
                ) if trades else 0
                remaining = order.quantity - sum(t.quantity for t in trades)
                if level.is_empty():
                    del self.asks[best_ask]
        else:
            # Sell market: match against bids (descending)
            while remaining > 0 and self.bids:
                best_bid = max(self.bids.keys())
                level    = self.bids[best_bid]
                trades += self._fill_from_level(
                    order, level, remaining, best_bid, "ask"
                )
                remaining = order.quantity - sum(t.quantity for t in trades)
                if level.is_empty():
                    del self.bids[best_bid]

        self.trades.extend(trades)
        self._record_state(order.timestamp)
        return trades

    def _match_limit(self, order: Order) -> list[Trade]:
        """Limit order: match if marketable, else rest in book."""
        trades = []
        remaining = order.quantity

        if order.side == "bid":
            # Marketable if bid >= best ask
            while remaining > 0 and self.asks:
                best_ask = min(self.asks.keys())
                if order.price < best_ask:
                    break
                level   = self.asks[best_ask]
                filled  = self._fill_from_level(order, level, remaining, best_ask, "bid")
                trades += filled
                remaining -= sum(t.quantity for t in filled)
                if level.is_empty():
                    del self.asks[best_ask]
            # Rest in book if unfilled
            if remaining > 0:
                resting = Order(
                    side="bid", order_type="limit",
                    price=order.price, quantity=remaining,
                    timestamp=order.timestamp,
                    is_informed=order.is_informed,
                    order_id=order.order_id,
                )
                if order.price not in self.bids:
                    self.bids[order.price] = PriceLevel(order.price)
                self.bids[order.price].add(resting)

        else:
            # Marketable if ask <= best bid
            while remaining > 0 and self.bids:
                best_bid = max(self.bids.keys())
                if order.price > best_bid:
                    break
                level   = self.bids[best_bid]
                filled  = self._fill_from_level(order, level, remaining, best_bid, "ask")
                trades += filled
                remaining -= sum(t.quantity for t in filled)
                if level.is_empty():
                    del self.bids[best_bid]
            if remaining > 0:
                resting = Order(
                    side="ask", order_type="limit",
                    price=order.price, quantity=remaining,
                    timestamp=order.timestamp,
                    is_informed=order.is_informed,
                    order_id=order.order_id,
                )
                if order.price not in self.asks:
                    self.asks[order.price] = PriceLevel(order.price)
                self.asks[order.price].add(resting)

        self.trades.extend(trades)
        self._record_state(order.timestamp)
        return trades

    def _fill_from_level(
        self,
        aggressor: Order,
        level: PriceLevel,
        max_qty: int,
        fill_price: float,
        aggressor_side: str,
    ) -> list[Trade]:
        """Fill as many orders from a price level as possible."""
        trades = []
        while level.orders and max_qty > 0:
            passive = level.orders[0]
            if passive.order_id in self._cancelled_ids:
                level.orders.popleft()
                continue
            fill_qty = min(passive.quantity, max_qty)
            t = Trade(
                timestamp=aggressor.timestamp,
                price=fill_price,
                quantity=fill_qty,
                aggressor=aggressor_side,
                bid_order_id=aggressor.order_id if aggressor_side == "bid" else passive.order_id,
                ask_order_id=passive.order_id if aggressor_side == "bid" else aggressor.order_id,
                is_informed=aggressor.is_informed or passive.is_informed,
            )
            trades.append(t)
            max_qty -= fill_qty
            level.total_qty -= fill_qty
            passive.quantity -= fill_qty
            if passive.quantity == 0:
                level.orders.popleft()
        return trades

    def _record_state(self, timestamp: float) -> None:
        mid = self.mid_price
        spd = self.spread
        if mid is not None:
            self.mid_price_history.append((timestamp, mid))
        if spd is not None:
            self.spread_history.append((timestamp, spd))
        self.depth_history.append((timestamp, self.bid_depth, self.ask_depth))

    def cancel(self, order_id: int) -> None:
        """Mark an order as cancelled (lazy deletion)."""
        self._cancelled_ids.add(order_id)
