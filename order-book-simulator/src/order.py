"""
order.py — Order dataclass and order types.

An order has:
- order_id   : unique identifier
- side       : 'bid' (buy) or 'ask' (sell)
- order_type : 'limit' or 'market'
- price      : limit price (None for market orders)
- quantity   : number of units
- timestamp  : arrival time (simulation clock)
- is_informed: True if order carries private information (adverse selection model)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import itertools

_id_counter = itertools.count(1)


@dataclass
class Order:
    side:        str            # 'bid' | 'ask'
    order_type:  str            # 'limit' | 'market'
    quantity:    int
    timestamp:   float
    price:       Optional[float] = None
    is_informed: bool           = False
    order_id:    int            = field(default_factory=lambda: next(_id_counter))

    def __post_init__(self):
        assert self.side in ("bid", "ask"), f"Invalid side: {self.side}"
        assert self.order_type in ("limit", "market"), f"Invalid type: {self.order_type}"
        assert self.quantity > 0, "Quantity must be positive"
        if self.order_type == "limit":
            assert self.price is not None, "Limit orders require a price"

    def __repr__(self):
        p = f"@{self.price:.4f}" if self.price else "@MKT"
        informed = " [INF]" if self.is_informed else ""
        return f"Order({self.order_id} {self.side.upper()} {self.quantity}{p}{informed} t={self.timestamp:.3f})"


@dataclass
class Trade:
    """Record of a matched trade."""
    timestamp:    float
    price:        float
    quantity:     int
    aggressor:    str    # 'bid' | 'ask' — which side initiated
    bid_order_id: int
    ask_order_id: int
    is_informed:  bool   # True if either leg was informed

    def __repr__(self):
        return (f"Trade(t={self.timestamp:.3f} qty={self.quantity} "
                f"px={self.price:.4f} aggr={self.aggressor}"
                f"{' [INF]' if self.is_informed else ''})")
