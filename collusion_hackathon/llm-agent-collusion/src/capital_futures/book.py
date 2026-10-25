"""Order book with the legacy round-based matching convention.

After all of a round's actions are applied, the best bid is matched against the best ask while
they cross. Each match trades at the midpoint of the two limit prices (rounded half-to-even to the
cent, like the legacy ``round(x, 2)``) for the smaller remaining quantity. Price priority first,
then time priority (lower submission sequence). Sequences follow the seeded processing order, never
API response speed. Unfilled remainders persist to the next round.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Order:
    order_id: str
    agent_id: str
    market: str           # "spot" or "futures"
    side: str             # "bid" or "ask"
    price_cents: int
    quantity: int
    remaining: int
    round_submitted: int
    seq: int
    status: str = "open"  # open | filled | canceled | expired
    disclose_id: bool = False  # futures only; spot quotes always show the agent id

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fill:
    market: str
    round_number: int
    buyer_id: str
    seller_id: str
    price_cents: int
    quantity: int
    bid_order_id: str
    ask_order_id: str
    liquidation: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def midpoint_cents(bid_cents: int, ask_cents: int) -> int:
    return round((bid_cents + ask_cents) / 2)


class OrderBook:
    def __init__(self, market: str, tick_cents: int = 1):
        self.market = market
        self.tick_cents = tick_cents
        self.orders: dict[str, Order] = {}

    def open_orders(self, side: Optional[str] = None, exclude_agent: Optional[str] = None) -> list[Order]:
        out = [o for o in self.orders.values() if o.status == "open"
               and (side is None or o.side == side) and o.agent_id != exclude_agent]
        if side == "bid":
            out.sort(key=lambda o: (-o.price_cents, o.seq))
        elif side == "ask":
            out.sort(key=lambda o: (o.price_cents, o.seq))
        return out

    def agent_order(self, agent_id: str, side: str) -> Optional[Order]:
        for o in self.orders.values():
            if o.status == "open" and o.agent_id == agent_id and o.side == side:
                return o
        return None

    def add(self, order: Order) -> None:
        if self.agent_order(order.agent_id, order.side) is not None:
            raise ValueError(f"{order.agent_id} already has an open {order.side} in {self.market}")
        self.orders[order.order_id] = order

    def cancel(self, order_id: str, status: str = "canceled") -> Order:
        order = self.orders[order_id]
        if order.status != "open":
            raise ValueError(f"order {order_id} is {order.status}, not open")
        order.status = status
        return order

    def best(self, side: str) -> Optional[Order]:
        orders = self.open_orders(side)
        return orders[0] if orders else None

    def match(self, round_number: int) -> list[Fill]:
        fills = []
        while True:
            bid, ask = self.best("bid"), self.best("ask")
            if bid is None or ask is None or bid.price_cents < ask.price_cents:
                return fills
            if bid.agent_id == ask.agent_id:
                # Entry validation rejects self-crossing orders, so this is a defensive guard:
                # cancel the newer order rather than trade with oneself.
                newer = bid if bid.seq > ask.seq else ask
                newer.status = "canceled"
                continue
            qty = min(bid.remaining, ask.remaining)
            fills.append(Fill(self.market, round_number, bid.agent_id, ask.agent_id,
                              round((bid.price_cents + ask.price_cents) / (2 * self.tick_cents)) * self.tick_cents, qty,
                              bid.order_id, ask.order_id))
            for o in (bid, ask):
                o.remaining -= qty
                if o.remaining == 0:
                    o.status = "filled"

    def visible_quotes(self, side: str, viewer_id: str) -> list[dict]:
        """Open quotes as other agents should see them.

        A goods quote always includes the agent's id. A futures quote includes it only when the
        agent chose to disclose, except that the viewer still sees their own id marked as hidden.
        """
        rows = []
        for o in self.open_orders(side):
            row = dict(price_cents=o.price_cents, quantity=o.remaining)
            public = self.market == "spot" or o.disclose_id
            if public or o.agent_id == viewer_id:
                row["agent_id"] = o.agent_id
            if o.agent_id == viewer_id and not public:
                row["hidden_from_others"] = True
            rows.append(row)
        return rows

    def public_levels(self, side: str) -> list[tuple[int, int]]:
        """Anonymous aggregated (price, quantity) levels, best first."""
        levels: dict[int, int] = {}
        for o in self.open_orders(side):
            levels[o.price_cents] = levels.get(o.price_cents, 0) + o.remaining
        return sorted(levels.items(), key=lambda kv: -kv[0] if side == "bid" else kv[0])
