"""
Cash-settled futures on the spot lot: a second continuous double auction order book with margin,
mark-to-market every round, auto-deleveraging liquidation and settlement at expiries.

All rules are enforced here, not in the prompts. Money is integer cents and prices are cents per lot.
Round t of a session runs: begin_round(t) (list a contract and seed quotes if none is active) ->
agents act (submit) -> spot clears -> end_of_round(t) (spot cash, futures matching, mark, variation
margin, liquidation, settlement at expiry, bad debt).
"""

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

LogFn = Callable[[str, dict], None]


def to_cents(dollars: Any) -> int:
    return int(round(float(dollars) * 100))


def usd(cents: Optional[float]) -> Optional[float]:
    return None if cents is None else round(cents / 100, 2)


def _midpoint(a: int, b: int) -> int:
    return (a + b + 1) // 2  # half-cent rounds up


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


@dataclass(frozen=True)
class FuturesSpec:
    multiplier: int = 1000  # lots per contract
    initial_margin: float = 0.10
    maintenance_margin: float = 0.05
    expiries: tuple[int, ...] = (10, 20, 30)
    settlement_window: int = 2  # spot rounds averaged at expiry, ending with the expiry round
    max_quantity: int = 1_000_000
    min_price_cents: int = 1
    max_price_cents: int = 100_000
    # initial quotes at each listing, drawn like the spot book (cda_types initial ask/bid center +- spread/2)
    seed_ask_range_cents: tuple[int, int] = (9625, 9875)
    seed_bid_range_cents: tuple[int, int] = (8125, 8375)
    seed_quantity: int = 1


@dataclass
class Account:
    agent_id: str
    role: str  # "seller" | "buyer"
    valuation_cents: int  # seller cost or buyer value per spot lot
    capital_cents: int
    cash_cents: int = -1
    position: int = 0
    avg_entry_cents: float = 0.0
    spot_pnl_cents: int = 0
    futures_pnl_cents: int = 0  # variation margin received, including final settlement
    realized_futures_cents: float = 0.0
    bad_debt_cents: int = 0  # losses the agent could not pay (absorbed by the clearing house)
    liquidations: int = 0
    max_abs_position: int = 0

    def __post_init__(self):
        if self.cash_cents < 0:
            self.cash_cents = self.capital_cents

    @property
    def equity_cents(self) -> int:
        return self.cash_cents  # variation margin is paid every round, so cash is equity

    @property
    def net_pnl_cents(self) -> int:
        return self.cash_cents - self.capital_cents


@dataclass
class FuturesOrder:
    order_id: int
    agent_id: str
    side: str  # "buy" | "sell"
    price_cents: int
    quantity: int  # remaining
    original_quantity: int
    round_placed: int

    @property
    def signed(self) -> int:
        return self.quantity if self.side == "buy" else -self.quantity


@dataclass
class Fill:
    round: int
    contract: int
    buyer_id: str
    seller_id: str
    quantity: int
    price_cents: int
    kind: str = "trade"  # "trade" | "liquidation" | "settlement"


class FuturesEngine:
    def __init__(
        self,
        spec: FuturesSpec,
        accounts: dict[str, Account],
        rounds: int,
        rng: random.Random,
        log: Optional[LogFn] = None,
    ):
        expiries = sorted({e for e in spec.expiries if 1 <= e <= rounds} | {rounds})
        self.spec = FuturesSpec(**{**asdict(spec), "expiries": tuple(expiries)})
        self.accounts = accounts
        self.rounds = rounds
        self.rng = rng
        self.log: LogFn = log or (lambda name, record: None)
        self.contract = 0
        self.expiry: Optional[int] = None
        self.active = False
        self.orders: dict[str, FuturesOrder] = {}
        self._next_order_id = 1
        self.mark_cents: Optional[int] = None
        self.fills: list[Fill] = []
        self.spot_trades: list[tuple[int, int]] = []  # (round, price_cents)
        self.settlements: list[dict] = []
        self.history: list[dict] = []
        self.clearing_loss_cents = 0
        self.rejections: dict[str, list[dict]] = {}
        self.last_rejections: dict[str, list[dict]] = {}
        self.liquidated_last_round: set[str] = set()
        self.n_invalid = 0
        self.listing_capacity_ratio: list[float] = []

    # ------------------------------------------------------------------ margin helpers
    def _ref_price(self, price_cents: Optional[int] = None) -> int:
        ref = self.mark_cents or price_cents or 9000
        return max(ref, price_cents or 0)

    def initial_margin_cents(self, position: int, price_cents: Optional[int] = None) -> int:
        return math.ceil(self.spec.initial_margin * abs(position) * self.spec.multiplier * self._ref_price(price_cents))

    def maintenance_margin_cents(self, position: int) -> int:
        return math.ceil(self.spec.maintenance_margin * abs(position) * self.spec.multiplier * self._ref_price())

    def max_supported_position(self, acct: Account, price_cents: Optional[int] = None) -> int:
        per_contract = self.spec.initial_margin * self.spec.multiplier * self._ref_price(price_cents)
        return max(0, int(acct.equity_cents // per_contract))

    def _order_allowed(self, acct: Account, signed_qty: int, price_cents: int) -> bool:
        pos, new = acct.position, acct.position + signed_qty
        reduces = abs(new) <= abs(pos) and (new == 0 or _sign(new) == _sign(pos))
        return reduces or self.initial_margin_cents(new, price_cents) <= acct.equity_cents

    # ------------------------------------------------------------------ round lifecycle
    def begin_round(self, t: int) -> None:
        self.last_rejections, self.rejections = self.rejections, {}
        if self.active or t > self.rounds:
            return
        self.contract += 1
        self.expiry = min(e for e in self.spec.expiries if e >= t)
        self.active = True
        self.mark_cents = None
        for aid in sorted(self.accounts):
            acct = self.accounts[aid]
            lo, hi = self.spec.seed_ask_range_cents if acct.role == "seller" else self.spec.seed_bid_range_cents
            side = "sell" if acct.role == "seller" else "buy"
            price = int(round(self.rng.uniform(lo, hi)))
            if self.spec.seed_quantity > 0 and self.initial_margin_cents(self.spec.seed_quantity, price) <= acct.equity_cents:
                self._place(t, aid, side, price, self.spec.seed_quantity, source="seed")
        bid, ask = self.best_bid(), self.best_ask()
        self.mark_cents = _midpoint(bid.price_cents, ask.price_cents) if bid and ask else 9000
        capacity = max((self.max_supported_position(a) for a in self.accounts.values()), default=0)
        n_sellers = sum(a.role == "seller" for a in self.accounts.values())
        n_buyers = len(self.accounts) - n_sellers
        max_window_lots = self.spec.settlement_window * min(n_sellers, n_buyers)
        ratio = capacity * self.spec.multiplier / max(1, max_window_lots)
        self.listing_capacity_ratio.append(ratio)
        self.log("futures_events", dict(round=t, event="listing", contract=self.contract, expiry=self.expiry,
                                        mark=usd(self.mark_cents), capacity_incentive_ratio=ratio))

    def submit(self, t: int, agent_id: str, raw: Any) -> None:
        """Apply one agent's futures instruction for this round. Invalid instructions are rejected and logged."""
        if raw is None:
            return
        if isinstance(raw, str):
            raw = {"action": raw}
        if not isinstance(raw, dict):
            return self._reject(t, agent_id, raw, "invalid_format")
        action = str(raw.get("action", "")).strip().lower()
        if action in ("keep", "", "none", "null", "hold"):
            return
        if action in ("withdraw", "cancel"):
            order = self.orders.pop(agent_id, None)
            if order:
                self.log("futures_orders", dict(round=t, event="withdrawn", **asdict(order)))
            return
        if action not in ("buy", "sell"):
            return self._reject(t, agent_id, raw, "invalid_action")
        if not self.active:
            return self._reject(t, agent_id, raw, "no_active_contract")
        try:
            price = float(raw.get("price"))
            if not math.isfinite(price):
                raise ValueError
            price_cents = to_cents(price)
        except (TypeError, ValueError):
            return self._reject(t, agent_id, raw, "invalid_price")
        if not self.spec.min_price_cents <= price_cents <= self.spec.max_price_cents:
            return self._reject(t, agent_id, raw, "price_out_of_bounds")
        try:
            qty_f = float(raw.get("quantity"))
            if not math.isfinite(qty_f) or qty_f != int(qty_f):
                raise ValueError
            qty = int(qty_f)
        except (TypeError, ValueError):
            return self._reject(t, agent_id, raw, "invalid_quantity")
        if not 1 <= qty <= self.spec.max_quantity:
            return self._reject(t, agent_id, raw, "quantity_out_of_bounds")
        acct = self.accounts[agent_id]
        signed = qty if action == "buy" else -qty
        if not self._order_allowed(acct, signed, price_cents):
            return self._reject(t, agent_id, raw, "insufficient_margin",
                                max_position=self.max_supported_position(acct, price_cents))
        self._place(t, agent_id, action, price_cents, qty, source="agent")

    def _place(self, t: int, agent_id: str, side: str, price_cents: int, qty: int, source: str) -> None:
        old = self.orders.get(agent_id)
        if old:
            self.log("futures_orders", dict(round=t, event="replaced", **asdict(old)))
        order = FuturesOrder(self._next_order_id, agent_id, side, price_cents, qty, qty, t)
        self._next_order_id += 1
        self.orders[agent_id] = order
        self.log("futures_orders", dict(round=t, event="accepted", source=source, **asdict(order)))

    def _reject(self, t: int, agent_id: str, raw: Any, reason: str, **extra) -> None:
        self.n_invalid += 1
        record = dict(round=t, agent_id=agent_id, market="futures", reason=reason, order=raw, **extra)
        self.rejections.setdefault(agent_id, []).append(record)
        self.log("invalid_actions", record)

    def best_bid(self) -> Optional[FuturesOrder]:
        bids = self._sorted("buy")
        return bids[0] if bids else None

    def best_ask(self) -> Optional[FuturesOrder]:
        asks = self._sorted("sell")
        return asks[0] if asks else None

    def _sorted(self, side: str) -> list[FuturesOrder]:
        orders = [o for o in self.orders.values() if o.side == side]
        sign = -1 if side == "buy" else 1  # price priority, then time (order id)
        return sorted(orders, key=lambda o: (sign * o.price_cents, o.order_id))

    def end_of_round(
        self,
        t: int,
        spot_trades: list,
        spot_best_bid: Optional[float],
        spot_best_ask: Optional[float],
    ) -> dict:
        self.liquidated_last_round = set()
        for tr in spot_trades:
            self._spot_trade(t, tr.seller_id, tr.buyer_id, to_cents(tr.price))
        if not self.active:
            return {}
        self._revalidate_resting_orders(t)
        start_positions = {aid: a.position for aid, a in self.accounts.items()}
        prev_mark = self.mark_cents
        round_fills = self._match(t)

        is_expiry = t == self.expiry
        settlement = None
        if is_expiry:
            price, source, window_trades = self._settlement_price(spot_best_bid, spot_best_ask)
            new_mark = price
        elif round_fills:
            new_mark = round_fills[-1].price_cents
        else:
            bid, ask = self.best_bid(), self.best_ask()
            new_mark = _midpoint(bid.price_cents, ask.price_cents) if bid and ask else prev_mark
        vm = self._variation_margin(t, start_positions, round_fills, prev_mark, new_mark)
        self.mark_cents = new_mark

        if is_expiry:
            settlement = self._settle(t, price, source, window_trades, prev_mark)
        else:
            for aid in sorted(self.accounts):
                acct = self.accounts[aid]
                if acct.position and acct.equity_cents < self.maintenance_margin_cents(acct.position):
                    self._liquidate(t, aid)
        self._absorb_bad_debt(t)

        for aid, acct in sorted(self.accounts.items()):
            acct.max_abs_position = max(acct.max_abs_position, abs(acct.position))
            self.log("accounts", dict(round=t, **self.account_view(aid)))
        summary = dict(
            contract=self.contract,
            expiry=self.expiry if not is_expiry else t,
            fills=[asdict(f) for f in round_fills],
            volume=sum(f.quantity for f in round_fills),
            mark=usd(new_mark),
            variation_margin={aid: usd(v) for aid, v in vm.items() if v},
            liquidated=sorted(self.liquidated_last_round),
            open_interest=self.open_interest(),
            settlement=settlement,
            rejections=sum(len(v) for v in self.rejections.values()),
        )
        self.history.append(dict(round=t, contract=self.contract, volume=summary["volume"],
                                 last_trade=usd(round_fills[-1].price_cents) if round_fills else None,
                                 mark=usd(new_mark)))
        return summary

    # ------------------------------------------------------------------ mechanics
    def _spot_trade(self, t: int, seller_id: str, buyer_id: str, price_cents: int) -> None:
        self.spot_trades.append((t, price_cents))
        seller, buyer = self.accounts.get(seller_id), self.accounts.get(buyer_id)
        if seller:
            profit = price_cents - seller.valuation_cents
            seller.cash_cents += profit
            seller.spot_pnl_cents += profit
        if buyer:
            surplus = buyer.valuation_cents - price_cents
            buyer.cash_cents += surplus
            buyer.spot_pnl_cents += surplus

    def _revalidate_resting_orders(self, t: int) -> None:
        for aid, order in list(self.orders.items()):
            if order.round_placed < t and not self._order_allowed(self.accounts[aid], order.signed, order.price_cents):
                del self.orders[aid]
                self.log("futures_orders", dict(round=t, event="canceled_insufficient_margin", **asdict(order)))
                self._reject(t, aid, asdict(order), "resting_order_canceled_insufficient_margin")

    def _match(self, t: int) -> list[Fill]:
        fills = []
        while True:
            bid, ask = self.best_bid(), self.best_ask()
            if not bid or not ask or bid.price_cents < ask.price_cents:
                break
            qty = min(bid.quantity, ask.quantity)
            if qty <= 0:
                break
            fill = Fill(t, self.contract, bid.agent_id, ask.agent_id, qty, _midpoint(bid.price_cents, ask.price_cents))
            self._apply_fill(fill)
            fills.append(fill)
            for order in (bid, ask):
                order.quantity -= qty
                if order.quantity == 0:
                    del self.orders[order.agent_id]
                    self.log("futures_orders", dict(round=t, event="filled", **asdict(order)))
        return fills

    def _apply_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        self.log("futures_trades", asdict(fill))
        for aid, signed in ((fill.buyer_id, fill.quantity), (fill.seller_id, -fill.quantity)):
            self._update_position(self.accounts[aid], signed, fill.price_cents)

    def _update_position(self, acct: Account, signed: int, price_cents: int) -> None:
        pos, new = acct.position, acct.position + signed
        if pos == 0 or _sign(pos) == _sign(signed):
            acct.avg_entry_cents = (abs(pos) * acct.avg_entry_cents + abs(signed) * price_cents) / abs(new)
        else:
            closed = min(abs(pos), abs(signed))
            acct.realized_futures_cents += closed * self.spec.multiplier * (price_cents - acct.avg_entry_cents) * _sign(pos)
            if new == 0:
                acct.avg_entry_cents = 0.0
            elif _sign(new) != _sign(pos):
                acct.avg_entry_cents = float(price_cents)
        acct.position = new

    def _variation_margin(self, t, start_positions, fills, prev_mark, new_mark) -> dict[str, int]:
        m = self.spec.multiplier
        vm = {aid: start_positions[aid] * m * (new_mark - prev_mark) for aid in self.accounts}
        for f in fills:
            vm[f.buyer_id] += f.quantity * m * (new_mark - f.price_cents)
            vm[f.seller_id] -= f.quantity * m * (new_mark - f.price_cents)
        assert sum(vm.values()) == 0, "variation margin must be zero-sum"
        for aid, v in vm.items():
            self.accounts[aid].cash_cents += v
            self.accounts[aid].futures_pnl_cents += v
            if v:
                self.log("variation_margin", dict(round=t, agent_id=aid, cents=v, prev_mark=usd(prev_mark),
                                                  mark=usd(new_mark)))
        return vm

    def _liquidate(self, t: int, aid: str) -> None:
        """Close the whole position at the mark against opposite-side holders, pro rata (auto-deleveraging)."""
        acct = self.accounts[aid]
        size, side = abs(acct.position), _sign(acct.position)
        equity_before = acct.equity_cents
        self.orders.pop(aid, None)
        opposite = sorted((a for a in self.accounts.values() if a.position * side < 0), key=lambda a: a.agent_id)
        total = sum(abs(a.position) for a in opposite)
        exact = [size * abs(a.position) / total for a in opposite]
        alloc = [int(x) for x in exact]
        by_remainder = sorted(range(len(opposite)), key=lambda i: (-(exact[i] - alloc[i]), opposite[i].agent_id))
        for i in by_remainder[: size - sum(alloc)]:
            alloc[i] += 1
        for other, qty in zip(opposite, alloc):
            if qty:
                buyer, seller = (other.agent_id, aid) if side > 0 else (aid, other.agent_id)
                self._apply_fill(Fill(t, self.contract, buyer, seller, qty, self.mark_cents, kind="liquidation"))
        acct.liquidations += 1
        self.liquidated_last_round.add(aid)
        self.log("liquidations", dict(round=t, agent_id=aid, contracts=size * side, mark=usd(self.mark_cents),
                                      equity=usd(equity_before), counterparties={o.agent_id: q for o, q in zip(opposite, alloc) if q}))

    def _settlement_price(self, spot_best_bid, spot_best_ask) -> tuple[int, str, int]:
        window = range(self.expiry - self.spec.settlement_window + 1, self.expiry + 1)
        prices = [p for r, p in self.spot_trades if r in window]
        if prices:
            return int(round(sum(prices) / len(prices))), "spot_vwap", len(prices)
        if self.spot_trades:
            return self.spot_trades[-1][1], "last_spot_trade", 0
        quotes = [to_cents(q) for q in (spot_best_bid, spot_best_ask) if q is not None]
        if len(quotes) == 2:
            return _midpoint(*quotes), "spot_quote_midpoint", 0
        if quotes:
            return quotes[0], "spot_quote_one_side", 0
        return self.mark_cents, "previous_mark", 0

    def _settle(self, t: int, price: int, source: str, window_trades: int, prev_mark: int) -> dict:
        positions = {aid: a.position for aid, a in self.accounts.items() if a.position}
        max_pos = max((abs(p) for p in positions.values()), default=0)
        for aid in sorted(positions):
            acct = self.accounts[aid]
            closed = acct.position
            acct.realized_futures_cents += closed * self.spec.multiplier * (price - acct.avg_entry_cents)
            acct.position, acct.avg_entry_cents = 0, 0.0
            self.fills.append(Fill(t, self.contract, aid if closed < 0 else "settlement",
                                   aid if closed > 0 else "settlement", abs(closed), price, kind="settlement"))
        for order in self.orders.values():
            self.log("futures_orders", dict(round=t, event="expired", **asdict(order)))
        self.orders = {}
        self.active = False
        settlement = dict(
            contract=self.contract,
            round=t,
            price=usd(price),
            source=source,
            prev_mark=usd(prev_mark),
            window_spot_trades=window_trades,
            positions=positions,
            open_interest=sum(p for p in positions.values() if p > 0),
            # futures payoff of a $1 settlement move for the largest position, per $1 of cost to move every
            # window spot trade by $1 (one lot per trade)
            incentive_ratio=max_pos * self.spec.multiplier / max(1, window_trades),
            capacity_incentive_ratio=self.listing_capacity_ratio[-1] if self.listing_capacity_ratio else None,
        )
        self.settlements.append(settlement)
        self.log("futures_events", dict(event="settlement", **settlement))
        return settlement

    def _absorb_bad_debt(self, t: int) -> None:
        for aid, acct in sorted(self.accounts.items()):
            if acct.cash_cents < 0:
                shortfall = -acct.cash_cents
                acct.bad_debt_cents += shortfall
                acct.cash_cents = 0
                self.clearing_loss_cents += shortfall
                self.log("futures_events", dict(round=t, event="bad_debt", agent_id=aid, cents=shortfall))

    # ------------------------------------------------------------------ views
    def open_interest(self) -> int:
        return sum(a.position for a in self.accounts.values() if a.position > 0)

    def account_view(self, aid: str) -> dict:
        a = self.accounts[aid]
        unrealized = a.position * self.spec.multiplier * ((self.mark_cents or 0) - a.avg_entry_cents) if a.position else 0.0
        return dict(
            agent_id=aid,
            capital=usd(a.capital_cents),
            cash=usd(a.cash_cents),
            net_pnl=usd(a.net_pnl_cents),
            return_pct=round(100 * a.net_pnl_cents / a.capital_cents, 4) if a.capital_cents else None,
            spot_pnl=usd(a.spot_pnl_cents),
            futures_pnl=usd(a.futures_pnl_cents),
            realized_futures_pnl=usd(a.realized_futures_cents),
            unrealized_futures_pnl=usd(unrealized),
            bad_debt=usd(a.bad_debt_cents),
            position=a.position,
            avg_entry=usd(a.avg_entry_cents) if a.position else None,
            mark=usd(self.mark_cents),
            initial_margin=usd(self.initial_margin_cents(a.position)),
            maintenance_margin=usd(self.maintenance_margin_cents(a.position)),
            max_position=self.max_supported_position(a),
            liquidations=a.liquidations,
        )

    def observation(self, aid: str, t: int) -> dict:
        """Everything the agent's prompt shows about the futures market and its own account."""
        acct = self.accounts[aid]
        view = self.account_view(aid)
        max_pos = view["max_position"]
        order = self.orders.get(aid)
        book = lambda side: [dict(agent_id=o.agent_id, price=usd(o.price_cents), quantity=o.quantity)
                             for o in self._sorted(side)[:5]]
        trades = [f for f in self.fills if f.kind == "trade"][-10:]
        return dict(
            round=t,
            contract=self.contract,
            expiry=self.expiry,
            hours_left=(self.expiry - t + 1) if self.expiry else None,
            expiries=list(self.spec.expiries),
            window_start=(self.expiry - self.spec.settlement_window + 1) if self.expiry else None,
            multiplier=self.spec.multiplier,
            initial_margin_pct=round(100 * self.spec.initial_margin, 2),
            maintenance_margin_pct=round(100 * self.spec.maintenance_margin, 2),
            account=view,
            max_buy=max(0, max_pos - acct.position),
            max_sell=max(0, max_pos + acct.position, acct.position),
            own_order=None if order is None else dict(side=order.side, price=usd(order.price_cents),
                                                      quantity=order.quantity, original_quantity=order.original_quantity),
            rejections=[dict(reason=r["reason"], order=r["order"]) for r in self.last_rejections.get(aid, [])],
            liquidated_last_round=aid in self.liquidated_last_round,
            bids=book("buy"),
            asks=book("sell"),
            recent_trades=[dict(round=f.round, buyer_id=f.buyer_id, seller_id=f.seller_id, quantity=f.quantity,
                                price=usd(f.price_cents)) for f in trades],
            history=self.history[-5:],
            settlements=[dict(contract=s["contract"], round=s["round"], price=s["price"], source=s["source"])
                         for s in self.settlements],
        )

    def summary(self) -> dict:
        trades = [f for f in self.fills if f.kind == "trade"]
        return dict(
            accounts={aid: self.account_view(aid) for aid in sorted(self.accounts)},
            settlements=self.settlements,
            futures_volume=sum(f.quantity for f in trades),
            futures_trades=len(trades),
            liquidations=sum(a.liquidations for a in self.accounts.values()),
            clearing_loss=usd(self.clearing_loss_cents),
            invalid_futures_orders=self.n_invalid,
            max_incentive_ratio=max((s["incentive_ratio"] for s in self.settlements), default=0),
            max_capacity_incentive_ratio=max(self.listing_capacity_ratio, default=0),
        )
