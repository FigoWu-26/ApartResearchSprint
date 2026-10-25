"""Round engine: spot call auction + futures order book + cash, margin and settlement accounting.

Order of operations in round t (documented in README, identical in every condition):

 1. (t = 1) seed the spot books with the paper's random initial bids/asks.
 2. Re-check resting orders against current funds; cancel underfunded ones (logged).
 3. Build an immutable observation for every agent from the state at the start of the round.
 4. Collect every agent's action (in parallel for LLMs). Nothing is applied yet.
 5. Apply actions agent by agent in a seeded random order (independent of response speed):
    cancels first, then spot, futures bid, futures ask. Each new order is checked for funding
    against the agent's cash, shared across spot and futures; rejections are logged with reasons.
 6. Clear spot (legacy convention); settle cash and external flows.
 7. From hour 6, clear that hour's price bet (same convention, partial fills allowed); charge fees.
    Hours 1–5 have no bet. Each bet is paid at the end of its hour and then closed.
 8. Set the mark (futures VWAP, else carry forward); pay variation margin.
 9. Stake check: accounts below the stake have orders canceled and positions closed
    against resting orders only. Unfilled residuals and cash shortfalls are recorded.
10. Pay this hour's bet on the goods average of the window ending this hour, then cancel its offers.
    On the last hour, expire any goods offers still open.
11. If the regulator is on, it reviews the public chat, every agent's profit and loss, and
    the trade history. If that review suspects collusion, it may also read the private
    messages of at most two participants. It may suspend agents, warn them, set a price
    band, suspend a market, change the leverage, or fine them. Each action is announced
    in public and takes effect next hour. Seller-only and buyer-only chats stay closed.
12. After the last hour, the regulator audits every public message, every private message,
    every agent's profit and loss, and the full trade history, and records how many agents
    it suspects of colluding. That report does not change the market.
13. Write ledgers. Messages sent in round t are delivered at the start of round t + 1.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from src.capital_futures.accounts import (
    Account,
    Clearing,
    stake_cents,
    settle_cash_shortfall,
    worst_case_position,
)
from src.capital_futures.actions import OrderInstruction, TraderAction, keep_all
from src.capital_futures.agents import (
    PRIVATE_INSPECTION_LIMIT,
    REGULATOR_POLICY,
    audit_summary,
    build_regulator,
    empty_decision,
    keyword_audit,
    suspects_collusion,
)
from src.capital_futures.book import Fill, Order, OrderBook
from src.capital_futures.config import ExperimentConfig
from src.capital_futures.ledger import Recorder
from src.capital_futures.rng import round_stream, stream
from src.capital_futures.settlement import external_path, hour_settlement_price, next_mark, vwap_cents


@dataclass
class AgentInfo:
    agent_id: str
    slot: str
    role: str
    model: str
    policy: str


@dataclass
class RoundResult:
    round_number: int
    spot_fills: list[Fill] = field(default_factory=list)
    futures_fills: list[Fill] = field(default_factory=list)
    mark_cents: int = 0
    mark_source: str = ""


def usd(cents: Optional[int]) -> Optional[float]:
    return None if cents is None else cents / 100


class Engine:
    def __init__(self, cfg: ExperimentConfig, agents: dict, infos: dict[str, AgentInfo],
                 recorder: Optional[Recorder] = None, max_workers: int = 10):
        self.cfg = cfg.validate()
        self.agents = agents                 # agent_id -> object with .act(observation) -> TraderAction
        self.infos = infos                   # agent_id -> AgentInfo
        self.rec = recorder or Recorder()
        self.max_workers = max_workers
        cash = cfg.initial_cash()
        self.accounts = {aid: Account(aid, info.slot, info.role, cash[info.slot], cash[info.slot],
                                      last_vm_price=cfg.futures.initial_reference_cents)
                         for aid, info in infos.items()}
        self.clearing = Clearing()
        self.spot_book = OrderBook("spot")
        self.futures_book = OrderBook("futures", cfg.futures.tick_cents)
        self.seq = 0
        self.mark = cfg.futures.initial_reference_cents
        self.marks: list[int] = []
        self.spot_fills: list[Fill] = []
        self.futures_fills: list[Fill] = []
        self.round_results: list[RoundResult] = []
        self.history: list[dict] = []        # public per-round summary
        self.pending_messages: dict[str, dict[str, str]] = {"seller": {}, "buyer": {}}
        self.pending_public: dict[str, str] = {}
        self.pending_private: list[dict] = []
        self.regulator = build_regulator(self.cfg, self.rec)
        self.banned_until: dict[str, int] = {}
        self.price_floor_cents: Optional[int] = None
        self.price_ceiling_cents: Optional[int] = None
        self.price_control_until: int = 0
        self.spot_suspended_until: int = 0
        self.futures_suspended_until: int = 0
        self.ban_count: int = 0
        self.warning_count: int = 0
        self.fines: list[dict] = []
        self._offers_this_round: dict[str, list] = {"spot": [], "futures": []}
        self._bet_positions: dict[str, int] = {}
        self.last_rejections: dict[str, list[dict]] = {aid: [] for aid in infos}
        self.external = (external_path(cfg.seed, cfg.rounds, cfg.futures)
                         if cfg.futures_enabled and cfg.futures.settlement == "external" else None)
        self.settlement: Optional[dict] = None
        self.hourly_settlements: list[dict] = []
        self.round_number = 0
        self.outgoing: dict[str, dict[str, str]] = {"seller": {}, "buyer": {}}
        self.outgoing_public: dict[str, str] = {}
        self.outgoing_private: list[dict] = []
        self.public_log: list[dict] = []
        self.private_log: list[dict] = []
        self._liquidating: Optional[str] = None

    # ------------------------------------------------------------------ helpers
    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _new_order(self, aid: str, market: str, side: str, price: int, qty: int, t: int,
                   disclose_id: bool = False) -> Order:
        seq = self._next_seq()
        # A goods quote always carries the agent's id. Only a futures quote may hide it.
        shown = True if market == "spot" else bool(disclose_id)
        return Order(f"{market[0].upper()}{t:03d}-{seq:05d}", aid, market, side, price, qty, qty, t, seq,
                     disclose_id=shown)

    def _log_order(self, t: int, event: str, order: Optional[Order] = None, aid: Optional[str] = None,
                   **extra) -> None:
        rec = dict(round=t, event=event)
        if order is not None:
            rec.update(order.to_dict())
        if aid is not None:
            rec["agent_id"] = aid
        rec.update(extra)
        self.rec.write("orders", rec)

    def _public_futures_trade(self, fill: Fill) -> dict:
        """A futures fill names a side only when that side's quote chose to show its id."""
        bid = self.futures_book.orders.get(fill.bid_order_id)
        ask = self.futures_book.orders.get(fill.ask_order_id)
        row = dict(round=fill.round_number, price_cents=fill.price_cents, quantity=fill.quantity,
                   liquidation=fill.liquidation)
        if bid is not None and bid.disclose_id:
            row["buyer"] = fill.buyer_id
        if ask is not None and ask.disclose_id:
            row["seller"] = fill.seller_id
        return row

    def bet_open(self, t: int) -> bool:
        """A new bet exists only once a full run of goods prices can be averaged. With a 5-hour
        average, that is hour 6 through the last hour. Each of those hours is its own bet."""
        return self.cfg.futures_enabled and t > self.cfg.futures.settlement_window

    @staticmethod
    def is_reducing(acct, side, quantity):
        """A closing order may shrink a bet. It may not grow it or turn it around."""
        return (acct.position != 0 and quantity <= abs(acct.position)
                and side == ("ask" if acct.position > 0 else "bid"))

    def spot_side(self, role: str) -> str:
        return "ask" if role == "seller" else "bid"

    def spot_reservation(self, spot: Optional[Order], role: str = "seller") -> int:
        """Cash set aside for one goods offer. A sell offer from a seller sets aside the supplier's price.
        A trader has no supplier cost. A buy offer sets aside the price offered."""
        if spot is None:
            return 0
        fee = self.cfg.spot.fee_per_lot_cents
        if spot.side == "ask" and role != "trader":
            return self.cfg.spot.seller_cost_cents * spot.remaining + fee
        if spot.side == "ask":
            return fee
        return spot.price_cents * spot.remaining + fee

    def requirement(self, acct: Account, spot: Optional[Order], fbid: Optional[Order],
                    fask: Optional[Order]) -> dict:
        """Cash that must be free to support these open orders plus the current position."""
        fs = self.cfg.futures
        spot_res = self.spot_reservation(spot, acct.role)
        bq = fbid.remaining if fbid else 0
        aq = fask.remaining if fask else 0
        worst = worst_case_position(acct.position, bq, aq)
        im = stake_cents(worst, fs.multiplier, fs.stake_per_lot_cents)
        fees = fs.fee_per_contract_cents * (bq + aq)
        return dict(spot=spot_res, futures_margin=im, fees=fees, total=spot_res + im + fees,
                    worst_case_position=worst)

    def open_orders(self, aid: str) -> tuple[Optional[Order], Optional[Order], Optional[Order]]:
        spot = self.spot_book.agent_order(aid, "bid") or self.spot_book.agent_order(aid, "ask")
        return (spot, self.futures_book.agent_order(aid, "bid"), self.futures_book.agent_order(aid, "ask"))

    def max_supported_position(self, acct: Account, spot: Optional[Order]) -> int:
        """Largest bet the free cash can cover at the fixed stake."""
        fs = self.cfg.futures
        free = acct.equity - self.spot_reservation(spot, acct.role)
        per_contract = fs.multiplier * fs.stake_per_lot_cents
        return max(0, int(free // per_contract)) if per_contract > 0 else 0

    # ------------------------------------------------------------------ round steps
    def seed_initial_orders(self) -> None:
        if not self.cfg.spot.initial_orders:
            return
        rng = stream(self.cfg.seed, "initial_orders")
        lo_b, hi_b = self.cfg.spot.initial_bid_range_cents
        lo_a, hi_a = self.cfg.spot.initial_ask_range_cents
        for i, aid in enumerate(sorted(self.accounts, key=lambda a: self.infos[a].slot)):
            acct = self.accounts[aid]
            if acct.role == "trader":
                side = "ask" if i % 2 == 0 else "bid"
            else:
                side = self.spot_side(acct.role)
            price = rng.randint(lo_a, hi_a) if side == "ask" else rng.randint(lo_b, hi_b)
            order = self._new_order(aid, "spot", side, price, self.cfg.spot.lots_per_order, 1)
            if self.requirement(acct, order, None, None)["total"] <= acct.equity:
                self.spot_book.add(order)
                self._log_order(1, "initial", order)

    def revalidate(self, t: int) -> None:
        for aid, acct in self.accounts.items():
            spot, fbid, fask = self.open_orders(aid)
            if self.requirement(acct, spot, fbid, fask)["total"] <= acct.equity:
                continue
            for o in (fbid, fask):
                if o and worst_case_position(acct.position, o.remaining if o.side == "bid" else 0,
                                             o.remaining if o.side == "ask" else 0) > abs(acct.position):
                    self.futures_book.cancel(o.order_id)
                    self._log_order(t, "canceled_underfunded", o)
            spot, fbid, fask = self.open_orders(aid)
            if spot and self.requirement(acct, spot, fbid, fask)["total"] > acct.equity:
                self.spot_book.cancel(spot.order_id)
                self._log_order(t, "canceled_underfunded", spot)

    def observation(self, aid: str, t: int) -> dict:
        cfg, fs = self.cfg, self.cfg.futures
        acct, info = self.accounts[aid], self.infos[aid]
        spot, fbid, fask = self.open_orders(aid)
        req = self.requirement(acct, spot, fbid, fask)
        own_fills = [f for f in self.spot_fills if aid in (f.buyer_id, f.seller_id)]
        own_fut = [f for f in self.futures_fills if aid in (f.buyer_id, f.seller_id)]
        msgs = {}
        if info.role == "seller" and cfg.seller_comms_enabled:
            msgs = dict(self.pending_messages["seller"])
        if info.role == "buyer" and cfg.buyer_comms_enabled:
            msgs = dict(self.pending_messages["buyer"])
        msgs.pop(aid, None)
        public = dict(self.pending_public)
        public.pop(aid, None)
        private = [dict(sender=m["sender"], text=m["text"]) for m in self.pending_private if m["recipient"] == aid]
        obs = dict(
            round=t, rounds=None if cfg.hide_num_rounds else cfg.rounds,
            rounds_remaining=None if cfg.hide_num_rounds else cfg.rounds - t + 1,
            agent_id=aid, role=info.role,
            unit_value_cents=cfg.spot.buyer_value_cents if info.role == "buyer" else None,
            unit_cost_cents=cfg.spot.seller_cost_cents if info.role == "seller" else None,
            futures_enabled=cfg.futures_enabled,
            contract=dict(multiplier=fs.multiplier, tick_cents=fs.tick_cents,
                          stake_per_lot_cents=fs.stake_per_lot_cents,
                          leverage=fs.leverage,
                          fee_per_contract_cents=fs.fee_per_contract_cents,
                          settlement=fs.settlement, settlement_window=fs.settlement_window,
                          bet_open=self.bet_open(t),
                          first_bet_round=fs.settlement_window + 1,
                          window_start=(t - fs.settlement_window + 1) if fs.settlement_window > 0 else t,
                          window_end=t,
                          fallback_reference_cents=fs.fallback_reference_cents,
                          initial_reference_cents=fs.initial_reference_cents,
                          external_reference_now_cents=self.external[t - 1] if self.external else None,
                          spot_lots_per_round=cfg.spot_lots_per_round(),
                          position_limit_contracts=cfg.position_limit_contracts(),
                          position_limit_lots=cfg.position_limit_lots(),
                          relative_size=cfg.relative_size),
            account=dict(initial_cash_cents=acct.initial_cash, cash_cents=acct.cash,
                         liabilities_cents=acct.liabilities, equity_cents=acct.equity,
                         net_pnl_cents=acct.net_pnl, spot_pnl_cents=acct.spot_pnl,
                         futures_pnl_cents=acct.futures_pnl, fees_cents=acct.fees,
                         position_contracts=acct.position, mark_cents=self.mark,
                         unsettled_marked_value_cents=0,
                         stake_cents=stake_cents(acct.position, fs.multiplier, fs.stake_per_lot_cents),
                         reserved_cents=req["total"], reserved_breakdown=req,
                         available_cents=acct.equity - req["total"],
                         max_supported_position_contracts=self.max_supported_position(acct, spot),
                         position_limit_contracts=cfg.position_limit_contracts(),
                         in_liquidation=acct.in_liquidation, defaulted=acct.defaulted),
            own_orders=dict(spot=spot.to_dict() if spot else None, futures_bid=fbid.to_dict() if fbid else None,
                            futures_ask=fask.to_dict() if fask else None),
            spot_book=dict(bids=self.spot_book.visible_quotes("bid", aid),
                           asks=self.spot_book.visible_quotes("ask", aid)),
            futures_book=dict(bids=self.futures_book.visible_quotes("bid", aid),
                              asks=self.futures_book.visible_quotes("ask", aid)),
            history=copy.deepcopy(self.history[-10:]),
            spot_trades=[dict(round=f.round_number, buyer=f.buyer_id, seller=f.seller_id,
                               price_cents=f.price_cents, quantity=f.quantity)
                         for f in self.spot_fills[-30:]],
            futures_trades=[self._public_futures_trade(f) for f in self.futures_fills[-20:]],
            own_spot_trades=[dict(round=f.round_number, side="buy" if f.buyer_id == aid else "sell",
                                  counterparty=f.seller_id if f.buyer_id == aid else f.buyer_id,
                                  price_cents=f.price_cents) for f in own_fills],
            own_futures_trades=[dict(round=f.round_number, side="buy" if f.buyer_id == aid else "sell",
                                     price_cents=f.price_cents, quantity=f.quantity, liquidation=f.liquidation)
                                for f in own_fut[-20:]],
            messages=msgs,
            public_messages=public if cfg.public_chat_enabled else {},
            private_messages=private if cfg.private_chat_enabled else [],
            other_agents=[a for a in sorted(self.accounts) if a != aid],
            comms_enabled=cfg.seller_comms_enabled if info.role == "seller" else cfg.buyer_comms_enabled,
            public_chat_enabled=cfg.public_chat_enabled,
            private_chat_enabled=cfg.private_chat_enabled,
            regulator_enabled=cfg.regulator_enabled,
            regulator_policy=REGULATOR_POLICY if cfg.regulator_enabled else None,
            regulator_announcement=self.pending_public.get("Regulator"),
            suspended=self._agent_suspended(aid, t),
            suspended_until=self.banned_until.get(aid) if self._agent_suspended(aid, t) else None,
            price_floor_cents=self.price_floor_cents,
            price_ceiling_cents=self.price_ceiling_cents,
            spot_trading_suspended=self._market_suspended("spot", t),
            futures_trading_suspended=self._market_suspended("futures", t),
            max_message_length=cfg.max_message_length,
            last_round_rejections=list(self.last_rejections.get(aid, [])),
            disclosed_capital=({a: self.accounts[a].initial_cash for a in sorted(self.accounts)}
                               if cfg.disclose_wealth else None),
            initial_orders_note=(t == 1 and cfg.spot.initial_orders),
        )
        return copy.deepcopy(obs)

    def collect(self, observations: dict[str, dict]) -> dict[str, TraderAction]:
        def call(aid):
            try:
                return aid, self.agents[aid].act(observations[aid])
            except Exception as e:  # noqa: BLE001 - a failing agent holds; the error is logged
                self._log_order(observations[aid]["round"], "agent_error", aid=aid, error=repr(e))
                return aid, keep_all()
        ids = list(observations)
        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(ids)))) as ex:
            return dict(ex.map(call, ids))

    def _reject(self, t: int, aid: str, market: str, side: str, ins: OrderInstruction, reason: str, **extra):
        self.accounts[aid].rejected_orders += 1
        rec = dict(market=market, side=side, price=ins.price, quantity=ins.quantity, reason=reason, **extra)
        self.last_rejections[aid].append(rec)
        self._log_order(t, "rejected", aid=aid, **rec)

    def apply(self, aid: str, action: TraderAction, t: int) -> None:
        acct, cfg = self.accounts[aid], self.cfg
        spot_side = self.spot_side(acct.role)
        chosen_spot_side = action.spot_order.side or spot_side
        plan = [("spot", chosen_spot_side, action.spot_order), ("futures", "bid", action.futures_bid),
                ("futures", "ask", action.futures_ask)]
        books = {"spot": self.spot_book, "futures": self.futures_book}

        for market, side, ins in plan:
            if ins.action == "cancel":
                o = books[market].agent_order(aid, side)
                if o:
                    books[market].cancel(o.order_id)
                    self._log_order(t, "canceled", o)
                else:
                    self._log_order(t, "cancel_noop", aid=aid, market=market, side=side)

        for market, side, ins in plan:
            if ins.action != "place":
                continue
            if market == "futures" and not cfg.futures_enabled:
                self._reject(t, aid, market, side, ins, "futures_disabled")
                continue
            if market == "futures" and not self.bet_open(t):
                self._reject(t, aid, market, side, ins, "no_price_bet_this_hour",
                             first_bet_round=cfg.futures.settlement_window + 1)
                continue
            if self._agent_suspended(aid, t):
                self._reject(t, aid, market, side, ins, "suspended_by_regulator",
                             until_round=self.banned_until.get(aid))
                continue
            if self._market_suspended(market, t):
                self._reject(t, aid, market, side, ins, "trading_suspended")
                continue
            tick = cfg.futures.tick_cents if market == "futures" else 1
            price = int(round(ins.price * 100 / tick)) * tick
            if abs(ins.price * 100 - price) > 1e-6:
                self._reject(t, aid, market, side, ins, "price_not_on_tick")
                continue
            lo, hi = cfg.spot.min_price_cents, cfg.spot.max_price_cents
            if not lo <= price <= hi:
                self._reject(t, aid, market, side, ins, "price_out_of_bounds", min_cents=lo, max_cents=hi)
                continue
            if not self._price_allowed(price):
                self._reject(t, aid, market, side, ins, "outside_price_control",
                             floor_cents=self.price_floor_cents, ceiling_cents=self.price_ceiling_cents)
                continue
            qty = cfg.spot.lots_per_order if market == "spot" and ins.quantity is None else int(ins.quantity)
            if qty < 1 or qty > cfg.futures.max_order_quantity:
                self._reject(t, aid, market, side, ins, "quantity_above_limit", limit=cfg.futures.max_order_quantity)
                continue
            if market == "futures":
                other = self.futures_book.agent_order(aid, "ask" if side == "bid" else "bid")
                if other and ((side == "bid" and price >= other.price_cents) or
                              (side == "ask" and price <= other.price_cents)):
                    self._reject(t, aid, market, side, ins, "self_trade_prevention",
                                 own_opposite_order=other.order_id)
                    continue
            new = self._new_order(aid, market, side, price, qty, t,
                                  disclose_id=market == "spot" or bool(ins.disclose_id))
            spot, fbid, fask = self.open_orders(aid)
            old = {"spot": spot, "bid": fbid, "ask": fask}[side if market == "futures" else "spot"]
            proposed = dict(spot=spot, bid=fbid, ask=fask)
            proposed["spot" if market == "spot" else side] = new
            req = self.requirement(acct, proposed["spot"], proposed["bid"], proposed["ask"])
            reduce_only = market == "futures" and self.is_reducing(acct, side, qty)
            blocked = market == "futures" and (acct.in_liquidation or acct.defaulted) and not reduce_only
            if blocked:
                self._reject(t, aid, market, side, ins, "reduce_only_while_in_liquidation_or_default")
                continue
            limit = cfg.position_limit_contracts()
            if market == "futures" and limit is not None and req["worst_case_position"] > limit and not reduce_only:
                self._reject(t, aid, market, side, ins, "position_limit", limit_contracts=limit,
                             worst_case_position=req["worst_case_position"])
                continue
            if req["total"] > acct.equity and not reduce_only:
                self._reject(t, aid, market, side, ins, "insufficient_collateral",
                             required_cents=req["total"], equity_cents=acct.equity,
                             existing_order_unchanged=old.order_id if old else None)
                continue
            if old:
                books[market].cancel(old.order_id)
                self._log_order(t, "replaced", old, replaced_by=new.order_id)
            if market == "spot":
                opposite = self.spot_book.agent_order(aid, "ask" if side == "bid" else "bid")
                if opposite:
                    self.spot_book.cancel(opposite.order_id)
                    self._log_order(t, "canceled", opposite, reason="one goods offer at a time")
            books[market].add(new)
            self._log_order(t, "accepted", new, required_cents=req["total"], reduce_only=reduce_only)

        if action.message_to_sellers and acct.role == "seller" and cfg.seller_comms_enabled:
            self._send(t, aid, "seller", action.message_to_sellers)
        if action.message_to_buyers and acct.role == "buyer" and cfg.buyer_comms_enabled:
            self._send(t, aid, "buyer", action.message_to_buyers)
        if action.message_public and cfg.public_chat_enabled:
            self._send(t, aid, "public", action.message_public)
        if action.message_private and action.private_to and cfg.private_chat_enabled:
            target = str(action.private_to).strip()
            if target in self.accounts and target != aid:
                self._send(t, aid, "private", action.message_private, recipient=target)

    def _send(self, t: int, aid: str, channel: str, text: str, recipient: Optional[str] = None) -> None:
        text = str(text)[: self.cfg.max_message_length]
        if text.strip().lower() in ("", "null", "none"):
            return
        record = dict(round=t, sender=aid, channel=channel, text=text, delivered_round=t + 1)
        if channel == "public":
            self.outgoing_public[aid] = text
            self.public_log.append(dict(round=t, sender=aid, text=text))
        elif channel == "private":
            self.outgoing_private.append(dict(sender=aid, recipient=recipient, text=text))
            self.private_log.append(dict(round=t, sender=aid, recipient=recipient, text=text))
            record["recipient"] = recipient
        else:
            self.outgoing[channel][aid] = text
        self.rec.write("messages", record)

    def _agent_suspended(self, aid: str, t: int) -> bool:
        return t <= self.banned_until.get(aid, 0)

    def _market_suspended(self, market: str, t: int) -> bool:
        until = self.spot_suspended_until if market == "spot" else self.futures_suspended_until
        return t <= until

    def _price_allowed(self, price: int) -> bool:
        if self.price_floor_cents is not None and price < self.price_floor_cents:
            return False
        if self.price_ceiling_cents is not None and price > self.price_ceiling_cents:
            return False
        return True

    def _cancel_orders(self, orders: list, t: int, event: str) -> None:
        for order in orders:
            if order.status != "open":
                continue
            book = self.spot_book if order.market == "spot" else self.futures_book
            book.cancel(order.order_id)
            self._log_order(t, event, order)

    def enforce_restrictions(self, t: int) -> None:
        """Drop offers that a restriction already in force will not allow to trade this hour."""
        if not self.cfg.regulator_enabled:
            return
        if self.price_control_until and t > self.price_control_until:
            self.price_floor_cents = None
            self.price_ceiling_cents = None
            self.price_control_until = 0
        if self._market_suspended("spot", t):
            self._cancel_orders(list(self.spot_book.open_orders()), t, "canceled_trading_suspended")
        if self._market_suspended("futures", t):
            self._cancel_orders(list(self.futures_book.open_orders()), t, "canceled_trading_suspended")
        banned = [o for book in (self.spot_book, self.futures_book) for o in book.open_orders()
                  if self._agent_suspended(o.agent_id, t)]
        self._cancel_orders(banned, t, "canceled_suspended")
        if self.price_floor_cents is not None or self.price_ceiling_cents is not None:
            outside = [o for book in (self.spot_book, self.futures_book) for o in book.open_orders()
                       if not self._price_allowed(o.price_cents)]
            self._cancel_orders(outside, t, "canceled_price_control")

    def _offer_rows(self, book: OrderBook) -> list[dict]:
        return [dict(agent_id=o.agent_id, side=o.side, price_cents=o.price_cents, quantity=o.remaining)
                for o in book.open_orders()]

    def _regulator_dossier(self, t: int, result: RoundResult) -> dict:
        fs = self.cfg.futures
        return dict(
            agent_ids=sorted(self.accounts),
            ban_count=self.ban_count,
            warning_count=self.warning_count,
            fines=list(self.fines),
            leverage=fs.leverage,
            stake_per_lot_cents=fs.stake_per_lot_cents,
            opening_price_cents=fs.initial_reference_cents,
            price_floor_cents=self.price_floor_cents,
            price_ceiling_cents=self.price_ceiling_cents,
            price_control_until=self.price_control_until or None,
            spot_suspended_until=self.spot_suspended_until if self.spot_suspended_until >= t else None,
            futures_suspended_until=self.futures_suspended_until if self.futures_suspended_until >= t else None,
            active_bans=[dict(agent_id=aid, until_round=until)
                         for aid, until in sorted(self.banned_until.items()) if until >= t],
            public_messages=dict(self.outgoing_public),
            spot_trades=[dict(buyer=f.buyer_id, seller=f.seller_id, price_cents=f.price_cents, quantity=f.quantity)
                         for f in result.spot_fills],
            futures_trades=[dict(buyer=f.buyer_id, seller=f.seller_id, price_cents=f.price_cents,
                                 quantity=f.quantity, liquidation=f.liquidation) for f in result.futures_fills],
            spot_offers=list(self._offers_this_round["spot"]),
            futures_offers=list(self._offers_this_round["futures"]),
            positions=[dict(agent_id=aid, contracts=self._bet_positions.get(aid, 0)) for aid in sorted(self.accounts)],
            history=copy.deepcopy(self.history[-5:]),
            agent_pnl=self._pnl_rows(),
            public_history=list(self.public_log),
            trade_history=self._trade_history(),
        )

    def _pnl_rows(self) -> list[dict]:
        return [dict(agent_id=aid, net_pnl_cents=acct.net_pnl, spot_pnl_cents=acct.spot_pnl,
                     futures_pnl_cents=acct.futures_pnl, fees_cents=acct.fees, cash_cents=acct.cash)
                for aid, acct in sorted(self.accounts.items())]

    def _trade_history(self) -> list[dict]:
        rows = [dict(round=f.round_number, market="spot", buyer=f.buyer_id, seller=f.seller_id,
                     price_cents=f.price_cents, quantity=f.quantity, liquidation=False)
                for f in self.spot_fills]
        rows += [dict(round=f.round_number, market="futures", buyer=f.buyer_id, seller=f.seller_id,
                      price_cents=f.price_cents, quantity=f.quantity, liquidation=f.liquidation)
                 for f in self.futures_fills]
        return rows

    def _audit_dossier(self) -> dict:
        return dict(
            round=self.cfg.rounds, agent_ids=sorted(self.accounts),
            public_log=list(self.public_log), private_log=list(self.private_log),
            agent_pnl=self._pnl_rows(), trade_history=self._trade_history(),
        )

    def _inspection_targets(self, decision: dict) -> list[str]:
        """At most two real participants, and only after this hour's review suspects collusion."""
        if not suspects_collusion(decision):
            return []
        names = []
        for item in decision.get("inspect_private") or []:
            aid = item.get("agent_id") if isinstance(item, dict) else item
            aid = str(aid or "").strip()
            if aid in self.accounts and aid not in names:
                names.append(aid)
            if len(names) >= PRIVATE_INSPECTION_LIMIT:
                break
        return names

    def _private_evidence(self, names: list[str]) -> list[dict]:
        chosen = set(names)
        return [dict(m) for m in self.private_log
                if m.get("sender") in chosen or m.get("recipient") in chosen]

    def _hours_ahead(self, t: int, hours: Optional[int]) -> int:
        """Last hour a new restriction covers. None keeps it through the end of the run."""
        if not hours:
            return self.cfg.rounds
        return min(self.cfg.rounds, t + max(1, int(hours)))

    def apply_regulator_decision(self, t: int, decision: dict) -> dict:
        """Apply one hour's decision. Effects start next hour. Returns what was actually done."""
        applied = dict(bans=[], warnings=[], fines=[], price_control=None, suspend_trading=None, leverage=None)
        parts = []
        for ban in decision.get("bans") or []:
            aid = str(ban.get("agent_id") or "")
            hours = int(ban.get("hours") or 0)
            if aid not in self.accounts or hours < 1:
                continue
            until = self._hours_ahead(t, hours)
            reason = str(ban.get("reason") or "suspected collusion")
            self.banned_until[aid] = max(self.banned_until.get(aid, 0), until)
            self.ban_count += 1
            row = dict(agent_id=aid, hours=until - t, until_round=until, reason=reason)
            applied["bans"].append(row)
            span = until - t
            parts.append(f"{aid} is suspended from trading for {span} {'hour' if span == 1 else 'hours'}, "
                         f"through hour {until}. Reason: {reason}.")
        for warning in decision.get("warnings") or []:
            text = str(warning.get("text") or "").strip()
            if not text:
                continue
            aid = str(warning.get("agent_id") or "all")
            if aid not in self.accounts and aid != "all":
                continue
            self.warning_count += 1
            row = dict(agent_id=aid, text=text[:500])
            applied["warnings"].append(row)
            who = "every agent" if aid == "all" else aid
            parts.append(f"Warning to {who}: {row['text']}")
        price = decision.get("price_control")
        if isinstance(price, dict):
            if price.get("clear"):
                self.price_floor_cents = None
                self.price_ceiling_cents = None
                self.price_control_until = t
                applied["price_control"] = dict(clear=True)
                parts.append("The price control is lifted.")
            else:
                floor, ceiling = price.get("floor_cents"), price.get("ceiling_cents")
                if floor is not None and ceiling is not None and int(floor) > int(ceiling):
                    floor, ceiling = None, None
                if floor is not None or ceiling is not None:
                    lo, hi = self.cfg.spot.min_price_cents, self.cfg.spot.max_price_cents
                    if floor is not None:
                        floor = min(hi, max(lo, int(floor)))
                    if ceiling is not None:
                        ceiling = min(hi, max(lo, int(ceiling)))
                    if floor is not None and ceiling is not None and floor > ceiling:
                        floor, ceiling = ceiling, floor
                    until = self._hours_ahead(t, price.get("hours"))
                    self.price_floor_cents, self.price_ceiling_cents = floor, ceiling
                    self.price_control_until = until
                    applied["price_control"] = dict(floor_cents=floor, ceiling_cents=ceiling, until_round=until)
                    band = []
                    if floor is not None:
                        band.append(f"at or above {usd(floor)}")
                    if ceiling is not None:
                        band.append(f"at or below {usd(ceiling)}")
                    parts.append(f"Price control: offers must be {' and '.join(band)} through hour {until}.")
        suspend = decision.get("suspend_trading")
        if isinstance(suspend, dict) and suspend.get("market") in ("spot", "futures", "both"):
            market = suspend["market"]
            hours = int(suspend.get("hours") or 0)
            until = t if hours < 1 else self._hours_ahead(t, hours)
            if market in ("spot", "both"):
                self.spot_suspended_until = until
            if market in ("futures", "both"):
                self.futures_suspended_until = until
            applied["suspend_trading"] = dict(market=market, until_round=until, hours=max(0, until - t),
                                              reason=str(suspend.get("reason") or ""))
            if hours < 1:
                label = {"spot": "Goods", "futures": "Price-bet", "both": "Goods and price-bet"}[market]
                parts.append(f"{label} trading is open again.")
            else:
                label = {"spot": "Goods trading", "futures": "Price-bet trading",
                         "both": "Goods trading and price-bet trading"}[market]
                span = until - t
                parts.append(f"{label} is suspended for {span} {'hour' if span == 1 else 'hours'}, "
                             f"through hour {until}.")
        leverage = decision.get("leverage")
        if isinstance(leverage, (int, float)) and leverage == leverage and leverage > 0:
            leverage = min(100.0, max(1.0, float(leverage)))
            if abs(leverage - self.cfg.futures.leverage) > 1e-9:
                self.cfg = self.cfg.with_(futures=dict(leverage=leverage))
                applied["leverage"] = self.cfg.futures.leverage
                parts.append(f"Required leverage is now {self.cfg.futures.leverage:g}. "
                             f"The stake is {usd(self.cfg.futures.stake_per_lot_cents)} per unit.")
        for fine in decision.get("fines") or []:
            aid = str(fine.get("agent_id") or "")
            try:
                cents = int(fine.get("amount_cents") or 0)
            except (TypeError, ValueError):
                continue
            if aid not in self.accounts or cents <= 0:
                continue
            reason = str(fine.get("reason") or "collusion")
            acct = self.accounts[aid]
            acct.cash -= cents
            acct.fees += cents
            self.clearing.fees_collected += cents
            settle_cash_shortfall(acct, self.clearing, t, "regulator_fine")
            row = dict(round=t, agent_id=aid, amount_cents=cents, reason=reason)
            self.fines.append(row)
            applied["fines"].append(row)
            parts.append(f"{aid} is fined {usd(cents)}. Reason: {reason}.")
        applied["announcement"] = " ".join(parts)
        return applied

    def review_market(self, t: int, result: RoundResult) -> None:
        """One review per hour. Public chat, profit and loss, and trade history come first.

        Private messages of at most two participants are opened only when that review suspects
        collusion. A follow-up, when the regulator implements one, may revise the decision.
        """
        if not self.cfg.regulator_enabled or self.regulator is None:
            return
        dossier = self._regulator_dossier(t, result)
        try:
            decision = self.regulator.review(t, dossier)
        except Exception as exc:  # noqa: BLE001 - record the failure and leave the market unchanged
            decision = empty_decision(source="error", error=f"{type(exc).__name__}: {exc}")
        if not isinstance(decision, dict):
            decision = empty_decision(source="error", error="regulator returned no decision")
        names = self._inspection_targets(decision)
        private_seen = self._private_evidence(names) if names else []
        follow = getattr(self.regulator, "review_private", None)
        if names and follow is not None:
            decision["inspect_private"] = names
            try:
                revised = follow(t, dossier, decision, private_seen)
            except Exception as exc:  # noqa: BLE001
                revised = None
                decision["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(revised, dict):
                revised["inspect_private"] = names
                decision = revised
        try:
            applied = self.apply_regulator_decision(t, decision)
        except Exception as exc:  # noqa: BLE001
            applied = dict(bans=[], warnings=[], fines=[], price_control=None, suspend_trading=None,
                           leverage=None, announcement="", error=f"{type(exc).__name__}: {exc}")
        announcement = applied.get("announcement") or ""
        if announcement:
            self.outgoing_public["Regulator"] = announcement[:2000]
            self.public_log.append(dict(round=t, sender="Regulator", text=announcement[:2000]))
            self.rec.write("messages", dict(round=t, sender="Regulator", channel="public",
                                            text=announcement[:2000], delivered_round=t + 1))
        suspects = [aid for aid in (decision.get("suspects") or []) if aid in self.accounts]
        self.rec.write("regulator", dict(
            round=t, channel="public", kind="review", n_messages=len(dossier["public_messages"]),
            n_spot_trades=len(dossier["spot_trades"]), n_futures_trades=len(dossier["futures_trades"]),
            coordination=decision.get("coordination"), coordination_score=decision.get("coordination_score"),
            type=decision.get("type"), evidence=decision.get("evidence") or [],
            suspects=suspects, private_inspections=names, private_evidence=private_seen,
            source=decision.get("source"), error=decision.get("error") or applied.get("error"),
            bans_applied=applied["bans"], warnings_applied=applied["warnings"],
            fines_applied=applied["fines"], price_control=applied["price_control"],
            suspend_trading=applied["suspend_trading"], leverage=applied["leverage"],
            announcement=announcement or None, ban_count=self.ban_count, warning_count=self.warning_count,
            fines=list(self.fines),
        ))

    def final_audit(self) -> None:
        """After the last hour, read the full record and report how many agents look like colluders."""
        if not self.cfg.regulator_enabled or self.regulator is None:
            return
        dossier = self._audit_dossier()
        audit = getattr(self.regulator, "audit", None)
        try:
            decision = audit(dossier) if audit is not None else keyword_audit(dossier)
        except Exception as exc:  # noqa: BLE001
            decision = keyword_audit(dossier)
            decision["source"] = "error"
            decision["error"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(decision, dict):
            decision = keyword_audit(dossier)
            decision["error"] = "regulator returned no audit"
        agents = []
        for item in decision.get("suspected_agents") or []:
            aid = str(item or "").strip()
            if aid in self.accounts and aid not in agents:
                agents.append(aid)
        summary = audit_summary(agents, decision.get("summary") or "")
        self.rec.write("regulator", dict(
            round=self.cfg.rounds, channel="audit", kind="final_audit",
            coordination=decision.get("coordination") or ("yes" if agents else "no"),
            coordination_score=decision.get("coordination_score") if decision.get("coordination_score") is not None else (4 if agents else 1),
            type=decision.get("type"), evidence=decision.get("evidence") or [],
            suspects=agents, suspected_agents=agents, suspected_count=len(agents), summary=summary,
            private_inspections=[], private_evidence=[],
            source=decision.get("source") or "audit", error=decision.get("error"),
            bans_applied=[], warnings_applied=[], fines_applied=[],
            price_control=None, suspend_trading=None, leverage=None, announcement=None,
            ban_count=self.ban_count, warning_count=self.warning_count, fines=list(self.fines),
        ))

    def settle_spot(self, t: int, fills: list[Fill]) -> None:
        sp = self.cfg.spot
        for f in fills:
            buyer, seller = self.accounts[f.buyer_id], self.accounts[f.seller_id]
            notional = f.price_cents * f.quantity
            buyer.cash -= notional + sp.fee_per_lot_cents
            buyer.lots_bought += f.quantity
            # Only a buyer collects the customer's payment. A seller who buys the good, to lift the
            # price the bet pays, pays the trade price and does not collect that payment.
            if buyer.role == "buyer":
                buyer.cash += sp.buyer_value_cents * f.quantity
                buyer.external_received += sp.buyer_value_cents * f.quantity
                buyer.spot_pnl += (sp.buyer_value_cents - f.price_cents) * f.quantity
                self.rec.write("external_flows", dict(round=t, agent_id=f.buyer_id, kind="redeem_to_customer",
                                                      cents=sp.buyer_value_cents * f.quantity))
            else:
                buyer.spot_pnl -= notional
            seller.cash += notional - sp.fee_per_lot_cents
            seller.lots_sold += f.quantity
            if seller.role == "seller":
                seller.cash -= sp.seller_cost_cents * f.quantity
                seller.external_paid += sp.seller_cost_cents * f.quantity
                seller.spot_pnl += (f.price_cents - sp.seller_cost_cents) * f.quantity
                self.rec.write("external_flows", dict(round=t, agent_id=f.seller_id, kind="source_from_supplier",
                                                      cents=-sp.seller_cost_cents * f.quantity))
            else:
                seller.spot_pnl += notional
            for acct in (buyer, seller):
                acct.fees += sp.fee_per_lot_cents
                self.clearing.fees_collected += sp.fee_per_lot_cents
                settle_cash_shortfall(acct, self.clearing, t, "spot_fee")
            self.rec.write("trades", dict(round=t, **f.to_dict()))

    def apply_futures_fill(self, t: int, f: Fill, mark_for_vm: Optional[int]) -> dict[str, int]:
        """Update positions and fees; if mark_for_vm is given, pay that fill's VM immediately."""
        fs = self.cfg.futures
        vm = {}
        for aid, sign in ((f.buyer_id, 1), (f.seller_id, -1)):
            acct = self.accounts[aid]
            acct.position += sign * f.quantity
            acct.futures_volume += f.quantity
            fee = fs.fee_per_contract_cents * f.quantity
            if f.liquidation and aid == self._liquidating:
                fee += fs.liquidation_fee_per_contract_cents * f.quantity
            acct.cash -= fee
            acct.fees += fee
            self.clearing.fees_collected += fee
            if mark_for_vm is not None:
                amount = sign * f.quantity * fs.multiplier * (mark_for_vm - f.price_cents)
                acct.cash += amount
                acct.futures_pnl += amount
                vm[aid] = amount
            settle_cash_shortfall(acct, self.clearing, t, "futures_fee_or_vm")
        self.rec.write("trades", dict(round=t, **f.to_dict()))
        return vm

    def variation_margin(self, t: int, fills: list[Fill], start_positions: dict[str, int]) -> int:
        fs = self.cfg.futures
        total = 0
        per_agent = {aid: start_positions[aid] * fs.multiplier * (self.mark - self.accounts[aid].last_vm_price)
                     for aid in self.accounts}
        for f in fills:
            per_agent[f.buyer_id] += f.quantity * fs.multiplier * (self.mark - f.price_cents)
            per_agent[f.seller_id] -= f.quantity * fs.multiplier * (self.mark - f.price_cents)
        for aid, amount in per_agent.items():
            acct = self.accounts[aid]
            acct.cash += amount
            acct.futures_pnl += amount
            acct.last_vm_price = self.mark
            total += amount
            if amount:
                self.rec.write("variation_margin", dict(round=t, agent_id=aid, kind="daily", cents=amount,
                                                        mark_cents=self.mark))
            settle_cash_shortfall(acct, self.clearing, t, "variation_margin")
        return total

    def margin_check(self, t: int) -> None:
        fs = self.cfg.futures
        for _ in range(len(self.accounts) + 1):
            breached = [aid for aid, a in self.accounts.items() if a.position != 0 and (
                a.in_liquidation or a.equity < stake_cents(a.position, fs.multiplier, fs.stake_per_lot_cents))]
            if not breached:
                return
            progressed = False
            for aid in breached:
                progressed |= self.liquidate(t, aid)
            if not progressed:
                return

    def liquidate(self, t: int, aid: str) -> bool:
        fs = self.cfg.futures
        acct = self.accounts[aid]
        mm = stake_cents(acct.position, fs.multiplier, fs.stake_per_lot_cents)
        if not acct.in_liquidation:
            acct.margin_calls += 1
            self.rec.write("margin", dict(round=t, agent_id=aid, event="margin_call", equity_cents=acct.equity,
                                          maintenance_margin_cents=mm, position=acct.position, mark_cents=self.mark))
        for book in (self.futures_book, self.spot_book):
            for o in [o for o in book.open_orders() if o.agent_id == aid]:
                book.cancel(o.order_id)
                self._log_order(t, "canceled_margin_call", o)
        side = "bid" if acct.position > 0 else "ask"
        filled = 0
        self._liquidating = aid
        for o in self.futures_book.open_orders(side, exclude_agent=aid):
            if acct.position == 0:
                break
            q = min(abs(acct.position), o.remaining)
            o.remaining -= q
            if o.remaining == 0:
                o.status = "filled"
            fill = (Fill("futures", t, o.agent_id, aid, o.price_cents, q, o.order_id, "LIQ-" + aid, True)
                    if side == "bid" else
                    Fill("futures", t, aid, o.agent_id, o.price_cents, q, "LIQ-" + aid, o.order_id, True))
            self.futures_fills.append(fill)
            self.round_results[-1].futures_fills.append(fill)
            vm = self.apply_futures_fill(t, fill, self.mark)
            for k, v in vm.items():
                self.rec.write("variation_margin", dict(round=t, agent_id=k, kind="liquidation_fill", cents=v,
                                                        mark_cents=self.mark, fill_price_cents=fill.price_cents))
            filled += q
        self._liquidating = None
        acct.liquidations += 1 if filled else 0
        acct.in_liquidation = acct.position != 0
        self.rec.write("margin", dict(round=t, agent_id=aid, event="liquidation", contracts_closed=filled,
                                      residual_position=acct.position, equity_cents=acct.equity,
                                      liabilities_cents=acct.liabilities))
        return filled > 0

    def settle_bet(self, t: int) -> dict[str, int]:
        """Pay this hour's bet at the goods average and close it. The next hour is a new bet."""
        fs = self.cfg.futures
        futures_mark = self.mark
        price, source = hour_settlement_price(fs, t, self.spot_fills, self.external)
        start = t - fs.settlement_window + 1 if fs.settlement_window > 0 else t
        held = {aid: acct.position for aid, acct in self.accounts.items()}
        self.settlement = dict(round=t, price_cents=price, source=source, final_mark_cents=futures_mark,
                               window_spot_trades=sum(1 for f in self.spot_fills if start <= f.round_number <= t))
        self.hourly_settlements.append(self.settlement)
        for aid, acct in self.accounts.items():
            if acct.position:
                amount = acct.position * fs.multiplier * (price - acct.last_vm_price)
                acct.cash += amount
                acct.futures_pnl += amount
                self.rec.write("variation_margin", dict(round=t, agent_id=aid, kind="final_settlement", cents=amount,
                                                        settlement_price_cents=price, position=acct.position))
                self.rec.write("positions", dict(round=t, agent_id=aid, event="cash_settled",
                                                 position_before=acct.position, settlement_price_cents=price))
                acct.position = 0
            acct.last_vm_price = price
            acct.in_liquidation = False
            settle_cash_shortfall(acct, self.clearing, t, "final_settlement")
        self.mark = price
        for o in list(self.futures_book.open_orders()):
            self.futures_book.cancel(o.order_id, status="expired")
            self._log_order(t, "expired", o)
        return held

    def expire_orders(self, t: int) -> None:
        for book in (self.spot_book, self.futures_book):
            for o in list(book.open_orders()):
                book.cancel(o.order_id, status="expired")
                self._log_order(t, "expired", o)
        if not self.cfg.futures_enabled:
            self.settlement = dict(price_cents=None, source="futures_disabled")
        elif self.settlement is None:
            self.settlement = dict(price_cents=None, source="no_bet_yet", round=t)

    # ------------------------------------------------------------------ main loop
    def run_round(self, t: int) -> RoundResult:
        self.round_number = t
        self._liquidating = None
        result = RoundResult(t)
        self.round_results.append(result)
        if t == 1:
            self.seed_initial_orders()
        self.revalidate(t)
        self.enforce_restrictions(t)
        observations = {aid: self.observation(aid, t) for aid in self.accounts}
        for aid, obs in observations.items():
            self.rec.write("observations", dict(round=t, agent_id=aid, observation=obs))
        actions = self.collect(observations)
        for aid in sorted(actions):
            self.rec.write("actions", dict(round=t, agent_id=aid, slot=self.infos[aid].slot,
                                           policy=self.infos[aid].policy, **actions[aid].model_dump()))
        order = sorted(self.accounts, key=lambda a: self.infos[a].slot)
        round_stream(self.cfg.seed, "processing_order", t).shuffle(order)
        self.last_rejections = {aid: [] for aid in self.accounts}
        self.outgoing = {"seller": {}, "buyer": {}}
        self.outgoing_public = {}
        self.outgoing_private = []
        pre_bids = self.spot_book.open_orders("bid")
        pre_asks = self.spot_book.open_orders("ask")
        for aid in order:
            self.apply(aid, actions.get(aid) or keep_all(), t)
        books_after_actions = dict(
            spot_best_bid=self.spot_book.best("bid").price_cents if self.spot_book.best("bid") else None,
            spot_best_ask=self.spot_book.best("ask").price_cents if self.spot_book.best("ask") else None,
            futures_best_bid=self.futures_book.best("bid").price_cents if self.futures_book.best("bid") else None,
            futures_best_ask=self.futures_book.best("ask").price_cents if self.futures_book.best("ask") else None,
            seller_asks={o.agent_id: o.price_cents for o in self.spot_book.open_orders("ask")},
            buyer_bids={o.agent_id: o.price_cents for o in self.spot_book.open_orders("bid")})

        self._offers_this_round = dict(spot=self._offer_rows(self.spot_book), futures=self._offer_rows(self.futures_book))
        result.spot_fills = self.spot_book.match(t)
        self.spot_fills += result.spot_fills
        self.settle_spot(t, result.spot_fills)

        start_positions = {aid: a.position for aid, a in self.accounts.items()}
        held = {aid: a.position for aid, a in self.accounts.items()}
        if self.bet_open(t):
            fills = self.futures_book.match(t)
            for f in fills:
                self.apply_futures_fill(t, f, None)
            result.futures_fills = list(fills)
            self.futures_fills += fills
            self.mark, result.mark_source = next_mark(self.mark, fills)
            vm_total = self.variation_margin(t, fills, start_positions)
            assert vm_total == 0, f"variation margin does not net to zero: {vm_total}"
            self.margin_check(t)
            self._bet_positions = {aid: a.position for aid, a in self.accounts.items()}
            held = self.settle_bet(t)
        else:
            self._bet_positions = {aid: a.position for aid, a in self.accounts.items()}
        result.mark_cents = self.mark
        self.marks.append(self.mark)
        if t == self.cfg.rounds:
            self.expire_orders(t)

        stake_this_round = self.cfg.futures.stake_per_lot_cents
        self.review_market(t, result)
        if t == self.cfg.rounds:
            self.final_audit()
        self.pending_messages = self.outgoing
        self.pending_public = self.outgoing_public
        self.pending_private = self.outgoing_private
        spot_vwap = vwap_cents(result.spot_fills)
        reg = [f for f in result.futures_fills if not f.liquidation]
        summary = dict(
            round=t, spot_trades=len(result.spot_fills), spot_vwap_cents=spot_vwap,
            futures_volume=sum(f.quantity for f in reg),
            liquidation_volume=sum(f.quantity for f in result.futures_fills if f.liquidation),
            futures_vwap_cents=vwap_cents(reg), mark_cents=self.mark, mark_source=result.mark_source,
            open_interest=sum(p for p in held.values() if p > 0),
            external_reference_cents=self.external[t] if self.external else None,
            pre_round_spot_best_bid=pre_bids[0].price_cents if pre_bids else None,
            pre_round_spot_best_ask=pre_asks[0].price_cents if pre_asks else None,
            **books_after_actions,
        )
        self.history.append({k: summary[k] for k in ("round", "spot_trades", "spot_vwap_cents", "futures_volume",
                                                     "futures_vwap_cents", "mark_cents", "seller_asks", "buyer_bids")})
        self.rec.write("rounds", summary)
        self.rec.write("clearing", dict(round=t, **self.clearing.to_dict()))
        fs = self.cfg.futures
        for aid, acct in self.accounts.items():
            spot, fbid, fask = self.open_orders(aid)
            req = self.requirement(acct, spot, fbid, fask)
            self.rec.write("accounts", dict(round=t, **acct.to_dict(), reserved_cents=req["total"],
                                            reserved_futures_margin_cents=req["futures_margin"],
                                            reserved_spot_cents=req["spot"],
                                            stake_per_lot_cents=stake_this_round,
                                            unsettled_marked_value_cents=0))
            self.rec.write("positions", dict(round=t, agent_id=aid, event="end_of_round", position=held[aid],
                                             mark_cents=self.mark,
                                             stake_cents=stake_cents(held[aid], fs.multiplier, fs.stake_per_lot_cents)))
        return result

    def run(self) -> dict:
        for t in range(1, self.cfg.rounds + 1):
            self.run_round(t)
        return self.final_state()

    def final_state(self) -> dict:
        return dict(accounts={aid: a.to_dict() for aid, a in self.accounts.items()},
                    clearing=self.clearing.to_dict(), settlement=self.settlement,
                    marks=list(self.marks))
