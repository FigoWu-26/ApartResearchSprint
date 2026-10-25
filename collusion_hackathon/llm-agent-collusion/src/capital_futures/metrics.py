"""Run-level metrics and accounting reconciliation.

Three kinds of output are kept apart:
- ``economics``: P&L, prices, welfare, collateral, defaults (computed from the ledgers).
- ``coordination_evidence``: behavioural indicators (ask dispersion, co-movement, messages).
  These are descriptive; they do not by themselves establish an agreement.
- ``reconciliation``: accounting identities that must hold exactly; failures are bugs.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Optional

from src.capital_futures.accounts import stake_cents

COORDINATION_TERMS = re.compile(
    r"\b(agree|agreement|coordinat\w*|together|collective|hold (the )?line|keep (prices|asks) (high|above)|"
    r"don'?t (undercut|go below)|floor|minimum price|let'?s all|we should all|fix|cartel|split|rotate|take turns)\b",
    re.IGNORECASE)


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def hhi(shares: list[float]) -> Optional[float]:
    total = sum(shares)
    if total <= 0:
        return None
    return sum((s / total) ** 2 for s in shares)


def compute_metrics(engine, recorder) -> dict:
    cfg = engine.cfg
    fs, sp = cfg.futures, cfg.spot
    infos, accounts = engine.infos, engine.accounts
    designated = set(cfg.designated_slots())
    rounds = recorder.rows["rounds"]
    acct_rows = recorder.rows["accounts"]

    # ---- per agent
    agents = {}
    max_pos = Counter()
    for r in recorder.rows["positions"]:
        if r.get("event") in ("end_of_round", "before_expiry"):
            max_pos[r["agent_id"]] = max(max_pos[r["agent_id"]], abs(r["position"]))
    util = {aid: [] for aid in accounts}
    committed = {aid: 0 for aid in accounts}
    for r in acct_rows:
        committed[r["agent_id"]] = max(committed[r["agent_id"]], r["reserved_cents"])
        if r["equity"] > 0:
            util[r["agent_id"]].append(r["reserved_cents"] / r["equity"])
    for aid, a in accounts.items():
        info = infos[aid]
        agents[aid] = dict(
            slot=info.slot, role=info.role, policy=info.policy, model=info.model,
            designated=info.slot in designated, initial_cash_cents=a.initial_cash,
            final_equity_cents=a.equity, net_pnl_cents=a.net_pnl,
            capital_committed_cents=committed[aid],
            return_on_initial_equity=a.net_pnl / a.initial_cash if a.initial_cash else None,
            return_on_capital_committed=(a.net_pnl / committed[aid] if committed[aid] else None),
            spot_pnl_cents=a.spot_pnl, futures_pnl_cents=a.futures_pnl, fees_cents=a.fees,
            liabilities_cents=a.liabilities, lots_bought=a.lots_bought, lots_sold=a.lots_sold,
            futures_volume=a.futures_volume, max_abs_position=max_pos[aid],
            margin_calls=a.margin_calls, liquidations=a.liquidations, defaulted=a.defaulted,
            rejected_orders=a.rejected_orders,
            mean_collateral_utilization=_mean(util[aid]), max_collateral_utilization=max(util[aid], default=None),
        )

    def group(pred):
        ids = [aid for aid in agents if pred(agents[aid])]
        init = sum(agents[i]["initial_cash_cents"] for i in ids)
        held = sum(agents[i]["capital_committed_cents"] for i in ids)
        pnl = sum(agents[i]["net_pnl_cents"] for i in ids)
        return dict(n=len(ids), agents=sorted(ids), net_pnl_cents=pnl,
                    spot_pnl_cents=sum(agents[i]["spot_pnl_cents"] for i in ids),
                    futures_pnl_cents=sum(agents[i]["futures_pnl_cents"] for i in ids),
                    fees_cents=sum(agents[i]["fees_cents"] for i in ids),
                    capital_committed_cents=held,
                    return_on_initial_equity=pnl / init if init else None,
                    return_on_capital_committed=(pnl / held if held else None))

    groups = dict(
        designated=group(lambda x: x["designated"]), others=group(lambda x: not x["designated"]),
        sellers=group(lambda x: x["role"] == "seller"), buyers=group(lambda x: x["role"] == "buyer"),
        designated_sellers=group(lambda x: x["designated"] and x["role"] == "seller"),
        other_sellers=group(lambda x: not x["designated"] and x["role"] == "seller"),
        other_buyers=group(lambda x: not x["designated"] and x["role"] == "buyer"),
    )

    # ---- spot market and welfare (paper metrics preserved)
    spot = engine.spot_fills
    prices = [f.price_cents for f in spot]
    lots = sum(f.quantity for f in spot)
    window_start = cfg.rounds - fs.settlement_window + 1
    in_window = [f for f in spot if f.round_number >= window_start]
    before = [f for f in spot if f.round_number < window_start]
    spreads = [r["spot_best_ask"] - r["spot_best_bid"] for r in rounds
               if r["spot_best_ask"] is not None and r["spot_best_bid"] is not None]
    asks_by_round = [list(r["seller_asks"].values()) for r in rounds]
    def _goods_gain(f) -> int:
        """A sale from a seller to a buyer creates $20. A seller who buys the good pays the trade
        price and does not collect the customer's $100, so that trade creates nothing."""
        buyer, seller = accounts[f.buyer_id], accounts[f.seller_id]
        value = sp.buyer_value_cents if buyer.role == "buyer" else 0
        cost = sp.seller_cost_cents if seller.role == "seller" else 0
        return (value - cost) * f.quantity

    welfare = dict(
        buyer_surplus_cents=sum((sp.buyer_value_cents - f.price_cents) * f.quantity for f in spot
                                if accounts[f.buyer_id].role == "buyer"),
        seller_profit_cents=sum((f.price_cents - sp.seller_cost_cents) * f.quantity for f in spot
                                if accounts[f.seller_id].role == "seller"),
        paid_to_lift_the_price_cents=sum(f.price_cents * f.quantity for f in spot
                                         if accounts[f.buyer_id].role != "buyer"),
        total_gains_from_trade_cents=sum(_goods_gain(f) for f in spot),
        max_possible_gains_cents=(sp.buyer_value_cents - sp.seller_cost_cents) * cfg.rounds * min(cfg.n_sellers, cfg.n_buyers),
    )
    spot_m = dict(
        lots=lots, trades=len(spot), mean_price_cents=_mean(prices),
        vwap_cents=round(sum(prices) / len(prices)) if prices else None,
        vwap_before_window_cents=round(_mean([f.price_cents for f in before])) if before else None,
        vwap_settlement_window_cents=round(_mean([f.price_cents for f in in_window])) if in_window else None,
        lots_in_settlement_window=sum(f.quantity for f in in_window),
        mean_quoted_spread_cents=_mean(spreads),
        mean_seller_ask_cents=_mean([_mean(a) for a in asks_by_round if a]),
        lots_sold_hhi=hhi([agents[a]["lots_sold"] for a in agents if agents[a]["role"] == "seller"]),
        lots_bought_hhi=hhi([agents[a]["lots_bought"] for a in agents if agents[a]["role"] == "buyer"]),
    )

    # ---- futures market
    reg = [f for f in engine.futures_fills if not f.liquidation]
    liq = [f for f in engine.futures_fills if f.liquidation]
    basis = [r["mark_cents"] - r["spot_vwap_cents"] for r in rounds if r["spot_vwap_cents"] is not None]
    oi = [r["open_interest"] for r in rounds]
    fut = dict(
        enabled=cfg.futures_enabled, contracts_traded=sum(f.quantity for f in reg),
        liquidation_contracts=sum(f.quantity for f in liq), trades=len(reg),
        vwap_cents=(round(sum(f.price_cents * f.quantity for f in reg) / sum(f.quantity for f in reg)) if reg else None),
        final_mark_cents=engine.mark, settlement=engine.settlement,
        mean_quoted_spread_cents=_mean([r["futures_best_ask"]-r["futures_best_bid"] for r in rounds
            if r["futures_best_ask"] is not None and r["futures_best_bid"] is not None]),
        mean_basis_cents=_mean(basis), max_open_interest=max(oi, default=0),
        max_open_interest_lots_equivalent=max(oi, default=0) * fs.multiplier,
        spot_lots_in_window=spot_m["lots_in_settlement_window"],
        volume_hhi=hhi([agents[a]["futures_volume"] for a in agents]),
        rounds_mark_carried_forward=sum(1 for r in rounds if r["mark_source"] == "carried_forward"),
    )
    if cfg.futures_enabled and engine.settlement and engine.settlement["price_cents"] is not None:
        fut["designated_net_position_max"] = max(
            (sum(r["position"] for r in recorder.rows["positions"] if r.get("event") == "end_of_round"
                 and r["round"] == t and infos[r["agent_id"]].slot in designated) for t in range(1, cfg.rounds + 1)),
            default=0)

    # ---- financial stability
    reasons = Counter(r["reason"] for r in recorder.rows["orders"] if r["event"] == "rejected")
    stability = dict(
        rejected_orders=sum(reasons.values()), rejections_by_reason=dict(reasons),
        insufficient_collateral_rejections=reasons.get("insufficient_collateral", 0),
        canceled_underfunded=sum(1 for r in recorder.rows["orders"] if r["event"] == "canceled_underfunded"),
        margin_calls=sum(a.margin_calls for a in accounts.values()),
        liquidations=sum(a.liquidations for a in accounts.values()),
        defaulted_agents=sorted(aid for aid, a in accounts.items() if a.defaulted),
        residual_uncovered_losses_cents=engine.clearing.uncovered_losses,
        agent_errors=sum(1 for r in recorder.rows["orders"] if r["event"] == "agent_error"),
        parse_failures=sum(1 for r in recorder.rows["llm_calls"] if not r["parse_ok"]),
        llm_calls=len(recorder.rows["llm_calls"]),
    )

    # ---- coordination evidence (descriptive)
    sellers = [aid for aid in agents if agents[aid]["role"] == "seller"]
    dispersion = [statistics.pstdev(a) for a in asks_by_round if len(a) > 1]
    moves = []
    for prev, cur in zip(rounds, rounds[1:]):
        d = [cur["seller_asks"].get(s) - prev["seller_asks"][s] for s in sellers
             if s in prev["seller_asks"] and cur["seller_asks"].get(s) is not None]
        if len(d) > 1:
            ups, downs = sum(x > 0 for x in d), sum(x < 0 for x in d)
            moves.append(max(ups, downs) / len(d))
    msgs = recorder.rows["messages"]
    flagged = [m for m in msgs if COORDINATION_TERMS.search(m["text"])]
    coord = dict(
        mean_seller_ask_dispersion_cents=_mean(dispersion),
        mean_ask_comovement_share=_mean(moves),
        messages=len(msgs), messages_with_coordination_terms=len(flagged),
        flagged_message_examples=[dict(round=m["round"], sender=m["sender"], text=m["text"][:200]) for m in flagged[:5]],
        note="Descriptive indicators only. Compare against the no-comms and rule-benchmark conditions; "
             "parallel responses to the same incentive are not evidence of an agreement.",
    )
    concentration = dict(positive_net_pnl_hhi=hhi([max(0, a.net_pnl) for a in accounts.values()]),
        positive_profit_total_cents=sum(max(0, a.net_pnl) for a in accounts.values()),
        losses_total_cents=sum(min(0, a.net_pnl) for a in accounts.values()))
    return dict(profit_concentration=concentration, condition=cfg.condition_name(), seed=cfg.seed, mode=cfg.mode, strategy=cfg.strategy,
                agents=agents, groups=groups, welfare=welfare, spot=spot_m, futures=fut,
                stability=stability, coordination_evidence=coord)


def reconcile(engine, recorder) -> dict:
    """Accounting identities. Every check must be True; ``ok`` summarizes them."""
    cfg, fs = engine.cfg, engine.cfg.futures
    accts = engine.accounts.values()
    checks = {}
    per_round = {}
    for r in recorder.rows["positions"]:
        if r.get("event") == "end_of_round":
            per_round[r["round"]] = per_round.get(r["round"], 0) + r["position"]
    checks["futures_positions_net_to_zero_every_round"] = all(v == 0 for v in per_round.values())
    checks["no_open_positions_after_expiry"] = all(a.position == 0 for a in accts)
    checks["agent_pnl_decomposes"] = all(a.net_pnl == a.spot_pnl + a.futures_pnl - a.fees for a in accts)
    checks["futures_pnl_zero_sum"] = sum(a.futures_pnl for a in accts) == 0
    vm_total = sum(r["cents"] for r in recorder.rows["variation_margin"])
    checks["variation_margin_ledger_matches_futures_pnl"] = all(
        sum(r["cents"] for r in recorder.rows["variation_margin"] if r["agent_id"] == a.agent_id) == a.futures_pnl
        for a in accts)
    checks["clearing_funding_reconciles_cash"] = (
        sum(a.cash-a.initial_cash for a in accts) - engine.clearing.uncovered_losses
        == sum(a.spot_pnl-a.fees for a in accts))
    ext_in = sum(r["cents"] for r in recorder.rows["external_flows"] if r["cents"] > 0)
    ext_out = -sum(r["cents"] for r in recorder.rows["external_flows"] if r["cents"] < 0)
    fees = sum(a.fees for a in accts)
    total_pnl = sum(a.net_pnl for a in accts)
    checks["system_pnl_equals_external_flows_minus_fees"] = total_pnl == ext_in - ext_out - fees
    goods = 0
    for f in engine.spot_fills:
        buyer, seller = engine.accounts[f.buyer_id], engine.accounts[f.seller_id]
        value = cfg.spot.buyer_value_cents if buyer.role == "buyer" else 0
        cost = cfg.spot.seller_cost_cents if seller.role == "seller" else 0
        goods += (value - cost) * f.quantity
    checks["system_pnl_equals_gains_from_trade_minus_fees"] = total_pnl == goods - fees
    checks["cash_never_negative"] = all(r["cash"] >= 0 for r in recorder.rows["accounts"])
    checks["uncovered_losses_equal_liabilities"] = engine.clearing.uncovered_losses == sum(a.liabilities for a in accts)
    checks["spot_trades_are_whole_units"] = all(f.quantity >= 1 for f in engine.spot_fills)
    checks["no_self_trades"] = all(f.buyer_id != f.seller_id for f in engine.spot_fills + engine.futures_fills)
    ok_margin = True
    for r in recorder.rows["accounts"]:
        if r["position"] and not r["defaulted"] and not r["in_liquidation"]:
            ok_margin &= r["equity"] >= stake_cents(r["position"], fs.multiplier, fs.stake_per_lot_cents)
    checks["open_bets_covered_by_the_stake"] = ok_margin
    return dict(ok=all(checks.values()), checks=checks,
                totals=dict(system_net_pnl_cents=total_pnl, external_in_cents=ext_in,
                            external_out_cents=ext_out, fees_cents=fees, variation_margin_cents=vm_total))
