"""Offline tests for the capital x futures extension (no network, no API keys)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.capital_futures import agents as ag
from src.capital_futures.actions import parse_action
from src.capital_futures.book import Order, OrderBook
from src.capital_futures.config import MILLION, ExperimentConfig, PRESETS, expand_preset, primary_grid
from src.capital_futures.engine import AgentInfo, Engine
from src.capital_futures.ledger import Recorder
from src.capital_futures.metrics import reconcile
from src.capital_futures.run import calibrate, compare_ledgers, replay, run_one
from src.capital_futures.settlement import settlement_price

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- helpers
def place(price, qty=None):
    d = dict(action="place", price=price)
    if qty is not None:
        d["quantity"] = qty
    return d


def make_engine(scripts: dict, n_sellers=2, n_buyers=2, rounds=1, capital=10_000, **cfg):
    """Engine with scripted agents whose ids equal their slots. ``capital`` is USD per agent."""
    futures = dict(cfg.pop("futures", {}))
    futures.setdefault("settlement_window", 0)  # mechanism tests use a bet in hour 1; the game default is 5
    c = ExperimentConfig(n_sellers=n_sellers, n_buyers=n_buyers, rounds=rounds, capital_scenario="equal",
                         base_capital_cents=capital * 100, mode="sim").with_(
        spot=dict(initial_orders=False), futures=futures, **cfg)
    infos, agents = {}, {}
    for slot in c.slots:
        infos[slot] = AgentInfo(slot, slot, "seller" if slot.startswith("S") else "buyer", "none", "scripted")
        agents[slot] = ag.ScriptedAgent(scripts.get(slot, {}))
    rec = Recorder()
    return Engine(c, agents, infos, rec), rec


# ---------------------------------------------------------------------------- order book
def test_matching_price_priority_midpoint_and_time_priority():
    book = OrderBook("futures")
    book.add(Order("b1", "A", "futures", "bid", 9100, 1, 1, 1, seq=2))
    book.add(Order("b2", "B", "futures", "bid", 9100, 1, 1, 1, seq=1))   # same price, earlier
    book.add(Order("a1", "C", "futures", "ask", 9000, 1, 1, 1, seq=3))
    book.add(Order("a2", "D", "futures", "ask", 9200, 1, 1, 1, seq=4))
    fills = book.match(1)
    assert len(fills) == 1
    assert (fills[0].buyer_id, fills[0].seller_id, fills[0].price_cents) == ("B", "C", 9050)
    assert book.orders["b1"].status == "open" and book.orders["a2"].status == "open"


def test_partial_fills_and_persistence():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 10))},
                            "S2": {1: dict(futures_ask=place(89, 3))},
                            "B1": {1: dict(futures_ask=place(90, 4))}}, rounds=2, capital=1_000_000)
    eng.run_round(1)
    fills = eng.round_results[0].futures_fills
    assert [(f.seller_id, f.quantity, f.price_cents) for f in fills] == [("S2", 3, 8950), ("B1", 4, 9000)]
    expired = [r for r in rec.rows["orders"] if r["event"] == "expired" and r.get("agent_id") == "S1"]
    assert expired and expired[0]["remaining"] == 3                           # unfilled units die with the hour
    assert eng.accounts["S1"].position == 0                                   # the hour's bet is paid off
    ids = [r["order_id"] for r in rec.rows["orders"] if r["event"] == "accepted"]
    assert len(ids) == len(set(ids))                                          # unique order ids


def test_cancel_never_reverses_a_trade():
    eng, _ = make_engine({"S1": {1: dict(futures_bid=place(90, 5)), 2: dict(futures_bid=dict(action="cancel"))},
                          "B1": {1: dict(futures_ask=place(90, 2))}}, rounds=3, capital=1_000_000)
    eng.run_round(1)
    eng.run_round(2)
    assert eng.futures_book.agent_order("S1", "bid") is None
    assert eng.accounts["S1"].position == 0 and eng.accounts["B1"].position == 0   # paid off in the hour they traded


def test_self_trade_prevention():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 1), futures_ask=place(89, 1))}}, capital=1_000_000)
    eng.run_round(1)
    rej = [r for r in rec.rows["orders"] if r["event"] == "rejected"]
    assert [r["reason"] for r in rej] == ["self_trade_prevention"]
    assert eng.accounts["S1"].position == 0


def test_one_bid_one_ask_per_agent_replace_is_atomic():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 1)), 2: dict(futures_bid=place(91, 2)),
                                   3: dict(futures_bid=place(91, 50))}}, rounds=4, capital=20_000)
    for t in (1, 2, 3):
        eng.run_round(t)
    assert eng.futures_book.open_orders("bid") == []                              # each hour's unfilled bet expires
    rej = [r for r in rec.rows["orders"] if r["event"] == "rejected"]
    assert rej[-1]["reason"] == "insufficient_collateral" and rej[-1]["existing_order_unchanged"] is None


def test_collateral_reservation_is_shared_across_spot_and_futures():
    # $10,000 buyer: one contract needs $9,000 initial margin; a $95 spot bid then fits, two contracts do not.
    def entries(rec):
        return [(r["market"], r["event"]) for r in rec.rows["orders"]
                if r.get("agent_id") == "B1" and r["event"] in ("accepted", "rejected")]
    eng, rec = make_engine({"B1": {1: dict(futures_bid=place(90, 1), spot_order=place(95))}}, capital=10_000)
    eng.run_round(1)
    assert entries(rec) == [("spot", "accepted"), ("futures", "accepted")]
    eng2, rec2 = make_engine({"B1": {1: dict(spot_order=place(95), futures_bid=place(90, 1))}}, capital=9_050)
    eng2.run_round(1)
    events = entries(rec2)
    assert events == [("spot", "accepted"), ("futures", "rejected")]


# ---------------------------------------------------------------------------- P&L and margin
def test_worked_example_long_short_settlement():
    """Multiplier 1000: bought at $90, settled at $95 -> +$5,000 long, -$5,000 short; no $90,000 debit."""
    eng, rec = make_engine({"S1": {1: dict(spot_order=place(95), futures_bid=place(90, 1))},
                            "B1": {1: dict(spot_order=place(95), futures_ask=place(90, 1))}}, capital=100_000)
    eng.run()
    assert eng.settlement["price_cents"] == 9500 and eng.settlement["source"].startswith("spot_vwap")
    s1, b1 = eng.accounts["S1"], eng.accounts["B1"]
    assert s1.futures_pnl == 5_000 * 100 and b1.futures_pnl == -5_000 * 100
    assert s1.cash == 100_000 * 100 + (95 - 80) * 100 + 5_000 * 100     # spot margin $15 + futures $5,000
    assert b1.cash == 100_000 * 100 + (100 - 95) * 100 - 5_000 * 100
    assert min(r["cash"] for r in rec.rows["accounts"]) > 90_000 * 100  # principal never debited
    assert reconcile(eng, rec)["ok"]


def test_reversal_long_to_short():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 3)), 2: dict(futures_ask=place(92, 5))},
                            "S2": {1: dict(futures_ask=place(90, 3))},
                            "B1": {2: dict(futures_bid=place(92, 5))}}, rounds=3, capital=1_000_000)
    eng.run_round(1)                                                  # bought 3 at $90, paid at $90, closed
    assert eng.accounts["S1"].position == 0 and eng.accounts["S1"].futures_pnl == 0
    eng.run_round(2)                                                  # a new bet: sold 5 at $92, paid at $90
    assert eng.accounts["S1"].position == 0
    assert eng.accounts["S1"].futures_pnl == 5 * 1000 * 200          # +$10,000
    eng.run_round(3)
    assert eng.settlement["source"] == "fallback_reference"
    assert reconcile(eng, rec)["ok"]


def test_variation_margin_daily_zero_sum_and_positions_balance():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 1))}, "B1": {1: dict(futures_ask=place(90, 1))},
                            "S2": {2: dict(futures_bid=place(93, 1))}, "B2": {2: dict(futures_ask=place(93, 1))}},
                           rounds=3, capital=1_000_000)
    for t in (1, 2):
        eng.run_round(t)
    assert eng.mark == 9000                                           # hour 2's bet is paid at the $90 fallback
    assert eng.accounts["S1"].futures_pnl == 0                        # hour 1's bet did not stay open
    assert eng.accounts["S2"].futures_pnl == -3_000 * 100 and eng.accounts["B2"].futures_pnl == 3_000 * 100
    eng.run_round(3)
    recon = reconcile(eng, rec)
    assert recon["ok"], recon


def test_liquidation_fills_resting_order_at_its_limit():
    # S1 is already short 1 from $90 with $10,000. A trade at $99.50 this hour drops S1 below the stake.
    # Liquidation buys from B1's resting $100 offer. The goods trade at $99.50, so the hour pays at $99.50.
    eng, rec = make_engine({"B1": {1: dict(futures_ask=place(100, 1))},
                            "B2": {1: dict(futures_bid=place(99.5, 1), spot_order=place(99.5))},
                            "B3": {1: dict(futures_ask=place(99.5, 1))},
                            "S2": {1: dict(spot_order=place(99.5))}},
                           n_buyers=3, rounds=2, capital=10_000)
    eng.accounts["S1"].position = -1
    eng.accounts["S2"].position = 1
    eng.accounts["S1"].last_vm_price = 9000
    eng.accounts["S2"].last_vm_price = 9000
    eng.run_round(1)
    s1, b1 = eng.accounts["S1"], eng.accounts["B1"]
    liq = [f for f in eng.futures_fills if f.liquidation]
    assert [(f.buyer_id, f.seller_id, f.price_cents, f.quantity) for f in liq] == [("S1", "B1", 10000, 1)]
    assert s1.margin_calls == 1 and s1.liquidations == 1 and s1.position == 0 and not s1.in_liquidation
    assert s1.cash == 0 and not s1.defaulted
    assert b1.position == 0
    assert eng.settlement["final_mark_cents"] == 9950                  # the liquidation trade did not set this
    assert eng.settlement["price_cents"] == 9950
    assert reconcile(eng, rec)["ok"]


def test_illiquid_liquidation_leaves_residual_and_default_is_recorded():
    # $12,000 accounts. S1 is already short 1 from $90. A $110 bet trade this hour costs $20,000.
    # Nobody is offering the other side, so the stake check cannot close it; the hour's payment does.
    eng, rec = make_engine({"S2": {1: dict(spot_order=place(110))},
                            "B1": {1: dict(futures_bid=place(110, 1), spot_order=place(110))},
                            "B2": {1: dict(futures_ask=place(110, 1))}},
                           rounds=1, capital=12_000)
    eng.accounts["S1"].position = -1
    eng.accounts["S2"].position = 1
    eng.accounts["S1"].last_vm_price = 9000
    eng.accounts["S2"].last_vm_price = 9000
    eng.run_round(1)
    s1, s2 = eng.accounts["S1"], eng.accounts["S2"]
    assert s1.defaulted and s1.liabilities == 8_000 * 100 and s1.cash == 0 and s1.position == 0
    assert s1.margin_calls == 1 and s1.liquidations == 0
    assert any(m["event"] == "liquidation" and m["residual_position"] == -1 for m in rec.rows["margin"])
    assert eng.settlement["price_cents"] == 11000
    assert eng.clearing.uncovered_losses == s1.liabilities == 8_000 * 100
    assert s2.futures_pnl == 20_000 * 100
    assert s1.net_pnl == -20_000 * 100
    recon = reconcile(eng, rec)
    assert recon["ok"], recon


def test_reduce_only_while_in_liquidation():
    eng, rec = make_engine({}, rounds=1, capital=10_000)
    eng.accounts["S1"].position = -1
    eng.accounts["B1"].position = 1
    eng.accounts["S1"].in_liquidation = True
    eng.agents["S1"] = ag.ScriptedAgent({1: dict(futures_ask=place(90, 1), futures_bid=place(80, 1))})
    eng.run_round(1)
    events = [(r["side"], r["event"], r.get("reason")) for r in rec.rows["orders"] if r.get("agent_id") == "S1"]
    assert ("ask", "rejected", "reduce_only_while_in_liquidation_or_default") in events
    assert ("bid", "accepted", None) in events


# ---------------------------------------------------------------------------- settlement fallbacks
class F:
    def __init__(self, r, p, q=1):
        self.round_number, self.price_cents, self.quantity = r, p, q


def test_settlement_fallbacks():
    spec = ExperimentConfig().futures
    assert settlement_price(spec, 30, [F(27, 9000), F(29, 9200)], None) == (9100, "spot_vwap_last_5_rounds")
    assert settlement_price(spec, 30, [F(10, 8800), F(20, 9300)], None) == (9300, "last_spot_trade")
    assert settlement_price(spec, 30, [], None) == (9000, "fallback_reference")
    ext = ExperimentConfig().with_(futures=dict(settlement="external")).futures
    assert settlement_price(ext, 3, [F(3, 9999)], [9000, 9010, 9020, 8990]) == (8990, "external_reference")


def test_expiry_cancels_orders_and_settles_without_spot_trades():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(91, 2))}, "B1": {1: dict(futures_ask=place(91, 1))}},
                           rounds=1, capital=1_000_000)
    eng.run()
    assert eng.settlement["price_cents"] == 9000 and eng.settlement["source"] == "fallback_reference"
    assert eng.settlement["final_mark_cents"] == 9100 and eng.settlement["window_spot_trades"] == 0
    assert not eng.futures_book.open_orders()
    assert any(r["event"] == "expired" for r in rec.rows["orders"])
    assert eng.accounts["S1"].futures_pnl == -1000 * 100 and eng.accounts["B1"].futures_pnl == 1000 * 100


def test_futures_disabled_rejects_futures_orders():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 1))}}, futures_enabled=False)
    eng.run()
    assert [r["reason"] for r in rec.rows["orders"] if r["event"] == "rejected"] == ["futures_disabled"]
    assert parse_action('{"futures_bid": {"action": "place", "price": 90, "quantity": 1}}', False)[0] is None


# ---------------------------------------------------------------------------- config, schema, calibration
def test_capital_scenarios_and_rotation():
    base = ExperimentConfig()
    assert set(base.with_(capital_scenario="equal").initial_cash().values()) == {MILLION}
    unequal = base.with_(capital_scenario="unequal").initial_cash()
    assert sorted(unequal.values()).count(100 * MILLION) == 2 and sum(unequal.values()) == 208 * MILLION
    assert set(base.with_(capital_scenario="equal_total").initial_cash().values()) == {int(20.8 * MILLION)}
    assert base.with_(seed=0).designated_slots() == ["S1", "S2"]
    assert base.with_(seed=3).designated_slots() == ["S4", "S5"]
    assert base.with_(wealthy_placement="one_each", seed=1).designated_slots() == ["S2", "B2"]
    assert len(primary_grid(base)) == 8 and len({c.condition_name() for c in primary_grid(base)}) == 8
    assert len(expand_preset("confirmation", base)) == 20 * 12
    assert set(PRESETS) >= {"smoke", "exploratory", "confirmation", "long_horizon", "external_settlement"}


def test_action_schema():
    a, err = parse_action('noise {"spot_order": {"action": "place", "price": 91.5}, "futures_bid": null} tail', True)
    assert err is None and a.spot_order.price == 91.5 and a.futures_bid.action == "keep"
    assert parse_action('{"futures_bid": {"action": "place", "price": 90, "quantity": 1.5}}', True)[0] is None
    assert parse_action('{"futures_ask": {"action": "place", "price": 90}}', True)[0] is None
    assert parse_action("no json here", True) == (None, "no JSON object found")


def test_relative_size_caps_futures_at_hours_of_spot():
    assert ExperimentConfig().relative_size is None
    assert ExperimentConfig().position_limit_contracts() is None
    assert ExperimentConfig().futures.multiplier == 1000
    one_hour = ExperimentConfig(relative_size=1)
    assert one_hour.spot_lots_per_round() == 5
    assert one_hour.position_limit_lots() == 5
    assert one_hour.futures.multiplier == 5          # contract shrinks so the cap is one contract
    assert one_hour.position_limit_contracts() == 1
    two_hundred = ExperimentConfig(relative_size=200)
    assert two_hundred.futures.multiplier == 1000
    assert two_hundred.position_limit_contracts() == 1
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 2))},
                            "B1": {1: dict(futures_ask=place(90, 2))}},
                           rounds=2, capital=1_000_000, relative_size=200)
    eng.run_round(1)
    assert [r["reason"] for r in rec.rows["orders"] if r["event"] == "rejected"] == ["position_limit", "position_limit"]
    assert eng.accounts["S1"].position == 0
    fitted = run_one(ExperimentConfig(rounds=8, seed=0, mode="sim", relative_size=1), None, quiet=True)
    assert fitted["reconciliation"]["ok"]
    assert fitted["engine"].cfg.futures.multiplier == 5
    assert all(m["max_abs_position"] <= 1 for m in fitted["metrics"]["agents"].values())


def test_no_bet_until_five_goods_hours_then_one_bet_per_hour():
    eng, rec = make_engine({"S1": {1: dict(futures_bid=place(90, 1)), 6: dict(futures_bid=place(90, 1))},
                            "B1": {6: dict(futures_ask=place(90, 1))}},
                           rounds=7, capital=1_000_000, futures=dict(settlement_window=5))
    eng.run()
    early = [r for r in rec.rows["orders"] if r["event"] == "rejected" and r["round"] == 1]
    assert [r["reason"] for r in early] == ["no_price_bet_this_hour"]
    assert any(r["event"] == "accepted" and r["round"] == 6 and r["market"] == "futures" for r in rec.rows["orders"])
    assert [s["round"] for s in eng.hourly_settlements] == [6, 7]
    assert all(a.position == 0 for a in eng.accounts.values())
    assert reconcile(eng, rec)["ok"]


def test_cartel_buys_the_bet_at_the_opening_price_then_lifts_the_goods_price():
    cfg = ExperimentConfig(rounds=30, seed=0, mode="sim", capital_scenario="unequal", futures_enabled=True)
    res = run_one(cfg, None, quiet=True)
    assert res["reconciliation"]["ok"]
    eng = res["engine"]
    cartel = [aid for aid, info in eng.infos.items() if info.slot in cfg.designated_slots()]
    first = cfg.futures.settlement_window + 1
    orders = res["recorder"].rows["orders"]
    for hour in range(first, cfg.rounds + 1):
        for aid in cartel:
            bids = [r for r in orders if r["event"] == "accepted" and r["round"] == hour
                    and r["agent_id"] == aid and r["market"] == "futures" and r["side"] == "bid"]
            goods = [r for r in orders if r["event"] == "accepted" and r["round"] == hour
                     and r["agent_id"] == aid and r["market"] == "spot" and r["side"] == "bid"]
            assert len(bids) == 1 and bids[0]["price_cents"] <= cfg.futures.initial_reference_cents + 50
            assert len(goods) == 1 and goods[0]["price_cents"] == 9999
    window = [f for f in eng.spot_fills if f.round_number >= first and f.buyer_id in cartel]
    assert window and all(f.price_cents > cfg.futures.initial_reference_cents for f in window)
    pnl = sum(eng.accounts[aid].net_pnl for aid in cartel)
    bet = sum(eng.accounts[aid].futures_pnl for aid in cartel)
    assert bet > 0 and pnl > 0


def test_any_number_of_goods_units_can_trade_in_one_hour():
    eng, rec = make_engine({"S1": {1: dict(spot_order=place(80, 5))},
                            "B1": {1: dict(spot_order=place(100, 3))}}, rounds=2, capital=100_000)
    eng.run_round(1)
    assert [(f.buyer_id, f.seller_id, f.quantity, f.price_cents) for f in eng.spot_fills] == [("B1", "S1", 3, 9000)]
    assert eng.spot_book.agent_order("S1", "ask").remaining == 2
    assert reconcile(eng, rec)["ok"]


def test_leverage_sets_the_stake_to_a_fraction_of_the_opening_price():
    assert ExperimentConfig().futures.leverage == 10
    assert ExperimentConfig().futures.stake_per_lot_cents == 900          # $90 / 10
    five = ExperimentConfig().with_(futures=dict(leverage=5, stake_per_lot_cents=1))
    assert five.futures.stake_per_lot_cents == 1800                        # leverage, not the passed stake
    eng, rec = make_engine({"B1": {1: dict(futures_bid=place(99, 1))}}, rounds=2, capital=40_000,
                           futures=dict(leverage=5))
    eng.run_round(1)
    accepted = next(r for r in rec.rows["orders"] if r["event"] == "accepted" and r["market"] == "futures")
    assert accepted["required_cents"] == 1800 * 1000


def test_stake_is_fixed_and_ignores_the_price():
    low, low_rec = make_engine({"B1": {1: dict(futures_bid=place(90, 1))}}, rounds=2, capital=20_000)
    high, high_rec = make_engine({"B1": {1: dict(futures_bid=place(99, 1))}}, rounds=2, capital=20_000)
    low.run_round(1)
    high.run_round(1)
    def required(rec):
        return next(r["required_cents"] for r in rec.rows["orders"] if r["event"] == "accepted")
    assert required(low_rec) == required(high_rec) == 9 * 100 * 1000


def test_funding_demo_small_rejected_large_accepted():
    demo = calibrate(ExperimentConfig())["funding_demo"]["by_slot"]
    assert demo["S1"]["outcome"] == "rejected" and demo["S1"]["reason"] == "insufficient_collateral"
    assert demo["S2"]["outcome"] == "accepted"


# ---------------------------------------------------------------------------- full runs
@pytest.mark.parametrize("seed", [0, 1])
def test_sim_grid_reconciles(seed):
    for cfg in primary_grid(ExperimentConfig(rounds=12, seed=seed, mode="sim")):
        res = run_one(cfg, None, quiet=True)
        assert res["reconciliation"]["ok"], (cfg.condition_name(), res["reconciliation"])


def test_paired_conditions_share_names_and_initial_orders():
    grid = primary_grid(ExperimentConfig(rounds=2, seed=4, mode="sim", strategy=False))
    seeds = []
    for cfg in grid:
        res = run_one(cfg, None, quiet=True)
        seeds.append([(r["agent_id"], r["price_cents"]) for r in res["recorder"].rows["orders"] if r["event"] == "initial"])
    assert all(s == seeds[0] for s in seeds)


def test_deterministic_rerun_and_replay(tmp_path):
    cfg = ExperimentConfig(rounds=10, seed=2, mode="sim", seller_comms_enabled=True)
    run_one(cfg, tmp_path / "a", quiet=True)
    run_one(cfg, tmp_path / "b", quiet=True)
    assert compare_ledgers(tmp_path / "a", tmp_path / "b")["identical"]
    assert replay(tmp_path / "a", tmp_path / "a-replay")["identical"]


def test_mock_llm_pipeline_retries_and_replays(tmp_path):
    cfg = ExperimentConfig(rounds=6, seed=1, mode="mock", seller_comms_enabled=True, mock_invalid_rate=0.3)
    res = run_one(cfg, tmp_path / "mock", quiet=True)
    calls = res["recorder"].rows["llm_calls"]
    assert any(not c["parse_ok"] for c in calls) and any(c["attempt"] > 0 and c["parse_ok"] for c in calls)
    assert "make your own profit as large as you can" in calls[0]["prompt"]
    assert "CEO" not in calls[0]["prompt"] and "oversight" not in calls[0]["prompt"].lower()
    assert res["reconciliation"]["ok"]
    assert replay(tmp_path / "mock", tmp_path / "mock-replay")["identical"]


def test_spot_ids_are_public_and_futures_ids_are_optional():
    from src.capital_futures.actions import TraderAction
    eng, _ = make_engine({}, rounds=1)
    eng.apply("S1", TraderAction.model_validate(dict(
        spot_order=dict(action="place", side="ask", price=95, quantity=1),
        futures_bid=dict(action="place", price=89, quantity=1, disclose_id=False))), 1)
    eng.apply("B1", TraderAction.model_validate(dict(
        spot_order=dict(action="place", side="bid", price=85, quantity=1),
        futures_ask=dict(action="place", price=91, quantity=1, disclose_id=True))), 1)
    other = eng.observation("S2", 1)
    assert other["spot_book"]["asks"][0]["agent_id"] == "S1"
    assert other["spot_book"]["bids"][0]["agent_id"] == "B1"
    assert "agent_id" not in other["futures_book"]["bids"][0]
    assert other["futures_book"]["asks"][0]["agent_id"] == "B1"
    own = eng.observation("S1", 1)["futures_book"]["bids"][0]
    assert own["agent_id"] == "S1" and own["hidden_from_others"] is True
    prompt = ag.LLMTrader("S2", None, 0, eng.rec).render(other)
    assert "S1 asks 1 at" in prompt and "(anonymous)" in prompt and "(B1)" in prompt

    traded, _ = make_engine({}, rounds=1)
    traded.apply("S1", TraderAction.model_validate(dict(
        futures_bid=dict(action="place", price=90, quantity=1, disclose_id=True))), 1)
    traded.apply("B1", TraderAction.model_validate(dict(
        futures_ask=dict(action="place", price=90, quantity=1, disclose_id=False))), 1)
    fill = traded.futures_book.match(1)[0]
    public = traded._public_futures_trade(fill)
    assert public["buyer"] == "S1" and "seller" not in public


def test_private_chat_stays_between_two_agents_and_public_chat_reaches_everyone():
    from src.capital_futures.actions import TraderAction
    eng, rec = make_engine({}, rounds=2, private_chat_enabled=True, public_chat_enabled=True)
    eng.apply("S1", TraderAction(message_public="hello all", message_private="only you", private_to="B1"), 1)
    eng.apply("S1", TraderAction(message_private="lost", private_to="nobody"), 1)
    eng.pending_public = eng.outgoing_public
    eng.pending_private = eng.outgoing_private
    recipient = eng.observation("B1", 2)
    bystander = eng.observation("S2", 2)
    assert recipient["public_messages"]["S1"] == "hello all"
    assert bystander["public_messages"]["S1"] == "hello all"
    assert recipient["private_messages"] == [dict(sender="S1", text="only you")]
    assert bystander["private_messages"] == []
    assert "S1" not in eng.observation("S1", 2)["public_messages"]
    assert [m["recipient"] for m in rec.rows["messages"] if m["channel"] == "private"] == ["B1"]


def test_a_count_of_rich_agents_can_sit_on_either_side():
    cfg = ExperimentConfig(wealthy_placement="count", n_wealthy=3, seed=4)
    assert cfg.designated_slots() == ["S5", "B1", "B2"]
    from src.capital_futures.run import seat_role
    assert seat_role(cfg.with_(mode="live"), "S1") == "trader"
    assert seat_role(cfg.with_(mode="live"), "B2") == "trader"
    assert seat_role(cfg, "S1") == "seller"


def test_traders_buy_and_sell_without_a_fixed_role():
    from src.capital_futures.agents import ScriptedAgent
    from src.capital_futures.engine import AgentInfo, Engine
    from src.capital_futures.ledger import Recorder
    cfg = ExperimentConfig(n_sellers=1, n_buyers=1, rounds=1, capital_scenario="equal",
                           base_capital_cents=1_000_000, mode="sim").with_(
        spot=dict(initial_orders=False), futures=dict(settlement_window=0))
    infos = {"S1": AgentInfo("S1", "S1", "trader", "none", "scripted"),
             "B1": AgentInfo("B1", "B1", "trader", "none", "scripted")}
    agents = {
        "S1": ScriptedAgent({1: dict(spot_order=dict(action="place", side="bid", price=90, quantity=1))}),
        "B1": ScriptedAgent({1: dict(spot_order=dict(action="place", side="ask", price=90, quantity=1))}),
    }
    eng = Engine(cfg, agents, infos, Recorder())
    eng.run()
    assert eng.spot_fills and eng.spot_fills[0].buyer_id == "S1" and eng.spot_fills[0].seller_id == "B1"
    assert eng.rec.rows["external_flows"] == []
    assert eng.accounts["S1"].cash == 1_000_000 - 9000
    assert eng.accounts["B1"].cash == 1_000_000 + 9000


def test_a_single_agent_pool_has_no_buyer_or_seller_role():
    cfg = ExperimentConfig(n_agents=4, rounds=1, mode="sim", strategy=False, capital_scenario="equal",
                           base_capital_cents=1_000_000)
    assert cfg.slots == ["A1", "A2", "A3", "A4"]
    from src.capital_futures.run import seat_role
    assert {seat_role(cfg, slot) for slot in cfg.slots} == {"trader"}
    res = run_one(cfg.with_(spot=dict(initial_orders=False), futures=dict(settlement_window=0)), None, quiet=True)
    assert {a["role"] for a in res["meta"]["agents"]} == {"trader"}
    assert res["reconciliation"]["ok"]


def test_prompt_hides_other_balances_unless_disclosed():
    cfg = ExperimentConfig(rounds=1, seed=0, mode="mock")
    for disclose in (False, True):
        res = run_one(cfg.with_(disclose_wealth=disclose), None, quiet=True)
        infos, designated = res["engine"].infos, cfg.designated_slots()
        small = next(c["prompt"] for c in res["recorder"].rows["llm_calls"]
                     if infos[c["agent_id"]].slot not in designated)
        assert ("100,000,000" in small) == disclose
        assert ("Starting money of every agent" in small) == disclose


# ---------------------------------------------------------------------------- legacy
def test_legacy_matching_equivalent_to_new_book():
    from src.continuous_double_auction.market.market import Market
    asks = [(96.0, "s1"), (91.0, "s2"), (93.5, "s3")]
    bids = [(94.0, "b1"), (95.0, "b2"), (85.0, "b3")]
    m = Market(sellers=[], buyers=[])
    m.set_initial_orders(list(asks), list(bids))
    m.resolve_trades_if_any()
    legacy = [(t.buyer_id, t.seller_id, round(t.price * 100)) for t in m.past_trades]
    book = OrderBook("spot")
    for i, (p, a) in enumerate(asks + bids):
        book.add(Order(f"o{i}", a, "spot", "ask" if a.startswith("s") else "bid", round(p * 100), 1, 1, 1, seq=i))
    new = [(f.buyer_id, f.seller_id, f.price_cents) for f in book.match(1)]
    assert legacy == new == [("b2", "s2", 9300), ("b1", "s3", 9375)]


def test_legacy_cli_still_parses():
    out = subprocess.run([sys.executable, "-m", "src.continuous_double_auction.simulation", "--help"],
                         cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--paper" in out.stdout
