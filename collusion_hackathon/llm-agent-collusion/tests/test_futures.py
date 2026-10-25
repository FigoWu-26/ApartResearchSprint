"""Engine tests for the cash-settled futures book, and the spot path it must not change. No model calls."""

import random

from src.continuous_double_auction.cda_types import ExperimentParams
from src.continuous_double_auction.market.agents import Agent
from src.continuous_double_auction.market.futures import Account, FuturesEngine, FuturesSpec
from src.continuous_double_auction.market.market import Market
from src.continuous_double_auction.util.jinja_util import render_prompt

TEMPLATE_DIR = "src/continuous_double_auction/prompt_templates/"


class Spot:
    def __init__(self, seller_id, buyer_id, price):
        self.seller_id, self.buyer_id, self.price = seller_id, buyer_id, price


def accounts(cash=1_000_000_00, n=2):
    return {
        "S1": Account("S1", "seller", 8000, cash),
        "B1": Account("B1", "buyer", 10000, cash),
        **({} if n < 3 else {"S2": Account("S2", "seller", 8000, cash), "B2": Account("B2", "buyer", 10000, cash)}),
    }


def engine(accts, rounds=30, seed=0, **spec):
    return FuturesEngine(FuturesSpec(**spec), accts, rounds, random.Random(seed))


def test_seeded_quotes_do_not_cross_and_set_the_mark_at_their_midpoint():
    e = engine(accounts())
    e.begin_round(1)
    ask, bid = e.best_ask(), e.best_bid()
    assert ask.agent_id == "S1" and 96.25 <= ask.price_cents / 100 <= 98.75
    assert bid.agent_id == "B1" and 81.25 <= bid.price_cents / 100 <= 83.75
    e.end_of_round(1, [], None, None)
    assert not e.fills and e.mark_cents == (ask.price_cents + bid.price_cents + 1) // 2


def test_partial_fill_and_midpoint_price():
    e = engine(accounts(), seed_quantity=0)
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 92, "quantity": 3})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 5})
    e.end_of_round(1, [], None, None)
    fill = e.fills[0]
    assert fill.quantity == 3 and fill.price_cents == 9100
    assert e.accounts["S1"].position == -3 and e.orders["S1"].quantity == 2


def test_margin_rejects_what_capital_cannot_support_and_allows_reducing():
    e = engine(accounts(cash=90_000_00), seed_quantity=0)  # 10% of $90 x 1000 = $9,000 per contract -> 10 max
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 11})
    assert e.rejections["B1"][0]["reason"] == "insufficient_margin"
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 10})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 10})
    e.end_of_round(1, [], None, None)
    assert e.accounts["B1"].position == 10
    e.begin_round(2)
    e.submit(2, "B1", {"action": "sell", "price": 90, "quantity": 4})  # reducing is always allowed
    e.submit(2, "S1", {"action": "buy", "price": 90, "quantity": 4})
    e.end_of_round(2, [], None, None)
    assert e.accounts["B1"].position == 6


def test_invalid_orders_are_logged_and_do_not_change_the_book():
    e = engine(accounts(), seed_quantity=0)
    e.begin_round(1)
    for raw in ("nope", {"action": "buy", "price": "x", "quantity": 1},
                {"action": "buy", "price": 90, "quantity": 1.5}, {"action": "buy", "price": 0, "quantity": 1}):
        e.submit(1, "B1", raw)
    assert e.n_invalid == 4 and not e.orders


def test_mark_to_market_is_zero_sum_and_shorting_pays_when_the_mark_falls():
    e = engine(accounts(), seed_quantity=0)
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 2})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 2})
    e.end_of_round(1, [], None, None)
    e.begin_round(2)
    e.submit(2, "B1", {"action": "sell", "price": 80, "quantity": 1})
    e.submit(2, "S1", {"action": "buy", "price": 80, "quantity": 1})
    e.end_of_round(2, [], None, None)
    long, short = e.accounts["B1"], e.accounts["S1"]
    assert long.position == 1 and short.position == -1
    # both contracts are marked from $90 to $80; the one that was closed is also realized
    assert long.futures_pnl_cents == -short.futures_pnl_cents == -2 * 1000 * 1000
    assert long.realized_futures_cents == -1 * 1000 * 1000


def test_liquidation_closes_the_position_pro_rata_against_the_other_side():
    e = engine(accounts(n=3), seed_quantity=0)
    e.mark_cents = 8400
    e.accounts["B1"].position, e.accounts["B2"].position = 5, 5
    e.accounts["S1"].position, e.accounts["S2"].position = -6, -4
    e._liquidate(1, "B1")
    assert e.accounts["B1"].position == 0 and e.accounts["B2"].position == 5
    assert e.accounts["S1"].position == -3 and e.accounts["S2"].position == -2  # 5 of 10, split 6:4
    assert e.open_interest() == 5 and sum(a.position for a in e.accounts.values()) == 0


def test_a_mark_drop_below_maintenance_triggers_liquidation():
    e = engine(accounts(n=3, cash=30_000_00), seed_quantity=0)
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 2})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 2})
    e.end_of_round(1, [], None, None)
    e.begin_round(2)
    # B2/S2 print $75, so B1's 2 contracts lose $30,000 and equity falls to 0
    e.submit(2, "B2", {"action": "buy", "price": 75, "quantity": 1})
    e.submit(2, "S2", {"action": "sell", "price": 75, "quantity": 1})
    e.end_of_round(2, [], None, None)
    assert "B1" in e.liquidated_last_round and e.accounts["B1"].position == 0
    assert e.accounts["B1"].cash_cents == 0
    assert sum(a.futures_pnl_cents for a in e.accounts.values()) == 0


def test_settlement_uses_the_window_vwap_then_the_fallbacks():
    e = engine(accounts(), seed_quantity=0, expiries=(2,), settlement_window=2)
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 1})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 1})
    e.end_of_round(1, [Spot("S1", "B1", 88)], 80, 100)
    e.end_of_round(2, [Spot("S1", "B1", 92)], 80, 100)
    s = e.settlements[0]
    assert s["source"] == "spot_vwap" and s["price"] == 90.0 and s["window_spot_trades"] == 2
    assert e.accounts["B1"].position == 0 and not e.orders and not e.active

    e = engine(accounts(), seed_quantity=0, expiries=(3,), settlement_window=2)
    e.begin_round(1)
    e.end_of_round(1, [Spot("S1", "B1", 91.5)], None, None)  # before the window
    e.end_of_round(2, [], None, None)
    e.end_of_round(3, [], None, None)
    assert e.settlements[0]["source"] == "last_spot_trade" and e.settlements[0]["price"] == 91.5

    e = engine(accounts(), seed_quantity=0, expiries=(1,))
    e.begin_round(1)
    e.end_of_round(1, [], 81.25, 98.75)
    assert e.settlements[0]["source"] == "spot_quote_midpoint" and e.settlements[0]["price"] == 90.0


def test_contracts_roll_and_quotes_are_reseeded():
    e = engine(accounts(), rounds=20, expiries=(10, 20))
    for t in range(1, 12):
        e.begin_round(t)
        e.end_of_round(t, [Spot("S1", "B1", 90)], None, None)
    assert [s["round"] for s in e.settlements] == [10]
    assert e.contract == 2 and e.expiry == 20 and e.orders  # reseeded at round 11


def test_a_loss_larger_than_equity_becomes_bad_debt_and_cash_never_goes_negative():
    e = engine(accounts(cash=20_000_00), seed_quantity=0)
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 2})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 2})
    e.end_of_round(1, [], None, None)
    e.begin_round(2)
    e.submit(2, "B1", {"action": "sell", "price": 50, "quantity": 1})
    e.submit(2, "S1", {"action": "buy", "price": 50, "quantity": 1})
    e.end_of_round(2, [], None, None)  # 2 contracts x 1000 x -$40 = -$80,000 on $20,000
    assert e.accounts["B1"].cash_cents == 0 and e.accounts["B1"].bad_debt_cents == 60_000_00
    assert e.clearing_loss_cents == 60_000_00
    assert all(a.cash_cents >= 0 for a in e.accounts.values())


def test_spot_profit_and_futures_pnl_are_reported_separately():
    e = engine(accounts(), seed_quantity=0, expiries=(1,))
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 1})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 1})
    e.end_of_round(1, [Spot("S1", "B1", 95)], None, None)
    seller, buyer = e.summary()["accounts"]["S1"], e.summary()["accounts"]["B1"]
    assert seller["spot_pnl"] == 15.0 and buyer["spot_pnl"] == 5.0
    assert seller["futures_pnl"] == -5000.0 and buyer["futures_pnl"] == 5000.0  # the seller was short
    assert seller["net_pnl"] == -4985.0
    assert e.summary()["max_incentive_ratio"] == 1000  # 1 contract x 1000 lots / 1 window trade


def test_a_mark_drop_cancels_a_resting_order_the_account_can_no_longer_margin():
    e = engine(accounts(n=3, cash=100_000_00), seed_quantity=0)
    e.begin_round(1)
    e.submit(1, "B1", {"action": "buy", "price": 90, "quantity": 5})
    e.submit(1, "S1", {"action": "sell", "price": 90, "quantity": 5})
    e.end_of_round(1, [], None, None)
    e.begin_round(2)
    e.submit(2, "B1", {"action": "buy", "price": 75, "quantity": 6})  # 11 contracts fits the $90 mark
    e.submit(2, "B2", {"action": "buy", "price": 80, "quantity": 1})
    e.submit(2, "S2", {"action": "sell", "price": 80, "quantity": 1})
    e.end_of_round(2, [], None, None)  # mark $80; B1 loses $50,000 and keeps the resting bid
    assert e.orders["B1"].quantity == 6 and e.accounts["B1"].cash_cents == 50_000_00
    e.begin_round(3)
    e.end_of_round(3, [], None, None)  # rechecked at the $80 mark: 11 contracts need $88,000
    assert "B1" not in e.orders
    assert e.rejections["B1"][0]["reason"] == "resting_order_canceled_insufficient_margin"


class Scripted(Agent):
    steps: list[dict]
    n: int = 0

    def generate_bid_response(self, **kwargs):
        step = self.steps[self.n]
        self.n += 1
        return step


def test_the_spot_market_is_unchanged_when_futures_are_off():
    params = ExperimentParams(seller_valuations=[80], buyer_valuations=[100], seller_models=["gpt-4.1"],
                              buyer_models=["gpt-4.1"], rounds=1)
    seller = Scripted(id="Seller Ada", company="A", valuation=80, expt_params=params, steps=[{"ask": 91}])
    buyer = Scripted(id="Buyer Bo", company="B", valuation=100, expt_params=params, steps=[{"bid": 95}])
    market = Market(sellers=[seller], buyers=[buyer])
    market.run_round()
    assert market.past_trades[0].price == 93.0 and market.futures is None


def test_a_scripted_round_clears_spot_and_futures_together():
    params = ExperimentParams(seller_valuations=[80], buyer_valuations=[100], seller_models=["gpt-4.1"],
                              buyer_models=["gpt-4.1"], rounds=1, futures_enabled=True, seller_comms_enabled=True)
    seller = Scripted(id="Seller Ada", company="A", valuation=80, expt_params=params, steps=[{
        "ask": 96, "message_to_sellers": "x" * 20,
        "futures_order": {"action": "sell", "price": 90, "quantity": 1}}])
    buyer = Scripted(id="Buyer Bo", company="B", valuation=100, expt_params=params, steps=[{
        "bid": 100, "futures_order": {"action": "buy", "price": 90, "quantity": 1}}])
    market = Market(sellers=[seller], buyers=[buyer], futures=engine(
        {"Seller Ada": Account("Seller Ada", "seller", 8000, 1_000_000_00),
         "Buyer Bo": Account("Buyer Bo", "buyer", 10000, 1_000_000_00)},
        rounds=1, seed_quantity=0))
    market.run_round()
    assert market.past_trades[0].price == 98.0
    seller = market.futures.accounts["Seller Ada"]
    assert seller.spot_pnl_cents == 1800  # sold the lot at $98 against an $80 cost
    # one round means the contract expires this hour, so the short settles at the $98 spot trade
    assert seller.position == 0 and seller.futures_pnl_cents == -800_000
    assert market.futures.settlements[0]["price"] == 98.0
    assert market.rounds[0].seller_messages["Seller Ada"] == "x" * 20  # no cap unless one is set


def test_prompts_say_maximize_pnl_and_mention_futures_only_when_it_is_on():
    common = dict(template_dir=TEMPLATE_DIR, seller_id="Seller Ada", buyer_id="Buyer Bo", company="C",
                  valuation=80, round_number=1, num_rounds=30, hide_num_rounds=False, memory="",
                  seller_comms_enabled=True, buyer_comms_enabled=False, seller_messages={}, buyer_messages={},
                  bid_queue=[], ask_queue=[], past_bids_and_asks="", past_trades="", agent_successful_trades="",
                  scratch_pad="", initial_condition=True, max_message_length=None, futures=None,
                  is_gagged=False, boss_pressure=False, oversight_enabled=False, demonym=None)
    seller = render_prompt(prompt_template="seller_prompt_base.jinja2", **common)
    buyer = render_prompt(prompt_template="buyer_prompt_base.jinja2", **{k: v for k, v in common.items() if k != "seller_id"})
    for prompt in (seller, buyer):
        assert "maximize PnL" in prompt and "secure maximum profitability" not in prompt
        assert "maximum length" not in prompt and "Futures Market" not in prompt
    futures = engine(accounts(), seed_quantity=0)
    futures.begin_round(1)
    with_futures = render_prompt(prompt_template="seller_prompt_base.jinja2", **{**common, "futures": futures.observation("S1", 1)})
    assert "Futures Market" in with_futures and "futures_order" in with_futures and "1000" in with_futures
