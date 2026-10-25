"""Adversarial funding, reproducibility and real HTTP lifecycle tests; no external API calls."""
import json
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.capital_futures.actions import parse_action
from src.capital_futures.accounts import position_margin
from src.capital_futures.analyze import load_runs, paired_effects
from src.capital_futures.config import ExperimentConfig
from src.capital_futures.dashboard.server import make_handler, JOBS
from src.capital_futures.run import run_one
from test_capital_futures import make_engine, place


def test_stake_ignores_the_quoted_price():
    eng, _ = make_engine({}, capital=1_000_000)
    spec = eng.cfg.futures
    assert position_margin(1, 9000, spec) == 900_000
    assert position_margin(1, 10000, spec, True) == 900_000
    order = eng._new_order('S1', 'futures', 'bid', 10000, 1, 1)
    req = eng.requirement(eng.accounts['S1'], None, order, None)
    assert req['futures_margin'] == 900_000
    assert req['total'] == 900_000


def test_a_price_move_does_not_raise_the_stake():
    eng, rec = make_engine({
        'S1': {1: dict(futures_ask=place(90, 1))}, 'S2': {1: dict(futures_bid=place(90, 1))},
    }, rounds=1, capital=10_000)
    eng.run_round(1)
    a = eng.accounts['S1']
    assert a.position == 0 and a.margin_calls == 0
    assert position_margin(1, 9100, eng.cfg.futures) == position_margin(1, 9000, eng.cfg.futures)


def test_closing_permission_cannot_reverse_an_unfunded_position():
    eng, rec = make_engine({}, rounds=2, capital=10_000)
    a = eng.accounts['S1']; a.position = 1; a.in_liquidation = True
    from src.capital_futures.actions import TraderAction
    eng.apply('S1', TraderAction(futures_ask=place(90, 2)), 1)
    assert rec.rows['orders'][-1]['reason'] == 'reduce_only_while_in_liquidation_or_default'


def test_spot_gross_funding_and_no_double_counted_utility():
    eng, rec = make_engine({'B1': {1: dict(spot_order=place(90))},
                            'S1': {1: dict(spot_order=place(80))}}, capital=85, futures_enabled=False)
    eng.run()
    assert not eng.spot_fills
    assert any(r.get('reason') == 'insufficient_collateral' for r in rec.rows['orders'])
    assert eng.accounts['B1'].cash == 8500


def test_off_tick_and_nonfinite_actions_rejected():
    for value in ('Infinity', 'NaN'):
        assert parse_action('{"futures_bid":{"action":"place","price":'+value+',"quantity":1}}', True)[0] is None
    assert parse_action('{"futures_bid":{"action":"place","price":90,"quantity":true}}', True)[0] is None
    eng, rec = make_engine({'S1': {1: dict(futures_bid=place(90.01, 1))}}, futures=dict(tick_cents=5))
    eng.run()
    assert rec.rows['orders'][0]['reason'] == 'price_not_on_tick'


def test_immutable_observations_and_latency_independent_execution():
    class Delayed:
        policy = 'scripted'
        def __init__(self, inner, delay): self.inner, self.delay = inner, delay
        def act(self, obs):
            time.sleep(self.delay)
            return self.inner.act(obs)
    scripts = {'S1': {1: dict(futures_ask=place(90, 1))},
               'B1': {1: dict(futures_bid=place(90, 1))}, 'B2': {1: dict(futures_bid=place(90, 1))}}
    a, _ = make_engine(scripts, capital=1_000_000)
    b, _ = make_engine(scripts, capital=1_000_000)
    for i, aid in enumerate(b.agents): b.agents[aid] = Delayed(b.agents[aid], (4-i)*0.005)
    obs = a.observation('S1', 1)
    obs['account']['cash_cents'] = 0
    assert a.accounts['S1'].cash == 100_000_000
    a.run(); b.run()
    for name in ('trades', 'orders', 'accounts', 'observations'):
        assert a.rec.rows[name] == b.rec.rows[name]


def test_exactly_two_wealthy_and_invalid_configuration():
    for config in (dict(wealthy_placement='explicit', wealthy_slots=('S1',)),
                   dict(wealthy_placement='explicit', wealthy_slots=('S1', 'S1')), dict(n_sellers=1),
                   dict(rounds=True), dict(base_capital_cents=-1)):
        with pytest.raises(ValueError): ExperimentConfig(**config).validate()
    with pytest.raises(ValueError): ExperimentConfig().with_(futures=dict(maintenance_margin_rate=.2))


def test_analysis_separates_designs_and_rejects_duplicate_seeds(tmp_path):
    base = ExperimentConfig(rounds=2)
    run_one(base.with_(capital_scenario='equal'), tmp_path/'equal', quiet=True)
    run_one(base.with_(capital_scenario='unequal', futures=dict(multiplier=2000)), tmp_path/'other_design', quiet=True)
    assert paired_effects(load_runs(tmp_path)) == []
    run_one(base, tmp_path/'unequal', quiet=True)
    effects = paired_effects(load_runs(tmp_path))
    assert effects and all(e['ci_low'] is None and e['sign'] == 'insufficient replication' for e in effects)
    run_one(base, tmp_path/'duplicate', quiet=True)
    with pytest.raises(ValueError, match='duplicate'): paired_effects(load_runs(tmp_path))
    with pytest.raises(FileExistsError): run_one(base, tmp_path/'equal', quiet=True)


def test_dashboard_job_pair_refresh_ledgers_and_request_validation(tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def call(path, data=None, **headers):
        if data is not None: headers['Content-Type'] = 'application/json'
        req = Request(base+path, data=json.dumps(data).encode() if data is not None else None, headers=headers)
        with urlopen(req, timeout=10) as response: return json.loads(response.read())
    try:
        assert call('/api/status')['server']
        j = call('/api/run', dict(mode='mock', rounds=2, batch='pair_capital', comms=True))
        for _ in range(200):
            state = call('/api/jobs/'+j['job'])
            if state['status'] != 'running': break
            time.sleep(.02)
        assert state['status'] == 'done', state
        assert state['rounds_done'] == 4 and state['completed_runs'] == 2
        assert len(call('/api/runs')) == 2
        results = [call('/api/runs/'+rid) for rid in state['results']]
        assert all(r['reconciliation']['ok'] for r in results)
        assert {r['meta']['capital'] for r in results} == {'equal', 'unequal'}
        with urlopen(base+'/api/artifacts/'+state['results'][0]+'/orders.jsonl') as r:
            assert r.status == 200 and r.read()
        assert len(call('/api/aggregate')) == 2
        for bad in (dict(rounds=0), dict(mode='replay'), dict(futures='false'), dict(n_buyers=21)):
            with pytest.raises(HTTPError) as e: call('/api/run', bad)
            assert e.value.code == 400
        with pytest.raises(HTTPError) as e:
            call('/api/run', dict(mode='sim'), Origin='https://untrusted.example')
        assert e.value.code == 403
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_regulator_penalizes_public_coordination_and_ignores_other_channels():
    from src.capital_futures.metrics import reconcile
    eng, rec = make_engine({
        "S1": {1: dict(message_to_sellers="we should all keep the ask at 99", message_public="prices look high"),
               2: dict(message_public="we should all keep the ask at 99", spot_order=place(96)),
               3: dict(spot_order=place(96), message_public="still here")},
        "B1": {2: dict(message_private="we should all keep the bid at 81", private_to="B2",
                       message_public="hello")},
    }, rounds=3, seller_comms_enabled=True, private_chat_enabled=True, public_chat_enabled=True,
       regulator_enabled=True)
    eng.run()
    assert reconcile(eng, rec)["ok"]
    seller = [m["text"] for m in rec.rows["messages"] if m["channel"] == "seller"]
    assert seller == ["we should all keep the ask at 99"]
    assert eng.banned_until.get("S1") == 3
    assert "B1" not in eng.banned_until
    review = [r for r in rec.rows["regulator"] if r["round"] == 2][0]
    assert review["channel"] == "public"
    assert review["ban_count"] == 1 and review["warning_count"] == 1
    assert review["fines"][0]["agent_id"] == "S1" and review["fines"][0]["amount_cents"] == 10_000
    assert "n_spot_trades" in review and "private" not in review["source"]
    obs = [r["observation"] for r in rec.rows["observations"] if r["round"] == 3 and r["agent_id"] == "S1"][0]
    assert obs["regulator_enabled"] is True and obs["suspended"] is True and obs["suspended_until"] == 3
    assert "suspended from trading" in obs["regulator_announcement"]
    assert "Regulator" in obs["public_messages"]
    rejected = [r for r in rec.rows["orders"] if r.get("agent_id") == "S1" and r["event"] == "rejected"
                and r["round"] == 3]
    assert rejected and rejected[0]["reason"] == "suspended_by_regulator"
    assert eng.accounts["S1"].fees == 10_000
    assert eng.accounts["S1"].cash == eng.accounts["S1"].initial_cash - 10_000


def test_regulator_leaves_ordinary_public_messages_alone():
    eng, rec = make_engine({
        "S1": {1: dict(message_public="prices look high"),
               2: dict(message_public="still watching the book")},
    }, rounds=2, public_chat_enabled=True, regulator_enabled=True)
    eng.run()
    assert [m["text"] for m in rec.rows["messages"] if m["sender"] != "Regulator"] == [
        "prices look high", "still watching the book"]
    assert eng.ban_count == 0 and eng.warning_count == 0 and eng.fines == []
    assert all(not r["bans_applied"] and not r["fines_applied"] for r in rec.rows["regulator"])


def test_traders_are_told_a_regulator_is_supervising():
    from src.capital_futures.agents import LLMTrader
    eng, rec = make_engine({}, rounds=1, regulator_enabled=True, private_chat_enabled=True,
                           public_chat_enabled=True, seller_comms_enabled=True)
    eng.run_round(1)
    obs = [r["observation"] for r in rec.rows["observations"] if r["agent_id"] == "S1"][0]
    text = LLMTrader("S1", None, 0, rec).render(obs)
    assert obs["regulator_policy"] in text
    assert "at most two participants" in text
    assert "profit and loss" in text and "trade history" in text
    assert "After the last hour" in text
    assert "does not read private chats" not in text
    assert "does not read this channel" in text
    assert "announced in public" in text
    quiet, quiet_rec = make_engine({}, rounds=1)
    quiet.run_round(1)
    off = LLMTrader("S1", None, 0, quiet_rec).render(quiet.observation("S1", 1))
    assert "A regulator is present" not in off


def test_regulator_sees_names_on_anonymous_price_bets_and_not_private_chat():
    from src.capital_futures.agents import empty_decision
    captured = {}

    class Capture:
        policy = "regulator:capture"

        def review(self, round_num, dossier):
            captured[round_num] = dossier
            return empty_decision(source="capture")

    eng, rec = make_engine({
        "S1": {1: dict(futures_bid=dict(action="place", price=90, quantity=1, disclose_id=False),
                       message_to_sellers="we should all keep the ask at 99",
                       message_public="watching")},
        "B1": {1: dict(futures_ask=dict(action="place", price=90, quantity=1, disclose_id=False))},
    }, rounds=2, seller_comms_enabled=True, public_chat_enabled=True, regulator_enabled=True,
       capital=1_000_000)
    eng.regulator = Capture()
    eng.run()
    trade = captured[1]["futures_trades"][0]
    assert (trade["buyer"], trade["seller"]) == ("S1", "B1")
    assert captured[1]["public_messages"] == {"S1": "watching"}
    assert "seller" not in captured[1]
    later = [r["observation"] for r in rec.rows["observations"] if r["round"] == 2 and r["agent_id"] == "S2"][0]
    assert "buyer" not in later["futures_trades"][0] and "seller" not in later["futures_trades"][0]
    from src.capital_futures.agents import TEMPLATE_DIR, _usd
    from src.continuous_double_auction.util.jinja_util import render_prompt
    prompt = render_prompt(TEMPLATE_DIR, "regulator.jinja2", usd=_usd, round=1, **captured[1])
    assert "S1" in prompt and "B1" in prompt and "watching" in prompt
    assert "Profit and loss" in prompt and "Trade history" in prompt
    assert "at most two" in prompt
    assert "we should all keep the ask" not in prompt
    assert "Temporary bans so far: 0" in prompt


def test_regulator_can_ban_warn_limit_prices_suspend_change_leverage_and_fine():
    from src.capital_futures.agents import ScriptedRegulator, empty_decision
    from src.capital_futures.metrics import reconcile
    decision = empty_decision(
        source="script", coordination="yes", coordination_score=4,
        bans=[dict(agent_id="S1", hours=1, reason="suspected collusion"),
              dict(agent_id="Nobody", hours=2, reason="unknown")],
        warnings=[dict(agent_id="all", text="Do not coordinate prices.")],
        price_control=dict(floor_cents=8500, ceiling_cents=9500, hours=2),
        suspend_trading=dict(market="futures", hours=1, reason="disorderly trading"),
        leverage=5,
        fines=[dict(agent_id="S1", amount_cents=50_000, reason="collusion")],
    )
    eng, rec = make_engine({
        "S1": {1: dict(spot_order=place(99)),
               2: dict(spot_order=place(99), futures_bid=place(90, 1), futures_ask=place(91, 1))},
        "S2": {2: dict(spot_order=place(90))},
        "B1": {2: dict(futures_ask=place(90, 1))},
        "B2": {2: dict(spot_order=place(99))},
    }, rounds=2, public_chat_enabled=True, regulator_enabled=True, capital=1_000_000)
    eng.regulator = ScriptedRegulator({1: decision})
    eng.run()
    assert reconcile(eng, rec)["ok"]
    assert eng.ban_count == 1 and eng.warning_count == 1 and len(eng.fines) == 1
    assert eng.accounts["S1"].fees == 50_000
    assert eng.cfg.futures.leverage == 5 and eng.cfg.futures.stake_per_lot_cents == 1800
    assert eng.price_floor_cents == 8500 and eng.price_ceiling_cents == 9500
    assert eng.futures_suspended_until == 2 and eng.spot_suspended_until == 0
    obs = [r["observation"] for r in rec.rows["observations"] if r["round"] == 2 and r["agent_id"] == "S2"][0]
    assert obs["contract"]["leverage"] == 5 and obs["contract"]["stake_per_lot_cents"] == 1800
    assert obs["price_floor_cents"] == 8500 and obs["futures_trading_suspended"] is True
    assert "Required leverage is now 5" in obs["regulator_announcement"]
    assert "Warning to every agent" in obs["public_messages"]["Regulator"]
    reasons = {(r["agent_id"], r["market"], r["reason"]) for r in rec.rows["orders"]
               if r["round"] == 2 and r["event"] == "rejected"}
    assert ("S1", "spot", "suspended_by_regulator") in reasons
    assert ("B1", "futures", "trading_suspended") in reasons
    assert ("B2", "spot", "outside_price_control") in reasons
    assert any(r["event"] == "accepted" and r["agent_id"] == "S2" and r["market"] == "spot"
               for r in rec.rows["orders"] if r["round"] == 2)
    s1 = [r["observation"] for r in rec.rows["observations"] if r["round"] == 2 and r["agent_id"] == "S1"][0]
    text = __import__("src.capital_futures.agents", fromlist=["LLMTrader"]).LLMTrader(
        "S1", None, 0, rec).render(s1)
    assert "You are suspended from trading through hour 2" in text
    assert "at or above $85.00" in text and "at or below $95.00" in text


def test_private_inspection_is_capped_at_two_and_the_audit_counts_the_game():
    from src.capital_futures.agents import empty_decision
    from src.capital_futures.metrics import reconcile
    seen = {}

    class TwoStep:
        policy = "regulator:test"

        def review(self, round_num, dossier):
            blob = json.dumps(dossier)
            assert "secret-s2" not in blob and "private_log" not in dossier
            assert dossier["agent_pnl"] and "trade_history" in dossier and dossier["public_history"]
            seen["first"] = dossier
            return empty_decision(source="script", coordination="yes", coordination_score=4,
                                  suspects=["B1", "B2", "S1"], inspect_private=["B1", "B2", "S1"])

        def review_private(self, round_num, dossier, decision, private_messages):
            seen["private"] = private_messages
            assert [m["text"] for m in private_messages if "secret-s2" in m["text"]] == []
            return empty_decision(source="script", coordination="yes", coordination_score=4,
                                  suspects=["B1"], fines=[dict(agent_id="B1", amount_cents=100, reason="private")])

        def audit(self, dossier):
            assert any(m["text"] == "secret-s2" for m in dossier["private_log"])
            assert dossier["agent_pnl"] and dossier["trade_history"] is not None
            seen["audit"] = dossier
            return dict(kind="final_audit", source="script", suspected_agents=["B1", "S2"],
                        summary="B1 and S2 split the orders.")

    eng, rec = make_engine({
        "B1": {1: dict(message_private="secret-b1", private_to="B2", message_public="hello")},
        "B2": {1: dict(message_private="secret-b2", private_to="S1")},
        "S1": {1: dict(message_private="secret-s1", private_to="B1")},
        "S2": {1: dict(message_private="secret-s2", private_to="S1", message_public="watching")},
    }, rounds=1, public_chat_enabled=True, private_chat_enabled=True, regulator_enabled=True,
       capital=1_000_000)
    eng.regulator = TwoStep()
    eng.run()
    assert reconcile(eng, rec)["ok"]
    review = [r for r in rec.rows["regulator"] if r.get("kind") == "review"][0]
    assert review["private_inspections"] == ["B1", "B2"]
    assert eng.accounts["B1"].fees == 100
    assert "S2" not in eng.banned_until
    audit = [r for r in rec.rows["regulator"] if r.get("kind") == "final_audit"][0]
    assert audit["suspected_count"] == 2 and audit["suspected_agents"] == ["B1", "S2"]
    assert audit["summary"].startswith("The regulator suspects 2 agent(s) of colluding during the game: B1, S2.")
    assert "B1 and S2 split the orders." in audit["summary"]


def test_keyword_regulator_opens_two_private_channels_and_audits_private_phrases():
    eng, rec = make_engine({
        "S1": {1: dict(message_public="we should all keep the ask at 99",
                       message_private="note from s1", private_to="B1")},
        "S2": {1: dict(message_public="lets all fix the price",
                       message_private="note from s2", private_to="B1")},
        "B1": {1: dict(message_public="we should all hold the price",
                       message_private="note from b1", private_to="B2")},
        "B2": {1: dict(message_public="hello",
                       message_private="we should all keep the bid at 81", private_to="B1")},
    }, rounds=2, public_chat_enabled=True, private_chat_enabled=True, regulator_enabled=True)
    eng.run()
    review = [r for r in rec.rows["regulator"] if r.get("kind") == "review" and r["round"] == 1][0]
    inspected = set(review["private_inspections"])
    assert len(inspected) == 2 and inspected <= {"S1", "S2", "B1"}
    for msg in review["private_evidence"]:
        assert msg["sender"] in inspected or msg["recipient"] in inspected
    assert "B2" not in eng.banned_until
    assert eng.banned_until["S1"] == 2 and eng.banned_until["B1"] == 2
    audit = [r for r in rec.rows["regulator"] if r.get("kind") == "final_audit"][0]
    assert audit["suspected_count"] == 4
    assert set(audit["suspected_agents"]) == {"S1", "S2", "B1", "B2"}
    assert "suspects 4 agent(s)" in audit["summary"]


def test_regulator_does_not_open_private_messages_without_suspicion():
    eng, rec = make_engine({
        "S1": {1: dict(message_public="prices look high")},
        "B1": {1: dict(message_public="hello", message_private="we should all keep the bid at 81", private_to="B2")},
    }, rounds=1, public_chat_enabled=True, private_chat_enabled=True, regulator_enabled=True)
    eng.run()
    review = [r for r in rec.rows["regulator"] if r.get("kind") == "review"][0]
    assert review["private_inspections"] == [] and review["private_evidence"] == []
    assert "B1" not in eng.banned_until and eng.ban_count == 0
    audit = [r for r in rec.rows["regulator"] if r.get("kind") == "final_audit"][0]
    assert audit["suspected_agents"] == ["B1"] and audit["suspected_count"] == 1


def test_modal_model_id_selects_the_deployed_endpoint():
    from src.capital_futures.agents import modal_model_id, parse_regulator_response
    from modal_infra.models import missing_modal_env, url_env_for
    assert modal_model_id("qwen3-32b") == "modal:Qwen/Qwen3-32B-FP8"
    assert modal_model_id("kimi") == "modal:moonshotai/Kimi-VL-A3B-Instruct"
    assert modal_model_id("gemma") == "modal:google/gemma-3-27b-it"
    assert modal_model_id("modal:Qwen/Qwen3-32B-FP8") == "modal:Qwen/Qwen3-32B-FP8"
    assert url_env_for("kimi") == "MODAL_KIMI_BASE_URL"
    assert url_env_for("modal:google/gemma-3-27b-it") == "MODAL_GEMMA_BASE_URL"
    assert url_env_for("gpt-4.1") == "MODAL_LLM_BASE_URL"
    serve = Path(__file__).resolve().parents[1] / "modal_infra" / "serve_vllm.py"
    text = serve.read_text()
    for repo in ("Qwen/Qwen3-32B-FP8", "moonshotai/Kimi-VL-A3B-Instruct", "google/gemma-3-27b-it"):
        assert repo in text
    assert missing_modal_env(["kimi"], getenv=lambda name, default="": {"MODAL_LLM_API_KEY": "k"}.get(name, default)) == [
        "MODAL_KIMI_BASE_URL"]
    verdict = parse_regulator_response(
        '{"coordination":"yes","coordination_score":4,"type":"overt","evidence":[],'
        '"bans":[{"agent_id":"S1","hours":2,"reason":"collusion"}],'
        '"fines":[{"agent_id":"S1","amount_usd":50,"reason":"collusion"}],'
        '"leverage":5,"warnings":[{"agent_id":"all","text":"Stop."}]}')
    assert verdict["source"] == "model" and verdict["bans"][0]["hours"] == 2
    assert verdict["fines"][0]["amount_cents"] == 5000 and verdict["leverage"] == 5
    assert "gag" not in verdict
    score_only = parse_regulator_response('{"coordination":"no","coordination_score":1,"type":"null","evidence":[]}')
    assert score_only["bans"] == [] and score_only["fines"] == []
    opened = parse_regulator_response(
        '{"coordination":"yes","coordination_score":4,"suspects":["A","B","C"],'
        '"inspect_private":["A","B","C"]}')
    assert opened["inspect_private"] == ["A", "B"] and opened["suspects"] == ["A", "B", "C"]
    refused = parse_regulator_response(
        '{"coordination":"no","coordination_score":1,"inspect_private":["A","B"]}')
    assert refused["inspect_private"] == []


def test_modal_client_calls_the_server_for_that_model(monkeypatch):
    monkeypatch.setenv("MODAL_LLM_BASE_URL", "https://qwen.example/v1")
    monkeypatch.setenv("MODAL_KIMI_BASE_URL", "https://kimi.example/v1")
    monkeypatch.setenv("MODAL_GEMMA_BASE_URL", "https://gemma.example/v1")
    monkeypatch.setenv("MODAL_LLM_API_KEY", "test-key")
    from src.resources.model_wrappers import OpenRouterClient

    def base(client):
        return str(client.client.base_url).rstrip("/")

    kimi = OpenRouterClient("modal:moonshotai/Kimi-VL-A3B-Instruct")
    gemma = OpenRouterClient("modal:google/gemma-3-27b-it")
    qwen = OpenRouterClient("modal:Qwen/Qwen3-32B-FP8")
    assert kimi.model_name == "moonshotai/Kimi-VL-A3B-Instruct" and base(kimi) == "https://kimi.example/v1"
    assert gemma.model_name == "google/gemma-3-27b-it" and base(gemma) == "https://gemma.example/v1"
    assert "extra_body" not in kimi.additional_params and "extra_body" not in gemma.additional_params
    assert base(qwen) == "https://qwen.example/v1"
    assert qwen.additional_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_dashboard_channels_regulator_and_modal_route(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MODAL_LLM_API_KEY", raising=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = Request(base + path, data=body, headers=headers)
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read())

    try:
        assert call("/api/status")["modal_ready"] is False
        job = call("/api/run", dict(mode="mock", rounds=1, n_sellers=2, n_buyers=2, capital="equal",
                                    futures=False, comms=True, buyer_comms=True, public_chat=True,
                                    private_chat=True, regulator=True))
        state = {"status": "running"}
        for _ in range(200):
            state = call("/api/jobs/" + job["job"])
            if state["status"] != "running":
                break
            time.sleep(0.02)
        assert state["status"] == "done", state
        payload = call("/api/runs/" + state["results"][0])
        assert payload["meta"]["comms"] is False
        assert payload["meta"]["buyer_comms"] is False
        assert payload["meta"]["public_chat"] is True
        assert payload["meta"]["private_chat"] is True
        assert payload["meta"]["regulator"] is True
        assert payload["meta"]["inference_backend"] == "api"
        assert payload["regulator"]
        assert payload["regulator"][0]["channel"] in ("public", "private")
        with pytest.raises(HTTPError) as denied:
            call("/api/run", dict(mode="live", backend="modal", confirm=True, rounds=1, capital="equal"))
        assert denied.value.code == 400
        monkeypatch.setenv("MODAL_LLM_BASE_URL", "https://qwen.example/v1")
        monkeypatch.setenv("MODAL_LLM_API_KEY", "test-key")
        with pytest.raises(HTTPError) as kimi_denied:
            call("/api/run", dict(mode="live", backend="modal", confirm=True, rounds=1, capital="equal",
                                  seller_models=["kimi"], buyer_models=["kimi"]))
        assert kimi_denied.value.code == 400
        assert "MODAL_KIMI_BASE_URL" in kimi_denied.value.read().decode()
        with pytest.raises(HTTPError) as bad:
            call("/api/run", dict(mode="sim", backend="gpu", capital="equal"))
        assert bad.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_fees_are_expenses_and_collateral_is_not():
    from src.capital_futures.metrics import reconcile
    eng, rec = make_engine({'S1': {1: dict(spot_order=place(95), futures_bid=place(90, 1))},
        'B1': {1: dict(spot_order=place(95), futures_ask=place(90, 1))}}, capital=100_000,
        futures=dict(fee_per_contract_cents=250))
    eng.cfg = eng.cfg.with_(spot=dict(fee_per_lot_cents=25))
    eng.run()
    assert sum(a.fees for a in eng.accounts.values()) == 550
    assert eng.accounts['S1'].net_pnl == 1500 + 500_000 - 275
    assert eng.clearing.fees_collected == 550
    assert reconcile(eng, rec)['ok']


def test_default_advances_and_recovery_are_explicit():
    from src.capital_futures.accounts import Account, Clearing, settle_cash_shortfall
    a = Account('s', 'S1', 'seller', 10000, -500)
    c = Clearing()
    settle_cash_shortfall(a, c, 1, 'test')
    assert a.cash == 0 and a.liabilities == 500
    assert c.to_dict()['funding_cash_cents'] == -500
    a.cash = 200
    settle_cash_shortfall(a, c, 2, 'recovery')
    assert a.liabilities == c.uncovered_losses == 300 and c.recovered == 200


def test_futures_execution_tick_and_last_round_exposure():
    eng, rec = make_engine({'S1': {1: dict(futures_bid=place(90.05, 1))},
                            'B1': {1: dict(futures_ask=place(90, 1))}},
                           capital=1_000_000, futures=dict(tick_cents=5))
    eng.run()
    assert eng.futures_fills[0].price_cents % 5 == 0
    assert rec.rows['rounds'][0]['open_interest'] == 1
    assert eng.accounts['S1'].position == 0 and eng.accounts['B1'].position == 0
