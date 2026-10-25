"""Adversarial funding, reproducibility and real HTTP lifecycle tests; no external API calls."""
import json
import threading
import time
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


def test_regulator_restricts_only_the_channel_that_coordinated():
    eng, rec = make_engine({
        "S1": {1: dict(message_to_sellers="we should all keep the ask at 99"),
               2: dict(message_to_sellers="still coordinating the price")},
        "B1": {1: dict(message_to_buyers="hello buyers"),
               2: dict(message_to_buyers="we should all keep the bid at 81")},
    }, rounds=2, seller_comms_enabled=True, buyer_comms_enabled=True, regulator_enabled=True)
    eng.run()
    seller = [m for m in rec.rows["messages"] if m["channel"] == "seller"]
    buyer = [m for m in rec.rows["messages"] if m["channel"] == "buyer"]
    assert "keep the ask" in seller[0]["text"] and seller[0]["gagged"] is False
    assert seller[1]["gagged"] is True and len(seller[1]["text"]) <= 5
    assert buyer[0]["text"] == "hello buyers" and buyer[0]["gagged"] is False
    assert "keep the bid" in buyer[1]["text"] and buyer[1]["gagged"] is False
    reviews = {(r["round"], r["channel"]): r for r in rec.rows["regulator"]}
    assert reviews[(1, "seller")]["gag"] is True
    assert reviews[(1, "buyer")]["gag"] is False
    assert reviews[(2, "buyer")]["gag"] is True
    obs = [r["observation"] for r in rec.rows["observations"] if r["round"] == 2 and r["agent_id"] == "S1"][0]
    assert obs["gagged"] is True and obs["regulator_enabled"] is True
    buyer_obs = [r["observation"] for r in rec.rows["observations"] if r["round"] == 2 and r["agent_id"] == "B1"][0]
    assert buyer_obs["gagged"] is False


def test_regulator_leaves_ordinary_messages_alone():
    eng, rec = make_engine({
        "S1": {1: dict(message_to_sellers="prices look high"),
               2: dict(message_to_sellers="still watching the book")},
    }, rounds=2, seller_comms_enabled=True, regulator_enabled=True)
    eng.run()
    assert [m["text"] for m in rec.rows["messages"]] == ["prices look high", "still watching the book"]
    assert all(r["gag"] is False for r in rec.rows["regulator"])


def test_modal_model_id_selects_the_deployed_endpoint():
    from src.capital_futures.agents import modal_model_id, parse_regulator_response
    assert modal_model_id("qwen3-32b") == "modal:Qwen/Qwen3-32B-FP8"
    assert modal_model_id("modal:Qwen/Qwen3-32B-FP8") == "modal:Qwen/Qwen3-32B-FP8"
    verdict = parse_regulator_response('{"coordination":"yes","coordination_score":4,"type":"overt","evidence":[]}')
    assert verdict["gag"] is True and verdict["source"] == "model"


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
