"""Command-line entry point for the capital x futures extension.

    uv run python -m src.capital_futures.run run --preset smoke --mode sim
    uv run python -m src.capital_futures.run run --mode sim --capital unequal --futures on --seed 3
    uv run python -m src.capital_futures.run run --preset exploratory --mode live --yes
    uv run python -m src.capital_futures.run estimate --preset confirmation
    uv run python -m src.capital_futures.run calibrate
    uv run python -m src.capital_futures.run replay  --run-dir results/capital_futures/<run>
    uv run python -m src.capital_futures.run diagnose --run-dir results/capital_futures/<run> --replace S1
    uv run python -m src.capital_futures.run episode --run-dir results/capital_futures/<run>

Result labels written to every config.json (``result_type``):
  sim    -> rule-based simulation (ZI benchmark agents, STRATEGY policy on designated seats); no LLM
  mock   -> mocked LLM pipeline (prompt, parser, retries run; responses are synthetic)
  live   -> real LLM responses
  replay -> deterministic re-execution of recorded responses/actions
"""

from __future__ import annotations

import argparse
import json
import hashlib
from importlib.metadata import version
import shutil
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.capital_futures import agents as ag
from src.capital_futures.accounts import stake_cents
from src.capital_futures.config import PRESETS, ExperimentConfig, expand_preset, load_config, primary_grid, equal_total_grid
from src.capital_futures.engine import AgentInfo, Engine
from src.capital_futures.ledger import LEDGERS, Recorder, read_jsonl
from src.capital_futures.metrics import compute_metrics, reconcile
from src.capital_futures.rng import stream


def _version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "not-installed"

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results" / "capital_futures"

RESULT_TYPES = dict(sim="rule-based simulation (no LLM)", mock="mocked LLM pipeline (synthetic responses, no model)",
                    live="real LLM responses", replay="replay of a recorded run")

# Per-million-token USD prices used only for estimates. They are assumptions; check current pricing.
PRICE_ASSUMPTIONS = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.4, 1.6), "gpt-4o": (2.5, 10.0),
                     "claude-3.7-sonnet": (3.0, 15.0), "claude-sonnet-4": (3.0, 15.0), "default": (3.0, 15.0)}


# ---------------------------------------------------------------------------- provenance
def git_info() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    status = git("status", "--porcelain", "--", ".")
    return dict(commit=git("rev-parse", "HEAD"), dirty_src=bool(status), branch=git("rev-parse", "--abbrev-ref", "HEAD"))


def effective_model(name: str) -> str:
    """Alias expansion (cda_types) followed by the repo-level resolution used by the legacy runner."""
    from src.continuous_double_auction.cda_types import resolve_model
    resolved = resolve_model(name)
    try:
        from poseidon_compute import resolve_model as cfg_resolve
        return cfg_resolve(resolved)
    except Exception:  # noqa: BLE001 - poseidon_compute is optional outside the hackathon env
        return resolved


def source_hashes():
    paths = list((REPO / "src" / "capital_futures").rglob("*.py"))
    paths += list((REPO / "src" / "capital_futures").rglob("*.jinja2"))
    paths += [REPO / p for p in ("src/resources/model_wrappers.py", "src/continuous_double_auction/cda_types.py",
        "src/continuous_double_auction/identities.py", "src/continuous_double_auction/util/misc.py",
        "src/continuous_double_auction/util/jinja_util.py")]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


# ---------------------------------------------------------------------------- building a run
def seat_role(cfg: ExperimentConfig, slot: str, recorded: Optional[str] = None) -> str:
    """Live agents are traders: any of them may buy or sell. Rule agents keep a seller or buyer seat."""
    if cfg.mode == "replay" and recorded:
        return recorded
    if cfg.n_agents is not None or cfg.mode == "live":
        return "trader"
    return "seller" if slot.startswith("S") else "buyer"


def agent_ids(cfg: ExperimentConfig) -> dict[str, str]:
    """slot -> display name. Depends only on the seed, so paired conditions share names."""
    from src.continuous_double_auction.identities import AGENT_NAMES
    names = stream(cfg.seed, "identities").sample(AGENT_NAMES, len(cfg.slots))
    return dict(zip(cfg.slots, names))


def build_agents(cfg: ExperimentConfig, recorder: Recorder, replay_dir: Optional[Path] = None,
                 overrides: Optional[dict] = None):
    """Returns (agents, infos). ``overrides`` maps slot -> agent object (diagnostics, tests)."""
    ids = agent_ids(cfg)
    designated = set(cfg.designated_slots())
    original_infos = {}
    if replay_dir:
        original_infos = {i["slot"]: i for i in json.loads((Path(replay_dir) / "config.json").read_text())["agents"]}
    agents, infos = {}, {}
    for slot, aid in ids.items():
        recorded = original_infos.get(slot, {}).get("role")
        role = seat_role(cfg, slot, recorded)
        model = cfg.model_for(slot)
        if overrides and slot in overrides:
            agent = overrides[slot]
        elif cfg.mode == "sim":
            agent = (ag.StrategyAgent(cfg.seed, slot, capacity_fraction=cfg.strategy_capacity_fraction)
                     if cfg.strategy and slot in designated else
                     ag.ZIAgent(cfg.seed, slot, cfg.zi_futures_participation, cfg.zi_futures_capacity_fraction))
            model = "none"
        elif cfg.mode == "mock":
            agent = ag.LLMTrader(aid, ag.MockClient(cfg.seed, slot, cfg.mock_invalid_rate), cfg.max_parse_retries, recorder)
            model = ag.MockClient.model_name
        elif cfg.mode == "live":
            # #region agent log
            try:
                import json as _json, time as _time
                _pth = Path("/Users/figowu/Documents/Apart Research Sprint/collusion_hackathon/llm-agent-collusion/.venv/lib/python3.12/site-packages/_editable_impl_poseidon_compute.pth")
                _pth_target = _pth.read_text().strip() if _pth.is_file() else None
                with open("/Users/figowu/Documents/Apart Research Sprint/.cursor/debug-1665b2.log", "a") as _f:
                    _f.write(_json.dumps({"sessionId": "1665b2", "hypothesisId": "A", "location": "run.py:build_agents", "message": "before live client import", "data": {"mode": cfg.mode, "pythonpath": os.environ.get("PYTHONPATH"), "pth_target": _pth_target, "pth_exists": bool(_pth_target and Path(_pth_target).is_dir()), "new_util_exists": Path("/Users/figowu/Documents/Apart Research Sprint/collusion_hackathon/util/poseidon_compute").is_dir(), "sys_path_has_util": any("util" in p or "poseidon" in p for p in sys.path)}, "timestamp": int(_time.time() * 1000)}) + "\n")
            except Exception:
                pass
            # #endregion
            from src.continuous_double_auction.util.misc import get_client
            # #region agent log
            try:
                import json as _json, time as _time
                import poseidon_compute as _pc
                with open("/Users/figowu/Documents/Apart Research Sprint/.cursor/debug-1665b2.log", "a") as _f:
                    _f.write(_json.dumps({"sessionId": "1665b2", "runId": "post-fix", "hypothesisId": "A", "location": "run.py:build_agents", "message": "poseidon_compute imported", "data": {"file": getattr(_pc, "__file__", None)}, "timestamp": int(_time.time() * 1000)}) + "\n")
            except Exception as _e:
                try:
                    import json as _json, time as _time
                    with open("/Users/figowu/Documents/Apart Research Sprint/.cursor/debug-1665b2.log", "a") as _f:
                        _f.write(_json.dumps({"sessionId": "1665b2", "runId": "post-fix", "hypothesisId": "A", "location": "run.py:build_agents", "message": "poseidon_compute import still failed", "data": {"error": f"{type(_e).__name__}: {_e}"}, "timestamp": int(_time.time() * 1000)}) + "\n")
                except Exception:
                    pass
            # #endregion
            if cfg.inference_backend == "modal":
                model = ag.modal_model_id(model)
            client = get_client(model, cfg.temperature, max_tokens=cfg.max_output_tokens, max_retries=1)
            if hasattr(getattr(client, "client", None), "max_retries"):
                client.client.max_retries = 0
            agent = ag.LLMTrader(aid, client, cfg.max_parse_retries, recorder)
            model = client.model_name
        elif cfg.mode == "replay":
            orig = original_infos.get(slot, {})
            if orig.get("policy") == "llm":
                agent = ag.LLMTrader(aid, ag.ReplayClient(replay_dir, aid), cfg.max_parse_retries, recorder)
            else:
                agent = ag.ActionReplayAgent(replay_dir, aid)
                agent.policy = orig.get("policy", agent.policy)
            model = orig.get("model", model)
        else:
            raise ValueError(cfg.mode)
        agents[aid] = agent
        infos[aid] = AgentInfo(aid, slot, role, model, getattr(agent, "policy", "custom"))
    return agents, infos


def run_one(cfg: ExperimentConfig, out_dir: Optional[Path], replay_dir: Optional[Path] = None,
            overrides: Optional[dict] = None, quiet: bool = False, progress=None) -> dict:
    cfg = cfg.validate()
    rec = Recorder(out_dir)
    agents, infos = build_agents(cfg, rec, replay_dir, overrides)
    meta = dict(config=cfg.to_dict(), result_type=RESULT_TYPES[cfg.mode], git=git_info(),
                engine_version="capital-futures-v2",
                initial_inventory=dict(lots=0, economic_value_cents=0),
                source_hashes=source_hashes(),
                dependencies={name: _version(name) for name in ("pydantic", "jinja2", "openai", "numpy")},
                provider=os.getenv("LLM_PROVIDER", "openrouter"),
                model_override=os.getenv("LLM_MODEL"),
                python=platform.python_version(), started=datetime.now().astimezone().isoformat(timespec="seconds"),
                command=" ".join(sys.argv), replay_of=str(replay_dir) if replay_dir else None,
                agents=[dict(agent_id=i.agent_id, slot=i.slot, role=i.role, model=i.model, policy=i.policy,
                             initial_cash_cents=cfg.initial_cash()[i.slot],
                             designated=i.slot in cfg.designated_slots()) for i in infos.values()])
    rec.write_json("config.json", meta)
    start = time.time()
    engine = Engine(cfg, agents, infos, rec)
    try:
        for t in range(1, cfg.rounds + 1):
            engine.run_round(t)
            if progress:
                progress(t, cfg.rounds)
    except BaseException:
        rec.close()
        raise
    metrics = compute_metrics(engine, rec)
    recon = reconcile(engine, rec)
    metrics["runtime_s"] = round(time.time() - start, 2)
    if not recon["ok"]:
        rec.write_json("reconciliation.json", recon)
        rec.close()
        raise RuntimeError(f"accounting reconciliation failed: {recon['checks']}")
    rec.write_json("summary.json", metrics)
    rec.write_json("reconciliation.json", recon)
    rec.write_json("dashboard.json", dashboard_payload(engine, rec, metrics, recon, meta))
    if out_dir:
        for rel in meta["source_hashes"]:
            dest = Path(out_dir) / "source_snapshot" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, dest)
    rec.close()
    if not quiet:
        g = metrics["groups"]
        print(f"[{cfg.mode}] {cfg.condition_name()} seed={cfg.seed}: designated P&L "
              f"${g['designated']['net_pnl_cents'] / 100:,.0f}, others ${g['others']['net_pnl_cents'] / 100:,.0f}, "
              f"spot lots {metrics['spot']['lots']}, futures {metrics['futures']['contracts_traded']} contracts, "
              f"settlement {(engine.settlement or {}).get('source')}, reconciliation {'OK' if recon['ok'] else 'FAILED'}"
              + (f" -> {out_dir}" if out_dir else ""))
    return dict(engine=engine, recorder=rec, metrics=metrics, reconciliation=recon, meta=meta)


def dashboard_payload(engine, rec, metrics, recon, meta) -> dict:
    """Compact per-round series for the dashboard."""
    series = {}
    for r in rec.rows["accounts"]:
        s = series.setdefault(r["agent_id"], dict(equity=[], net_pnl=[], spot_pnl=[], futures_pnl=[], position=[],
                                                  reserved=[], cash=[]))
        s["equity"].append(r["equity"])
        s["net_pnl"].append(r["net_pnl"])
        s["spot_pnl"].append(r["spot_pnl"])
        s["futures_pnl"].append(r["futures_pnl"])
        s["position"].append(r["position"])
        s["reserved"].append(r["reserved_cents"])
        s["cash"].append(r["cash"])
    rounds = [{k: r[k] for k in ("round", "spot_trades", "spot_vwap_cents", "futures_volume", "liquidation_volume",
                                 "futures_vwap_cents", "mark_cents", "open_interest", "external_reference_cents",
                                 "spot_best_bid", "spot_best_ask", "seller_asks", "buyer_bids")}
              for r in rec.rows["rounds"]]
    events = [dict(round=r["round"], kind="margin", agent_id=r["agent_id"], event=r["event"]) for r in rec.rows["margin"]]
    events += [dict(round=r["round"], kind="rejected", agent_id=r["agent_id"], event=r["reason"], market=r["market"])
               for r in rec.rows["orders"] if r["event"] == "rejected"]
    notes = [dict(round=r["round"], agent_id=r["agent_id"], note=r.get("strategy_note"))
             for r in rec.rows["actions"] if r.get("strategy_note")]

    def _contract(r) -> Optional[str]:
        hour = r.get("round", r.get("round_number"))
        if r.get("market") != "futures" or hour is None:
            return None
        if not engine.cfg.futures_enabled or hour <= engine.cfg.futures.settlement_window:
            return None
        return f"H{hour}"

    trades = [dict(round=r.get("round", r.get("round_number")), market=r["market"], buyer=r["buyer_id"],
                   seller=r["seller_id"], price_cents=r["price_cents"], quantity=r["quantity"],
                   liquidation=bool(r.get("liquidation")), contract=_contract(r))
              for r in rec.rows["trades"]]
    offers = [dict(round=r["round"], market=r["market"], agent_id=r["agent_id"], side=r["side"],
                   price_cents=r["price_cents"], quantity=r.get("quantity") or 1, contract=_contract(r))
              for r in rec.rows["orders"] if r.get("event") in ("accepted", "initial")
              and r.get("market") in ("spot", "futures")]
    return dict(meta=dict(engine_version=meta["engine_version"], full_config=meta["config"], condition=metrics["condition"], seed=metrics["seed"], mode=metrics["mode"],
                          strategy=metrics["strategy"], result_type=meta["result_type"], git=meta["git"],
                          agents=meta["agents"], rounds=engine.cfg.rounds,
                          settlement_window=engine.cfg.futures.settlement_window,
                          futures_enabled=engine.cfg.futures_enabled, multiplier=engine.cfg.futures.multiplier,
                          placement=engine.cfg.wealthy_placement, settlement_rule=engine.cfg.futures.settlement,
                          capital=engine.cfg.capital_scenario, comms=engine.cfg.seller_comms_enabled,
                          buyer_comms=engine.cfg.buyer_comms_enabled,
                          private_chat=engine.cfg.private_chat_enabled, public_chat=engine.cfg.public_chat_enabled,
                          regulator=engine.cfg.regulator_enabled,
                          inference_backend=engine.cfg.inference_backend,
                          relative_size=engine.cfg.relative_size,
                          leverage=engine.cfg.futures.leverage,
                          stake_per_lot_cents=engine.cfg.futures.stake_per_lot_cents,
                          hourly_bets=True,
                          spot_lots_per_round=engine.cfg.spot_lots_per_round(),
                          position_limit_contracts=engine.cfg.position_limit_contracts()),
                series=series, rounds=rounds, settlement=engine.settlement,
                settlements=list(engine.hourly_settlements), metrics=metrics,
                reconciliation=recon, events=events[:500], messages=rec.rows["messages"][:300],
                regulator=rec.rows["regulator"][:300],
                strategy_notes=notes[:300], trades=trades, offers=offers)


# ---------------------------------------------------------------------------- commands
def base_config(args) -> ExperimentConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else ExperimentConfig()
    changes = {}
    for key, attr in (("seed", "seed"), ("rounds", "rounds"), ("n_sellers", "sellers"), ("n_buyers", "buyers"),
                      ("mode", "mode"), ("wealthy_placement", "placement"), ("capital_scenario", "capital")):
        v = getattr(args, attr, None)
        if v is not None:
            changes[key] = v
    if getattr(args, "n_wealthy", None) is not None:
        changes["n_wealthy"] = args.n_wealthy
        changes["wealthy_placement"] = "count"
    if getattr(args, "agents", None) is not None:
        changes["n_agents"] = args.agents
    if getattr(args, "futures", None) is not None:
        changes["futures_enabled"] = args.futures == "on"
    if getattr(args, "comms", None) is not None:
        changes["seller_comms_enabled"] = args.comms == "on"
    if getattr(args, "buyer_comms", None) is not None:
        changes["buyer_comms_enabled"] = args.buyer_comms == "on"
    if getattr(args, "private_chat", None) is not None:
        changes["private_chat_enabled"] = args.private_chat == "on"
    if getattr(args, "public_chat", None) is not None:
        changes["public_chat_enabled"] = args.public_chat == "on"
    if getattr(args, "regulator", None) is not None:
        changes["regulator_enabled"] = args.regulator == "on"
    if getattr(args, "backend", None):
        changes["inference_backend"] = args.backend
    if getattr(args, "strategy", None) is not None:
        changes["strategy"] = args.strategy == "on"
    if getattr(args, "seller_models", None):
        changes["seller_models"] = tuple(args.seller_models)
    if getattr(args, "buyer_models", None):
        changes["buyer_models"] = tuple(args.buyer_models)
    if getattr(args, "wealthy_capital", None):
        changes["wealthy_capital_cents"] = int(args.wealthy_capital * 100)
    if getattr(args, "base_capital", None):
        changes["base_capital_cents"] = int(args.base_capital * 100)
    if getattr(args, "disclose_wealth", False):
        changes["disclose_wealth"] = True
    fut = {}
    if getattr(args, "settlement", None):
        fut["settlement"] = args.settlement
    if getattr(args, "multiplier", None):
        fut["multiplier"] = args.multiplier
    if getattr(args, "relative_size", None) is not None:
        changes["relative_size"] = args.relative_size
    if getattr(args, "leverage", None) is not None:
        fut["leverage"] = args.leverage
    return cfg.with_(**changes, futures=fut)


def configs_for(args) -> list[ExperimentConfig]:
    base = base_config(args)
    if args.preset:
        configs = expand_preset(args.preset, base)
        if args.rounds is not None:
            configs = [c.with_(rounds=args.rounds) for c in configs]
        return configs
    if args.grid:
        seeds = args.seeds or [base.seed]
        return [c for s in seeds for c in (primary_grid(base.with_(seed=s)) +
            (equal_total_grid(base.with_(seed=s)) if args.equal_total_control else []))]
    return [base]


def estimate(configs: list[ExperimentConfig], retry_rate: float = 0.05, output_tokens: int = 450) -> dict:
    """Calls and cost for live mode, from the rendered prompt length (about 4 characters per token)."""
    total_calls, total_cost, rows = 0, 0.0, []
    for cfg in configs:
        live = cfg.with_(mode="sim")
        rec = Recorder()
        eng = Engine(live, *build_agents(live, rec), rec)
        trader = ag.LLMTrader("x", None, 0, rec)
        obs = eng.observation(next(iter(eng.accounts)), 1)
        chars_round1 = len(trader.render(obs))
        prompt_tokens = (chars_round1 + 2500) / 4    # history, trades and memory grow over the run
        calls = len(cfg.slots) * cfg.rounds * (1 + retry_rate)
        reg_calls = 0.0
        if cfg.regulator_enabled:
            channels = (int(cfg.seller_comms_enabled) + int(cfg.buyer_comms_enabled)
                        + int(cfg.public_chat_enabled) + int(cfg.private_chat_enabled))
            reg_calls = channels * cfg.rounds * (1 + retry_rate)
        cost = 0.0
        for slot in cfg.slots:
            pin, pout = PRICE_ASSUMPTIONS.get(cfg.model_for(slot), PRICE_ASSUMPTIONS["default"])
            cost += cfg.rounds * (1 + retry_rate) * (prompt_tokens * pin + output_tokens * pout) / 1e6
        if reg_calls:
            reg_name = cfg.seller_models[0] if cfg.inference_backend == "modal" else cfg.regulator_model
            pin, pout = PRICE_ASSUMPTIONS.get(reg_name, PRICE_ASSUMPTIONS["default"])
            cost += reg_calls * (prompt_tokens * pin + output_tokens * pout) / 1e6
        calls += reg_calls
        total_calls += calls
        total_cost += cost
        rows.append(dict(condition=cfg.condition_name(), seed=cfg.seed, calls=round(calls), est_usd=round(cost, 2)))
    def _budget(cfg: ExperimentConfig) -> int:
        calls = len(cfg.slots) * cfg.rounds * (1 + cfg.max_parse_retries)
        if cfg.regulator_enabled:
            channels = (int(cfg.seller_comms_enabled) + int(cfg.buyer_comms_enabled)
                        + int(cfg.public_chat_enabled) + int(cfg.private_chat_enabled))
            calls += channels * cfg.rounds * (1 + cfg.max_parse_retries)
        return calls

    return dict(runs=len(configs), calls=round(total_calls),
                maximum_response_requests=sum(_budget(c) for c in configs), est_usd=round(total_cost, 2), per_run=rows,
                assumptions=dict(chars_per_token=4, retry_rate=retry_rate, output_tokens_per_call=output_tokens,
                                 usd_per_million_tokens=PRICE_ASSUMPTIONS,
                                 note="Historical assumed prices, not current quotes or a spending cap; context grows over time."))


def calibrate(cfg: ExperimentConfig) -> dict:
    """Funding demo (same scripted order, small vs large account) and an exposure table."""
    fs = cfg.futures
    per_contract = stake_cents(1, fs.multiplier, fs.stake_per_lot_cents)
    demo_qty = int(cfg.base_capital_cents // per_contract) * 2
    c = cfg.with_(n_sellers=3, n_buyers=1, rounds=1, capital_scenario="unequal", wealthy_placement="explicit",
                  wealthy_slots=("S2", "S3"), futures_enabled=True, mode="sim", spot=dict(initial_orders=False),
                  futures=dict(settlement_window=0))
    order = {1: dict(futures_bid=dict(action="place", price=fs.initial_reference_cents / 100, quantity=demo_qty))}
    res = run_one(c, None, overrides={"S1": ag.ScriptedAgent(order), "S2": ag.ScriptedAgent(order),
                                       "B1": ag.ScriptedAgent({}), "S3": ag.ScriptedAgent({})}, quiet=True)
    orders = res["recorder"].rows["orders"]
    by_slot = {}
    for aid, info in res["engine"].infos.items():
        ev = [o for o in orders if o.get("agent_id") == aid and o["event"] in ("accepted", "rejected")]
        by_slot[info.slot] = dict(initial_cash_usd=c.initial_cash()[info.slot] / 100,
                                  outcome=ev[0]["event"] if ev else "none",
                                  reason=ev[0].get("reason") if ev else None,
                                  required_usd=(ev[0].get("required_cents") or 0) / 100 if ev else None)
    supply = cfg.rounds * min(cfg.n_sellers, cfg.n_buyers)
    window_supply = fs.settlement_window * min(cfg.n_sellers, cfg.n_buyers)
    table = []
    for cap in sorted({cfg.base_capital_cents, cfg.wealthy_capital_cents,
                       cfg.with_(capital_scenario="equal_total").initial_cash()[cfg.slots[0]]}):
        n = int(cap // per_contract)
        table.append(dict(capital_usd=cap / 100, max_bets_at_the_stake=n,
                          lots_equivalent=n * fs.multiplier,
                          multiple_of_total_spot_supply=round(n * fs.multiplier / supply, 1),
                          pnl_per_1usd_settlement_move_usd=n * fs.multiplier,
                          pnl_per_1usd_move_pct_of_capital=round(100 * n * fs.multiplier * 100 / cap, 2)))
    return dict(
        funding_demo=dict(order=f"futures bid {demo_qty} contracts @ ${fs.initial_reference_cents / 100:.2f}",
                          stake_per_bet_usd=per_contract / 100, by_slot=by_slot),
        exposure=table,
        spot=dict(max_lots_total=supply, max_lots_in_settlement_window=window_supply,
                  max_spot_cost_to_move_window_vwap_by_1usd=window_supply * 1.0,
                  spot_pnl_per_lot_range_usd=[0, (cfg.spot.buyer_value_cents - cfg.spot.seller_cost_cents) / 100]),
        note=("Derivatives exposure dwarfs spot supply under these settings: settlement depends on at most "
              f"{window_supply} spot lots, while one contract references {fs.multiplier} lots. This is a property "
              "of the specified calibration and makes spot-VWAP settlement highly manipulable; use "
              "--settlement external or a smaller --multiplier as robustness checks."),
    )


def compare_ledgers(dir_a: Path, dir_b: Path) -> dict:
    diffs = {}
    for name in LEDGERS:
        if name == "llm_calls":
            continue
        a, b = read_jsonl(dir_a / f"{name}.jsonl"), read_jsonl(dir_b / f"{name}.jsonl")
        if a != b:
            diffs[name] = dict(len_a=len(a), len_b=len(b),
                               first_diff=next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
    return dict(identical=not diffs, differences=diffs)


def replay(run_dir: Path, out_dir: Path) -> dict:
    meta = json.loads((run_dir / "config.json").read_text())
    if meta.get("source_hashes") and meta["source_hashes"] != source_hashes():
        raise ValueError("source differs from recorded run; use its source_snapshot for exact replay")
    cfg = ExperimentConfig.from_dict({**meta["config"], "mode": "replay", "replay_dir": str(run_dir)})
    run_one(cfg, out_dir, replay_dir=run_dir)
    result = compare_ledgers(run_dir, out_dir)
    (out_dir / "replay_check.json").write_text(json.dumps(result, indent=2))
    return result


def diagnose(run_dir: Path, replace: list[str], replacement: str, out_dir: Path) -> dict:
    """Deviation/replacement diagnostic. Re-runs the recorded run with the given seats replaced by a
    ZI (or STRATEGY) agent. Rule agents in the original are re-instantiated (closed loop); LLM agents
    replay their recorded actions (open loop, labeled)."""
    meta = json.loads((run_dir / "config.json").read_text())
    orig_cfg = ExperimentConfig.from_dict(meta["config"])
    orig = json.loads((run_dir / "summary.json").read_text())
    cfg = orig_cfg if orig_cfg.mode == "sim" else orig_cfg.with_(mode="replay", replay_dir=str(run_dir))
    make = {"zi": lambda s: ag.ZIAgent(cfg.seed, s, cfg.zi_futures_participation, cfg.zi_futures_capacity_fraction),
            "strategy": lambda s: ag.StrategyAgent(cfg.seed, s, capacity_fraction=cfg.strategy_capacity_fraction)}
    res = run_one(cfg, out_dir, replay_dir=None if cfg.mode == "sim" else run_dir,
                  overrides={s: make[replacement](s) for s in replace}, quiet=True)
    new = res["metrics"]
    ids = {a["slot"]: a["agent_id"] for a in meta["agents"]}
    rep = [ids[s] for s in replace]
    def pnl(m, who):
        return sum(m["agents"][a]["net_pnl_cents"] for a in who)
    others = [a for a in orig["agents"] if a not in rep]
    out = dict(replaced=dict(zip(replace, rep)), replacement=replacement,
               loop="closed (rule agents react)" if orig_cfg.mode == "sim" else "open (other agents replay recorded actions)",
               replaced_pnl_original_cents=pnl(orig, rep), replaced_pnl_counterfactual_cents=pnl(new, rep),
               others_pnl_original_cents=pnl(orig, others), others_pnl_counterfactual_cents=pnl(new, others),
               spot_price_original_cents=orig["spot"]["mean_price_cents"], spot_price_counterfactual_cents=new["spot"]["mean_price_cents"],
               settlement_original=orig["futures"]["settlement"], settlement_counterfactual=new["futures"]["settlement"])
    (out_dir / "diagnostic.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_cfg(sp):
        sp.add_argument("--config", help="JSON config file (ExperimentConfig fields)")
        sp.add_argument("--preset", choices=sorted(PRESETS))
        sp.add_argument("--grid", action="store_true", help="run the 8-condition primary grid")
        sp.add_argument("--equal-total-control", action="store_true")
        sp.add_argument("--seeds", type=int, nargs="+")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--rounds", type=int)
        sp.add_argument("--sellers", type=int)
        sp.add_argument("--buyers", type=int)
        sp.add_argument("--agents", type=int, help="one pool of traders; each may buy or sell")
        sp.add_argument("--mode", choices=["sim", "mock", "live"])
        sp.add_argument("--strategy", choices=["on", "off"])
        sp.add_argument("--capital", choices=["equal", "unequal", "equal_total"])
        sp.add_argument("--placement", choices=["two_sellers", "two_buyers", "one_each"])
        sp.add_argument("--n-wealthy", type=int, help="how many agents start rich; they may buy or sell")
        sp.add_argument("--futures", choices=["on", "off"])
        sp.add_argument("--comms", choices=["on", "off"], help="seller communication")
        sp.add_argument("--buyer-comms", choices=["on", "off"], help="buyer communication")
        sp.add_argument("--private-chat", choices=["on", "off"], help="one agent may write privately to one other")
        sp.add_argument("--public-chat", choices=["on", "off"], help="every agent may read and write one shared chat")
        sp.add_argument("--regulator", choices=["on", "off"], help="regulator that can restrict a message channel")
        sp.add_argument("--backend", choices=["api", "modal"], help="live inference: hosted API or the Modal vLLM endpoint")
        sp.add_argument("--settlement", choices=["spot_vwap", "external"])
        sp.add_argument("--multiplier", type=int, help="lots per futures contract (default 1000)")
        sp.add_argument("--relative-size", type=float,
                        help="cap each price bet at this many hours of goods trading")
        sp.add_argument("--leverage", type=float,
                        help="opening goods price divided by the stake per unit (default 10)")
        sp.add_argument("--wealthy-capital", type=float, help="USD")
        sp.add_argument("--base-capital", type=float, help="USD")
        sp.add_argument("--disclose-wealth", action="store_true")
        sp.add_argument("--seller-models", nargs="+")
        sp.add_argument("--buyer-models", nargs="+")

    r = sub.add_parser("run")
    add_cfg(r)
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    r.add_argument("--yes", action="store_true", help="confirm a live (paid) run after the estimate")
    e = sub.add_parser("estimate")
    add_cfg(e)
    c = sub.add_parser("calibrate")
    add_cfg(c)
    for name in ("replay", "diagnose", "episode"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", type=Path, required=True)
        if name == "diagnose":
            sp.add_argument("--replace", nargs="+", required=True, help="slots, e.g. S1 S2")
            sp.add_argument("--with", dest="replacement", choices=["zi", "strategy"], default="zi")
    args = p.parse_args(argv)

    if args.cmd == "calibrate":
        print(json.dumps(calibrate(base_config(args)), indent=2))
        return
    if args.cmd == "estimate":
        print(json.dumps(estimate([c.with_(mode="live") for c in configs_for(args)]), indent=2))
        return
    if args.cmd == "replay":
        out = args.run_dir.parent / (args.run_dir.name + "-replay")
        print(json.dumps(replay(args.run_dir, out), indent=2))
        return
    if args.cmd == "diagnose":
        out = args.run_dir.parent / (args.run_dir.name + f"-diag-{'-'.join(args.replace)}-{args.replacement}")
        print(json.dumps(diagnose(args.run_dir, args.replace, args.replacement, out), indent=2, default=str))
        return
    if args.cmd == "episode":
        from src.capital_futures.episode import write_episode
        print(write_episode(args.run_dir))
        return

    configs = configs_for(args)
    if any(c.mode == "live" for c in configs):
        est = estimate(configs)
        print(f"Live run: {est['runs']} run(s), about {est['calls']} LLM calls, estimated ${est['est_usd']} "
              "(price assumptions in `estimate`).")
        live = [c for c in configs if c.mode == "live"]
        modal_live = any(c.inference_backend == "modal" for c in live)
        api_live = any(c.inference_backend != "modal" for c in live)
        if modal_live and not (os.getenv("MODAL_LLM_BASE_URL", "").strip() and os.getenv("MODAL_LLM_API_KEY", "").strip()):
            sys.exit("Modal endpoint is not configured. Set MODAL_LLM_BASE_URL and MODAL_LLM_API_KEY "
                     "(see modal_infra/serve_vllm.py). No calls were made.")
        if api_live and not os.getenv("OPENROUTER_API_KEY") and os.getenv("LLM_PROVIDER", "openrouter") == "openrouter":
            sys.exit("OPENROUTER_API_KEY is not set (put it in compute.env or the environment). No calls were made.")
        if not args.yes:
            sys.exit("Re-run with --yes to confirm the paid run.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch = args.out / f"{stamp}_{args.preset or ('grid' if args.grid else 'single')}_{configs[0].mode}"
    for cfg in configs:
        run_one(cfg, batch / f"{cfg.condition_name()}-strat_{'on' if cfg.strategy else 'off'}-seed{cfg.seed}")
    print(f"batch written to {batch}")


if __name__ == "__main__":
    main()
