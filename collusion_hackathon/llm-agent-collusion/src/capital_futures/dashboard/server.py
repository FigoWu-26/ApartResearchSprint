"""Local experiment dashboard. All runs are background jobs with progress and durable results.

    python -m src.capital_futures.dashboard.server --port 8765

No API keys are accepted from the browser. The server binds only to loopback.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from src.capital_futures.analyze import load_runs
from src.capital_futures.config import ExperimentConfig, primary_grid, equal_total_grid
from src.capital_futures.ledger import LEDGERS, read_jsonl
from src.capital_futures.run import DEFAULT_OUT, REPO, estimate, run_one

HERE = Path(__file__).resolve().parent
JOBS: dict[str, dict] = {}
LOCK = threading.RLock()
AGG_KEYS = ("designated_net_pnl_usd", "others_net_pnl_usd", "designated_futures_pnl_usd", "designated_spot_pnl_usd",
            "other_buyers_net_pnl_usd", "mean_spot_price", "window_spot_vwap", "settlement_price", "spot_lots",
            "buyer_surplus_usd", "seller_profit_usd", "gains_from_trade_usd", "futures_contracts", "max_open_interest",
            "margin_calls", "liquidations", "defaults", "uncovered_losses_usd", "collateral_rejections",
            "ask_comovement", "flagged_messages")
ARTIFACTS = {f"{name}.jsonl" for name in LEDGERS} | {
    "config.json", "summary.json", "reconciliation.json", "dashboard.json", "episode.md"}


def load_env():
    from dotenv import load_dotenv
    for p in (REPO.parent / "compute.env", REPO / ".env"):
        load_dotenv(p, override=False)


def key_present():
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    var = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "google": "GOOGLE_API_KEY"}.get(provider, "OPENROUTER_API_KEY")
    return bool(os.getenv(var, "").strip() or os.getenv("LLM_API_KEY", "").strip())


def modal_ready():
    """True when at least one deployed vLLM server (modal_infra/serve_vllm.py) is configured. The key stays on the server."""
    from modal_infra.models import status_rows
    return any(row["ready"] for row in status_rows())


def load_trades(run_dir):
    return read_jsonl(Path(run_dir) / "trades.jsonl")


def list_runs(root):
    out = []
    for p in sorted(root.rglob("dashboard.json")):
        try:
            meta = json.loads(p.read_text())["meta"]
            out.append(dict(id=str(p.parent.relative_to(root)), **{k: meta.get(k) for k in (
                "condition", "seed", "mode", "strategy", "result_type", "placement", "settlement_rule",
                "relative_size", "engine_version", "comms", "buyer_comms", "private_chat", "public_chat",
                "regulator", "inference_backend")}))
        except (ValueError, KeyError, OSError):
            continue
    return out


def aggregate(root):
    # Repeated execution of the same seed is not a new replication. Latest recorded
    # run wins for descriptive dashboard means; formal analysis rejects duplicates.
    rows = load_runs(root)
    def cell_key(r):
        return (r["design_id"], r["capital"], r["futures"], r["comms"],
                bool(r.get("buyer_comms")), bool(r.get("private_chat")), bool(r.get("public_chat")),
                bool(r.get("regulator")))

    unique = {(r["design_id"], r["seed"], r["capital"], r["futures"], r["comms"],
               bool(r.get("buyer_comms")), bool(r.get("private_chat")), bool(r.get("public_chat")),
               bool(r.get("regulator"))): r for r in rows}
    groups = defaultdict(list)
    for r in unique.values():
        if r["reconciliation_ok"]:
            groups[cell_key(r)].append(r)
    out = []
    for (design, cap, fut, comms, buyer_comms, private_chat, public_chat, regulator), rs in sorted(groups.items(), key=str):
        first = rs[0]
        d = {k: first[k] for k in ("result_type", "strategy", "placement", "settlement_rule", "rounds")}
        d.update(design_id=design, capital=cap, futures=fut, comms=comms, buyer_comms=buyer_comms,
                 private_chat=private_chat, public_chat=public_chat,
                 regulator=regulator, n_runs=len(rs),
                 seeds=sorted(r["seed"] for r in rs), reconciliation_failures=0)
        for k in AGG_KEYS:
            xs = [r[k] for r in rs if r[k] is not None]
            d[k] = statistics.fmean(xs) if xs else None
            d[k + "_se"] = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else None
        out.append(d)
    return out


def config_from(body):
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    cfg = ExperimentConfig()
    changes = {}
    for field, key in (("mode", "mode"), ("seed", "seed"), ("rounds", "rounds"),
        ("n_sellers", "n_sellers"), ("n_buyers", "n_buyers"), ("capital_scenario", "capital"),
        ("futures_enabled", "futures"), ("seller_comms_enabled", "comms"),
        ("buyer_comms_enabled", "buyer_comms"),
        ("private_chat_enabled", "private_chat"), ("public_chat_enabled", "public_chat"),
        ("regulator_enabled", "regulator"),
        ("inference_backend", "backend"), ("strategy", "strategy"),
        ("wealthy_placement", "placement"), ("disclose_wealth", "disclose_wealth"), ("relative_size", "relative_size")):
        if key in body and body[key] is not None:
            changes[field] = body[key]
    if "n_wealthy" in body and body["n_wealthy"] is not None:
        n = body["n_wealthy"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("number of rich agents must be a positive integer")
        changes["n_wealthy"] = n
        changes["wealthy_placement"] = "count"
    if changes.get("mode", "sim") not in ("sim", "mock", "live"):
        raise ValueError("dashboard mode must be sim, mock or live")
    for field, key in (("base_capital_cents", "base_capital_usd"), ("wealthy_capital_cents", "wealthy_capital_usd")):
        if key in body:
            value = body[key]
            if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
                raise ValueError("capital must be positive dollars")
            changes[field] = round(value * 100)
    for key in ("seller_models", "buyer_models", "wealthy_slots"):
        if key in body:
            if not isinstance(body[key], list) or not all(isinstance(v, str) and v.strip() for v in body[key]):
                raise ValueError(f"{key} must be a list of strings")
            changes[key] = tuple(body[key])
    if "n_agents" in body and body["n_agents"] is not None:
        n = body["n_agents"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("number of agents must be an integer of at least 2")
        changes["n_agents"] = n
    fut = {k: body[k] for k in ("settlement", "multiplier") if k in body}
    if body.get("leverage") not in (None, ""):
        fut["leverage"] = float(body["leverage"])
    cfg = cfg.with_(**changes, futures=fut)
    if cfg.rounds > 200 or (cfg.n_agents or 0) > 20 or cfg.n_sellers > 20 or cfg.n_buyers > 20:
        raise ValueError("dashboard limit: 200 rounds and 20 agents")
    return cfg


def without_group_chats(configs):
    """The dashboard cannot open seller-only or buyer-only chats. Drop the duplicate grid cells."""
    kept, seen = [], set()
    for cfg in configs:
        cfg = cfg.with_(seller_comms_enabled=False, buyer_comms_enabled=False)
        key = (cfg.condition_name(), cfg.seed, cfg.strategy)
        if key in seen:
            continue
        seen.add(key)
        kept.append(cfg)
    return kept


def requested_configs(body):
    cfg = config_from(body)
    batch = body.get("batch", "single")
    if body.get("pair_strategy"):
        configs = [cfg.with_(strategy=s) for s in (False, True)]
    elif batch == "single":
        configs = [cfg]
    elif batch == "pair_capital":
        configs = [cfg.with_(capital_scenario=s) for s in ("equal", "unequal")]
    elif batch == "grid":
        configs = primary_grid(cfg)
    elif batch == "grid_controls":
        configs = primary_grid(cfg) + equal_total_grid(cfg)
    else:
        raise ValueError("unknown batch")
    return without_group_chats(configs)


def start_job(root, configs):
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise ValueError("a simulation is already running; wait for it to complete")
        job_id = uuid.uuid4().hex[:12]
        batch = root / "dashboard_runs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + job_id)
        JOBS[job_id] = dict(status="running", completed_runs=0, total_runs=len(configs),
            rounds_done=0, rounds=sum(c.rounds for c in configs), results=[], error=None)

    def work():
        try:
            completed_rounds = 0
            for i, cfg in enumerate(configs):
                out = batch / (cfg.condition_name() + f"-strat_{'on' if cfg.strategy else 'off'}-seed{cfg.seed}")
                def progress(t, total):
                    with LOCK: JOBS[job_id]["rounds_done"] = completed_rounds + t
                run_one(cfg, out, quiet=True, progress=progress)
                from src.capital_futures.episode import write_episode
                write_episode(out)
                completed_rounds += cfg.rounds
                with LOCK:
                    JOBS[job_id]["completed_runs"] = i + 1
                    JOBS[job_id]["results"].append(str(out.relative_to(root)))
            with LOCK: JOBS[job_id]["status"] = "done"
        except Exception as exc:
            # #region agent log
            try:
                import json as _json, time as _time, traceback as _tb
                with open("/Users/figowu/Documents/Apart Research Sprint/.cursor/debug-1665b2.log", "a") as _f:
                    _f.write(_json.dumps({"sessionId": "1665b2", "hypothesisId": "D", "location": "server.py:start_job", "message": "job failed", "data": {"error_type": type(exc).__name__, "error": str(exc), "traceback": _tb.format_exc()[-1500:]}, "timestamp": int(_time.time() * 1000)}) + "\n")
            except Exception:
                pass
            # #endregion
            with LOCK:
                JOBS[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    threading.Thread(target=work, daemon=True).start()
    return job_id


def make_handler(root):
    root = Path(root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj=None, body=None, ctype="application/json"):
            data = body if body is not None else json.dumps(obj, default=str, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def local_request(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return host in allowed and (origin is None or origin == "http://" + host)

        def log_message(self, fmt, *args): pass

        def run_path(self, rid, filename):
            p = (root / unquote(rid) / filename).resolve()
            if root not in p.parents or not p.is_file():
                raise ValueError("run artifact not found")
            return p

        def do_GET(self):
            if not self.local_request(): return self._send(403, dict(error="local same-origin requests only"))
            path = urlsplit(self.path).path
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, body=(HERE / "index.html").read_bytes(), ctype="text/html; charset=utf-8")
                if path == "/api/status":
                    from modal_infra.models import status_rows
                    return self._send(200, dict(server=True, api_key_present=key_present(),
                        modal_ready=modal_ready(), modal_models=status_rows(),
                        provider=os.getenv("LLM_PROVIDER", "openrouter"), results_root=str(root)))
                if path == "/api/runs": return self._send(200, list_runs(root))
                if path.startswith("/api/runs/"):
                    p = self.run_path(path[len("/api/runs/" ):], "dashboard.json")
                    payload = json.loads(p.read_text())
                    if not payload.get("trades"):
                        payload["trades"] = load_trades(p.parent)
                    if "offers" not in payload:
                        window = (payload.get("meta") or {}).get("settlement_window", 5)
                        hourly = bool((payload.get("meta") or {}).get("hourly_bets")) or payload.get("settlements") is not None
                        offers = []
                        for r in read_jsonl(p.parent / "orders.jsonl"):
                            if r.get("event") not in ("accepted", "initial") or r.get("market") not in ("spot", "futures"):
                                continue
                            hour = r.get("round")
                            name = None
                            if r.get("market") == "futures" and hourly and hour is not None and hour > window:
                                name = f"H{hour}"
                            offers.append(dict(round=hour, market=r["market"], agent_id=r["agent_id"], side=r["side"],
                                               price_cents=r["price_cents"], quantity=r.get("quantity") or 1, contract=name))
                        payload["offers"] = offers
                    return self._send(200, payload)
                if path.startswith("/api/artifacts/"):
                    rid, filename = path[len("/api/artifacts/"):].rsplit("/", 1)
                    if filename not in ARTIFACTS: raise ValueError("unknown artifact")
                    return self._send(200, body=self.run_path(rid, filename).read_bytes(), ctype="text/plain; charset=utf-8")
                if path == "/api/aggregate": return self._send(200, aggregate(root))
                if path.startswith("/api/jobs/"):
                    with LOCK:
                        job = JOBS.get(path.rsplit("/", 1)[-1])
                        if job: return self._send(200, dict(job))
                return self._send(404, dict(error="not found"))
            except (ValueError, OSError) as exc:
                return self._send(404, dict(error=str(exc)))

        def do_POST(self):
            if not self.local_request(): return self._send(403, dict(error="local same-origin requests only"))
            if self.path not in ("/api/run", "/api/estimate"):
                return self._send(404, dict(error="not found"))
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Content-Type must be application/json")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536: raise ValueError("request body limit is 64 KB")
                body = json.loads(self.rfile.read(size))
                configs = requested_configs(body)
                est = estimate(configs)
                if self.path == "/api/estimate": return self._send(200, est)
                if configs[0].mode == "live":
                    if not body.get("confirm"):
                        return self._send(200, dict(needs_confirm=True, estimate=est))
                    if configs[0].inference_backend == "modal":
                        from modal_infra.models import missing_modal_env
                        names = [m for cfg in configs for m in (*cfg.seller_models, *cfg.buyer_models)]
                        missing = missing_modal_env(names)
                        if missing:
                            raise ValueError("Modal endpoint is not configured for this model. Set "
                                             + ", ".join(missing)
                                             + " in compute.env (see modal_infra/serve_vllm.py). No calls made")
                    elif not key_present():
                        raise ValueError("No provider key configured on server; no calls made")
                return self._send(202, dict(job=start_job(root, configs), estimate=est))
            except (ValueError, TypeError, KeyError) as exc:
                return self._send(400, dict(error=f"{type(exc).__name__}: {exc}"))
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--root", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    load_env()
    args.root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.root))
    print(f"Dashboard: http://127.0.0.1:{args.port} (results: {args.root})", flush=True)
    server.serve_forever()


if __name__ == "__main__": main()
