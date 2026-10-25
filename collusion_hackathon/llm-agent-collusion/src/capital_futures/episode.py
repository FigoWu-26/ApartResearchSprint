"""Episode reconstruction: a round-by-round, human-readable timeline built only from the ledgers.

Messages are shown next to what the sender did before and after, so a reader can check whether a
message was followed by matching price behaviour (supplementary evidence, not proof of agreement).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.capital_futures.ledger import read_jsonl


def _usd(c):
    return "n/a" if c is None else f"${c / 100:,.2f}"


def write_episode(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    led = {name: read_jsonl(run_dir / f"{name}.jsonl") for name in
           ("rounds", "trades", "messages", "margin", "orders", "actions", "variation_margin", "positions", "accounts")}
    by_round = defaultdict(lambda: defaultdict(list))
    for name, rows in led.items():
        if name == "rounds":
            continue
        for r in rows:
            by_round[r["round"]][name].append(r)
    regulator_by_round = defaultdict(list)
    for review in read_jsonl(run_dir / "regulator.jsonl"):
        regulator_by_round[review.get("round")].append(review)
    slot = {a["agent_id"]: a["slot"] + ("*" if a["designated"] else "") for a in meta["agents"]}
    cfg = meta["config"]
    lines = [f"# Episode: {summary['condition']} seed {summary['seed']} ({meta['result_type']})", "",
             f"Commit `{meta['git'].get('commit')}`; * marks designated (large-account) seats; "
             f"settlement {summary['futures']['settlement']}.", "",
             "| Agent | Seat | Policy | Model | Start | Net P&L | Spot | Futures |", "|---|---|---|---|---|---|---|---|"]
    for a in meta["agents"]:
        m = summary["agents"][a["agent_id"]]
        lines.append(f"| {a['agent_id']} | {slot[a['agent_id']]} | {a['policy']} | {a['model']} | "
                     f"{_usd(a['initial_cash_cents'])} | {_usd(m['net_pnl_cents'])} | {_usd(m['spot_pnl_cents'])} | "
                     f"{_usd(m['futures_pnl_cents'])} |")
    for r in led["rounds"]:
        t = r["round"]
        d = by_round[t]
        lines += ["", f"## Round {t}", "",
                  f"- Spot: {r['spot_trades']} trade(s), VWAP {_usd(r['spot_vwap_cents'])}; asks "
                  + ", ".join(f"{slot.get(a, a)} {_usd(p)}" for a, p in r["seller_asks"].items())
                  + "; bids " + ", ".join(f"{slot.get(a, a)} {_usd(p)}" for a, p in r["buyer_bids"].items())]
        if cfg["futures_enabled"]:
            lines.append(f"- Futures: {r['futures_volume']} contracts, mark {_usd(r['mark_cents'])} ({r['mark_source']}), "
                         f"open interest {r['open_interest']}"
                         + (f", liquidation volume {r['liquidation_volume']}" if r["liquidation_volume"] else ""))
        for tr in d["trades"]:
            lines.append(f"  - {tr['market']}: {slot.get(tr['buyer_id'], tr['buyer_id'])} bought {tr['quantity']} from "
                         f"{slot.get(tr['seller_id'], tr['seller_id'])} @ {_usd(tr['price_cents'])}"
                         + (" (liquidation)" if tr["liquidation"] else ""))
        for o in d["orders"]:
            if o["event"] == "rejected":
                lines.append(f"  - rejected {slot.get(o['agent_id'])} {o['market']} {o['side']}: {o['reason']}")
        for o in d["orders"]:
            if o["event"] != "rejected":
                lines.append(f"  - order {o.get('order_id', '(none)')} {slot.get(o.get('agent_id'), o.get('agent_id'))}: "
                    f"{o['event']} {o.get('market', '')} {o.get('side', '')} "
                    f"qty {o.get('quantity', '')} @ {_usd(o.get('price_cents'))}")
        for position in d["positions"]:
            if position["event"] in ("before_expiry", "cash_settled"):
                lines.append(f"  - position {slot.get(position['agent_id'])}: {position['event']}, "
                    f"{position.get('position', position.get('position_before'))} contracts")
        for vm in d["variation_margin"]:
            lines.append(f"  - cash VM {slot.get(vm['agent_id'])}: {vm['kind']} {_usd(vm['cents'])}")
        for account in d["accounts"]:
            lines.append(f"  - account {slot.get(account['agent_id'])}: equity {_usd(account['equity'])}, "
                f"cumulative P&L {_usd(account['net_pnl'])}, liabilities {_usd(account['liabilities'])}")
        for m in d["margin"]:
            lines.append(f"  - margin: {slot.get(m['agent_id'])} {m['event']}"
                         + (f", residual position {m['residual_position']}" if "residual_position" in m else ""))
        for m in d["messages"]:
            gag = " [restricted]" if m.get("gagged") else ""
            who = "everyone" if m.get("channel") == "public" else (
                slot.get(m.get("recipient"), m.get("recipient")) if m.get("channel") == "private" else f"{m['channel']}s")
            lines.append(f"  - message {slot.get(m['sender'])} -> {who} (delivered round {m['delivered_round']}){gag}: "
                         f"\"{m['text'][:300]}\"")
        for review in regulator_by_round[t]:
            lines.append(f"  - regulator {review.get('channel')}: score {review.get('coordination_score')} "
                         f"({review.get('source')}), restrict next hour={bool(review.get('gag'))}")
        for a in d["actions"]:
            if a.get("strategy_note") and not a["policy"].startswith("rule"):
                lines.append(f"  - note {slot.get(a['agent_id'])}: {a['strategy_note'][:200]}")
    out = run_dir / "episode.md"
    out.write_text("\n".join(lines) + "\n")
    return out
