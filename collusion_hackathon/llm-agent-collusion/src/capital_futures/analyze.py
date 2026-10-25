"""Cross-run analysis: paired treatment effects with seed-bootstrap intervals, CSVs and plots.

    uv run python -m src.capital_futures.analyze results/capital_futures/<batch> [--out <dir>]

The run is the unit of replication. Effects are paired within seed (same names, initial orders,
valuations and random streams), e.g. capital effect = unequal - equal in the same futures x comms
cell. Intervals are percentile bootstrap over seeds; with few seeds they are wide and indicative.
Output is labeled with the result type of the runs (sim / mock / live); never mix them.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("designated_net_pnl_usd", "Designated-seat net P&L ($)"),
    ("others_net_pnl_usd", "Other agents' net P&L ($)"),
    ("designated_futures_pnl_usd", "Designated-seat futures P&L ($)"),
    ("designated_spot_pnl_usd", "Designated-seat spot P&L ($)"),
    ("other_buyers_net_pnl_usd", "Other buyers' net P&L ($)"),
    ("mean_spot_price", "Mean spot price ($)"),
    ("window_spot_vwap", "Settlement-window spot VWAP ($)"),
    ("spot_lots", "Spot lots traded"),
    ("buyer_surplus_usd", "Buyer surplus ($)"),
    ("seller_profit_usd", "Seller profit ($)"),
    ("gains_from_trade_usd", "Total gains from trade ($)"),
    ("futures_contracts", "Futures contracts traded"),
    ("max_open_interest", "Max open interest (contracts)"),
    ("settlement_price", "Settlement price ($)"),
    ("margin_calls", "Margin calls"),
    ("liquidations", "Liquidations"),
    ("defaults", "Defaulted agents"),
    ("uncovered_losses_usd", "Residual uncovered losses ($)"),
    ("collateral_rejections", "Insufficient-collateral rejections"),
    ("ask_dispersion", "Mean seller ask dispersion ($)"),
    ("ask_comovement", "Seller ask co-movement share"),
    ("flagged_messages", "Messages with coordination terms"),
]


def design_id(meta):
    # Pair only the randomized factors; preserve every other configured rule and model.
    c = dict(meta["config"])
    for key in ("seed", "capital_scenario", "futures_enabled", "seller_comms_enabled", "tag",
                "designated_slots", "initial_cash_cents", "condition", "replay_dir"):
        c.pop(key, None)
    models = [(a["slot"], a["model"]) for a in meta["agents"]]
    return hashlib.sha256(json.dumps([c, models, meta.get("source_hashes"),
        meta.get("engine_version", "unversioned")], sort_keys=True).encode()).hexdigest()[:16]


def load_runs(root: Path) -> list[dict]:
    rows = []
    for summary in sorted(Path(root).rglob("summary.json")):
        d = summary.parent
        if d.name.endswith("-replay") or "-diag-" in d.name:
            continue
        s = json.loads(summary.read_text())
        if not (d / "config.json").exists() or not (d / "reconciliation.json").exists():
            continue
        meta = json.loads((d / "config.json").read_text())
        if "config" not in meta or "groups" not in s:
            continue
        c = meta["config"]
        g, w, sp, fu, st, co = (s["groups"], s["welfare"], s["spot"], s["futures"], s["stability"],
                                s["coordination_evidence"])
        usd = lambda x: None if x is None else x / 100  # noqa: E731
        rows.append(dict(
            run=str(d), design_id=design_id(meta), result_type=meta["result_type"], mode=c["mode"], seed=c["seed"], strategy=c["strategy"],
            capital=c["capital_scenario"], futures=c["futures_enabled"], comms=c["seller_comms_enabled"],
            buyer_comms=bool(c.get("buyer_comms_enabled", False)),
            private_chat=bool(c.get("private_chat_enabled", False)),
            public_chat=bool(c.get("public_chat_enabled", False)),
            regulator=bool(c.get("regulator_enabled", False)),
            inference_backend=c.get("inference_backend", "api"),
            placement=c["wealthy_placement"], settlement_rule=c["futures"]["settlement"], rounds=c["rounds"],
            condition=s["condition"],
            designated_net_pnl_usd=usd(g["designated"]["net_pnl_cents"]),
            designated_return=g["designated"]["return_on_initial_equity"],
            designated_futures_pnl_usd=usd(g["designated"]["futures_pnl_cents"]),
            designated_spot_pnl_usd=usd(g["designated"]["spot_pnl_cents"]),
            others_net_pnl_usd=usd(g["others"]["net_pnl_cents"]),
            others_return=g["others"]["return_on_initial_equity"],
            other_buyers_net_pnl_usd=usd(g["other_buyers"]["net_pnl_cents"]),
            mean_spot_price=usd(sp["mean_price_cents"]), window_spot_vwap=usd(sp["vwap_settlement_window_cents"]),
            spot_lots=sp["lots"], buyer_surplus_usd=usd(w["buyer_surplus_cents"]),
            seller_profit_usd=usd(w["seller_profit_cents"]), gains_from_trade_usd=usd(w["total_gains_from_trade_cents"]),
            futures_contracts=fu["contracts_traded"], max_open_interest=fu["max_open_interest"],
            settlement_price=usd((fu["settlement"] or {}).get("price_cents")),
            settlement_source=(fu["settlement"] or {}).get("source"),
            margin_calls=st["margin_calls"], liquidations=st["liquidations"], defaults=len(st["defaulted_agents"]),
            uncovered_losses_usd=usd(st["residual_uncovered_losses_cents"]),
            collateral_rejections=st["insufficient_collateral_rejections"],
            ask_dispersion=usd(co["mean_seller_ask_dispersion_cents"]), ask_comovement=co["mean_ask_comovement_share"],
            flagged_messages=co["messages_with_coordination_terms"],
            reconciliation_ok=json.loads((d / "reconciliation.json").read_text())["ok"],
        ))
    return rows


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float | None, float | None]:
    if len(xs) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


CONTRASTS = {
    # name: (factor, treated value, control value, fixed factors)
    "capital (unequal - equal)": ("capital", "unequal", "equal", ("futures", "comms")),
    "capital (unequal - equal_total)": ("capital", "unequal", "equal_total", ("futures", "comms")),
    "futures (on - off)": ("futures", True, False, ("capital", "comms")),
    "seller comms (on - off)": ("comms", True, False, ("capital", "futures")),
}


def paired_effects(rows: list[dict]) -> list[dict]:
    key = lambda r: (r["mode"], r["strategy"], r["placement"], r["settlement_rule"], r["rounds"], r["design_id"])  # noqa: E731
    index = {(key(r), r["seed"], r["capital"], r["futures"], r["comms"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate condition/seed in the same design; choose one batch, do not count reruns as independent")
    if any(not r["reconciliation_ok"] for r in rows):
        raise ValueError("cannot analyze runs with failed reconciliation")
    out = []
    for name, (factor, treated, control, fixed) in CONTRASTS.items():
        cells = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r[factor] != treated:
                continue
            ctrl_key = dict(capital=r["capital"], futures=r["futures"], comms=r["comms"])
            ctrl_key[factor] = control
            c = index.get((key(r), r["seed"], ctrl_key["capital"], ctrl_key["futures"], ctrl_key["comms"]))
            if c is None:
                continue
            cell = (key(r),) + tuple((f, r[f]) for f in fixed)
            for m, _ in METRICS:
                if r[m] is not None and c[m] is not None:
                    cells[cell][m].append(r[m] - c[m])
        for cell, metrics in cells.items():
            (mode, strategy, placement, rule, rounds, design), *fixed_vals = cell
            for m, label in METRICS:
                xs = metrics.get(m, [])
                if not xs:
                    continue
                lo, hi = bootstrap_ci(xs)
                out.append(dict(contrast=name, mode=mode, strategy=strategy, placement=placement,
                                settlement_rule=rule, rounds=rounds, design_id=design,
                                cell=", ".join(f"{f}={v}" for f, v in fixed_vals), metric=m, label=label,
                                n_seeds=len(xs), mean_diff=statistics.fmean(xs), ci_low=lo, ci_high=hi,
                                sign=("insufficient replication" if lo is None else "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive; CI spans zero")))
    # interaction: capital effect with futures on minus with futures off (same comms)
    by = defaultdict(dict)
    for r in rows:
        by[(key(r), r["seed"], r["comms"])][(r["capital"], r["futures"])] = r
    inter = defaultdict(lambda: defaultdict(list))
    for (k, seed, comms), cells in by.items():
        need = [("unequal", True), ("equal", True), ("unequal", False), ("equal", False)]
        if not all(n in cells for n in need):
            continue
        u1, e1, u0, e0 = (cells[n] for n in need)
        for m, _ in METRICS:
            vals = [u1[m], e1[m], u0[m], e0[m]]
            if None not in vals:
                inter[(k, comms)][m].append((u1[m] - e1[m]) - (u0[m] - e0[m]))
    for ((mode, strategy, placement, rule, rounds, design), comms), metrics in inter.items():
        for m, label in METRICS:
            xs = metrics.get(m, [])
            if xs:
                lo, hi = bootstrap_ci(xs)
                out.append(dict(contrast="capital x futures interaction", mode=mode, strategy=strategy,
                                placement=placement, settlement_rule=rule, rounds=rounds, design_id=design, cell=f"comms={comms}",
                                metric=m, label=label, n_seeds=len(xs), mean_diff=statistics.fmean(xs),
                                ci_low=lo, ci_high=hi, sign=("insufficient replication" if lo is None else "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive; CI spans zero")))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plots(rows: list[dict], effects: list[dict], out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    label = sorted({r["result_type"] for r in rows})
    conds = sorted({(r["capital"], r["futures"], r["comms"]) for r in rows})
    for strat in sorted({r["strategy"] for r in rows}):
        sub = [r for r in rows if r["strategy"] == strat]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        xs = range(len(conds))
        for off, (m, name, color) in enumerate((("designated_net_pnl_usd", "designated seats", "#b3261e"),
                                                ("others_net_pnl_usd", "other agents", "#1f5fa8"))):
            means, errs = [], []
            for c in conds:
                v = [r[m] for r in sub if (r["capital"], r["futures"], r["comms"]) == c]
                means.append(statistics.fmean(v) if v else 0)
                errs.append(statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0)
            ax.bar([x + (off - 0.5) * 0.38 for x in xs], means, 0.38, yerr=errs, label=name, color=color, alpha=0.85)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{c[0]}\nfut {'on' if c[1] else 'off'}\ncomms {'on' if c[2] else 'off'}" for c in conds],
                           fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("net P&L per run ($), mean ± s.e.")
        ax.set_xlabel("Capital / futures / seller communication condition")
        ax.set_title(f"Group P&L by condition (strategy={'on' if strat else 'off'}; {', '.join(label)})", fontsize=10)
        ax.legend()
        fig.tight_layout()
        p = out / f"group_pnl_strategy_{'on' if strat else 'off'}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    for m in ("designated_net_pnl_usd", "others_net_pnl_usd", "mean_spot_price", "gains_from_trade_usd"):
        sel = [e for e in effects if e["metric"] == m and e["ci_low"] is not None]
        if not sel:
            continue
        fig, ax = plt.subplots(figsize=(10, max(3, 0.32 * len(sel) + 1)))
        names = [f"{e['contrast']} | {e['cell']} | strat={'on' if e['strategy'] else 'off'}" for e in sel]
        ys = range(len(sel))
        ax.errorbar([e["mean_diff"] for e in sel], list(ys),
                    xerr=[[e["mean_diff"] - e["ci_low"] for e in sel], [e["ci_high"] - e["mean_diff"] for e in sel]],
                    fmt="o", color="#333", ecolor="#999", capsize=3)
        ax.axvline(0, color="#b3261e", lw=0.8)
        ax.set_xlabel(sel[0]["label"] + " — treatment minus control")
        ax.set_ylabel("Paired contrast")
        ax.set_yticks(list(ys))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"Paired effect on {sel[0]['label']} (mean, 95% seed-bootstrap CI; {', '.join(label)})", fontsize=9)
        fig.tight_layout()
        p = out / f"effects_{m}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.root)
    if not rows:
        raise SystemExit(f"no runs under {args.root}")
    types = sorted({r["result_type"] for r in rows})
    if len(types) > 1:
        print(f"WARNING: mixing result types {types}; effects are still paired within mode.")
    effects = paired_effects(rows)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "effects.csv", effects)
    figs = []
    for design in sorted({r["design_id"] for r in rows}):
        folder = out / design
        folder.mkdir(exist_ok=True)
        figs += plots([r for r in rows if r["design_id"] == design],
                      [e for e in effects if e["design_id"] == design], folder)
    bad = [r["run"] for r in rows if not r["reconciliation_ok"]]
    lines = [f"# Analysis of {len(rows)} runs ({', '.join(types)})", "",
             f"Reconciliation failures: {len(bad)}", "",
             "Paired effects (mean difference, 95% seed-bootstrap CI). Intervals require at least two independent seeds. Spanning zero is inconclusive, not evidence of equivalence. No multiplicity correction; confirmation endpoints must be prespecified.", "",
             "| contrast | cell | strategy | metric | n | mean | CI | sign |", "|---|---|---|---|---|---|---|---|"]
    key_metrics = {"designated_net_pnl_usd", "others_net_pnl_usd", "mean_spot_price", "window_spot_vwap",
                   "gains_from_trade_usd", "margin_calls", "defaults", "ask_comovement"}
    for e in effects:
        if e["metric"] in key_metrics and e["ci_low"] is not None:
            lines.append(f"| {e['contrast']} | {e['cell']} | {'on' if e['strategy'] else 'off'} | {e['label']} | "
                         f"{e['n_seeds']} | {e['mean_diff']:,.2f} | [{e['ci_low']:,.2f}, {e['ci_high']:,.2f}] | {e['sign']} |")
    lines += ["", "Figures: " + ", ".join(f.name for f in figs)]
    (out / "effects.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} runs, {len(effects)} effect rows -> {out}")


if __name__ == "__main__":
    main()
