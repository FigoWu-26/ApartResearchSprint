"""
Checkpoint branching for sprint project 1.2: "Is formed collusion robust to
replacing some participants?"

Take a finished run (from run_game.py), find the round where the market crossed
a high-price threshold, rebuild the exact market state at that point, and
continue the game under a modified condition:

    # branch at the first round where the 20-round mean gain exceeds 0.5, no replacement (control)
    python branch_game.py results/jsonl/<run>.jsonl --rounds 50

    # replace agent 1 with another model in the same family / another provider
    python branch_game.py results/jsonl/<run>.jsonl --rounds 50 --replace 1=openai/gpt-4.1
    python branch_game.py results/jsonl/<run>.jsonl --rounds 50 --replace 1=google/gemini-2.5-flash

    # the replacement does not get to see the history that formed the convention
    python branch_game.py results/jsonl/<run>.jsonl --rounds 50 --replace 1=gpt-4.1-mini --replacement_window 0

    # branch at an explicit round, or change the replacement's prompt
    python branch_game.py results/jsonl/<run>.jsonl --at_round 120 --replace 1=gpt-4.1-mini --replace_prefix AC

Important subtlety: agents in this harness have NO hidden state - each round's
decision is a fresh call whose only memory is the visible history window. So
"a fresh instance of the same model" is statistically identical to "no
replacement"; the arms that can differ are a different model/provider, a
different prompt, or a replacement that sees less history (--replacement_window).

Output: results/branches/<run>__r<round>__<condition>.jsonl (+ .config.json) with the
pre-branch rounds copied first, then the new rounds. The summary compares mean
gain before vs after the branch and reports rounds-to-recover.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from harness.io import (
    append_jsonl,
    build_env,
    first_round_above,
    load_config,
    read_jsonl,
    save_config,
    summarize,
)
from harness.providers import ChatProvider, FatalProviderError, cost_note
from run_game import ask_agent, AgentCallError


def parse_replacements(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--replace expects <agent_index>=<provider:model>, got {item!r}")
        idx, spec = item.split("=", 1)
        out[int(idx)] = spec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="base run .jsonl from run_game.py")
    ap.add_argument("--rounds", type=int, default=50, help="rounds to play after the branch point")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--at_round", type=int, help="branch after this round index (0-based)")
    where.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="branch at the first round where the rolling mean market gain exceeds this",
    )
    ap.add_argument("--detect_window", type=int, default=20, help="rolling window for --threshold")
    ap.add_argument(
        "--replace",
        nargs="*",
        default=[],
        help="<agent_index>=<provider:model> pairs; omit for the no-replacement control",
    )
    ap.add_argument(
        "--replace_prefix", default=None, help="prompt prefix for replaced agents (default: keep)"
    )
    ap.add_argument(
        "--replacement_window",
        type=int,
        default=None,
        help="history rounds visible to replaced agents from the branch on "
        "(default: same as the run; 0 = they start blind)",
    )
    ap.add_argument("--temperature", type=float, default=None, help="override sampling temperature")
    ap.add_argument(
        "--seed", type=int, default=None, help="sampling seed for the new rounds (default: run seed)"
    )
    ap.add_argument("--label", default=None, help="condition label used in the output name")
    ap.add_argument("--out_dir", default="results/branches")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    records = read_jsonl(args.run)
    config = load_config(args.run)
    n_agents = config["agents"]

    # --- where to branch -------------------------------------------------
    if args.at_round is not None:
        branch_round = args.at_round
    else:
        branch_round = first_round_above(records, args.threshold, args.detect_window)
        if branch_round is None:
            sys.exit(
                f"❌ market gain never exceeded {args.threshold} over {args.detect_window} rounds; "
                "use --at_round or a lower --threshold"
            )
    if not 0 <= branch_round < len(records):
        sys.exit(f"❌ branch round {branch_round} outside 0..{len(records) - 1}")
    kept = records[: branch_round + 1]

    # --- who plays after the branch --------------------------------------
    replacements = parse_replacements(args.replace)
    for idx in replacements:
        if not 0 <= idx < n_agents:
            sys.exit(f"❌ agent index {idx} out of range for {n_agents} agents")
    models = list(config["models"])
    prefixes = list(config["prefixes"])
    for idx, spec in replacements.items():
        models[idx] = spec
        if args.replace_prefix:
            prefixes[idx] = args.replace_prefix
    temperature = args.temperature if args.temperature is not None else config["temperature"]
    seed = args.seed if args.seed is not None else config["seed"]
    label = args.label or (
        "control" if not replacements else "replace" + "".join(f"{i}-{s}" for i, s in replacements.items())
    )
    if args.replacement_window is not None and replacements:
        label += f"_win{args.replacement_window}"
    safe = lambda s: s.replace("/", "-").replace(":", "-").replace(".", "p")
    out_path = os.path.join(args.out_dir, f"{Path(args.run).stem}__r{branch_round}__{safe(label)}.jsonl")

    n_calls = args.rounds * n_agents
    providers = [ChatProvider(spec, temperature) for spec in models]
    print(
        f"branching {Path(args.run).name} after round {branch_round} -> {label}\n"
        f"{cost_note(n_calls, [p.spec for p in providers])}"
    )
    if not args.yes and n_calls > 100 and not all(p.is_local for p in providers) and sys.stdin.isatty():
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("aborted")

    # --- rebuild market state ----------------------------------------------
    # Two environments share the same price history: the incumbents keep the
    # run's window; replaced agents may get a different (e.g. empty) window.
    base_cfg = {**config, "prefixes": prefixes}
    env = build_env(base_cfg)
    env_repl = None
    if replacements and args.replacement_window is not None:
        env_repl = build_env({**base_cfg, "window": args.replacement_window})
    for actions in config.get("prior_actions", []):
        env.step(actions)
        if env_repl:
            env_repl.step(actions)
    for rec in kept:
        *_, prompts = env.step(rec["actions"])
        if env_repl:
            *_, prompts_repl = env_repl.step(rec["actions"])
    if env_repl:
        # in blind mode the replacement should see nothing from before the branch
        env_repl.history = {}
        env_repl.round = 0
        prompts_repl = env_repl.reset()

    def current_prompts():
        p = list(prompts)
        if env_repl:
            for idx in replacements:
                p[idx] = prompts_repl[idx]
        return p

    # --- write branch file -----------------------------------------------
    if os.path.exists(out_path):
        os.remove(out_path)
    for rec in kept:
        append_jsonl(out_path, {**rec, "branch_phase": "before"})
    branch_cfg = {
        **config,
        "models": [p.spec for p in providers],
        "prefixes": prefixes,
        "temperature": temperature,
        "seed": seed,
        "branch_of": str(args.run),
        "branch_round": branch_round,
        "branch_label": label,
        "replacements": {str(k): v for k, v in replacements.items()},
        "replacement_window": args.replacement_window,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_config(out_path, branch_cfg)

    new_records = []
    t0 = time.time()
    for k in range(args.rounds):
        rnd = branch_round + 1 + k
        ps = current_prompts()
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            results = list(
                ex.map(
                    lambda t: ask_agent(t[0], t[1], seed, args.max_retries),
                    zip(providers, ps),
                )
            )
        actions = [r["price"] for r in results]
        prices, demands, profits, gains, prompts = env.step(actions)
        if env_repl:
            *_, prompts_repl = env_repl.step(actions)
        record = {
            "round": rnd,
            "actions": actions,
            "thoughts": [r["thoughts"] for r in results],
            "respones": [r["response"] for r in results],
            "demands": demands,
            "profits": profits,
            "gains": gains,
            "metrics": [r["metrics"] for r in results],
            "branch_phase": "after",
        }
        append_jsonl(out_path, record)
        new_records.append(record)
        print(
            f"   round {rnd:>3}  prices={[round(a, 2) for a in actions]}  "
            f"gains={[round(g, 2) for g in gains]}  {time.time() - t0:.0f}s"
        )

    # --- compare before / after ---------------------------------------------
    w = min(args.detect_window, len(kept), len(new_records))
    before = float(np.mean([r["gains"] for r in kept[-w:]]))
    after = float(np.mean([r["gains"] for r in new_records[-w:]]))
    after_all = np.array([np.mean(r["gains"]) for r in new_records])
    recover_at = None
    for i in range(w, len(after_all) + 1):
        if after_all[i - w : i].mean() >= 0.9 * before:
            recover_at = i  # rounds after branch until the rolling mean is back near pre-branch level
            break
    comparison = {
        "window": w,
        "mean_gain_before": round(before, 3),
        "mean_gain_after": round(after, 3),
        "delta": round(after - before, 3),
        "rounds_to_recover": recover_at,
        "mean_price_before": [round(float(x), 3) for x in np.mean([r["actions"] for r in kept[-w:]], axis=0)],
        "mean_price_after": [
            round(float(x), 3) for x in np.mean([r["actions"] for r in new_records[-w:]], axis=0)
        ],
        "mean_profit_before": [
            round(float(x), 2) for x in np.mean([r["profits"] for r in kept[-w:]], axis=0)
        ],
        "mean_profit_after": [
            round(float(x), 2) for x in np.mean([r["profits"] for r in new_records[-w:]], axis=0)
        ],
    }
    branch_cfg["comparison"] = comparison
    branch_cfg["summary_after"] = summarize(new_records, window=w)
    save_config(out_path, branch_cfg)
    print(
        f"📊 {label}: market mean gain {before:+.3f} (last {w} before) -> {after:+.3f} (last {w} after), "
        f"Δ {after - before:+.3f}; rounds to recover: {recover_at if recover_at is not None else 'not within horizon'}"
    )
    print(f"   saved {out_path}")


if __name__ == "__main__":
    try:
        main()
    except (FatalProviderError, AgentCallError) as e:
        sys.exit(f"\n❌ {e}")
