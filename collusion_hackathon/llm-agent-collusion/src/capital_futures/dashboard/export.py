"""Export a self-contained static dashboard: index.html with selected runs and aggregates embedded.

    uv run python -m src.capital_futures.dashboard.export --out ../spot-futures-dashboard.html \
        --roots results/capital_futures/sim_sweep results/capital_futures/sim_placements ...

Runs are selected by glob patterns on their directory names (default: seed 0 of every condition).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from datetime import datetime
from pathlib import Path

from src.capital_futures.dashboard.server import aggregate, load_trades
from src.capital_futures.run import git_info

HERE = Path(__file__).resolve().parent


def collect(roots: list[Path], patterns: list[str]) -> list[dict]:
    runs = []
    for root in roots:
        for p in sorted(root.rglob("dashboard.json")):
            name = p.parent.name
            if any(fnmatch.fnmatch(name, pat) for pat in patterns) and not name.endswith("-replay") and "-diag-" not in name:
                payload = json.loads(p.read_text())
                payload["trades"] = load_trades(p.parent)
                runs.append(dict(id=f"{root.name}/{p.parent.relative_to(root)}", payload=payload))
    return runs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--patterns", nargs="+", default=["*-seed0"])
    p.add_argument("--default-a", default="cap_unequal-fut_on-comms_off-strat_on-seed0")
    p.add_argument("--default-b", default="cap_unequal-fut_on-comms_off-strat_off-seed0")
    args = p.parse_args(argv)
    runs = collect(args.roots, args.patterns)
    agg = [row for root in args.roots for row in aggregate(root)]

    def find(name):
        return next((r["id"] for r in runs if r["id"].endswith("/" + name)), runs[0]["id"] if runs else None)
    snap = dict(generated=datetime.now().isoformat(timespec="seconds"), git=git_info(), runs=runs, aggregate=agg,
                defaults=dict(a=find(args.default_a), b=find(args.default_b)))
    html = (HERE / "index.html").read_text()
    marker = '<script id="snapshot" type="application/json">null</script>'
    assert marker in html
    data = json.dumps(snap, separators=(",", ":"), default=str).replace("</", "<\\/")
    args.out.write_text(html.replace(marker, f'<script id="snapshot" type="application/json">{data}</script>'))
    print(f"{len(runs)} runs, {len(agg)} aggregate rows -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
