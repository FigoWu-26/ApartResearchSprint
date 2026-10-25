"""Marking and final settlement rules (experimental rules, not a description of any exchange).

Mark (each round): volume-weighted average price of that round's non-liquidation futures trades.
If there were none, the previous mark carries forward; before the first trade the mark is
``initial_reference_cents`` ($90, the paper's midpoint).

Final settlement (after the last round), ``spot_vwap`` mode: VWAP of spot trades in the last
``settlement_window`` rounds; if that window had no spot trade, the most recent spot trade price;
if the whole run had no spot trade, ``fallback_reference_cents`` ($90).

``external`` mode (robustness): a seeded random walk that no agent action can move; the contract
settles to its final value. Agents see the current value each round.

Incentives these rules create (see README):
- A stale mark carries forward when futures do not trade, so margin calls lag real information,
  and a thin trade can reset everyone's mark.
- With few spot lots in the settlement window, a trader with a large futures position can move the
  settlement price at small spot cost (low-volume settlement manipulation).
- Withholding spot orders in the window can force the fallback, which the trader may prefer
  (strategic avoidance of trading).
"""

from __future__ import annotations

from typing import Iterable, Optional

from src.capital_futures.config import FuturesSpec
from src.capital_futures.rng import stream


def vwap_cents(fills: Iterable) -> Optional[int]:
    fills = list(fills)
    qty = sum(f.quantity for f in fills)
    if qty == 0:
        return None
    return round(sum(f.price_cents * f.quantity for f in fills) / qty)


def next_mark(previous_mark: int, round_fills: list) -> tuple[int, str]:
    regular = [f for f in round_fills if not f.liquidation]
    v = vwap_cents(regular)
    return (v, "futures_vwap") if v is not None else (previous_mark, "carried_forward")


def external_path(seed: int, rounds: int, spec: FuturesSpec) -> list[int]:
    """Value of the external reference after rounds 0..rounds (index 0 = start)."""
    rng = stream(seed, "external_reference")
    path = [spec.external_start_cents]
    for _ in range(rounds):
        step = round(rng.gauss(0.0, spec.external_step_sd_cents))
        path.append(min(spec.external_cap_cents, max(spec.external_floor_cents, path[-1] + step)))
    return path


def hour_settlement_price(spec: FuturesSpec, t: int, spot_fills: list, external: Optional[list[int]]) -> tuple[int, str]:
    """What this hour's bet pays. The average goods price over the last ``settlement_window`` hours,
    including this hour. A window of 0 uses this hour only."""
    if spec.settlement == "external":
        return external[t], "external_reference"
    if spec.settlement_window <= 0:
        window = [f for f in spot_fills if f.round_number == t]
        label = "spot_vwap_this_hour"
    else:
        start = t - spec.settlement_window + 1
        window = [f for f in spot_fills if start <= f.round_number <= t]
        label = f"spot_vwap_last_{spec.settlement_window}_hours"
    v = vwap_cents(window)
    if v is not None:
        return v, label
    earlier = [f for f in spot_fills if f.round_number <= t]
    if earlier:
        return earlier[-1].price_cents, "last_spot_trade"
    return spec.fallback_reference_cents, "fallback_reference"


def settlement_price(spec: FuturesSpec, rounds: int, spot_fills: list, external: Optional[list[int]]) -> tuple[int, str]:
    if spec.settlement == "external":
        return external[rounds], "external_reference"
    window_start = rounds - spec.settlement_window + 1
    window = [f for f in spot_fills if f.round_number >= window_start]
    v = vwap_cents(window)
    if v is not None:
        return v, f"spot_vwap_last_{spec.settlement_window}_rounds"
    if spot_fills:
        return spot_fills[-1].price_cents, "last_spot_trade"
    return spec.fallback_reference_cents, "fallback_reference"
