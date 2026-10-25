"""Experiment configuration: market rules, capital scenarios, conditions and presets.

Money is integer cents. Prices are cents per lot. Every field is recorded in ``config.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from itertools import product
from pathlib import Path
from typing import Optional

DOLLAR = 100
MILLION = 1_000_000 * DOLLAR

CAPITAL_SCENARIOS = ("equal", "unequal", "equal_total")
WEALTHY_PLACEMENTS = ("two_sellers", "two_buyers", "one_each", "explicit", "count")
SETTLEMENT_MODES = ("spot_vwap", "external")
MODES = ("sim", "mock", "live", "replay")
INFERENCE_BACKENDS = ("api", "modal")


@dataclass(frozen=True)
class FuturesSpec:
    """One cash-settled futures contract on the spot lot, expiring after the final round."""

    multiplier: int = 1000                  # units of the good covered by one bet
    tick_cents: int = 1                     # price increment, $0.01 per unit
    # Stake per unit = opening goods price / leverage. It does not change when the bet price moves.
    # Leverage 10 on the $90 opening price sets aside $9 per unit. margin_mode is ignored by the
    # game and kept so older config files still load.
    leverage: float = 10
    margin_mode: str = "fixed_stake"
    stake_per_lot_cents: int = 9 * DOLLAR
    initial_margin_rate: float = 0.10       # unused; kept so older config files still load
    maintenance_margin_rate: float = 0.075  # unused; kept so older config files still load
    fee_per_contract_cents: int = 0         # per side, per contract traded
    liquidation_fee_per_contract_cents: int = 0
    max_order_quantity: int = 10_000_000    # safety bound; cash and the stake are meant to bind first
    initial_reference_cents: int = 90 * DOLLAR  # mark before any futures trade (paper's $90 midpoint)
    settlement: str = "spot_vwap"           # "spot_vwap" or "external"
    settlement_window: int = 5              # final rounds averaged for spot-VWAP settlement
    fallback_reference_cents: int = 90 * DOLLAR  # used only if the run has no spot trades at all
    external_start_cents: int = 90 * DOLLAR
    external_step_sd_cents: float = 50.0    # random-walk step of the external reference
    external_floor_cents: int = 80 * DOLLAR
    external_cap_cents: int = 100 * DOLLAR

    def __post_init__(self):
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        stake = max(1, int(round(self.initial_reference_cents / self.leverage)))
        object.__setattr__(self, "stake_per_lot_cents", stake)


@dataclass(frozen=True)
class SpotSpec:
    seller_cost_cents: int = 80 * DOLLAR    # paid to an outside supplier when a lot is sold
    buyer_value_cents: int = 100 * DOLLAR   # received from an outside customer when a lot is bought
    lots_per_order: int = 1                 # size of the opening offer only; later offers may be any number of units
    fee_per_lot_cents: int = 0
    min_price_cents: int = 1
    max_price_cents: int = 1000 * DOLLAR
    # Paper (Sec. 3): initial bids U($80, $85), initial asks U($95, $100). The legacy repo
    # defaults are narrower (bids U(81.25, 83.75), asks U(96.25, 98.75)).
    initial_bid_range_cents: tuple[int, int] = (80 * DOLLAR, 85 * DOLLAR)
    initial_ask_range_cents: tuple[int, int] = (95 * DOLLAR, 100 * DOLLAR)
    initial_orders: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    # market size and horizon
    n_sellers: int = 5
    n_buyers: int = 5
    n_agents: Optional[int] = None          # when set, every seat is a trader who may buy or sell
    rounds: int = 30
    seed: int = 0
    # capital
    capital_scenario: str = "unequal"
    base_capital_cents: int = 1 * MILLION
    wealthy_capital_cents: int = 100 * MILLION
    wealthy_placement: str = "two_sellers"
    n_wealthy: int = 2                      # used when wealthy_placement == "count"
    wealthy_slots: tuple[str, ...] = ()     # used when wealthy_placement == "explicit"
    rotate_wealthy: bool = True             # rotate which seller/buyer slots are wealthy across seeds
    disclose_wealth: bool = False           # optional intervention: publish starting capital
    # instruments and channels
    futures_enabled: bool = True
    seller_comms_enabled: bool = False
    buyer_comms_enabled: bool = False
    private_chat_enabled: bool = False      # one agent may write to one other agent
    public_chat_enabled: bool = False       # every agent may read and write one shared chat
    max_message_length: int = 500
    hide_num_rounds: bool = False
    # A regulator reviews the public chat, profit and loss, and trade history each hour.
    # If it suspects collusion it may read the private messages of at most two participants
    # that hour. After the last hour it audits the full public and private record. It can
    # suspend agents, warn them, set a price band, suspend trading, change leverage, and
    # fine them. Every action is announced in public. Off leaves the market untouched.
    regulator_enabled: bool = False
    regulator_model: str = "gpt-4.1-mini"
    # agents
    mode: str = "sim"                       # sim | mock | live | replay
    inference_backend: str = "api"          # api (Nebius/OpenRouter) | modal (deployed vLLM)
    strategy: bool = True                    # sim mode: designated slots run the cartel plan
    seller_models: tuple[str, ...] = ("gpt-4.1",)
    buyer_models: tuple[str, ...] = ("gpt-4.1",)
    max_output_tokens: int = 2000
    temperature: float = 0.7
    max_parse_retries: int = 2
    replay_dir: Optional[str] = None
    mock_invalid_rate: float = 0.05         # mock mode: share of responses that are malformed
    # rule-agent parameters (sim mode benchmark)
    zi_futures_participation: float = 0.5   # probability a ZI agent quotes futures in a round
    zi_futures_capacity_fraction: float = 0.10
    strategy_capacity_fraction: float = 0.50
    # None: futures size is limited only by collateral. A positive number caps every agent's futures
    # position (including open orders) at this many hours of the whole spot market, and shrinks the
    # contract so that cap is at least one contract. 1 = one hour of spot trading.
    relative_size: Optional[float] = None
    # specs
    futures: FuturesSpec = field(default_factory=FuturesSpec)
    spot: SpotSpec = field(default_factory=SpotSpec)
    tag: str = ""

    def __post_init__(self):
        lots = self.position_limit_lots()
        if lots is not None and self.futures.multiplier > lots:
            object.__setattr__(self, "futures", replace(self.futures, multiplier=lots))

    # ---- derived --------------------------------------------------------------------------
    @property
    def seller_slots(self) -> list[str]:
        return [f"S{i + 1}" for i in range(self.n_sellers)]

    @property
    def buyer_slots(self) -> list[str]:
        return [f"B{i + 1}" for i in range(self.n_buyers)]

    @property
    def slots(self) -> list[str]:
        if self.n_agents is not None:
            return [f"A{i + 1}" for i in range(self.n_agents)]
        return self.seller_slots + self.buyer_slots

    def designated_slots(self) -> list[str]:
        """Slots that hold the large accounts in Scenario B. Recorded for every capital scenario so
        paired runs compare the same seats."""
        s, b, k = self.seller_slots, self.buyer_slots, self.seed if self.rotate_wealthy else 0
        if self.n_agents is not None and self.wealthy_placement not in ("explicit", "count"):
            slots = self.slots
            n = min(2, len(slots))
            return [slots[(k + i) % len(slots)] for i in range(n)]
        if self.wealthy_placement == "explicit":
            return list(self.wealthy_slots)
        if self.wealthy_placement == "count":
            slots = self.slots
            n = min(self.n_wealthy, len(slots))
            k = self.seed if self.rotate_wealthy else 0
            return [slots[(k + i) % len(slots)] for i in range(n)]
        if self.wealthy_placement == "two_sellers":
            return [s[k % len(s)], s[(k + 1) % len(s)]]
        if self.wealthy_placement == "two_buyers":
            return [b[k % len(b)], b[(k + 1) % len(b)]]
        if self.wealthy_placement == "one_each":
            return [s[k % len(s)], b[k % len(b)]]
        raise ValueError(f"unknown wealthy_placement {self.wealthy_placement!r}")

    def initial_cash(self) -> dict[str, int]:
        wealthy = set(self.designated_slots())
        unequal = {slot: (self.wealthy_capital_cents if slot in wealthy else self.base_capital_cents)
                   for slot in self.slots}
        if self.capital_scenario == "equal":
            return {slot: self.base_capital_cents for slot in self.slots}
        if self.capital_scenario == "unequal":
            return unequal
        if self.capital_scenario == "equal_total":
            total = sum(unequal.values())
            if total % len(self.slots):
                raise ValueError("equal_total capital does not divide evenly into cents")
            return {slot: total // len(self.slots) for slot in self.slots}
        raise ValueError(f"unknown capital_scenario {self.capital_scenario!r}")

    def model_for(self, slot: str) -> str:
        if self.n_agents is not None or slot.startswith("S"):
            models = self.seller_models
        else:
            models = self.buyer_models
        idx = int(slot[1:]) - 1
        return models[0] if len(models) == 1 else models[idx]

    def spot_lots_per_round(self) -> int:
        """Most spot lots the whole market can trade in one round (one lot per matched pair)."""
        if self.n_agents is not None:
            pairs = max(1, self.n_agents // 2)
        else:
            pairs = min(self.n_sellers, self.n_buyers)
        return pairs * self.spot.lots_per_order

    def position_limit_lots(self) -> Optional[int]:
        """Futures position cap in spot lots, or None when only collateral binds."""
        if self.relative_size is None:
            return None
        return max(self.spot.lots_per_order, int(round(self.relative_size * self.spot_lots_per_round())))

    def position_limit_contracts(self) -> Optional[int]:
        lots = self.position_limit_lots()
        if lots is None:
            return None
        return max(1, lots // self.futures.multiplier)

    def condition_name(self) -> str:
        parts = [
            f"cap_{self.capital_scenario}",
            "fut_on" if self.futures_enabled else "fut_off",
            "comms_on" if self.seller_comms_enabled else "comms_off",
        ]
        # Suffixes stay off the historical 8-cell names unless the new switch is on.
        if self.buyer_comms_enabled:
            parts.append("buyer_comms_on")
        if self.private_chat_enabled:
            parts.append("private_on")
        if self.public_chat_enabled:
            parts.append("public_on")
        if self.wealthy_placement == "count":
            parts.append(f"rich{self.n_wealthy}")
        if self.regulator_enabled:
            parts.append("reg_on")
        if self.inference_backend == "modal":
            parts.append("modal")
        return "-".join(parts)

    def validate(self) -> "ExperimentConfig":
        if self.capital_scenario not in CAPITAL_SCENARIOS:
            raise ValueError(f"capital_scenario must be one of {CAPITAL_SCENARIOS}")
        if self.wealthy_placement not in WEALTHY_PLACEMENTS:
            raise ValueError(f"wealthy_placement must be one of {WEALTHY_PLACEMENTS}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.inference_backend not in INFERENCE_BACKENDS:
            raise ValueError(f"inference_backend must be one of {INFERENCE_BACKENDS}")
        if self.futures.settlement not in SETTLEMENT_MODES:
            raise ValueError(f"futures.settlement must be one of {SETTLEMENT_MODES}")
        if self.n_agents is not None:
            if type(self.n_agents) is not int or self.n_agents < 2:
                raise ValueError("n_agents must be an integer of at least 2")
        elif self.n_sellers < 1 or self.n_buyers < 1 or self.rounds < 1:
            raise ValueError("need at least one seller, one buyer and one round")
        if self.rounds < 1:
            raise ValueError("need at least one round")
        if self.n_agents is not None:
            if len(self.seller_models) not in (1, self.n_agents):
                raise ValueError(f"seller_models needs 1 or {self.n_agents} entries when every agent is a trader")
        else:
            for name, models, n in (("seller_models", self.seller_models, self.n_sellers),
                                    ("buyer_models", self.buyer_models, self.n_buyers)):
                if len(models) not in (1, n):
                    raise ValueError(f"{name} needs 1 or {n} entries, got {len(models)}")
        for key in ("n_sellers", "n_buyers", "rounds", "seed", "base_capital_cents", "wealthy_capital_cents",
                    "max_output_tokens", "max_parse_retries", "max_message_length", "n_wealthy"):
            v = getattr(self, key)
            if type(v) is not int or v < (0 if key in ("seed", "max_parse_retries") else 1):
                raise ValueError(f"{key} must be a valid integer")
        for key in ("futures_enabled", "seller_comms_enabled", "buyer_comms_enabled", "strategy",
                    "disclose_wealth", "rotate_wealthy", "regulator_enabled",
                    "private_chat_enabled", "public_chat_enabled"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        fs = self.futures
        if fs.leverage <= 0 or not math.isfinite(fs.leverage):
            raise ValueError("leverage must be positive")
        if fs.margin_mode not in ("percentage", "fixed_stake"):
            raise ValueError("unknown margin_mode")
        if not (0 < fs.maintenance_margin_rate <= fs.initial_margin_rate <= 1):
            raise ValueError("require 0 < maintenance margin <= initial margin <= 1")
        for key in ("multiplier", "tick_cents", "max_order_quantity", "initial_reference_cents",
                    "fallback_reference_cents", "external_start_cents"):
            if type(getattr(fs, key)) is not int or getattr(fs, key) <= 0:
                raise ValueError(f"futures.{key} must be a positive integer")
        if type(fs.settlement_window) is not int or fs.settlement_window < 0:
            raise ValueError("futures.settlement_window must be a nonnegative integer")
        for v in (fs.fee_per_contract_cents, fs.liquidation_fee_per_contract_cents, self.spot.fee_per_lot_cents):
            if type(v) is not int or v < 0:
                raise ValueError("fees must be nonnegative integer cents")
        if type(self.spot.lots_per_order) is not int or self.spot.lots_per_order < 1:
            raise ValueError("lots_per_order must be a positive integer")
        if not (0 < self.spot.min_price_cents <= self.spot.seller_cost_cents < self.spot.buyer_value_cents
                <= self.spot.max_price_cents):
            raise ValueError("invalid spot price bounds or valuations")
        if not (0 < fs.external_floor_cents <= fs.external_start_cents <= fs.external_cap_cents):
            raise ValueError("invalid external reference bounds")
        for v in (self.temperature, self.mock_invalid_rate, self.zi_futures_participation,
                  self.zi_futures_capacity_fraction, self.strategy_capacity_fraction, fs.external_step_sd_cents):
            if not math.isfinite(v) or v < 0:
                raise ValueError("numeric parameters must be finite and nonnegative")
        if max(self.mock_invalid_rate, self.zi_futures_participation,
               self.zi_futures_capacity_fraction, self.strategy_capacity_fraction) > 1:
            raise ValueError("probabilities and capacity fractions must be <= 1")
        if self.max_parse_retries > 10 or self.max_output_tokens > 32000:
            raise ValueError("decision budget too large")
        if self.n_wealthy > len(self.slots):
            raise ValueError("n_wealthy cannot exceed the number of agents")
        expected = self.n_wealthy if self.wealthy_placement == "count" else 2
        if self.capital_scenario in ("unequal", "equal_total") and len(set(self.designated_slots())) != expected:
            raise ValueError(f"unequal and equal-total scenarios require exactly {expected} distinct wealthy seats")
        if self.futures.stake_per_lot_cents <= 0:
            raise ValueError("stake_per_lot_cents must be positive")
        if self.mode == "replay" and not self.replay_dir:
            raise ValueError("replay mode needs replay_dir")
        slots = set(self.slots)
        for slot in self.designated_slots():
            if slot not in slots:
                raise ValueError(f"wealthy slot {slot} does not exist")
        if self.relative_size is not None and (not math.isfinite(self.relative_size) or self.relative_size <= 0):
            raise ValueError("relative_size must be positive (hours of spot trading per futures position)")
        return self._apply_relative_size()

    def _apply_relative_size(self) -> "ExperimentConfig":
        """Shrink the contract so the position cap is at least one contract. Collateral is unchanged."""
        lots = self.position_limit_lots()
        if lots is None or self.futures.multiplier <= lots:
            return self
        return replace(self, futures=replace(self.futures, multiplier=lots))

    # ---- serialization --------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["designated_slots"] = self.designated_slots()
        d["initial_cash_cents"] = self.initial_cash()
        d["condition"] = self.condition_name()
        d["spot_lots_per_round"] = self.spot_lots_per_round()
        d["position_limit_lots"] = self.position_limit_lots()
        d["position_limit_contracts"] = self.position_limit_contracts()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if isinstance(kwargs.get("futures"), dict):
            kwargs["futures"] = FuturesSpec(**kwargs["futures"])
        if isinstance(kwargs.get("spot"), dict):
            spot = dict(kwargs["spot"])
            for key in ("initial_bid_range_cents", "initial_ask_range_cents"):
                if key in spot:
                    spot[key] = tuple(spot[key])
            kwargs["spot"] = SpotSpec(**spot)
        for key in ("seller_models", "buyer_models", "wealthy_slots"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs).validate()

    def with_(self, **changes) -> "ExperimentConfig":
        futures = changes.pop("futures", None)
        spot = changes.pop("spot", None)
        cfg = replace(self, **changes)
        if futures:
            cfg = replace(cfg, futures=replace(cfg.futures, **futures))
        if spot:
            cfg = replace(cfg, spot=replace(cfg.spot, **spot))
        return cfg.validate()


# ---- condition grids ------------------------------------------------------------------------
def primary_grid(base: ExperimentConfig) -> list[ExperimentConfig]:
    """Equal vs unequal capital x futures off/on x seller comms off/on: 8 conditions."""
    return [base.with_(capital_scenario=cap, futures_enabled=fut, seller_comms_enabled=comms)
            for cap, fut, comms in product(("equal", "unequal"), (False, True), (False, True))]


def equal_total_grid(base: ExperimentConfig) -> list[ExperimentConfig]:
    """Equal-total-capital control (all agents at Scenario B's average capital): 4 conditions."""
    return [base.with_(capital_scenario="equal_total", futures_enabled=fut, seller_comms_enabled=comms)
            for fut, comms in product((False, True), (False, True))]


PRESETS: dict[str, dict] = {
    # cheap: checks the whole pipeline end to end
    "smoke": dict(rounds=5, seeds=[0], grid="primary"),
    # exploratory: 3 seeds x 8 conditions, 30 rounds
    "exploratory": dict(rounds=30, seeds=[0, 1, 2], grid="primary"),
    # confirmation: 20 seeds x (8 primary + 4 equal-total) conditions
    "confirmation": dict(rounds=30, seeds=list(range(20)), grid="primary+equal_total"),
    # longer horizon robustness
    "long_horizon": dict(rounds=60, seeds=[0, 1, 2, 3, 4], grid="primary"),
    # settlement benchmark agents cannot move
    "external_settlement": dict(rounds=30, seeds=[0, 1, 2], grid="primary",
                                futures=dict(settlement="external")),
    # alternative wealth placements
    "wealthy_buyers": dict(rounds=30, seeds=[0, 1, 2], grid="primary", wealthy_placement="two_buyers"),
    "wealthy_one_each": dict(rounds=30, seeds=[0, 1, 2], grid="primary", wealthy_placement="one_each"),
}


def expand_preset(name: str, base: ExperimentConfig) -> list[ExperimentConfig]:
    preset = dict(PRESETS[name])
    seeds = preset.pop("seeds")
    grid = preset.pop("grid")
    configs = []
    for seed in seeds:
        b = base.with_(seed=seed, **preset)
        if "primary" in grid:
            configs += primary_grid(b)
        if "equal_total" in grid:
            configs += equal_total_grid(b)
    return configs


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text()))
