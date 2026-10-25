"""Validated action schema shared by LLM, rule-based, scripted and replayed agents."""

from __future__ import annotations

import json
import math
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class OrderInstruction(BaseModel):
    """``keep`` leaves any open order unchanged (or holds with no order); ``place`` submits a new order,
    replacing your open order on that side; ``cancel`` withdraws your open order on that side.

    ``side`` is only used for the goods offer. Empty means the agent's usual side: sellers offer to
    sell, buyers offer to buy. The cartel's plan sets it to ``bid`` so a seller can buy the good."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    action: Literal["keep", "place", "cancel"] = "keep"
    price: Optional[float] = None
    quantity: Optional[int] = None
    side: Optional[Literal["bid", "ask"]] = None
    # Futures quotes only. True shows this agent's id on that quote. False leaves it anonymous.
    # Goods quotes ignore this and always show the id.
    disclose_id: Optional[bool] = None

    @field_validator("action", mode="before")
    @classmethod
    def _normalize(cls, v):
        if v is None:
            return "keep"
        v = str(v).strip().lower()
        return {"hold": "keep", "none": "keep", "replace": "place", "new": "place", "withdraw": "cancel"}.get(v, v)

    @field_validator("disclose_id", mode="before")
    @classmethod
    def _disclose(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            word = v.strip().lower()
            if word in ("true", "yes", "1", "disclose"):
                return True
            if word in ("false", "no", "0", "hide", "anonymous"):
                return False
        raise ValueError("disclose_id must be true or false")

    @field_validator("quantity", mode="before")
    @classmethod
    def _int_quantity(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            raise ValueError("quantity cannot be boolean")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError("quantity must be a whole number of contracts")
        return int(v)


class TraderAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spot_order: OrderInstruction = Field(default_factory=OrderInstruction)
    futures_bid: OrderInstruction = Field(default_factory=OrderInstruction)
    futures_ask: OrderInstruction = Field(default_factory=OrderInstruction)
    message_to_sellers: Optional[str] = None
    message_to_buyers: Optional[str] = None
    message_public: Optional[str] = None
    message_private: Optional[str] = None
    private_to: Optional[str] = None
    reflection: Optional[str] = None
    strategy_note: Optional[str] = None
    new_memory: Optional[str] = None
    scratch_pad_update: Optional[str] = None

    @field_validator("spot_order", "futures_bid", "futures_ask", mode="before")
    @classmethod
    def _null_is_keep(cls, v):
        return {"action": "keep"} if v in (None, "null", "keep", "hold") else v

    def check(self, futures_enabled: bool) -> list[str]:
        """Semantic checks beyond types. Returns human-readable problems (empty = valid)."""
        problems = []
        for name in ("spot_order", "futures_bid", "futures_ask"):
            ins: OrderInstruction = getattr(self, name)
            if ins.action != "place":
                continue
            if ins.price is None or not (math.isfinite(ins.price) and ins.price > 0):
                problems.append(f"{name}: 'place' needs a positive price")
            if name != "spot_order":
                if not futures_enabled:
                    problems.append(f"{name}: futures trading is disabled in this market; use 'keep'")
                elif ins.quantity is None or ins.quantity < 1:
                    problems.append(f"{name}: 'place' needs an integer quantity of at least 1 contract")
        return problems


def keep_all() -> TraderAction:
    return TraderAction()


def parse_action(text: str, futures_enabled: bool) -> tuple[Optional[TraderAction], Optional[str]]:
    """Parse a model response into an action. Returns (action, None) or (None, error message)."""
    if text is None:
        return None, "empty response"
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "no JSON object found"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(data, dict):
        return None, "top-level JSON must be an object"
    try:
        action = TraderAction.model_validate(data)
    except ValidationError as e:
        return None, "schema error: " + "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
    problems = action.check(futures_enabled)
    if problems:
        return None, "; ".join(problems)
    return action, None
