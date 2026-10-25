"""Agent accounts, margin arithmetic and the clearing/default ledger.

Accounting identities (integer cents, checked by ``reconcile``):

    equity          = cash - liabilities
    net P&L         = equity - initial cash = spot P&L + futures P&L - fees
    futures P&L     = cumulative variation margin, including liquidation fills and final settlement

Variation margin is paid in cash every round at the mark, so after each round there is no unsettled
futures value left to add (``unsettled_marked_value`` is reported and is always 0 at round end).
Collateral is a reservation against cash, not a payment: posting initial margin moves no money.
Cash never goes below zero; a shortfall is moved to ``liabilities`` and to the clearing ledger's
uncovered default losses, and later gains repay it first.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from dataclasses import asdict, dataclass, field


@dataclass
class Account:
    agent_id: str
    slot: str
    role: str                 # "seller" or "buyer"
    initial_cash: int
    cash: int
    liabilities: int = 0
    position: int = 0         # signed futures contracts
    last_vm_price: int = 0    # price the position was last marked to
    spot_pnl: int = 0
    futures_pnl: int = 0
    fees: int = 0
    external_paid: int = 0    # to outside suppliers (sellers' $80 sourcing)
    external_received: int = 0  # from outside customers (buyers' $100 redemption)
    lots_bought: int = 0
    lots_sold: int = 0
    futures_volume: int = 0
    defaulted: bool = False
    in_liquidation: bool = False
    margin_calls: int = 0
    liquidations: int = 0
    rejected_orders: int = 0

    @property
    def equity(self) -> int:
        return self.cash - self.liabilities

    @property
    def net_pnl(self) -> int:
        return self.equity - self.initial_cash

    def to_dict(self) -> dict:
        d = asdict(self)
        d["equity"] = self.equity
        d["net_pnl"] = self.net_pnl
        d["initial_equity"] = self.initial_cash
        d["realized_futures_pnl"] = self.futures_pnl
        return d


@dataclass
class Clearing:
    """Central counterparty ledger. It holds no positions; it guarantees variation margin and records
    any shortfall it could not collect."""

    fees_collected: int = 0
    uncovered_losses: int = 0     # currently outstanding, owed by defaulted accounts
    recovered: int = 0
    default_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Explicit unfunded clearing advance; a receivable is not collected cash.
        d.update(position=0, funding_cash_cents=-self.uncovered_losses,
                 default_receivables_cents=self.uncovered_losses,
                 loss_if_receivables_unrecoverable_cents=self.uncovered_losses)
        return d


def margin_cents(contracts: int, multiplier: int, price_cents: int, rate: float) -> int:
    """Round the required marked-notional margin up to the next cent."""
    return int((Decimal(abs(contracts) * multiplier * price_cents) * Decimal(str(rate))).to_integral_value(
        rounding=ROUND_CEILING))


def stake_cents(contracts: int, multiplier: int, stake_per_lot_cents: int) -> int:
    """Fixed money set aside for a bet: units covered times the stake per unit. Independent of price."""
    return abs(contracts) * multiplier * stake_per_lot_cents


def worst_case_position(position: int, bid_qty: int, ask_qty: int) -> int:
    """Largest absolute position reachable if any combination of the open orders fills."""
    return max(abs(position), abs(position + bid_qty), abs(position - ask_qty),
               abs(position + bid_qty - ask_qty))


def settle_cash_shortfall(acct: Account, clearing: Clearing, round_number: int, reason: str) -> None:
    """Keep cash >= 0: move a shortfall into liabilities, and use positive cash to repay liabilities."""
    if acct.cash < 0:
        shortfall = -acct.cash
        acct.cash = 0
        acct.liabilities += shortfall
        acct.defaulted = True
        clearing.uncovered_losses += shortfall
        clearing.default_events.append(dict(round=round_number, agent_id=acct.agent_id,
                                             shortfall_cents=shortfall, reason=reason))
    elif acct.liabilities > 0 and acct.cash > 0:
        repay = min(acct.cash, acct.liabilities)
        acct.cash -= repay
        acct.liabilities -= repay
        clearing.uncovered_losses -= repay
        clearing.recovered += repay


def position_margin(contracts, price_cents, spec, maintenance=False):
    """Money set aside for a bet. The stake is fixed from the opening price and ignores ``price_cents``."""
    del price_cents, maintenance
    return stake_cents(contracts, spec.multiplier, spec.stake_per_lot_cents)
