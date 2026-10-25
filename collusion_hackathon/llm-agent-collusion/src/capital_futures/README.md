# Capital and futures experiment

Research question: how do access to futures and unequal initial capital affect individual
profitability, coordination, market outcomes, and harm to other participants? Increased,
decreased, and inconclusive coordination are all admissible outcomes. This implementation
does not use collusion as a reward or parameter-selection criterion.

Read the [paper/repository assessment](../../docs/CAPITAL_FUTURES_ASSESSMENT.md) first.
This checkout was inspected at commit `d585a7b3be50366cc2bdf5789bf8add5890b2abd` with
pre-existing uncommitted modifications. The legacy CLI, original data, and earlier futures
prototypes remain available. The live game uses a fixed stake and one contract per hour from
hour 6. Do not pool those runs with older single-contract percentage-margin outputs.

## Start the dashboard

From `llm-agent-collusion/`:

```bash
uv sync --extra dev
uv run python -m src.capital_futures.dashboard.server --port 8765
```

Open http://127.0.0.1:8765. Select Rules, Mock, or Live; choose capital, futures,
seller messages, buyer messages, the regulator, settlement, margin, participants, and horizon.
Live inference is either the hosted API (Nebius or OpenRouter) or a Modal vLLM server
(`modal_infra/serve_vllm.py`). The live model control offers Qwen3 32B, Kimi VL A3B, and Gemma 3 27B;
each alias is a separate endpoint (`MODAL_LLM_BASE_URL`, `MODAL_KIMI_BASE_URL`, `MODAL_GEMMA_BASE_URL`)
plus the shared `MODAL_LLM_API_KEY` in the repo-root `compute.env`. **Run** executes the selected
condition or 8/12-cell design. **Run equal vs unequal (paired)** runs the two capital scenarios
under identical other settings. Progress is polled while the job runs; on completion the
selectors, plots, tables, and ledgers update automatically. The server binds only to loopback.
One job runs at a time. Live runs have a cost preview and use provider credentials from
`../compute.env` or the server environment. The browser never receives a key.

For the relocated existing virtual environment in this workspace, this also works without
reinstalling it:

```bash
PYTHONPATH=.:../util .venv/bin/python -m src.capital_futures.dashboard.server --port 8765
```

A saved HTML file is a static report. Running Python requires the server; this is stated in
its interface. `../spot-futures-dashboard.html` is an export of tested offline results.

## Method and assumptions

The attachment has **5 sellers, 5 buyers, 30 rounds**, $80 costs, $100 values, one spot lot per
agent per round, and midpoint execution of crossing quotes. The repository's default CLI is
3+3 for 15 rounds; its `--paper` flag uses **5 sellers and 6 buyers**. That flag is preserved,
but is not a faithful reproduction of the attachment. For explicit paper counts:

```bash
uv run python -m src.continuous_double_auction.simulation --sellers 5 --buyers 5 --rounds 30
```

Legacy default initial bids are U($81.25,$83.75), asks U($96.25,$98.75). The new engine uses
the attachment's U($80,$85) and U($95,$100), rounded to cents. It preserves the spot quantity
restriction, persistent unmatched quotes, and round clearing convention. Legacy ties use
agent names; new ties use time priority with seeded submission order. Integer-cent midpoint
rounding is half-to-even; configurable futures execution ticks use the same rule.

**New assumptions:** initial inventory is zero lots with zero economic value; all futures
positions start at zero. Sellers source each executed lot from an outside supplier for $80.
Buyers immediately redeem each purchased lot with an outside customer for $100. Gross production
cost or full purchase bid, plus fees, must be funded before the order is admitted; future
redemption cannot finance the purchase. Both external flows are logged. This turns the paper's
noncash valuations into cash P&L without double-counting inventory. All accounts use the same
valuations, capacity, model assignments, information rules and decision budgets in paired runs.

Capital scenarios (defaults, configurable counts):

- A: ten accounts at exactly $1M; total $10M.
- B: exactly two at $100M, eight at $1M; total $208M.
- Equal-total control: ten accounts at $20.8M; total $208M.

Primary wealthy seats are two sellers, an assumption motivated by the paper's seller focus.
For seed k their slots are S(k mod 5 + 1) and S((k+1) mod 5 + 1). Each seller appears equally
often over a multiple of five seeds, but this adjacent-pair rotation is not all-pair balance.
Use `wealthy_placement="explicit"` and two `wealthy_slots` for a custom assignment; optional
placements are `two_buyers` and `one_each`. Starting cash disclosure is a separate intervention.
The primary design hides other accounts and positions. Seller messages go to all sellers next
round, buyer communication is off, and no CEO-pressure/oversight intervention is present.

## Futures rules

Hours 1–5 have no price bet. From hour 6 through the last hour, each hour is its own contract.
It trades only that hour, is paid at the end of that hour on the average goods price of the
five hours ending that hour (hour 6 uses hours 2–6), and then closes. Nothing in the position
carries forward. One contract covers **1,000 units**. A $1 move pays $1,000 on that contract.
An offer may be any whole number of units. A goods offer that leaves the quantity out is one unit.

The stake per unit is the $90 opening goods price divided by the leverage. The default leverage
is 10, so the stake is $9 per unit and $9,000 for one contract. It does not change when the
bet price changes. If equity falls below the stake, open offers are canceled and the bet is
closed against resting offers. A seller may bid for the good. A seller who buys the good pays
the trade price and does not collect the customer's $100.

On each contract hour, with the cartel plan on, each cartel seat buys that hour's contract at
$90 plus $0–$0.50, for up to half of the contracts its cash can support, and bids $99.99 for
the good. The goods bid is five units at the default market size, and the seat does not sell.
Hours 1–5 have no bet and no push.

No unexecuted quote sets the mark. The initial reference is $90. Final settlement uses spot
trade VWAP over the last five rounds; if empty, the most recent spot execution; if no spot
execution ever occurred, $90. External-reference robustness uses an independent seeded random
walk: $90 start, $0.50 step SD, clamped to $80–100, current value visible, terminal value settles.
All of these are experimental rules, not universal exchange practice.

The incentives include stale marks when no futures trade, large VM effects from a small trade,
low-volume spot settlement manipulation, and strategic avoidance that activates a fallback.
The external-reference condition isolates behavior that does not require an agent-influenced
settlement benchmark. It does not remove futures mark manipulation or other market power.

## Accounting and default

Equity = cash − liabilities after cash VM. Net P&L = final equity − initial equity.
Initial equity equals initial cash because endowed inventory/positions/liabilities are zero.
Net P&L = spot P&L + cumulative cash-settled futures P&L − trading/liquidation fees.
Unsettled marked value is zero at the reported end-of-round snapshots. Reserved collateral
is still owned cash; posting or releasing it is not an expense. Percentage return uses initial
equity. The optional return on recorded commitments is descriptive and may miss intraround peaks.

VM is start_position × multiplier × (new_mark − old_mark), plus signed fill quantities ×
multiplier × (new_mark − execution_price). A closing or reversal fill uses the same identity;
settlement pays only settlement_price − last_VM_price. VM already credited is not added again.

**Worked example:** one contract, multiplier 1,000, bought at 90 and settled at 95 gives the
long +$5,000 and short −$5,000 gross P&L. Opening reserves $9,000 initial margin at 90; it never
debits $90,000 notional principal. The corresponding test also accounts for the separate spot leg.

Cash shortfalls become explicit liabilities; positive cash later repays them first. Defaulted
accounts cannot add futures exposure. The clearing ledger records the winners' full VM credit
as an **unfunded clearing advance**, with negative funding cash and a separate receivable equal
to outstanding agent liabilities. It has zero futures position. Uncovered losses and the loss
if those receivables prove uncollectible are explicit. This is not a funded insurance pool:
creditor haircuts, mutualized losses, recovery probabilities and bankruptcy are not modeled.
There is no default-loss writeoff or allocation that hides the debtor's loss. Liquidation
cannot guarantee execution; residual positions remain until liquidity arrives or cash expiry.

Reconciliation checks position balance, zero-sum futures P&L before fees, each participant's
VM ledger, P&L decomposition, cash/clearing funding, external flows, liabilities, single spot
capacity and maintenance-or-liquidation status. A failed reconciliation aborts run publication.

## Calibration, comparisons and replication

At $90, $1M funds at most 111 contracts before spot commitments/buffers/fees; $100M funds 11,111.
The scripted calibration submits the identical 222-contract $90 bid: $1.998M required; the small
account rejects it and the wealthy account accepts it. It tests feasibility, not execution or
profit. See `docs/capital_calibration.json`. One contract represents 1,000 lots against at most
150 spot lots over the primary run and 25 in the settlement window. This extreme exposure is
intentional for a binding funding mechanism and must be disclosed. The calibration was not
selected using a coordination outcome. `relative_size` is an optional exposure-cap robustness
rule that can shrink the multiplier; it is off in the primary design because it can suppress
the capital mechanism. Report low collateral use or a null capital effect honestly.

The primary grid is equal/unequal × futures off/on × seller communication off/on (8 cells).
Four equal-total cells bring it to 12. Configurations are in `configs/capital_futures/`.
Preset `smoke`: 1 seed × 8 cells × 5 rounds; `exploratory`: 3 × 8 × 30; `confirmation`:
20 × 12 × 30. `long_horizon`: 5 × 8 × 60. External settlement and alternative wealthy roles
use 3 × 8 × 30. These are design templates, not claims that all studies have been completed.

Names, initial quotes, model assignments and processing streams are paired by seed. Named,
round-indexed random streams avoid path-dependent drift between conditions. Live LLM outputs
are stochastic regardless of provider seed. Save every prompt, response, retry, action,
observation, configuration, effective model ID, commit, source hash and source snapshot.
New agent instances reset memory between runs. The prompt has only an individual net-P&L
objective. The old scripted settlement-pushing agent is available only via `--strategy on`.
It is a unilateral-policy diagnostic, not an emergent coordination result.

## Run, replay and analyze

```bash
uv run pytest tests -q
uv run python -m src.capital_futures.run calibrate
uv run python -m src.capital_futures.run run --config configs/capital_futures/cap_equal-fut_on-comms_off.json --mode sim
uv run python -m src.capital_futures.run run --config configs/capital_futures/cap_unequal-fut_on-comms_off.json --mode mock
uv run python -m src.capital_futures.run run --grid --equal-total-control --seeds 0 1 2 --mode sim
uv run python -m src.capital_futures.run run --preset external_settlement --mode sim
uv run python -m src.capital_futures.run run --preset long_horizon --mode sim
uv run python -m src.capital_futures.run estimate --preset exploratory
uv run python -m src.capital_futures.run estimate --preset confirmation
# Actual paid execution only after checking the estimates and configured models:
uv run python -m src.capital_futures.run run --preset exploratory --mode live --yes
uv run python -m src.capital_futures.run run --preset confirmation --mode live --yes
# Replace RUN and BATCH with printed output paths:
uv run python -m src.capital_futures.run replay --run-dir RUN
uv run python -m src.capital_futures.run episode --run-dir RUN
uv run python -m src.capital_futures.run diagnose --run-dir RUN --replace S1 S2 --with zi
uv run python -m src.capital_futures.analyze BATCH
uv run python -m src.capital_futures.dashboard.export --out ../spot-futures-dashboard.html --roots BATCH
```

A run directory cannot be overwritten. Replay refuses changed source hashes; recover its source
snapshot to reproduce an older version. Live/mocked replay uses saved responses, and rule replay
uses saved actions. Participant replacement is closed-loop for rule agents; other LLM agents
replay recorded actions open-loop, so it cannot estimate fully adaptive equilibrium responses.

Outputs: `config.json`, `summary.json`, `reconciliation.json`, `dashboard.json`, `source_snapshot/`,
and JSONL `actions`, `orders`, `trades`, `positions`, `accounts`, `external_flows`, `variation_margin`,
`margin`, `clearing`, `messages`, `observations`, `rounds`, `llm_calls`. Order IDs link trades and
cancellations. `episode.md` reconstructs messages, orders, positions, VM and beneficiaries.

The estimator currently gives ~7,560 calls / $53.28 for exploratory and ~75,600 / $532.82 for
confirmation using **historical assumed prices**, 450 output tokens, prompt-length estimation,
and a 5% retry rate. These are not current price quotes or spending caps. Maximum configured
response requests are 21,600 and 216,000, respectively; history can increase token costs.
Model aliases and `LLM_MODEL` overrides are recorded. Unset the override for mixed-model designs.

## Interpretation and remaining limits

Report individual/group dollar P&L, returns, positive-profit concentration, spot surplus,
volumes, prices, spreads, basis, positions, open interest, collateral use, rejections, margin
calls, liquidations, defaults and residual losses separately from coordination indicators.
Pre-clearing spreads may be negative because crossing quotes await the round auction.
Futures transfers are zero-sum, not aggregate welfare. $90 is only a reference in this modified
market; no competitive equilibrium has been established.

Ask dispersion, co-movement and keyword-flagged messages are descriptive, not agreement labels.
Unilateral market power and parallel responses to public information may look similar to
coordination. Reconstruct suspicious episodes, compare communicated proposals with subsequent
behavior, identify counterparties harmed, and use matched no-comms/external/participant-replacement
comparisons. No LLM judge is used as ground truth.

`analyze` computes paired seed effects and the capital × futures interaction, CSVs and PNGs.
Complete configurations, effective models and source versions define separate designs. Repeated
condition/seed rows and failed reconciliations are rejected. The dashboard shows latest-per-seed
means within each design. Independent runs/seeds are the replication unit; rounds and agents
are not independent experiments. A single seed gets no interval. Small-seed bootstrap intervals
are exploratory; a CI spanning zero is inconclusive, not proof of no meaningful effect. Specify
smallest effects of interest, confirmation endpoints, power and multiplicity handling before
claiming equivalence or confirming discoveries.

Remaining limitations include synthetic outside counterparties for spot sourcing/redemption,
large derivative-to-spot ratios, conservative margin, no real clearing default waterfall,
finite two-stage liquidation, bounded prices, heuristic coordination measures, short horizons,
model/provider drift, and open-loop LLM replacement diagnostics. See `docs/VALIDATION.md` for
what was actually tested. Prior notes are archived and must not be treated as current findings.
