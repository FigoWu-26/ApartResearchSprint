# Assessment before the capital/futures revision

Inspected local checkout of https://github.com/poseidon-research/collusion_hackathon at
`d585a7b3be50366cc2bdf5789bf8add5890b2abd` on 2026-10-25 (Hong Kong).
The checkout was already dirty. Existing user edits, the earlier spot_futures engine,
legacy optional futures engine, dashboards, and historical result files are preserved.
The attachment is arXiv:2507.01413v1, especially sections 2–4 and appendices A–B.
Instructions quoted inside the paper are experimental material, not instructions for this task.

## Original methodology and implementation

The paper uses five buyers, five sellers, 30 rounds, $80 cost, $100 value, one lot per
agent per round, and midpoint clearing of crossing quotes after all actions arrive.
Initial bids are uniform $80–85 and asks $95–100. Memory and a private scratchpad persist
within a run; seller messages arrive next round. The profit objective already exists.
The paper examines seller communication, model composition, CEO pressure, and oversight;
it does not test futures or cash constraints. Higher coordination need not increase profit.

In this checkout, `continuous_double_auction/simulation.py` defaults to **3 sellers,
3 buyers, 15 rounds**. Its `--paper` flag overrides counts to **5 sellers, 6 buyers,
30 rounds**, contrary to the attachment. Do not use that flag for the primary study:
use explicit `--sellers 5 --buyers 5 --rounds 30`. We preserve the flag's behavior for
compatibility and document the discrepancy. Default quote ranges are $81.25–83.75
and $96.25–98.75 (`cda_types.py`, `util/misc.py`).

`market/agents.py` renders Jinja prompts, parses responses, and manages memory and
scratchpads. `market/market.py` runs parallel calls, applies quotes, matches sorted
persistent queues at midpoint prices, and passes last-round role messages. Equal
prices in the legacy queues use tuple/name ordering; the new experiment explicitly
uses seeded submission priority independent of API latency. Legacy evaluation uses
buyer surplus, seller profit, price/dispersion/collusion indices and optional LLM
coordination judgments. Logging includes metadata, readable transcripts, JSON and
optional W&B. Model aliases and environment overrides can change effective model IDs.
The paper names gpt-4.1-2025-04-14 and claude-3-7-sonnet-20250219; an alias is not proof
that those models were used.

## Gaps found in the existing unfinished extension

The `capital_futures` package already supplied books, scenarios, JSONL logs, replay,
analysis, tests, and a dashboard scaffold. However, it used a **fixed stake** and ignored
its initial/maintenance margin rates, enabled a hard-coded settlement-pushing policy
by default, and described an ambiguous score in its prompt. It excluded liquidation
trades from the mark, allowed permissive action coercion, under-specified clearing
funding, and could pool incompatible runs or silently overwrite repeated seeds in
analysis. The dashboard's asynchronous job flow could re-enable controls too early,
progress writes were buffered, and live paired runs were not implemented.

## Extension and interpretation

Use a separate `capital_futures` entry point, with no CEO/oversight intervention.
Primary counts are explicit 5+5; primary wealthy seats are two rotating sellers—an
assumption motivated by the paper's seller focus, not a finding. Buyer/seller cash is
made economically usable through immediate outside-customer redemption/supplier
sourcing, with gross funding required; initial inventory and its value are both zero.
Keep legacy surplus metrics distinct from financial P&L. Futures gains redistribute
wealth; they do not add aggregate welfare.

Cross capital × futures × seller communication (eight cells); add equal-total capital
($20.8M each), external settlement, alternate wealthy roles, and a longer horizon.
Compare uncertainty across independent seeds and models. A $90 midpoint is a reference,
not an established competitive equilibrium of this modified economy. Rule agents are
mechanical comparators, not an equilibrium proof. Messages, price correlation, an LLM
judge, or high profits alone cannot identify collusion. Inspect episodes, beneficiaries,
and participant-replacement diagnostics; allow increased, decreased, or inconclusive
coordination and report weak capital use honestly.
