# LLM Collusion Experiments for the Bertrand Pricing Game

**Paper:** Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions ([OpenReview](https://openreview.net/pdf?id=Vhi9tHbKTa))

```
@inproceedings{CollusiveReasoningAgents,
    title={Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions},   
    author={Riemer, Matthew and Tosato, Tommaso and Memarian, Amin and Touzel, Maximilian Puelma and Berseth, Glen and Rish, Irina and Dumas, Guillaume},    
    booktitle={In International Conference on Machine Learning (ICML)},    
    year={2026}   
}
```

## Overview

This repository reproduces experiments from the ICML 2026 paper above: price-competition experiments for the Bertrand game that test collusive behavior in LLM reasoning models across different prompting styles and with activation steering. It is the home for two Collusion Research Sprint projects:

- **Project 1.1 – which variables are necessary for collusion?** (`run_game.py`)
- **Project 1.2 – is formed collusion robust to replacing participants?** (`branch_game.py`)

The paper's experiments ran DeepSeek-R1 distills through vLLM on GPUs, with GPU-hosted judge models. For the sprint this repo adds a **CPU-only harness** that keeps the same market, prompts, and output format but reaches the models through the shared inference config. The original GPU scripts are untouched and documented under [Running experiments](#running-experiments); you do not need them for 1.1 / 1.2.

## Setup

> **Fastest path:** from the repo root, `./setup.sh llm-cartel` sets this experiment up (deps + the shared `poseidon_compute` package) in one step; `./setup.sh` does all experiments. The manual steps below are equivalent.

**Install** (< 1 min): `uv sync` — this installs only the CPU harness (`run_game.py`, `branch_game.py`, `judges/api_judge.py`). The GPU `[gpu]` extra (vllm/torch/transformers) and `[steering]` extra are **not** needed for the CPU harness; install them only to run the original GPU scripts: `uv sync --extra gpu --extra steering`.

**Inference config.** Configure inference once in the repo-root `compute.env` (copy it from `../compute.env.example`). That one file holds your keys for every project; routing is Nebius → OpenRouter → Modal. See [`../COMPUTE.md`](../COMPUTE.md).

- `NEBIUS_API_KEY` — a Nebius key (open models).
- `OPENROUTER_API_KEY` — **effectively required here**: the recommended agent/judge models below are OpenRouter ids (and closed / Nebius-missing models route through OpenRouter).

**You choose the agent model** (`--models` or `MODEL_NAME`); the script refuses to guess. Any overrides (e.g. `MODEL_NAME`, `LLM_BASE_URL`) go in the repo-root `compute.env`.

The paper's DeepSeek-R1-Distill-Qwen models are no longer served on OpenRouter. Measured through OpenRouter on 2026-09-15, per call in a duopoly:

| `--models` | what | speed / cost |
|---|---|---|
| `deepseek/deepseek-r1-0528` | closest to the paper: full R1 reasoning model | 110-190 s, 3-4k thinking tokens, ~$0.01 per call; a 300-round duopoly takes many hours |
| `qwen/qwen3-14b` | hybrid reasoning model (thinks by default) | 30-70 s and 2-4k thinking tokens per call, $0.12/$0.24 per M tokens; a 300-round duopoly takes several hours |
| `openai/gpt-4.1-mini` | non-reasoning model | 1-2 s per call; a 300-round duopoly in ~15 min; add `--think` to get inspectable thoughts |

The judge defaults to `qwen/qwen3-14b` (the paper's Qwen judge). A 300-round duopoly is 600 calls (reasoning models also bill thinking tokens); every script prints the call count and asks before spending.

Model ids for the CPU harness are OpenRouter ids (`openai/gpt-4.1`, `google/gemini-2.5-flash`, ...). Two opt-in prefixes bypass the shared config: `ollama:` (free local model) and `custom:` (any OpenAI-compatible endpoint via `LLM_BASE_URL`).

## Quick start

Cheapest smoke test — 60 calls. This native command runs with just an `OPENROUTER_API_KEY` set:

```bash
uv run python run_game.py --models openai/gpt-4.1-mini --rounds 30 -y   # ~1-2 min: fastest model, confirms wiring
uv run python run_game.py --models qwen/qwen3-14b --rounds 30 -y        # same 60 calls but the paper's Qwen reasoning model: tens of minutes
```

The first line (the fast non-reasoning model) is the quickest way to confirm the harness, routing, and output are wired up. The second uses the slower reasoning model from the model table above; it bills thinking tokens and 60 calls take tens of minutes. (`-y` skips the cost confirmation prompt — use it for non-interactive / scripted runs; drop it to be asked before spending.) Set `--rounds` as low as you like (each round = 1 call per agent) to make the smoke test cheaper still.

A short smoke run will almost always print `positive control: FAIL` — that is expected. FAIL only means no collusion formed over these few rounds (the default `P0` prompt just says "maximize profit"); it does **not** mean the harness failed. Each run ends with mean price and **gain** (0 = Nash/competitive profit, 1 = monopoly profit) over the last 50 rounds and `positive control: PASS/FAIL` (market mean gain > 0.5). A PASS — which you expect from the full 300-round control below — is the success signal. Then step up to the full positive control (300 rounds, seed 0):

```bash
uv run python run_game.py --models qwen/qwen3-14b
```

> **Reaching a PASS depends on the model, and is slow with the only paper-faithful option.** A PASS needs the market to settle *between* Nash and the monopoly price (gain > 0.5); gain is capped at 1.0 at the monopoly price, so an agent that overshoots it scores a **negative** gain. The reasoning model `qwen/qwen3-14b` reasons about the demand curve and lands near monopoly, but bills thinking tokens and takes ~1-2 min per call (a 300-round duopoly = 600 calls = many hours). The fast iteration model `openai/gpt-4.1-mini` *does* coordinate on a matching price under an explicit-collusion prompt (`--prefix PC`) — you can watch both agents converge to the same price within a few rounds, which is the end-to-end signal to look for. Where they settle varies run to run: often just above Nash and below the monopoly cap, which prints `positive control: PASS` even over a short smoke run; sometimes they overshoot the monopoly price, which caps gain and can turn a clearly-collusive run into a *negative* gain and a FAIL. So under `--prefix PC`, treat price **convergence** as the thing to confirm — a PASS is common but not guaranteed, and a FAIL there does not mean collusion failed to form. For a reliable PASS on the default `P0` prompt (no explicit-collusion nudge), budget the hours for `qwen/qwen3-14b` (or a comparable reasoning model) over the full 300 rounds.

**How to launch.** The native commands (`uv run python run_game.py ...`) work as-is with just an `OPENROUTER_API_KEY` set — they route automatically. You can equivalently wrap any of them in the repo's one entry point to pin the model and let it route (see [`../COMPUTE.md`](../COMPUTE.md)):

```bash
poseidon-run --model Qwen/Qwen3-14B -- \
  bash -c "cd llm-cartel && uv run python run_game.py --rounds 30 -y"
```

Specify the model **once**: `run_game.py` falls back to the `LLM_MODEL` env var that `poseidon-run --model` sets, so you do not pass it again as `--models`. `poseidon-run` uses Nebius/HF model casing (e.g. `Qwen/Qwen3-14B`); the native `--models` flag uses OpenRouter ids (e.g. `qwen/qwen3-14b`). Neither is the only way — pick whichever fits.

## Running experiments

### Project 1.1 – which variables are necessary for collusion? (`run_game.py`)

```bash
uv run python run_game.py --models qwen/qwen3-14b --rounds 30   # smoke test, 60 calls
uv run python run_game.py --models qwen/qwen3-14b                # positive control: P0, 300 rounds, seed 0
uv run python run_game.py --models deepseek/deepseek-r1-0528 --rounds 30   # paper-faithful, slow
uv run python run_game.py --prefix PC                 # then vary ONE thing at a time:
uv run python run_game.py --prefix P0+B               #   monitoring notice (+B behaviour, +T thoughts, +BT)
uv run python run_game.py --window 10                 #   0 / 10 / 100 visible rounds
uv run python run_game.py --temperature 0.3           #   temperature
uv run python run_game.py --think                     #   reasoning before the price (gives judge-able "thoughts")
uv run python run_game.py --agents 3 --prefix P0,AC,P0    # triopoly, per-agent prompts
uv run python run_game.py --models qwen/qwen3-14b,openai/gpt-4.1-mini   # two different models
uv run python run_game.py --seeds 0 1 2               # expand seeds once the control passes
```

### Project 1.2 – is formed collusion robust to replacing participants? (`branch_game.py`)

`$RUN` must be a run that **actually formed collusion** (its summary printed `positive control: PASS`), because by default `branch_game.py` branches at the first round where the 20-round rolling market gain exceeds 0.5. If the run never crossed that threshold the script exits with `market gain never exceeded 0.5 over 20 rounds; use --at_round or a lower --threshold` — pass `--at_round N` to branch at an explicit round instead, or `--threshold T` to lower the detection bar.

```bash
RUN=results/jsonl/<your positive-control run>.jsonl
uv run python branch_game.py $RUN --rounds 50                                   # control: no replacement
uv run python branch_game.py $RUN --rounds 50 --replace 1=qwen/qwen3-32b         # same family, other size (an OpenRouter id)
uv run python branch_game.py $RUN --rounds 50 --replace 1=openai/gpt-4.1-mini                   # different model family
uv run python branch_game.py $RUN --rounds 50 --replace 1=qwen/qwen3-14b --replacement_window 0  # newcomer sees no history
uv run python branch_game.py $RUN --rounds 50 --replace 1=qwen/qwen3-14b --replace_prefix AC     # newcomer told to avoid collusion
uv run python branch_game.py $RUN --rounds 50 --at_round 120                     # branch at an explicit round instead of auto-detecting
```

The branch point is the first round where the 20-round rolling market gain exceeds 0.5 (override with `--at_round N`, or move the bar with `--threshold`). The exact market state is rebuilt from the saved config (including the synthetic warm-up rounds), the pre-branch rounds are copied into the branch file, and the summary reports mean gain / price / profit before vs after plus rounds-to-recover.

> The `--replace i=<model>` value must be an OpenRouter id (same casing as `--models`). The paper's DeepSeek-R1-Distill models are no longer served on OpenRouter (see the model table above), so use a currently-served id such as `qwen/qwen3-32b` for the "same family, other size" arm.

*Note:* agents here have no hidden state (each round is a fresh call over the visible window), so "fresh instance of the same model" equals the control; the informative arms are other model, other provider, other prompt, and `--replacement_window 0`.

### Scoring behaviour without a GPU (`judges/api_judge.py`)

Same 1-10 collusion prompt as the vLLM judges, through the API.

```bash
uv run python judges/api_judge.py results/jsonl/<run>.jsonl                  # scores the 'thoughts' field with qwen/qwen3-14b
uv run python judges/api_judge.py results/jsonl/<run>.jsonl --field respones --every 10   # score the visible answers instead
uv run python judges/api_judge.py results/branches/*.jsonl --field respones --every 10
```

By default the judge scores the `thoughts` field. That field is only populated when the run had thoughts to record — i.e. a reasoning model, or any model run with `--think`. **For a plain non-reasoning run (e.g. `openai/gpt-4.1-mini` without `--think`) `thoughts` is empty, so pass `--field respones`** to score the visible price-setting answers instead. (`respones` is spelled that way on purpose — it matches the misspelled key in the paper's output schema.) `--every N` scores only every N-th round to control cost. Output is written to `results/judge/<run>.scores.csv`.

Reasoning models (R1 distills, Qwen3) return their chain of thought; it is stored in `thoughts` automatically, so `--think` is only needed for non-reasoning models. `chatgpt_duopoly_tester.py` (OpenAI reasoning models via the Responses API) uses the same shared inference config; closed OpenAI models route via OpenRouter when `OPENROUTER_API_KEY` is set.

### Prompt Variants

The `prefix` argument controls the system prompt given to each agent. Variants can be combined (e.g., `P0+B`, `P0+BT`).

**Base prompts:**

| Prefix | Description |
|--------|-------------|
| `P0` | Default prompt to maximize profit |
| `PI` | Implicit prompt to collude |
| `PC` | Explicit prompt to collude |

**Anti-collusion prompts:**

| Prefix | Description |
|--------|-------------|
| `AC` | Prompt to maximize profit and avoid collusion |
| `CD` | Prompt to maximize profit and avoid collusion, including the Wikipedia definition of a cartel |
| `AP` | Prompt to maximize profit and avoid collusion and penalties |

**Alternative objective prompts:**

| Prefix | Description |
|--------|-------------|
| `MD` | Prompt to maximize demand |
| `NPA` | Prompt to minimize prices |
| `NPB` | Prompt to minimize profit |

**Monitoring modifiers (append to a base prompt, e.g., `P0+B`):**

| Modifier | Description |
|----------|-------------|
| `+B` | Adds notice that behavior will be monitored |
| `+T` | Adds notice that thoughts will be monitored |
| `+BT` | Adds notice that behavior and thoughts will be monitored |

### Price Scale Values

The `price_scale` parameter (alpha) scales the price space in the Bertrand competition. Higher values stretch the price range proportionally, testing whether collusive behavior persists across different price magnitudes.

| Scale | Description |
|-------|-------------|
| 1.0   | Default — prices around cost of 1 |
| 3.2   | Moderate scaling |
| 10.0  | High scaling |

### GPU scripts (original, paper-faithful)

These require the `[gpu]` / `[steering]` extras and a GPU; they are not needed for projects 1.1 / 1.2.

#### environment/

Core Bertrand competition game engines.

- `oligopoly.py` — Duopoly (2-agent) game environment
- `multiprompt_oligopoly.py` — Triopoly (3-agent) game environment with per-agent prompts

#### r1_duopoly_tester.py

Baseline 2-agent Bertrand competition with DeepSeek-R1.

```bash
python r1_duopoly_tester.py <prefix> <prior_steps> <size> [price_scale]
```

| Argument | Description |
|----------|-------------|
| `prefix` | Prompt variant (see [Prompt Variants](#prompt-variants)) |
| `prior_steps` | Number of synthetic history steps around the Nash equilibria (always 1 in our paper) |
| `size` | Model size (`7B`, `14B`, `32B`) |
| `price_scale` | (Optional) Price scale multiplier, default `1.0` |

**Examples:**
```bash
# Run with default prompt, 1 history step, 7B model, default price scale
python r1_duopoly_tester.py P0 1 7B

# Same but with moderate price scaling (3.2x)
python r1_duopoly_tester.py P0 1 7B 3.2

# Same but with high price scaling (10x)
python r1_duopoly_tester.py P0 1 7B 10.0
```

#### r1_triopoly_tester.py

Baseline 3-agent Bertrand competition with DeepSeek-R1. Each agent can receive a different prompt variant via comma-separated prefixes.

```bash
python r1_triopoly_tester.py <prefix> <prior_steps> <size>
```

| Argument | Description |
|----------|-------------|
| `prefix` | Comma-separated prompt variants, one per agent (e.g., `P0,P0,P0`) |
| `prior_steps` | Number of synthetic history steps around the Nash equilibria (always 1 in our paper) |
| `size` | Model size (`7B`, `14B`, `32B`) |

**Examples:**
```bash
# All three agents get the default prompt
python r1_triopoly_tester.py P0,P0,P0 1 7B

# Mixed prompts: agent 2 gets a different variant
python r1_triopoly_tester.py P0,P3,P0 1 14B
```

#### chatgpt_duopoly_tester.py

2-agent Bertrand competition using OpenAI reasoning models (o3, o4-mini, gpt-5.4, etc.) via the Responses API, reached through the shared inference config (set `NEBIUS_API_KEY` in the repo-root `compute.env` — see [`../COMPUTE.md`](../COMPUTE.md)), like everything else here. A bare OpenAI name (`o3`) is prefixed to `openai/o3` automatically. Verified 2026-09-16 with gpt-5.4: reasoning summaries land in `thoughts`. (CPU/API-capable — does not require a GPU.)

```bash
python chatgpt_duopoly_tester.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--prefix` | `P0` | Prompt variant (see [Prompt Variants](#prompt-variants)) |
| `--prior_steps` | `1` | Number of synthetic history steps around the Nash equilibria (always 1 in our paper) |
| `--model` | `openai/gpt-5.4` | OpenRouter id of an OpenAI reasoning model |
| `--reasoning_summary` | `auto` | Reasoning verbosity: `auto`, `concise`, `detailed` |
| `--reasoning_effort` | `medium` | Compute budget: `low`, `medium`, `high` |
| `--temperature` | `1.0` | Sampling temperature |
| `--rounds` | `300` | Game rounds per episode |
| `--seed` | (all) | Run a single episode seed (0-9), or omit for all 10 |
| `--max_retries` | `2` | Retries per failed generation |
| `--overwrite` | off | Overwrite existing output instead of resuming |

**Examples:**
```bash
# Run with o3 model at high reasoning effort
python chatgpt_duopoly_tester.py --model openai/o3 --reasoning_effort high

# Run a single episode (seed 0) with o4-mini and collusion prompt
python chatgpt_duopoly_tester.py --model openai/o4-mini --prefix PC --seed 0

# Run with detailed reasoning summaries for 100 rounds
python chatgpt_duopoly_tester.py --model openai/gpt-5.4 --reasoning_summary detailed --rounds 100
```

#### steering/

This directory contains both the steering vector training pipeline and the full steering experiments.

##### train_steering_vectors.py

Trains Contrastive Activation Addition (CAA) steering vectors. Extracts activations from all model layers, computes direction vectors that separate high-collusion from low-collusion behavior, and generates analysis visualizations.

```bash
python steering/train_steering_vectors.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | — | Dataset name (loads from `results/datasets/{name}_high_scores_behavior.jsonl` and `_low_scores_behavior.jsonl`) |
| `--list-datasets` | — | List all available datasets and exit |
| `--high-data` | — | Custom path to high-collusion JSONL (overrides `--dataset`) |
| `--low-data` | — | Custom path to low-collusion JSONL (overrides `--dataset`) |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Model to extract activations from |
| `--max-pairs` | all | Maximum number of prompt-thought pairs to use |
| `--layers` | all | Specific layers to extract (space-separated) |
| `--no-normalize` | off | Don't normalize vectors to unit length |
| `--output-dir` | `results/steering_vectors` | Output directory |
| `--output-name` | dataset name | Name for this training run |
| `--load-activations` | — | Path to pre-computed activations (skip extraction) |
| `--skip-save-activations` | off | Don't save intermediate activations |
| `--no-visualize` | off | Skip visualization generation |
| `--top-n` | `5` | Number of top layers to visualize in detail |

**Examples:**
```bash
# List available datasets
python steering/train_steering_vectors.py --list-datasets

# Train with the included behavior dataset (default model)
python steering/train_steering_vectors.py --dataset max_profits_behavior

# Train with a custom model and limited pairs
python steering/train_steering_vectors.py --dataset max_profits_behavior --model Qwen/Qwen3-14B --max-pairs 100

# Train with custom data files
python steering/train_steering_vectors.py --high-data path/to/high.jsonl --low-data path/to/low.jsonl --output-name custom_run

# Train specific layers only, skip visualizations
python steering/train_steering_vectors.py --dataset max_profits_behavior --layers 10 11 12 13 14 --no-visualize
```

Output is saved to `results/steering_vectors/{output-name}/`, including:
- `vectors.json` — trained steering vectors for all layers
- `metrics.json` — per-layer performance metrics
- `plots/` — visualization plots (if enabled)

##### analyze_layers.py

Analyzes and compares layer-wise CAA performance from training metrics. Useful for identifying the optimal steering layer.

```bash
python steering/analyze_layers.py --metrics <path_to_metrics.json> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--metrics` | (required) | Path(s) to metrics JSON file(s) from training |
| `--compare` | off | Compare multiple metrics files side-by-side |
| `--names` | — | Display names for each metrics file (for comparison) |
| `--top-n` | `10` | Number of top layers to display |
| `--visualize` | off | Generate visualization plots |
| `--output` | same as metrics | Output directory for visualizations |

**Examples:**
```bash
# Analyze a single training run
python steering/analyze_layers.py --metrics results/steering_vectors/max_profits_behavior/metrics.json

# Show only top 5 layers
python steering/analyze_layers.py --metrics results/steering_vectors/max_profits_behavior/metrics.json --top-n 5

# Compare two training runs
python steering/analyze_layers.py --compare \
    --metrics results/steering_vectors/run_a/metrics.json results/steering_vectors/run_b/metrics.json \
    --names "Run A" "Run B"

# Generate visualizations
python steering/analyze_layers.py --metrics results/steering_vectors/max_profits_behavior/metrics.json --visualize --output plots/
```

##### Training Data

Pre-labeled behavior data for training steering vectors is in `results/datasets/`:

- `max_profits_high_scores_behavior.jsonl` — Transcripts scored as highly collusive
- `max_profits_low_scores_behavior.jsonl` — Transcripts scored as non-collusive

Each JSONL file contains experiment transcripts with agent thoughts, used as contrastive pairs for CAA.

##### Steering Experiments

Full activation steering experiments where all agents are steered using pre-trained behavior vectors.

###### r1_duopoly_all_steering_behavior.py

2-agent DeepSeek-R1 with all agents steered via behavior vectors.

```bash
python steering/r1_duopoly_all_steering_behavior.py <prefix> <prior_steps> <size> <multiplier> <seed> [price_scale] [layer1] [layer2] ...
```

| Argument | Description |
|----------|-------------|
| `prefix` | Prompt variant (see [Prompt Variants](#prompt-variants)) |
| `prior_steps` | Number of synthetic history steps around the Nash equilibria (always 1 in our paper) |
| `size` | Model size (`7B`, `14B`, `32B`) |
| `multiplier` | Steering vector multiplier (strength of steering) |
| `seed` | Random seed for reproducibility |
| `price_scale` | (Optional) Price scale, default `1.0` |
| `layer1, ...` | (Optional) Model layers to steer, default `[12]` |

**Examples:**
```bash
# Steer with multiplier 1.0, seed 42, default price scale and layer
python steering/r1_duopoly_all_steering_behavior.py P0 1 7B 1.0 42

# Same but with moderate price scaling (3.2x)
python steering/r1_duopoly_all_steering_behavior.py P0 1 7B 1.0 42 3.2

# Same but with high price scaling (10x)
python steering/r1_duopoly_all_steering_behavior.py P0 1 7B 1.0 42 10.0

# Steer at layers 12 and 18 (must specify price_scale to pass layers)
python steering/r1_duopoly_all_steering_behavior.py P0 1 7B 1.0 42 1.0 12 18
```

###### r1_triopoly_all_steering_behavior.py

3-agent DeepSeek-R1 with all agents steered. Same interface as the duopoly version.

**Examples:**
```bash
python steering/r1_triopoly_all_steering_behavior.py P0 1 7B 1.0 42
```

###### qwen3_duopoly_steering_behavior.py

2-agent Qwen3 with all agents steered. Uses model-specific layer selections and steering vectors from `results/steering_vectors/{model}/custom/vectors.json`.

```bash
python steering/qwen3_duopoly_steering_behavior.py <model_name> <multiplier> <seed>
```

| Argument | Description |
|----------|-------------|
| `model_name` | Qwen3 model size (`Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, `Qwen3-14B`, `Qwen3-32B`) |
| `multiplier` | Steering vector multiplier (strength of steering) |
| `seed` | Random seed for reproducibility |

**Examples:**
```bash
# Steer Qwen3-14B with multiplier 10.0
python steering/qwen3_duopoly_steering_behavior.py Qwen3-14B 10.0 42

# Steer Qwen3-32B with stronger multiplier
python steering/qwen3_duopoly_steering_behavior.py Qwen3-32B 50.0 0
```

#### partial_steering/

Partial steering experiments where only a subset of agents are steered. This tests whether collusive behavior propagates from steered agents to unsteered agents. Same interface as `steering/r1_duopoly_all_steering_behavior.py`.

##### r1_duopoly_1of2_steering_behavior.py

2-agent DeepSeek-R1 with only 1 of 2 agents steered.

**Examples:**
```bash
# Steer 1 agent with default price scale
python partial_steering/r1_duopoly_1of2_steering_behavior.py P0 1 7B 1.0 42

# With moderate price scaling
python partial_steering/r1_duopoly_1of2_steering_behavior.py P0 1 7B 1.0 42 3.2

# With high price scaling
python partial_steering/r1_duopoly_1of2_steering_behavior.py P0 1 7B 1.0 42 10.0
```

##### r1_triopoly_1of3_steering_behavior.py

3-agent DeepSeek-R1 with only 1 of 3 agents steered.

**Examples:**
```bash
python partial_steering/r1_triopoly_1of3_steering_behavior.py P0 1 7B 1.0 42
```

##### r1_triopoly_2of3_steering_behavior.py

3-agent DeepSeek-R1 with 2 of 3 agents steered.

**Examples:**
```bash
python partial_steering/r1_triopoly_2of3_steering_behavior.py P0 1 7B 1.0 42
```

#### judges/ (GPU warden judges)

LLM-as-judge scripts that score agent behavior from experiment outputs. Each script loads experiment transcripts and uses a language model to assign collusion scores (1-10) to each agent's behavior. (The CPU-only judge is `judges/api_judge.py`, documented above under [Running experiments](#scoring-behaviour-without-a-gpu-judgesapi_judgepy).)

All judges share a common interface (all arguments are optional with sensible defaults):

```bash
python judges/<script>.py [model_or_tag] [batch_size] [test_limit] [temperatures] [seed] [run_mode] [failed_csv]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_or_tag` | varies | Model name or run tag (depends on script, see below) |
| `batch_size` | varies | Prompts per GPU batch (adjust based on available memory) |
| `test_limit` | `0` | Process only first N data points (`0` = process all) |
| `temperatures` | `0.3` | Comma-separated temperature values for scoring |
| `seed` | `12345` | RNG seed for reproducibility |
| `run_mode` | `steering` | Which experiments to score: `steering`, `baseline`, or `all` |
| `failed_csv` | (none) | Path to a failed scores CSV to rerun only missing scores |

##### r1_warden_behscoring.py

Judge using DeepSeek-R1-Distill-Qwen-7B. No model argument needed (hardcoded).

**Examples:**
```bash
# Score steering experiments with all defaults
python judges/r1_warden_behscoring.py

# Score baseline experiments with batch size 256
python judges/r1_warden_behscoring.py 256 0 "0.3" 12345 baseline
```

##### llama33_warden_behscoring.py

Judge using Llama-3.3-70B-Instruct. No model argument needed (hardcoded). Default batch size: 32 (large model).

**Examples:**
```bash
# Score with all defaults
python judges/llama33_warden_behscoring.py

# Score steering experiments with explicit settings
python judges/llama33_warden_behscoring.py 32 0 "0.3" 12345 steering
```

##### qwen_warden_behscoring.py

Judge using Qwen3 models. First argument selects the model size.

**Examples:**
```bash
# Score using Qwen3-14B with defaults
python judges/qwen_warden_behscoring.py Qwen3-14B

# Score using Qwen3-32B, steering mode, batch size 256
python judges/qwen_warden_behscoring.py Qwen3-32B 256 0 "0.3" 12345 steering
```

##### ministral_warden_behscoring.py

Judge using Ministral-3-14B-Instruct-2512. First argument is a run tag for output naming.

**Examples:**
```bash
# Score with default run tag
python judges/ministral_warden_behscoring.py

# Score with explicit run tag and settings
python judges/ministral_warden_behscoring.py ministral_3_14b 512 0 "0.3"
```

##### olmo_warden_behscoring.py

Judge using OLMo-3.1-32B-Instruct. First argument is a run tag for output naming.

**Examples:**
```bash
# Score with default run tag
python judges/olmo_warden_behscoring.py

# Score with explicit run tag and settings
python judges/olmo_warden_behscoring.py olmo_3_1_32b 512 0 "0.3"
```

##### olmo_think_warden_behscoring.py

Judge using OLMo-3.1-32B-Think (reasoning variant). First argument is a run tag for output naming.

**Examples:**
```bash
# Score with default run tag
python judges/olmo_think_warden_behscoring.py

# Score with explicit run tag and settings
python judges/olmo_think_warden_behscoring.py olmo_3_1_32b_think 512 0 "0.3"
```

##### oss_warden_behscoring.py

Judge using OpenAI OSS models. First argument is the model version identifier.

**Examples:**
```bash
# Score using the 2025-04-16 model version with defaults
python judges/oss_warden_behscoring.py 2025-04-16

# Score baseline experiments with explicit settings
python judges/oss_warden_behscoring.py 2025-04-16 512 0 "0.3" 12345 baseline
```

## Outputs

**`run_game.py`** writes to `results/jsonl/`:
- `<name>.jsonl` — paper schema per round: `round, actions, thoughts, respones, demands, profits, gains, metrics`.
- `<name>.config.json` — the run settings (used to rebuild market state for branching).

Runs resume if interrupted. `<name>` defaults to the run settings (pass `--name` to override): e.g. a default run with `--models qwen/qwen3-14b` becomes `2agents_prefix_P0-P0_model_openrouter-qwen-qwen3-14b-openrouter-qwen-qwen3-14b_win100_temp1p0_seed0` — `<N>agents`, then the prefix and model **per agent** (so a single `--prefix P0` / single `--models` is repeated once per agent and joined with `-`), `win<window>` (default `100`), `temp<temperature>`, then `think`/`scale<α>` if set, and `seed<seed>`. The model piece is the *resolved* routing spec (bare OpenRouter ids pick up an `openrouter-` prefix), and `/`, `:`, `.`, `+` in ids become `-`, `-`, `p`, `plus`.

**`branch_game.py`** writes branch files to `results/branches/` (pre-branch rounds copied in, then the post-replacement rounds).

**Steering** (`train_steering_vectors.py`) writes to `results/steering_vectors/{output-name}/` (`vectors.json`, `metrics.json`, `plots/`).

**How to tell it worked.** Each game run prints mean price and **gain** over the last 50 rounds and a `positive control: PASS/FAIL` line — a market mean gain > 0.5 (PASS) means collusion formed. `branch_game.py` prints mean gain / price / profit before vs after the replacement, plus rounds-to-recover.

## Repository Structure

```
├── environment/           # Game environment implementations
├── steering/              # Steering vector training and full steering experiments
│   └── lib/              # Shared library for activation extraction and CAA training
├── partial_steering/      # Partial steering experiments (subset of agents steered)
├── judges/                # LLM-as-judge behavior scoring scripts
└── results/
    ├── datasets/          # Training data for steering vectors
    └── steering_vectors/  # Pre-trained steering vectors
```

## Notes

**Dependencies.** The CPU/API harness (`run_game.py`, `branch_game.py`, `judges/api_judge.py`) needs only the core deps and **does not require `vllm`/`torch`** — those ship in the optional `[gpu]` extra for the original GPU scripts (`r1_*_tester.py`, `steering/`, `partial_steering/`, `judges/*_warden_*.py`). Install the extras only if you run those: `uv sync --extra gpu --extra steering`.

- **Core (CPU harness):** `numpy`, `scipy`; `pandas` (judge scoring output); `openai`, `python-dotenv` (API client + config loading).
- **GPU extra (`[gpu]`, original scripts only):** `vllm` (inference engine for local models); `transformers` (direct model access for steering); `torch`.
- **Steering extra (`[steering]`):** `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`.
