# focal-points (hackathon edition)

Schelling-style coordination / focal points for LLMs, measured as a **coordination index**.

**Paper:** Aharon, La Malfa, Wooldridge & Kraus, *Tacit Coordination of Large Language Models* ([arXiv:2601.22184](https://arxiv.org/abs/2601.22184))

Official upstream code: [github.com/EmanueleLM/focal-points](https://github.com/EmanueleLM/focal-points).

## Overview

LLMs are asked Schelling-style coordination questions ("pick a number between 1 and 100",
"where would you meet a friend in Paris?", the Bardsley Amsterdam / Nottingham
pick-guess-coordinate tasks, a bargaining table). Sampling many answers per prompt gives a
**coordination index**: the probability that two independent answers agree, i.e. how strong
the focal point is.

In the Collusion Research Sprint this is the cheap *benign* control — agents converging on the
same answer **without** any communication or agreement. Use it as:

- **Replacement (1.2):** the convention-recovery baseline when a participant is replaced — does a
  "newcomer" model land on the same focal point?
- **Detector calibration (1.3):** the negative example for detectors — the `results/` counts are
  agreement-without-communication, which a collusion detector should score low.
- **Swap detection (3.3):** the cheap convention task for swap detection — alternate two models over
  a sequence of prompts and check whether the distribution shift is detectable.

## Setup

> **Fastest path:** from the repo root, `./setup.sh focal-points` sets this experiment up (deps + the shared `poseidon_compute` package) in one step; `./setup.sh` does all experiments. The manual steps below are equivalent.

**Install** (< 1 min): `uv sync`.

Inference is configured once in the **repo-root** `compute.env` (one level up from this folder; copy
it from `../compute.env.example`): set `NEBIUS_API_KEY` for open models (the default), and add
`OPENROUTER_API_KEY` for closed ones (GPT / Gemini / Claude) — see [`../COMPUTE.md`](../COMPUTE.md).
The shared config picks **Nebius → OpenRouter → Modal** automatically. Any overrides (e.g.
`MODEL_NAME`) also go in that same repo-root `compute.env`.

The paper ran open models locally on GPUs (transformers) plus OpenAI / Gemini batch jobs. Here every
model call goes through the shared inference config, so it runs on any laptop. The default model is the
paper's default, `meta-llama/Llama-3.3-70B-Instruct` (the value baked into the code and used to name the
shipped reference folder); any model id works and the paper's HuggingFace ids are mapped automatically.
The output folder is the model string **verbatim**, so keep this exact casing if you want a run to line
up with (or be guarded against overwriting) the shipped `results/meta-llama/Llama-3.3-70B-Instruct/`.

## Quick start

Calls per run = prompts × samples; the script prints the count and asks before spending.

```bash
uv run python main.py -m openai/gpt-4o-mini --max-prompts 3 -s 5   # smoke test: 15 calls
```

This works with just an `OPENROUTER_API_KEY` set. The cost prompt only blocks on an interactive
terminal; in a non-interactive shell (pipes, CI) the run auto-proceeds, so the command above runs as-is.
Pass `-y` / `--yes` to skip the prompt explicitly.

> **Output folder is named after `-m`, not after the model that actually answers.** Always pass
> `-m <model-id>` so the label matches the model you run. If you omit `-m`, the folder defaults to the
> paper's `meta-llama/Llama-3.3-70B-Instruct`, so running a *different* model without `-m` would write
> into `results/meta-llama/Llama-3.3-70B-Instruct/` and **overwrite the shipped paper reference data**.
> As a safety net the run refuses to generate into an existing `results/<model>/…` folder unless you
> pass `--overwrite`; re-run with the correct `-m` instead of using `--overwrite` to fix a mismatch.

Success signal: the run writes `logs/<model>/<dataset>_responses_<tag>.jsonl` and a
`results/<model>/<dataset>_<tag>.jsonl` with per-prompt answer counts and a coordination index.

## Running experiments

`uv run python main.py …` runs directly off the repo-root `compute.env`:

```bash
uv run python main.py                                      # schelling: 18 prompts × 10 samples = 180 calls
uv run python main.py -s 30                                # paper's 30 samples per prompt
uv run python main.py -m meta-llama/Llama-3.1-8B-Instruct  # a cheaper open model
uv run python main.py -m openai/gpt-4o qwen/qwen2.5-72b-instruct   # several models (needs OPENROUTER_API_KEY for gpt-4o)
uv run python main.py -d amsterdam nottingham -p problem-pick problem-guess problem-coordinate   # Bardsley tasks
uv run python main.py --temperature 0.7 --seed 1           # sampling controls
uv run python main.py -d bargaining_table_realdata --bargaining-player blue                     # bargaining table
uv run python main.py -m openai/gpt-4o-mini -y             # -y/--yes: skip the cost prompt (non-interactive)
uv run python main.py -m openai/gpt-4o-mini --overwrite    # replace this model's own existing results
```

`-y` / `--yes` skips the interactive cost confirmation (use it for scripted / CI runs). `--overwrite`
lets a run replace an already-present `results/<model>/<dataset>_<tag>.jsonl`; without it the run refuses
to generate into a folder that already holds results, which protects the shipped reference data.

You can equivalently route through the repo's one entry point,
`poseidon-run --model <id> -- uv run python main.py …`, which picks Nebius → OpenRouter → Modal
automatically (see [`../COMPUTE.md`](../COMPUTE.md)); neither is the "only" way.

Prompts live in `data/amsterdam_nottingham_schelling/*.jsonl` and `data/bargaining_table_llms/*.json`;
human baselines in `data/Bardsley-humans/` and `data/Dor-humans/`.

## Outputs

Per model:

- `logs/<model>/<dataset>_responses_<tag>.jsonl` — every raw answer; runs resume from it.
- `results/<model>/<dataset>_<tag>.jsonl` — answer counts + coordination index per prompt.
- `images/<model>/<dataset>/<tag>/*.png` — bar plots.

Here `<tag>` is the **problem tag** (`-p` / `--problem-tag`, default `problem`; for Bardsley it is
`problem-pick` / `problem-guess` / `problem-coordinate`, and for bargaining it expands to the variant
keys in the dataset json) — not a timestamp.

It worked if `results/` holds a coordination index per prompt and `images/` holds the bar plots.

## Analysis

Comparison data from the paper is kept where small (`results/` for the 70B Llama / 72B Qwen / gpt-oss
/ GPT / Gemini runs, `plots/`); the bulky raw transcripts and batch files (~1.8 GB) were removed from
this copy and are available in the [upstream repository](https://github.com/EmanueleLM/focal-points).

## Notes

- Local GPU inference (`--backend original`, `uv sync --extra gpu`) and the OpenAI / Gemini batch
  scripts are untouched but not needed for the hackathon path.
- Extra batch entry points: `bash experiments.sh` runs experiment batches;
  `src/bargaining_table/bargaining_table.py` (and `.sh`) generate bargaining-table results;
  `src/plotting/plot_bargaining_table_averages.py` plots their averages.
- Exploratory notebook `src/bargaining_table/bargaining_agent.ipynb` steps through a single
  bargaining agent interactively. Jupyter isn't in the base install, so run it with the `dev`
  extra: `uv sync --extra dev` then
  `uv run jupyter nbconvert --to notebook --execute --inplace src/bargaining_table/bargaining_agent.ipynb`
  (or open it in `uv run jupyter lab`).
- `pyproject.toml` / `uv.lock` are the `uv` dependency set (the install path for all local use); `requirements.txt` is kept **only** for the optional `Dockerfile` (containerized GPU repro, which pins CUDA-indexed torch);
  `Dockerfile` provides a containerized environment.
