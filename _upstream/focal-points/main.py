"""
Focal-point / coordination experiments (hackathon edition).

Poses Schelling-style coordination problems (and the Amsterdam / Nottingham
Bardsley tasks, and the bargaining table) to an LLM, samples several answers per
prompt and computes a coordination index (how often two random answers agree).

All model calls go through the shared inference config (Nebius or OpenRouter, with
OpenRouter as fallback for closed/unavailable models), set via the repo-root
`compute.env`; the default model is the paper's default,
meta-llama/Llama-3.3-70B-Instruct.

    python main.py                                   # schelling, 18 prompts × 10 samples = 180 calls
    python main.py -s 30                             # paper's 30 samples per prompt
    python main.py -m meta-llama/llama-3.1-8b-instruct      # any OpenRouter id
    python main.py -d amsterdam -p problem-pick problem-guess problem-coordinate
    python main.py --max-prompts 3 -s 5              # tiny smoke test (15 calls)
    python main.py --temperature 0.7 --seed 1        # sampling controls

Outputs: logs/<model>/<dataset>_responses_<tag>.jsonl (raw answers, resumable),
results/<model>/<dataset>_<tag>.jsonl (answer counts + coordination index),
images/<model>/<dataset>/<tag>/*.png (bar plots).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


from src.utils.llm import LLM, load_model
from src.utils.prompt import Level0
from src.utils import iterate_data, load_bargaining_table_prompts, plot_block_frequencies

DEFAULT_MODEL = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_names",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="OpenRouter model id(s) (HuggingFace ids from the paper are mapped automatically). "
        "Default: env MODEL_NAME or the paper's meta-llama/llama-3.3-70b-instruct.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="datasets",
        nargs="+",
        default=["schelling"],
        help="Dataset name(s) (without .jsonl).",
    )
    parser.add_argument(
        "-p",
        "--problem-tag",
        dest="problem_tags",
        nargs="+",
        default=["problem"],
        help="Key(s) of the problem in the json data.",
    )
    parser.add_argument(
        "-t",
        "--trials",
        dest="trials",
        type=int,
        default=1,
        help="How many times to call the model per prompt.",
    )
    parser.add_argument(
        "-s",
        "--return-sequences",
        dest="sequences",
        type=int,
        default=10,
        help="Responses (samples) per prompt. Paper: 30. Default 10 to keep the first run cheap.",
    )
    parser.add_argument(
        "--max-prompts",
        dest="max_prompts",
        type=int,
        default=None,
        help="Only run the first N prompts of each dataset/tag (smoke tests).",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: provider default, usually 1.0).",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (default: provider default).",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Sampling seed forwarded to the provider (best effort; not all models honour it).",
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        default="openrouter",
        choices=["openrouter", "original"],
        help="openrouter (default, no GPU) or original (paper's OpenAI/Gemini/local-transformers dispatch).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Skip the cost confirmation (non-interactive runs).",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Allow overwriting an existing results/<model>/<dataset>_<tag>.jsonl. Without this, the "
        "run refuses to clobber already-present results (e.g. the shipped paper reference data under "
        "results/meta-llama/Llama-3.3-70B-Instruct/). Note: the output folder is named after -m, so a "
        "mismatched -m is what puts the reference data at risk.",
    )
    parser.add_argument(
        "-x",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate per response.",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        dest="quantization",
        default=None,
        help="None, 8bit or 4bit.",
    )
    parser.add_argument(
        "-g",
        "--plot-graphs",
        dest="plot_graph",
        type=lambda s: s.lower() == "true",
        default=True,
        help="Whether to plot or not the barplots for each model and problem.",
    )
    parser.add_argument(
        "-r",
        "--reasoning",
        dest="reasoning",
        default=None,
        help='Reasoning effort for API models (e.g., "low", "medium").',
    )
    parser.add_argument(
        "--bargaining-player",
        dest="bargaining_player",
        default="blue",
        type=lambda s: s.lower(),
        choices=["blue", "yellow"],
        help="Player role for bargaining table datasets.",
    )
    return parser.parse_args()


def ensure_list(values: str | List[str]) -> List[str]:
    return values if isinstance(values, list) else [values]


def prepare_directories(model_name: str, base_data_dir: str = "./data/") -> Tuple[Path, Path]:
    dataset_dir = Path(base_data_dir)
    logs_dir = Path("./logs") / model_name
    for d in (dataset_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return dataset_dir, logs_dir


def save_jsonl(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def build_prompt_plan(problems: Dict[int, List[str]], norm_factors: Dict[int, List[int]]) -> List[dict]:
    prompt_plan: List[dict] = []
    for idx, variants in problems.items():
        for var_idx, variant in enumerate(variants):
            prompt_plan.append(
                {
                    "idx": idx,
                    "variation_idx": str(var_idx),
                    "prompt": f"{Level0.prefix}{variant}{Level0.suffix}",
                    "normalization_factor": norm_factors[idx][var_idx],
                }
            )
    return prompt_plan


def assemble_log_entries(
    prompt_plan: List[dict], responses_by_prompt: Dict[Tuple[int, str], List[str]]
) -> List[dict]:
    logs: List[dict] = []
    for entry in prompt_plan:
        key = (entry["idx"], entry["prompt"])
        logs.append(
            {
                "idx": entry["idx"],
                "variation-idx": entry["variation_idx"],
                "prompt": entry["prompt"],
                "responses": responses_by_prompt.get(key, []),
                "normalization_factor": entry["normalization_factor"],
            }
        )
    return logs


def load_existing_logs(path: Path) -> Dict[Tuple[int, str], List[str]]:
    if not path.exists():
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[WARNING] Could not parse existing log at {path}; ignoring it.")
        return {}

    responses: Dict[Tuple[int, str], List[str]] = {}
    for block in data:
        idx = block.get("idx")
        prompt = block.get("prompt")
        if idx is None or prompt is None:
            continue
        stored_responses = block.get("responses") or []
        if isinstance(stored_responses, list):
            filtered = [r for r in stored_responses if isinstance(r, str)]
            responses[(idx, prompt)] = filtered
    return responses


def build_flat_prompt_list(
    problems: Dict[int, List[str]],
) -> Tuple[List[str], List[Tuple[int, str]]]:
    prompts, keys = [], []
    for idx, variants in problems.items():
        for v in variants:
            prompts.append(f"{Level0.prefix}{v}{Level0.suffix}")
            keys.append((idx, v))
    return prompts, keys


def generate_batch_responses(
    model: LLM,
    prompt_plan: List[dict],
    trials: int,
    seq_per_prompt: int,
    log_path: Path,
    existing_responses: Dict[Tuple[int, str], List[str]],
) -> Tuple[Dict[Tuple[int, str], List[str]], bool]:
    target_per_prompt = trials * seq_per_prompt
    responses_by_prompt: Dict[Tuple[int, str], List[str]] = {
        key: list(val) for key, val in existing_responses.items()
    }

    for entry in prompt_plan:
        responses_by_prompt.setdefault((entry["idx"], entry["prompt"]), [])

    new_data_generated = False
    total_prompts = len(prompt_plan)

    for i, entry in enumerate(prompt_plan, start=1):
        key = (entry["idx"], entry["prompt"])
        prompt_responses = responses_by_prompt[key]

        if len(prompt_responses) >= target_per_prompt:
            print(
                f"[SKIP] idx={entry['idx']} var={entry['variation_idx']} "
                f"already has {len(prompt_responses)}/{target_per_prompt} responses."
            )
            continue

        if prompt_responses:
            print(
                f"[RESUME] idx={entry['idx']} var={entry['variation_idx']} "
                f"continuing from {len(prompt_responses)}/{target_per_prompt}."
            )

        while len(prompt_responses) < target_per_prompt:
            remaining = target_per_prompt - len(prompt_responses)
            batch_size = min(seq_per_prompt, remaining)
            print(
                f"[{model.model_id}] Q {i}/{total_prompts} "
                f"idx={entry['idx']} var={entry['variation_idx']} "
                f"({len(prompt_responses)}/{target_per_prompt} done, "
                f"+{batch_size} requested)"
            )
            print("[PROMPT]")
            print(entry["prompt"])
            if model.is_api_model:
                print("[RESPONSES]")
                for _ in range(batch_size):
                    text = model.generate(entry["prompt"])
                    prompt_responses.append(text)
                    new_data_generated = True
                    print(text)
                    save_jsonl(
                        log_path,
                        assemble_log_entries(prompt_plan, responses_by_prompt),
                    )
                continue

            texts = model.generate_batch([entry["prompt"]], num_return_sequences=batch_size)[0]
            prompt_responses.extend(texts[:batch_size])
            new_data_generated = True
            print("[RESPONSES]")
            print("\n".join(texts[:batch_size]))

            save_jsonl(log_path, assemble_log_entries(prompt_plan, responses_by_prompt))

    return responses_by_prompt, new_data_generated


def nest_outputs(keys: List[Tuple[int, str]], outputs: List[List[str]]) -> Dict[int, Dict[str, List[str]]]:
    resp: Dict[int, Dict[str, List[str]]] = {}
    for (idx, prompt), outs in zip(keys, outputs):
        resp.setdefault(idx, {})[prompt] = outs
    return resp


def build_log_json(
    idx_to_responses: Dict[int, Dict[str, List[str]]],
    norm_factors: Dict[int, List[int]],
) -> List[dict]:
    logs: List[dict] = []
    for idx, variants in idx_to_responses.items():
        for (var_idx, prompt), norm in zip(enumerate(variants), norm_factors[idx]):
            logs.append(
                {
                    "idx": idx,
                    "variation-idx": str(var_idx),
                    "prompt": prompt,
                    "responses": variants[prompt],
                    "normalization_factor": norm,
                }
            )
    return logs


def run_single_job(
    model_name: str,
    dataset: str,
    problem_tag: str,
    args: argparse.Namespace,
    cached_model: LLM | None,
) -> LLM | None:
    start = time.time()

    # Case-fold onto an existing results folder: the folder name is the model string verbatim, so a
    # casing-only mismatch (e.g. `-m meta-llama/llama-3.3-70b-instruct` vs the shipped
    # `results/meta-llama/Llama-3.3-70B-Instruct/`) would otherwise create a confusing duplicate folder
    # AND (on a case-sensitive filesystem) silently bypass the overwrite-guard on the shipped reference
    # data. Walk the org/model path component-by-component and adopt the existing casing at each level, so
    # the same model maps to the same folder regardless of how `-m` was typed.
    _cur = Path("./results")
    _resolved: list[str] = []
    for _part in model_name.split("/"):
        _match = (
            next((d.name for d in _cur.iterdir() if d.is_dir() and d.name.lower() == _part.lower()), None)
            if _cur.is_dir()
            else None
        )
        if _match is None:
            _resolved = []  # no existing folder for this model — keep the id as typed
            break
        _resolved.append(_match)
        _cur = _cur / _match
    if _resolved:
        model_name = "/".join(_resolved)

    # filesystem prep
    dataset_dir, logs_dir = prepare_directories(model_name)
    log_suffix = problem_tag
    if dataset in ("bargaining_table", "bargaining_table_realdata"):
        log_suffix = f"{problem_tag}_{args.bargaining_player}"
    log_file = logs_dir / f"{dataset}_responses_{log_suffix}.jsonl"

    # Data-loss guard: output folders are keyed on model_name (i.e. on -m, not on the model that
    # actually answered). Running a different model without a matching -m would write under the default
    # folder and clobber the shipped paper reference data in results/meta-llama/Llama-3.3-70B-Instruct/.
    # We refuse to overwrite a results file that existed before this run with freshly generated answers,
    # unless --overwrite is passed. (A pure resume of the same model, which only skips/recomputes without
    # new answers, is still allowed.)
    output_dataset = dataset
    if dataset in ("bargaining_table", "bargaining_table_realdata"):
        output_dataset = f"{dataset}_{args.bargaining_player}"
    results_path = Path(f"./results/{model_name}/{output_dataset}_{problem_tag}.jsonl")
    results_preexisted = results_path.exists()

    # load dataset
    if dataset in ("bargaining_table", "bargaining_table_realdata"):
        ds_path = dataset_dir / "bargaining_table_llms" / f"{dataset}.json"
        problems, norm_factors = load_bargaining_table_prompts(ds_path, args.bargaining_player, problem_tag)
    else:
        ds_path = dataset_dir / "amsterdam_nottingham_schelling" / f"{dataset}.jsonl"
        with open(ds_path) as f:
            raw_data = json.load(f)
        problems, norm_factors = iterate_data(raw_data, problem_tag)

    # build prompt plan and check existing progress before loading the model
    prompt_plan = build_prompt_plan(problems, norm_factors)
    if args.max_prompts is not None:
        prompt_plan = prompt_plan[: args.max_prompts]
    target_per_prompt = args.trials * args.sequences
    existing_responses = load_existing_logs(log_file)
    total_prompts = len(prompt_plan)
    completed_prompts = sum(
        1
        for entry in prompt_plan
        if len(existing_responses.get((entry["idx"], entry["prompt"]), [])) >= target_per_prompt
    )
    if existing_responses:
        print(f"[INFO] Found stored responses for {completed_prompts}/{total_prompts} prompts in {log_file}.")

    all_complete = completed_prompts == total_prompts and total_prompts > 0

    # Data-loss guard (fire before any generation / log writes): if a results file already exists for
    # this folder and this run would still generate new answers into it, refuse unless --overwrite.
    # A fully-complete resume generates nothing and is allowed. This stops a different model run under a
    # mismatched -m from clobbering shipped reference data (e.g. results/meta-llama/Llama-3.3-70B-Instruct/).
    if results_preexisted and not all_complete and not args.overwrite:
        raise ValueError(
            f"Refusing to generate into existing results at {results_path} "
            f"(model_name={model_name!r}, {completed_prompts}/{total_prompts} prompts complete). "
            f"The output folder is named after -m, not the model that actually answers; if this is a "
            f"different model, re-run with a matching -m so it writes to its own folder. Pass --overwrite "
            f"only if you really mean to replace these results."
        )

    responses_by_prompt: Dict[Tuple[int, str], List[str]] = dict(existing_responses)
    model = cached_model
    new_data_generated = False

    # load model if additional work is required
    if not all_complete:
        if total_prompts == 0:
            print("[WARNING] No prompts generated from dataset; exiting early.")
            return model

        # load model once per model_name and reuse for subsequent jobs
        if model is None:
            model = load_model(
                model_id=model_name,
                num_return_sequences=args.sequences,
                max_new_tokens=args.max_new_tokens,
                quantization=args.quantization,
                reasoning=args.reasoning,
                backend=args.backend,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )

        # generate
        responses_by_prompt, new_data_generated = generate_batch_responses(
            model,
            prompt_plan,
            trials=args.trials,
            seq_per_prompt=args.sequences,
            log_path=log_file,
            existing_responses=existing_responses,
        )
    else:
        print("[INFO] All prompts already have the required number of responses.")

    # reshape & save
    jsonl_logs = assemble_log_entries(prompt_plan, responses_by_prompt)
    save_jsonl(log_file, jsonl_logs)
    print(f"[OK] Responses written to {log_file}")

    # print timing
    elapsed = int(time.time() - start)
    print(f"[{model_name}] elapsed {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")

    # Plot graphs and create result folders (output_dataset / results_path computed above; the
    # data-loss guard already ran before generation).
    if args.plot_graph:
        if new_data_generated or not results_path.exists():
            compute_metrics(model_name, output_dataset, jsonl_logs, problem_tag)
        else:
            print(f"[INFO] Results already exist at {results_path}; skipping recomputation.")

    return model


def count_pending_calls(args: argparse.Namespace) -> int:
    """Model calls still needed for every (model, dataset, tag) in this invocation."""
    from src.utils.openrouter_llm import estimate_calls

    total = 0
    target = args.trials * args.sequences
    for model in ensure_list(args.model_names):
        for dataset in ensure_list(args.datasets):
            tags = ensure_list(args.problem_tags)
            if dataset in ("bargaining_table", "bargaining_table_realdata") and tags == ["problem"]:
                with open(Path("./data") / "bargaining_table_llms" / f"{dataset}.json") as f:
                    tags = list(json.load(f).get("variants", {}).keys())
            for tag in tags:
                dataset_dir, logs_dir = prepare_directories(model)
                suffix = f"{tag}_{args.bargaining_player}" if dataset.startswith("bargaining_table") else tag
                log_file = logs_dir / f"{dataset}_responses_{suffix}.jsonl"
                if dataset.startswith("bargaining_table"):
                    problems, norms = load_bargaining_table_prompts(
                        dataset_dir / "bargaining_table_llms" / f"{dataset}.json", args.bargaining_player, tag
                    )
                else:
                    with open(dataset_dir / "amsterdam_nottingham_schelling" / f"{dataset}.jsonl") as f:
                        problems, norms = iterate_data(json.load(f), tag)
                plan = build_prompt_plan(problems, norms)
                if args.max_prompts is not None:
                    plan = plan[: args.max_prompts]
                total += estimate_calls(plan, load_existing_logs(log_file), target)
    return total


def confirm_cost(args: argparse.Namespace) -> None:
    from src.utils.openrouter_llm import cost_note

    n_calls = count_pending_calls(args)
    models = ensure_list(args.model_names)
    print(
        f"[COST] {n_calls:,} pending calls for {len(models)} model(s) × {len(ensure_list(args.datasets))} "
        f"dataset(s) × {args.sequences} samples/prompt."
    )
    for m in models:
        print(f"[COST] {cost_note(n_calls // max(1, len(models)), m)}")
    if n_calls == 0 or args.yes or args.backend != "openrouter" or not sys.stdin.isatty():
        return
    if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("aborted")


def run_job(args: argparse.Namespace) -> None:
    confirm_cost(args)
    for model in ensure_list(args.model_names):
        cached_model: LLM | None = None
        for dataset in ensure_list(args.datasets):
            problem_tags = ensure_list(args.problem_tags)
            if dataset in ("bargaining_table", "bargaining_table_realdata") and problem_tags == ["problem"]:
                ds_path = Path("./data") / "bargaining_table_llms" / f"{dataset}.json"
                with open(ds_path, "r") as f:
                    data = json.load(f)
                variants = data.get("variants", {})
                if not isinstance(variants, dict) or not variants:
                    raise ValueError(f"Dataset at {ds_path} must include a non-empty 'variants' mapping.")
                problem_tags = list(variants.keys())
            for problem_tag in problem_tags:
                print(f"[INFO] Running model={model} dataset={dataset} problem_tag={problem_tag}")
                cached_model = run_single_job(model, dataset, problem_tag, args, cached_model)
        if cached_model and hasattr(cached_model, "clear_cache"):
            cached_model.clear_cache()


def compute_metrics(model_name: str, dataset: str, data: List[dict], problem_tag: str) -> None:
    """Plot graphs and compute the results

    Args:
        model_name (str): the LLM name
        dataset (str): the dataset name
        problem_tag (str): pick, guess, or coordinate
    """
    os.makedirs(f"./images/{model_name}/{dataset}/{problem_tag}/", exist_ok=True)
    os.makedirs(f"./results/{model_name}/", exist_ok=True)

    plot_block_frequencies(data, dataset, model_name, problem_tag)


if __name__ == "__main__":
    # Parse CLI args and run the job
    cli_args = parse_arguments()

    try:
        run_job(cli_args)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"\n❌ {e}")
