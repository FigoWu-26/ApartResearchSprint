import argparse
import json
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bargaining_table import bargaining_table as bt


FILE_RE = re.compile(
    r"(?P<strategy>p[12]_llm)_(?P<color>blue|yellow)_"
    r"(?P<model>gpt-oss-(?:20b|120b))-(?P<reasoning>low|medium|high)_"
    r"(?P<dataset>bargaining_table(?:_realdata)?)_responses_"
    r"(?P<variant>[a-z-]+)_(?P=color)\.json$"
)

RESPONSE_FILE_RE = re.compile(
    r"(?P<dataset>bargaining_table(?:_realdata)?)_"
    r"(?P<color>blue|yellow)_(?P<variant>[a-z-]+)\.jsonl$"
)

DEFAULT_VARIANTS = [
    "vanilla",
    "saliency",
    "greedy",
    "cooperative",
    "all-features",
]

DEFAULT_REASONING = ["low", "medium", "high"]
DEFAULT_MODELS = ["gpt-oss-120b", "gpt-oss-20b"]
DEFAULT_STRATEGIES = ["p1_llm", "p2_llm"]

STRATEGY_LABELS = {
    "p1_llm": ("Orange Human", "Blue LLM"),
    "p2_llm": ("Orange LLM", "Blue Human"),
}


@dataclass(frozen=True)
class ResultMeta:
    strategy: str
    color: str
    model: str
    reasoning: str
    dataset: str
    variant: str
    path: Path


@dataclass(frozen=True)
class ResponseMeta:
    model: str
    dataset: str
    color: str
    variant: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged bargaining-table results."
    )
    parser.add_argument(
        "--results-dir",
        default="./results/bargaining_table",
        help="Base folder containing bargaining_table result JSONs.",
    )
    parser.add_argument(
        "--plots-dir",
        default="./plots/bargaining_table/averages",
        help="Output folder for plots.",
    )
    parser.add_argument(
        "--dataset",
        default="bargaining_table_realdata",
        help="Dataset prefix to filter on (e.g. bargaining_table_realdata).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Prompt variants to include (default: all available).",
    )
    parser.add_argument(
        "--reasoning-levels",
        nargs="+",
        default=DEFAULT_REASONING,
        help="Reasoning levels to include (default: low medium high).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to include.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="Strategies to include (default: p1_llm p2_llm).",
    )
    parser.add_argument(
        "--responses-root",
        default="",
        help="Folder containing per-model response JSONL files.",
    )
    parser.add_argument(
        "--model-regex",
        default="",
        help="Regex to filter model folders under --responses-root.",
    )
    parser.add_argument(
        "--group-tag",
        default="",
        help="Tag used in output paths for response-mode plots.",
    )
    parser.add_argument(
        "--group-label",
        default="",
        help="Readable label used in response-mode plots.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random pairs to sample (default: 100).",
    )
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Sample with replacement for payoff calculation.",
    )
    parser.add_argument(
        "--player1-data",
        default="./data/Dor-humans/stage-2-analysis/Number1players/",
        help="Folder path for player 1 human games.",
    )
    parser.add_argument(
        "--player2-data",
        default="./data/Dor-humans/stage-2-analysis/Number2players/",
        help="Folder path for player 2 human games.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def scan_results(results_dir: Path, dataset_filter: Optional[str]) -> List[ResultMeta]:
    entries: List[ResultMeta] = []
    for path in results_dir.rglob("*.json"):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        info = match.groupdict()
        if dataset_filter and info["dataset"] != dataset_filter:
            continue
        entries.append(
            ResultMeta(
                strategy=info["strategy"],
                color=info["color"],
                model=info["model"],
                reasoning=info["reasoning"],
                dataset=info["dataset"],
                variant=info["variant"],
                path=path,
            )
        )
    return entries


def build_index(entries: Iterable[ResultMeta]) -> Dict[Tuple[str, str, str, str, str], List[Path]]:
    index: Dict[Tuple[str, str, str, str, str], List[Path]] = {}
    for entry in entries:
        key = (entry.strategy, entry.model, entry.reasoning, entry.variant, entry.dataset)
        index.setdefault(key, []).append(entry.path)
    return index


def scan_response_files(
    responses_root: Path, dataset_filter: Optional[str]
) -> List[ResponseMeta]:
    entries: List[ResponseMeta] = []
    if not responses_root.exists():
        return entries
    for model_dir in sorted(responses_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for path in model_dir.glob("*.jsonl"):
            match = RESPONSE_FILE_RE.match(path.name)
            if not match:
                continue
            info = match.groupdict()
            if dataset_filter and info["dataset"] != dataset_filter:
                continue
            entries.append(
                ResponseMeta(
                    model=model_dir.name,
                    dataset=info["dataset"],
                    color=info["color"],
                    variant=info["variant"],
                    path=path,
                )
            )
    return entries


def build_response_index(entries: Iterable[ResponseMeta]) -> Dict[Tuple[str, str, str, str], Path]:
    index: Dict[Tuple[str, str, str, str], Path] = {}
    for entry in entries:
        key = (entry.model, entry.color, entry.variant, entry.dataset)
        if key in index:
            print(f"Warning: duplicate response file for {key}: {entry.path.name}")
            continue
        index[key] = entry.path
    return index


def resolve_variants(entries: Iterable[ResultMeta], requested: Optional[List[str]]) -> List[str]:
    available = sorted({entry.variant for entry in entries})
    if requested:
        return [v for v in requested if v in available]
    ordered = [v for v in DEFAULT_VARIANTS if v in available]
    ordered.extend([v for v in available if v not in ordered])
    return ordered


def resolve_response_variants(
    entries: Iterable[ResponseMeta], requested: Optional[List[str]]
) -> List[str]:
    available = sorted({entry.variant for entry in entries})
    if requested:
        return [v for v in requested if v in available]
    ordered = [v for v in DEFAULT_VARIANTS if v in available]
    ordered.extend([v for v in available if v not in ordered])
    return ordered


def pick_single_path(paths: Optional[List[Path]], key: Tuple[str, str, str, str, str]) -> Optional[Path]:
    if not paths:
        return None
    if len(paths) > 1:
        names = ", ".join(p.name for p in paths)
        print(f"Warning: multiple results for {key}: {names}. Using {paths[0].name}.")
    return paths[0]


def load_total_payoffs(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys_p1 = sorted(data["overall_payoff_p1"].keys(), key=int)
    keys_p2 = sorted(data["overall_payoff_p2"].keys(), key=int)
    games_p1 = np.array([data["overall_payoff_p1"][k] for k in keys_p1])
    games_p2 = np.array([data["overall_payoff_p2"][k] for k in keys_p2])
    return games_p1.sum(axis=0), games_p2.sum(axis=0)


def totals_from_results(results: dict) -> Tuple[np.ndarray, np.ndarray]:
    games_p1 = np.array([v for v in results["overall_payoff_p1"].values()])
    games_p2 = np.array([v for v in results["overall_payoff_p2"].values()])
    return games_p1.sum(axis=0), games_p2.sum(axis=0)


def average_arrays(arrays: List[np.ndarray], label: str) -> np.ndarray:
    if not arrays:
        raise ValueError(f"No arrays provided for {label}.")
    lengths = [len(a) for a in arrays]
    min_len = min(lengths)
    if len(set(lengths)) > 1:
        print(f"Warning: {label} arrays have lengths {lengths}; trimming to {min_len}.")
    stacked = np.vstack([a[:min_len] for a in arrays])
    return stacked.mean(axis=0)


def average_results(paths: List[Path], label: str) -> Tuple[np.ndarray, np.ndarray]:
    totals_p1: List[np.ndarray] = []
    totals_p2: List[np.ndarray] = []
    for path in paths:
        p1, p2 = load_total_payoffs(path)
        totals_p1.append(p1)
        totals_p2.append(p2)
    return (
        average_arrays(totals_p1, f"{label} p1"),
        average_arrays(totals_p2, f"{label} p2"),
    )


def build_labels(strategy: str, llm_detail: Optional[str]) -> Tuple[str, str]:
    orange_label, blue_label = STRATEGY_LABELS.get(strategy, ("Player 2", "Player 1"))
    if llm_detail:
        if strategy == "p1_llm":
            blue_label = f"{blue_label} ({llm_detail})"
        elif strategy == "p2_llm":
            orange_label = f"{orange_label} ({llm_detail})"
    return blue_label, orange_label


def plot_and_save(
    data_p1: np.ndarray,
    data_p2: np.ndarray,
    strategy: str,
    save_path: Path,
    llm_detail: Optional[str],
    variant_label: Optional[str] = None,
) -> None:
    label1, label2 = build_labels(strategy, llm_detail)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bt.plot_pretty_boxplot(
        data_p1,
        data_p2,
        label1=label1,
        label2=label2,
        annotate_stats=True,
        save_path=str(save_path),
        variant_label=variant_label,
    )


def expand_responses(responses: object) -> List[str]:
    if isinstance(responses, dict):
        expanded: List[str] = []
        for response, count in responses.items():
            expanded.extend([response] * int(count))
        return expanded
    if isinstance(responses, list):
        return list(responses)
    return []


def normalize_llm_data(data: List[dict]) -> List[dict]:
    def game_key(entry: dict) -> int:
        idx = str(entry.get("idx", ""))
        match = re.search(r"(\d+)", idx)
        return int(match.group(1)) if match else 0

    normalized: List[dict] = []
    for entry in sorted(data, key=game_key):
        responses = expand_responses(entry.get("responses", []))
        normalized.append({"responses": responses})
    return normalized


def write_llm_data(tmp_dir: Path, tag: str, data: List[dict]) -> Path:
    path = tmp_dir / f"{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def resolve_response_models(
    entries: Iterable[ResponseMeta], models: Optional[List[str]], model_regex: str
) -> List[str]:
    available = sorted({entry.model for entry in entries})
    if models:
        return [m for m in models if m in available]
    if model_regex:
        pattern = re.compile(model_regex)
        return [m for m in available if pattern.search(m)]
    return available


def default_group_tag(responses_root: Path, model_regex: str) -> str:
    root = responses_root.name.lower()
    family = "models"
    if "llama" in root:
        family = "llama"
    elif "qwen" in root:
        family = "qwen"
    size = ""
    match = re.search(r"(\d+[bB])", model_regex)
    if match:
        size = match.group(1).lower()
    return f"{family}-{size}" if size else family


def run_response_mode(args: argparse.Namespace) -> None:
    responses_root = Path(args.responses_root)
    entries = scan_response_files(responses_root, args.dataset)
    if not entries:
        print("No matching response files found.")
        return

    models = resolve_response_models(entries, args.models, args.model_regex)
    if not models:
        print("No matching models found.")
        return

    variants = resolve_response_variants(entries, args.variants)
    strategies = [s for s in args.strategies if s in DEFAULT_STRATEGIES]
    if not strategies:
        print("No matching strategies found.")
        return

    group_tag = args.group_tag or default_group_tag(responses_root, args.model_regex)
    group_label = args.group_label or group_tag

    player1_files = bt.player_files(args.player1_data, "BT")
    player2_files = bt.player_files(args.player2_data, "BT")
    if args.seed is not None:
        random.seed(args.seed)
    sampled_pairs = bt.sample_pairs(
        player1_files,
        player2_files,
        args.num_samples,
        with_replacement=args.sample_with_replacement,
    )

    index = build_response_index(entries)
    plots_dir = Path(args.plots_dir) / f"avg-models-{group_tag}"

    strategy_to_color = {"p1_llm": "blue", "p2_llm": "yellow"}

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for strategy in strategies:
            color = strategy_to_color[strategy]
            for variant in variants:
                totals_p1: List[np.ndarray] = []
                totals_p2: List[np.ndarray] = []
                for model in models:
                    key = (model, color, variant, args.dataset)
                    path = index.get(key)
                    if not path:
                        print(f"Missing response file for {key}. Skipping.")
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    llm_data = normalize_llm_data(raw_data)
                    if not llm_data:
                        print(f"No responses found in {path.name}. Skipping.")
                        continue
                    tmp_path = write_llm_data(
                        tmp_dir, f"{model}_{color}_{variant}", llm_data
                    )
                    if strategy == "p1_llm":
                        bt.LLM_AS_P1 = str(tmp_path)
                    else:
                        bt.LLM_AS_P2 = str(tmp_path)
                    results = bt.compute_payoff(
                        sampled_pairs,
                        args.player1_data,
                        args.player2_data,
                        strategy=strategy,
                    )
                    p1, p2 = totals_from_results(results)
                    totals_p1.append(p1)
                    totals_p2.append(p2)

                if not totals_p1:
                    print(f"No results to average for {strategy} {variant}. Skipping.")
                    continue
                avg_p1 = average_arrays(totals_p1, f"{strategy} {variant} p1")
                avg_p2 = average_arrays(totals_p2, f"{strategy} {variant} p2")
                save_path = (
                    plots_dir
                    / strategy
                    / f"{args.dataset}_{variant}_avg-{group_tag}.png"
                )
                plot_and_save(
                    avg_p1,
                    avg_p2,
                    strategy,
                    save_path,
                    llm_detail=f"{group_label} avg",
                    variant_label=variant,
                )


def main() -> None:
    args = parse_args()
    if args.responses_root:
        run_response_mode(args)
        return

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    entries = scan_results(results_dir, args.dataset)
    if not entries:
        print("No matching results found.")
        return

    variants = resolve_variants(entries, args.variants)
    strategies = [s for s in args.strategies if s in {e.strategy for e in entries}]
    if not strategies:
        print("No matching strategies found.")
        return
    available_models = sorted({e.model for e in entries})
    models_requested = args.models or DEFAULT_MODELS
    models = [m for m in models_requested if m in available_models]
    if not models:
        print("No matching models found.")
        return
    models_tag = "+".join(m.replace("gpt-oss-", "") for m in models)

    index = build_index(entries)

    # Average each model across reasoning levels for each variant.
    for strategy in strategies:
        for variant in variants:
            for model in models:
                paths: List[Path] = []
                for reasoning in args.reasoning_levels:
                    key = (strategy, model, reasoning, variant, args.dataset)
                    path = pick_single_path(index.get(key), key)
                    if path:
                        paths.append(path)
                if not paths:
                    print(
                        f"Missing {model} results for {strategy} {variant}. Skipping."
                    )
                    continue
                avg_p1, avg_p2 = average_results(
                    paths, f"{strategy} {model} {variant}"
                )
                save_path = (
                    plots_dir
                    / "avg-model-across-reasoning"
                    / strategy
                    / f"{args.dataset}_{variant}_avg-{model}-across-reasoning.png"
                )
                plot_and_save(
                    avg_p1,
                    avg_p2,
                    strategy,
                    save_path,
                    llm_detail=f"{model} avg reasoning",
                    variant_label=variant,
                )

    # Average across models for each reasoning level and variant.
    for strategy in strategies:
        for variant in variants:
            for reasoning in args.reasoning_levels:
                paths = []
                for model in models:
                    key = (strategy, model, reasoning, variant, args.dataset)
                    path = pick_single_path(index.get(key), key)
                    if path:
                        paths.append(path)
                if not paths:
                    print(
                        f"Missing model results for {strategy} {variant} {reasoning}. Skipping."
                    )
                    continue
                avg_p1, avg_p2 = average_results(
                    paths, f"{strategy} {variant} {reasoning}"
                )
                save_path = (
                    plots_dir
                    / "avg-models-by-reasoning"
                    / strategy
                    / f"{args.dataset}_{variant}_avg-{models_tag}-{reasoning}.png"
                )
                plot_and_save(
                    avg_p1,
                    avg_p2,
                    strategy,
                    save_path,
                    llm_detail=f"avg {models_tag} {reasoning}",
                    variant_label=variant,
                )


if __name__ == "__main__":
    main()
