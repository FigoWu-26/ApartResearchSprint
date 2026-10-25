import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import build_prompt_plan, load_existing_logs
from src.utils import iterate_data, load_bargaining_table_prompts


MODELS = (
    "gpt-4o",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
)

ALL_DATASETS = (
    "amsterdam",
    "amsterdam-instruct-all-features",
    "amsterdam-instruct-saliency",
    "amsterdam_numeric",
    "amsterdam_numeric-instruct-all-features",
    "amsterdam_numeric-instruct-saliency",
    "asymmetric_payoff",
    "asymmetric_payoff-instruct-all-features",
    "asymmetric_payoff-instruct-saliency",
    "nottingham",
    "nottingham-instruct-all-features",
    "nottingham-instruct-saliency",
    "nottingham_numeric",
    "nottingham_numeric-instruct-all-features",
    "nottingham_numeric-instruct-saliency",
    "amsterdam-instruct-culture",
    "nottingham-instruct-culture",
    "schelling",
    "schelling-instruct-all-features",
    "schelling-instruct-saliency",
    "bargaining_table_realdata",
)

STANDARD_PROBLEM_TAGS = ("problem-pick", "problem-guess", "problem-coordinate")
SCHELLING_PROBLEM_TAGS = ("problem",)
BARGAINING_PROBLEM_TAGS = (
    "greedy",
    "cooperative",
    "all-features",
    "saliency",
    "vanilla",
)
BARGAINING_PLAYERS = ("blue", "yellow")


def _slug(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "none"


def _iter_jobs() -> Iterable[tuple[str, str, str | None]]:
    for dataset in ALL_DATASETS:
        if dataset in {"schelling", "schelling-instruct-all-features", "schelling-instruct-saliency"}:
            for problem_tag in SCHELLING_PROBLEM_TAGS:
                yield dataset, problem_tag, None
        elif dataset == "bargaining_table_realdata":
            for problem_tag in BARGAINING_PROBLEM_TAGS:
                for player in BARGAINING_PLAYERS:
                    yield dataset, problem_tag, player
        else:
            for problem_tag in STANDARD_PROBLEM_TAGS:
                yield dataset, problem_tag, None


def _load_prompt_plan(data_dir: Path, dataset: str, problem_tag: str, player: str | None):
    if player is not None:
        ds_path = data_dir / "bargaining_table_llms" / f"{dataset}.json"
        problems, norm_factors = load_bargaining_table_prompts(
            ds_path, player, problem_tag
        )
    else:
        ds_path = data_dir / "amsterdam_nottingham_schelling" / f"{dataset}.jsonl"
        with open(ds_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        problems, norm_factors = iterate_data(raw_data, problem_tag)

    return build_prompt_plan(problems, norm_factors)


def _log_suffix(problem_tag: str, player: str | None) -> str:
    return f"{problem_tag}_{player}" if player is not None else problem_tag


def _request_id(
    dataset: str,
    problem_tag: str,
    player: str | None,
    idx: object,
    variation_idx: object,
    response_idx: int,
) -> str:
    parts = [
        _slug(dataset),
        _slug(problem_tag),
        _slug(player) if player is not None else "standard",
        f"idx-{_slug(idx)}",
        f"var-{_slug(variation_idx)}",
        f"resp-{response_idx:02d}",
    ]
    return "__".join(parts)


def _openai_request(model: str, request_id: str, prompt: str) -> dict:
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def _gemini_request(request_id: str, prompt: str) -> dict:
    return {
        "key": request_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        },
    }


def build_requests(
    model: str,
    data_dir: Path,
    completed_dir: Path,
    target_responses: int,
) -> tuple[list[dict], list[dict]]:
    provider = "openai" if model.startswith("gpt-") else "gemini"
    requests: list[dict] = []
    manifest: list[dict] = []

    for dataset, problem_tag, player in _iter_jobs():
        prompt_plan = _load_prompt_plan(data_dir, dataset, problem_tag, player)
        suffix = _log_suffix(problem_tag, player)
        log_file = completed_dir / model / f"{dataset}_responses_{suffix}.jsonl"
        existing_responses = load_existing_logs(log_file)

        for entry in prompt_plan:
            key = (entry["idx"], entry["prompt"])
            existing_count = len(existing_responses.get(key, []))
            missing_count = max(0, target_responses - existing_count)
            for response_idx in range(existing_count, existing_count + missing_count):
                request_id = _request_id(
                    dataset,
                    problem_tag,
                    player,
                    entry["idx"],
                    entry["variation_idx"],
                    response_idx,
                )
                prompt = entry["prompt"]
                if provider == "openai":
                    request = _openai_request(model, request_id, prompt)
                else:
                    request = _gemini_request(request_id, prompt)

                requests.append(request)
                manifest.append(
                    {
                        "request_id": request_id,
                        "provider": provider,
                        "model": model,
                        "dataset": dataset,
                        "problem_tag": problem_tag,
                        "bargaining_player": player,
                        "idx": entry["idx"],
                        "variation_idx": entry["variation_idx"],
                        "response_idx": response_idx,
                        "completed_log": str(log_file),
                    }
                )

    return requests, manifest


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--completed-dir", type=Path, default=Path("data/completed_prompts")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/batch_requests"))
    parser.add_argument("--target-responses", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    summary = []
    for model in args.models:
        requests, manifest = build_requests(
            model=model,
            data_dir=args.data_dir,
            completed_dir=args.completed_dir,
            target_responses=args.target_responses,
        )
        model_dir = args.output_dir / model
        request_path = model_dir / f"{model}.jsonl"
        manifest_path = model_dir / f"{model}.manifest.jsonl"
        write_jsonl(request_path, requests)
        write_jsonl(manifest_path, manifest)
        row = {
            "model": model,
            "requests": len(requests),
            "request_file": str(request_path),
            "manifest_file": str(manifest_path),
        }
        if model.startswith("gpt-") and len(requests) > 50000:
            row["parts"] = []
            for part_number, start in enumerate(range(0, len(requests), 50000), start=1):
                end = min(start + 50000, len(requests))
                part_request_path = model_dir / f"{model}.part-{part_number:03d}.jsonl"
                part_manifest_path = (
                    model_dir / f"{model}.part-{part_number:03d}.manifest.jsonl"
                )
                write_jsonl(part_request_path, requests[start:end])
                write_jsonl(part_manifest_path, manifest[start:end])
                row["parts"].append(
                    {
                        "requests": end - start,
                        "request_file": str(part_request_path),
                        "manifest_file": str(part_manifest_path),
                    }
                )

        summary.append(row)
        print(f"{model}: wrote {len(requests)} requests to {request_path}")

    summary_path = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
