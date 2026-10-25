import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import assemble_log_entries, load_existing_logs, save_jsonl
from src.batch_requests.prepare_batch_requests import _iter_jobs, _load_prompt_plan, _log_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge provider batch response JSONL files into completed-prompt style "
            "logs without overwriting data/completed_prompts."
        )
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["gemini", "openai", "gpt"],
        help="Provider response format.",
    )
    parser.add_argument("--model", required=True, help="Model name used for the batch.")
    parser.add_argument(
        "--response-files",
        nargs="+",
        type=Path,
        required=True,
        help="Downloaded batch response JSONL files.",
    )
    parser.add_argument(
        "--manifest-files",
        nargs="*",
        type=Path,
        default=None,
        help="Optional matching manifest JSONL files. If omitted, paths are inferred.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--completed-dir", type=Path, default=Path("data/completed_prompts")
    )
    parser.add_argument(
        "--batch-root", type=Path, default=Path("data/batch_requests")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("logs/Gemini"),
        help="Merged logs are written to <output-root>/<model>/.",
    )
    parser.add_argument("--target-responses", type=int, default=30)
    return parser.parse_args()


def normalize_provider(provider: str) -> str:
    return "openai" if provider == "gpt" else provider


def normalize_output(text: str) -> str:
    return "".join(
        c for c in text.strip().lower() if c.isalnum() or c.isspace() or c in "<>/"
    )


def infer_manifest_path(response_file: Path, batch_root: Path, model: str) -> Path:
    name = response_file.name
    if name.endswith(".responses.jsonl"):
        manifest_name = name[: -len(".responses.jsonl")] + ".manifest.jsonl"
    elif name.endswith(".jsonl"):
        manifest_name = name[: -len(".jsonl")] + ".manifest.jsonl"
    else:
        raise ValueError(f"Cannot infer manifest path for {response_file}")

    return batch_root / model / manifest_name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
    return rows


def load_manifest(paths: list[Path]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            request_id = row["request_id"]
            if request_id in manifest:
                raise ValueError(f"Duplicate request_id in manifests: {request_id}")
            manifest[request_id] = row
    return manifest


def gemini_response_id(row: dict[str, Any]) -> str:
    return row["key"]


def gemini_response_text(row: dict[str, Any]) -> str | None:
    response = row.get("response")
    if not response:
        return None
    candidates = response.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return "".join(texts) if texts else None


def openai_response_id(row: dict[str, Any]) -> str:
    return row["custom_id"]


def openai_response_text(row: dict[str, Any]) -> str | None:
    response = row.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return message.get("content")


def response_id_and_text(provider: str, row: dict[str, Any]) -> tuple[str, str | None]:
    if provider == "gemini":
        return gemini_response_id(row), gemini_response_text(row)
    if provider == "openai":
        return openai_response_id(row), openai_response_text(row)
    raise ValueError(f"Unsupported provider: {provider}")


def load_batch_responses(
    provider: str,
    paths: list[Path],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    responses: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            request_id, text = response_id_and_text(provider, row)
            if request_id in responses:
                raise ValueError(f"Duplicate response id in response files: {request_id}")
            if text is None:
                failures.append({"request_id": request_id, "row": row})
                continue
            responses[request_id] = normalize_output(text)
    return responses, failures


def completed_log_name(meta: dict[str, Any]) -> str:
    return Path(str(meta["completed_log"]).replace("\\", "/")).name


def prompt_key_by_idx_var(
    data_dir: Path,
    dataset: str,
    problem_tag: str,
    player: str | None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[Any, str]]]:
    prompt_plan = _load_prompt_plan(data_dir, dataset, problem_tag, player)
    lookup: dict[tuple[str, str], tuple[Any, str]] = {}
    for entry in prompt_plan:
        lookup[(str(entry["idx"]), str(entry["variation_idx"]))] = (
            entry["idx"],
            entry["prompt"],
        )
    return prompt_plan, lookup


def main() -> None:
    args = parse_args()
    provider = normalize_provider(args.provider)

    if args.manifest_files:
        if len(args.manifest_files) != len(args.response_files):
            raise ValueError("--manifest-files must match --response-files length")
        manifest_paths = args.manifest_files
    else:
        manifest_paths = [
            infer_manifest_path(path, args.batch_root, args.model)
            for path in args.response_files
        ]

    missing_manifests = [path for path in manifest_paths if not path.exists()]
    if missing_manifests:
        raise FileNotFoundError(f"Missing manifest files: {missing_manifests}")

    manifest = load_manifest(manifest_paths)
    responses, failures = load_batch_responses(provider, args.response_files)

    unknown_response_ids = sorted(set(responses) - set(manifest))
    if unknown_response_ids:
        raise ValueError(
            f"{len(unknown_response_ids)} response ids were not found in the manifests. "
            f"First id: {unknown_response_ids[0]}"
        )

    additions_by_log: dict[str, dict[tuple[Any, str], list[tuple[int, str]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    jobs: dict[str, tuple[str, str, str | None]] = {}

    prompt_lookup_cache: dict[
        tuple[str, str, str | None],
        tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[Any, str]]],
    ] = {}

    for request_id, text in responses.items():
        meta = manifest[request_id]
        dataset = meta["dataset"]
        problem_tag = meta["problem_tag"]
        player = meta.get("bargaining_player")
        job_key = (dataset, problem_tag, player)

        if job_key not in prompt_lookup_cache:
            prompt_lookup_cache[job_key] = prompt_key_by_idx_var(
                args.data_dir, dataset, problem_tag, player
            )

        _, lookup = prompt_lookup_cache[job_key]
        prompt_key = lookup[(str(meta["idx"]), str(meta["variation_idx"]))]
        log_name = completed_log_name(meta)
        jobs[log_name] = job_key
        additions_by_log[log_name][prompt_key].append((int(meta["response_idx"]), text))

    output_dir = args.output_root / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    total_added = 0
    total_baseline_responses = 0
    total_skipped_existing = 0
    total_gaps = 0

    expected_jobs: dict[str, tuple[str, str, str | None]] = {}
    for dataset, problem_tag, player in _iter_jobs():
        suffix = _log_suffix(problem_tag, player)
        log_name = f"{dataset}_responses_{suffix}.jsonl"
        expected_jobs[log_name] = (dataset, problem_tag, player)

    for log_name, (dataset, problem_tag, player) in sorted(expected_jobs.items()):
        job_key = (dataset, problem_tag, player)
        if job_key not in prompt_lookup_cache:
            prompt_lookup_cache[job_key] = prompt_key_by_idx_var(
                args.data_dir, dataset, problem_tag, player
            )

        prompt_plan, _ = prompt_lookup_cache[job_key]
        existing_path = args.completed_dir / args.model / log_name
        responses_by_prompt = {
            key: list(values) for key, values in load_existing_logs(existing_path).items()
        }

        added = 0
        baseline_responses = sum(len(values) for values in responses_by_prompt.values())
        skipped_existing = 0
        gaps = 0
        prompt_additions = additions_by_log.get(log_name, {})
        for prompt_key, additions in prompt_additions.items():
            current = responses_by_prompt.setdefault(prompt_key, [])
            for response_idx, text in sorted(additions, key=lambda item: item[0]):
                if len(current) > response_idx:
                    skipped_existing += 1
                    continue
                if response_idx > len(current):
                    gaps += 1
                if len(current) >= args.target_responses:
                    skipped_existing += 1
                    continue
                current.append(text)
                added += 1

        output_path = output_dir / log_name
        save_jsonl(output_path, assemble_log_entries(prompt_plan, responses_by_prompt))
        total_added += added
        total_baseline_responses += baseline_responses
        total_skipped_existing += skipped_existing
        total_gaps += gaps
        summary.append(
            {
                "log": str(output_path),
                "baseline": str(existing_path),
                "baseline_responses": baseline_responses,
                "added": added,
                "skipped_existing": skipped_existing,
                "gap_warnings": gaps,
            }
        )

    failure_report = None
    if failures:
        failure_report = output_dir / "batch_response_failures.json"
        with open(failure_report, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "provider": provider,
                "model": args.model,
                "response_files": [str(path) for path in args.response_files],
                "manifest_files": [str(path) for path in manifest_paths],
                "responses_loaded": len(responses),
                "failures": len(failures),
                "failure_report": str(failure_report) if failure_report else None,
                "logs_written": len(summary),
                "total_baseline_responses": total_baseline_responses,
                "total_added": total_added,
                "total_skipped_existing": total_skipped_existing,
                "total_gap_warnings": total_gaps,
                "logs": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
