import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.batch_requests.batch_api_common import (
    object_to_dict,
    openai_api_key,
    safe_path_part,
)
from src.SAR.create_sar_batch import (
    RESPONSES_ENDPOINT,
    append_job_log,
    build_request_body,
    write_jsonl,
)

DEFAULT_PROMPT_FILE = Path("data/SAR_prompts/sar_map_saliency_eval_prompts.json")
DEFAULT_OUTPUT_DIR = Path("data/SAR_prompts/batch_requests")
DEFAULT_JOB_LOG = Path("data/SAR_prompts/job_ids.jsonl")


@dataclass(frozen=True)
class EvalPrompt:
    custom_id: str
    incident: str
    incident_index: int
    image_version: str
    map_style: str
    image_path: Path
    prompt_name: str
    prompt_version: str
    prompt: str


@dataclass(frozen=True)
class RunContext:
    timestamp: datetime
    model_slug: str
    run_slug: str
    batch_path: Path
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and submit an OpenAI Batch API job for SAR map saliency "
            "evaluation prompts."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="OpenAI model id to use in each /v1/responses request, e.g. gpt-5.5.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Prompt JSON written by the SAR map saliency evaluation prompt generator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where batch JSONL and manifest JSONL files are written.",
    )
    parser.add_argument(
        "--job-log",
        type=Path,
        default=DEFAULT_JOB_LOG,
        help="Append-only JSONL file for submitted batch job ids.",
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "original", "auto"],
        default="original",
        help="detail value for the input_image block.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional cap for total generated tokens.",
    )
    parser.add_argument(
        "--completion-window",
        choices=["24h"],
        default="24h",
        help="OpenAI Batch API completion window.",
    )
    return parser.parse_args()


def require_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Prompt row is missing non-empty string field {field!r}: {row}"
        )
    return value


def require_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise ValueError(f"Prompt row is missing integer field {field!r}: {row}")
    return value


def load_eval_prompts(path: Path) -> list[EvalPrompt]:
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)

    rows = raw.get("prompts") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON object with a prompts list")

    prompts: list[EvalPrompt] = []
    seen_custom_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object prompt row: {row!r}")

        custom_id = require_string(row, "custom_id")
        if custom_id in seen_custom_ids:
            raise ValueError(f"{path} contains duplicate custom_id: {custom_id}")
        seen_custom_ids.add(custom_id)

        image_path = Path(require_string(row, "image_path"))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing terrain map: {image_path}")

        prompts.append(
            EvalPrompt(
                custom_id=custom_id,
                incident=require_string(row, "incident"),
                incident_index=require_int(row, "incident_index"),
                image_version=require_string(row, "image_version"),
                map_style=require_string(row, "map_style"),
                image_path=image_path,
                prompt_name=require_string(row, "prompt_name"),
                prompt_version=require_string(row, "prompt_version"),
                prompt=require_string(row, "prompt"),
            )
        )

    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts


def upload_vision_files(
    client: OpenAI,
    prompts: Iterable[EvalPrompt],
) -> dict[Path, dict[str, str]]:
    uploads: dict[Path, dict[str, str]] = {}
    for prompt in prompts:
        if prompt.image_path in uploads:
            continue

        with prompt.image_path.open("rb") as image_file:
            uploaded_file = client.files.create(file=image_file, purpose="vision")

        uploads[prompt.image_path] = {
            "file_id": uploaded_file.id,
            "path": str(prompt.image_path),
        }
        print(f"uploaded {prompt.image_path} -> {uploaded_file.id}")

    return uploads


def build_run_context(args: argparse.Namespace) -> RunContext:
    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    model_slug = safe_path_part(args.model)
    run_slug = f"sar_map_saliency_eval_{model_slug}_{run_id}"
    model_dir = args.output_dir / model_slug
    return RunContext(
        timestamp=run_timestamp,
        model_slug=model_slug,
        run_slug=run_slug,
        batch_path=model_dir / f"{run_slug}.jsonl",
        manifest_path=model_dir / f"{run_slug}.manifest.jsonl",
    )


def build_requests_and_manifest(
    *,
    args: argparse.Namespace,
    prompts: list[EvalPrompt],
    uploads: dict[Path, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for prompt in prompts:
        upload = uploads[prompt.image_path]
        body = build_request_body(
            model=args.model,
            prompt_text=prompt.prompt,
            file_id=upload["file_id"],
            image_detail=args.image_detail,
            max_output_tokens=args.max_output_tokens,
        )
        requests.append(
            {
                "custom_id": prompt.custom_id,
                "method": "POST",
                "url": RESPONSES_ENDPOINT,
                "body": body,
            }
        )
        manifest.append(
            {
                "custom_id": prompt.custom_id,
                "model": args.model,
                "incident": prompt.incident,
                "incident_index": prompt.incident_index,
                "image_version": prompt.image_version,
                "map_style": prompt.map_style,
                "map_path": upload["path"],
                "vision_file_id": upload["file_id"],
                "prompt_name": prompt.prompt_name,
                "prompt_version": prompt.prompt_version,
                "endpoint": RESPONSES_ENDPOINT,
                "image_detail": args.image_detail,
            }
        )

    return requests, manifest


def submit_batch_file(
    *,
    client: OpenAI,
    args: argparse.Namespace,
    context: RunContext,
    request_count: int,
) -> tuple[Any, Any]:
    with context.batch_path.open("rb") as batch_file:
        uploaded_batch_file = client.files.create(file=batch_file, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_batch_file.id,
        endpoint=RESPONSES_ENDPOINT,
        completion_window=args.completion_window,
        metadata={
            "source": "src/SAR/create_sar_map_saliency_eval_batch.py",
            "model": args.model,
            "request_count": str(request_count),
        },
    )
    return uploaded_batch_file, batch


def main() -> None:
    args = parse_args()
    prompts = load_eval_prompts(args.prompt_file)
    context = build_run_context(args)
    client = OpenAI(api_key=openai_api_key(None))

    uploads = upload_vision_files(client, prompts)
    requests, manifest = build_requests_and_manifest(
        args=args,
        prompts=prompts,
        uploads=uploads,
    )
    request_count = write_jsonl(context.batch_path, requests)
    manifest_count = write_jsonl(context.manifest_path, manifest)
    if request_count != manifest_count:
        raise RuntimeError(
            f"Request count ({request_count}) does not match manifest count ({manifest_count})"
        )

    uploaded_batch_file, batch = submit_batch_file(
        client=client,
        args=args,
        context=context,
        request_count=request_count,
    )

    job_record = {
        "timestamp_utc": context.timestamp.isoformat(),
        "batch_id": batch.id,
        "status": batch.status,
        "model": args.model,
        "endpoint": RESPONSES_ENDPOINT,
        "completion_window": args.completion_window,
        "request_count": request_count,
        "batch_input_file_id": uploaded_batch_file.id,
        "batch_input_path": str(context.batch_path),
        "manifest_path": str(context.manifest_path),
        "prompt_file": str(args.prompt_file),
        "unique_image_uploads": len(uploads),
        "image_detail": args.image_detail,
        "source": "src/SAR/create_sar_map_saliency_eval_batch.py",
        "batch": object_to_dict(batch),
        "uploaded_batch_file": object_to_dict(uploaded_batch_file),
    }
    append_job_log(args.job_log, job_record)

    print(f"batch_id: {batch.id}")
    print(f"status: {batch.status}")
    print(f"requests: {request_count}")
    print(f"unique_image_uploads: {len(uploads)}")
    print(f"batch_input_path: {context.batch_path}")
    print(f"manifest_path: {context.manifest_path}")
    print(f"job_log: {args.job_log}")


if __name__ == "__main__":
    main()
