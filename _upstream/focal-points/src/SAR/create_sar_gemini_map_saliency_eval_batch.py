import argparse
import json
import mimetypes
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.batch_requests.batch_api_common import gemini_api_key, safe_path_part

DEFAULT_PROMPT_FILE = Path("data/SAR_prompts/sar_map_saliency_eval_prompts.json")
DEFAULT_OUTPUT_DIR = Path("data/SAR_prompts/batch_requests")
DEFAULT_MODEL = "gemini-2.5-pro"
SOURCE = "src/SAR/create_sar_gemini_map_saliency_eval_batch.py"


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
    upload_cache_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Gemini Batch API JSONL file for SAR map saliency "
            "evaluation prompts. This script prepares inputs only; it does not "
            "submit a Gemini batch job."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model id to target. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"SAR saliency prompt JSON. Defaults to {DEFAULT_PROMPT_FILE}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for batch JSONL and manifest files. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--upload-cache",
        type=Path,
        default=None,
        help=(
            "JSON cache for Gemini uploaded image file metadata. Defaults to a "
            "stable file in the model output folder."
        ),
    )
    parser.add_argument(
        "--refresh-uploads",
        action="store_true",
        help="Ignore cached Gemini image uploads and upload images again.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate prompts/images and print the planned output paths without writing requests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional Gemini generation temperature. Defaults to unset.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Optional Gemini max output tokens per prompt. Defaults to unset, "
            "matching the OpenAI saliency-eval batch generator."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Gemini API key override. Otherwise GEMINI_API_KEY or GOOGLE_API_KEY is used.",
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


def build_run_context(args: argparse.Namespace) -> RunContext:
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    model_slug = safe_path_part(args.model)
    run_slug = f"sar_map_saliency_eval_{model_slug}_{run_id}"
    model_dir = args.output_dir / model_slug
    upload_cache_path = (
        args.upload_cache
        if args.upload_cache is not None
        else model_dir / f"sar_map_saliency_eval_{model_slug}.gemini_uploads.json"
    )
    return RunContext(
        timestamp=timestamp,
        model_slug=model_slug,
        run_slug=run_slug,
        batch_path=model_dir / f"{run_slug}.jsonl",
        manifest_path=model_dir / f"{run_slug}.manifest.jsonl",
        upload_cache_path=upload_cache_path,
    )


def mime_type_for(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        raise ValueError(f"Could not infer MIME type for {path}")
    return mime_type


def read_upload_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    uploads = raw.get("uploads")
    if not isinstance(uploads, dict):
        raise ValueError(f"{path} must contain an uploads object")
    return uploads


def write_upload_cache(path: Path, uploads: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": SOURCE,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "uploads": uploads,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cache_key(path: Path) -> str:
    return str(path.as_posix())


def cached_upload_is_current(path: Path, upload: dict[str, Any]) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    return (
        upload.get("size_bytes") == stat.st_size
        and upload.get("mtime_ns") == stat.st_mtime_ns
        and upload.get("mime_type") == mime_type_for(path)
        and isinstance(upload.get("file_uri"), str)
        and isinstance(upload.get("file_name"), str)
    )


def upload_images(
    *,
    prompts: Iterable[EvalPrompt],
    cache_path: Path,
    refresh_uploads: bool,
    api_key: str | None,
) -> dict[Path, dict[str, Any]]:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed in this Python environment. "
            "Use .venv/Scripts/python.exe or install requirements.txt."
        ) from exc

    key = gemini_api_key(api_key)
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required. "
            "This script will still not submit a batch job; it only uploads image files "
            "and writes the batch JSONL."
        )

    client = genai.Client(api_key=key)
    cached = {} if refresh_uploads else read_upload_cache(cache_path)
    uploads_by_path: dict[Path, dict[str, Any]] = {}

    unique_paths = sorted({prompt.image_path for prompt in prompts}, key=lambda p: p.as_posix())
    for image_path in unique_paths:
        key_path = cache_key(image_path)
        cached_upload = cached.get(key_path)
        if cached_upload and cached_upload_is_current(image_path, cached_upload):
            uploads_by_path[image_path] = cached_upload
            print(f"reused {image_path} -> {cached_upload['file_name']}")
            continue

        uploaded = client.files.upload(file=image_path)
        stat = image_path.stat()
        upload = {
            "path": str(image_path),
            "file_name": uploaded.name,
            "file_uri": uploaded.uri,
            "mime_type": uploaded.mime_type or mime_type_for(image_path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        cached[key_path] = upload
        uploads_by_path[image_path] = upload
        print(f"uploaded {image_path} -> {uploaded.name}")
        write_upload_cache(cache_path, cached)

    return uploads_by_path


def file_uri_image_part(upload: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_data": {
            "file_uri": upload["file_uri"],
            "mime_type": upload["mime_type"],
        }
    }


def request_for_prompt(
    *,
    prompt: EvalPrompt,
    args: argparse.Namespace,
    upload: dict[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    file_uri_image_part(upload),
                    {"text": prompt.prompt},
                ],
            }
        ],
    }
    generation_config: dict[str, Any] = {}
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
    if args.max_output_tokens is not None:
        generation_config["max_output_tokens"] = args.max_output_tokens
    if generation_config:
        request["generation_config"] = generation_config

    return {
        "key": prompt.custom_id,
        "request": request,
    }


def manifest_for_prompt(
    *,
    prompt: EvalPrompt,
    args: argparse.Namespace,
    upload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": prompt.custom_id,
        "custom_id": prompt.custom_id,
        "provider": "gemini",
        "model": args.model,
        "incident": prompt.incident,
        "incident_index": prompt.incident_index,
        "image_version": prompt.image_version,
        "map_style": prompt.map_style,
        "map_path": str(prompt.image_path),
        "prompt_name": prompt.prompt_name,
        "prompt_version": prompt.prompt_version,
        "image_mode": "file-uri",
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "gemini_file_name": upload["file_name"],
        "gemini_file_uri": upload["file_uri"],
        "gemini_file_mime_type": upload["mime_type"],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize_prompts(prompts: list[EvalPrompt]) -> dict[str, Any]:
    image_version_counts: dict[str, int] = {}
    prompt_name_counts: dict[str, int] = {}
    for prompt in prompts:
        image_version_counts[prompt.image_version] = image_version_counts.get(prompt.image_version, 0) + 1
        prompt_name_counts[prompt.prompt_name] = prompt_name_counts.get(prompt.prompt_name, 0) + 1
    return {
        "prompt_count": len(prompts),
        "incident_count": len({prompt.incident_index for prompt in prompts}),
        "unique_image_count": len({prompt.image_path for prompt in prompts}),
        "image_version_counts": dict(sorted(image_version_counts.items())),
        "prompt_name_counts": dict(sorted(prompt_name_counts.items())),
    }


def main() -> None:
    args = parse_args()
    prompts = load_eval_prompts(args.prompt_file)
    context = build_run_context(args)
    summary = summarize_prompts(prompts)

    if args.validate_only:
        print(json.dumps(
            {
                "valid": True,
                "summary": summary,
                "batch_path": str(context.batch_path),
                "manifest_path": str(context.manifest_path),
                "upload_cache_path": str(context.upload_cache_path),
                "source": SOURCE,
            },
            indent=2,
            ensure_ascii=False,
        ))
        return

    uploads = upload_images(
        prompts=prompts,
        cache_path=context.upload_cache_path,
        refresh_uploads=args.refresh_uploads,
        api_key=args.api_key,
    )

    context.batch_path.parent.mkdir(parents=True, exist_ok=True)
    request_count = 0
    manifest_count = 0
    with context.batch_path.open("w", encoding="utf-8") as batch_file, context.manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_file:
        for prompt in prompts:
            upload = uploads[prompt.image_path]
            request = request_for_prompt(prompt=prompt, args=args, upload=upload)
            manifest_row = manifest_for_prompt(prompt=prompt, args=args, upload=upload)
            batch_file.write(json.dumps(request, ensure_ascii=False) + "\n")
            manifest_file.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            request_count += 1
            manifest_count += 1

    if request_count != manifest_count:
        raise RuntimeError(
            f"Request count ({request_count}) does not match manifest count ({manifest_count})"
        )

    print(json.dumps(
        {
            "submitted": False,
            "model": args.model,
            "summary": summary,
            "image_mode": "file-uri",
            "batch_path": str(context.batch_path),
            "manifest_path": str(context.manifest_path),
            "upload_cache_path": str(context.upload_cache_path),
            "source": SOURCE,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
